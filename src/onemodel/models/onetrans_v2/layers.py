"""Stage visibility, mixed parameterization and OneTrans-V2 blocks.

Keep the attention/autograd boundaries in the model. Extract reusable components
to onemodel.modules only after their interfaces are established.
"""
import math
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from onemodel.models.onetrans_v2.config import OneTransV2Config
from onemodel.modules.normalize import RMSNorm


def _check_input(x, dim):
    if x.ndim != 3 or x.shape[-1] != dim:
        raise ValueError(f"x must have shape [B, T, {dim}]")


def _check_types(type_ids, length, num_types):
    """同一次调用的所有样本使用相同的语义槽位布局。"""
    if isinstance(type_ids, int):
        ids = [type_ids]
    elif isinstance(type_ids, (list, tuple)):
        if len(type_ids) != length:
            raise ValueError("one type_id is required per token position")
        ids = type_ids
    else:
        raise TypeError("type_ids must be an int, list, or tuple")
    for type_id in ids:
        if not isinstance(type_id, int) or not 0 <= type_id < num_types:
            raise ValueError(f"invalid token type: {type_id}")


def build_history_mask(valid):
    """valid: [B, L]；True 表示有效历史 key。"""
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean [B, L] tensor")
    length = valid.shape[1]
    causal = torch.ones(length, length, device=valid.device, dtype=torch.bool).tril()
    return causal[None] & valid[:, None, :]


def build_stage_mask(visible, stage_len):
    """一个 batch 行代表同一曝光的同一阶段；只读取可见历史和当前阶段前缀。"""
    if visible.ndim != 2 or visible.dtype != torch.bool:
        raise ValueError("visible must be a boolean [B, L] tensor")
    own = torch.ones(stage_len, stage_len, device=visible.device, dtype=torch.bool).tril()
    own = own[None].expand(visible.shape[0], -1, -1)
    return torch.cat((visible[:, None, :].expand(-1, stage_len, -1), own), dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, cfg: OneTransV2Config):
        super().__init__()
        self.up = nn.Linear(cfg.dim, 2 * cfg.ff_dim, bias=False)
        self.down = nn.Linear(cfg.ff_dim, cfg.dim, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, -1)
        return self.down(F.silu(a) * b)


class MixedLinear(nn.Module):
    """每个语义类型持有独立投影；历史的所有位置使用类型 0。"""

    def __init__(self, cfg: OneTransV2Config):
        super().__init__()
        self.dim = cfg.dim
        self.out_dim = cfg.dim if cfg.projection_dim is None else cfg.projection_dim
        self.num_types = cfg.num_types
        # 保留原来注意力投影的 bias 设置。
        self.linears = nn.ModuleList([
            nn.Linear(self.dim, self.out_dim) for _ in range(self.num_types)
        ])

    def forward(self, x, type_ids=0):
        _check_input(x, self.dim)
        _check_types(type_ids, x.shape[1], self.num_types)
        if isinstance(type_ids, int):
            return self.linears[type_ids](x)
        if x.shape[1] == 0:
            return x.new_empty(x.shape[0], 0, self.out_dim)
        return torch.cat([
            self.linears[type_id](x[:, i:i + 1, :])
            for i, type_id in enumerate(type_ids)
        ], dim=1)


class GatedAttention(nn.Module):
    def __init__(self, cfg: OneTransV2Config):
        super().__init__()
        self.d_model = cfg.dim
        self.num_heads = cfg.heads
        self.num_kv_heads = cfg.kv_heads

        if min(self.d_model, self.num_heads, self.num_kv_heads) < 1:
            raise ValueError("attention dimensions must be positive")
        if self.num_heads % self.num_kv_heads or self.d_model % self.num_heads:
            raise ValueError("dim/heads and heads/kv_heads must be integers")

        self.head_dim = self.d_model // self.num_heads
        self.d_kv = self.num_kv_heads * self.head_dim
        self.groups = self.num_heads // self.num_kv_heads

        if self.head_dim % 2:
            raise ValueError("rotary head dimension must be even")
        if not 0 < cfg.temporal_min_period <= cfg.temporal_max_period:
            raise ValueError("invalid temporal period range")

        model_cfg = replace(cfg, projection_dim=self.d_model)
        kv_cfg = replace(cfg, projection_dim=self.d_kv)
        self.Wq = MixedLinear(model_cfg)
        self.Wk = MixedLinear(kv_cfg)
        self.Wv = MixedLinear(kv_cfg)
        self.Wg = MixedLinear(model_cfg)
        self.Wo = nn.Linear(self.d_model, self.d_model)

        self.Q_norm = RMSNorm(self.head_dim)
        self.K_norm = RMSNorm(self.head_dim)
        self.register_buffer("temporal_periods", torch.logspace(
            math.log10(cfg.temporal_min_period), math.log10(cfg.temporal_max_period),
            self.head_dim // 2, dtype=torch.float64,
        ))

    def _head_norm(self, x, norm):
        return norm(x)

    def _temporal_rotate(self, x, times):
        """x: [B, H, T, dh]，times: [B, T]，时间戳单位为秒。"""
        if times.shape != (x.shape[0], x.shape[2]):
            raise ValueError("times must have shape [B, T]")
        # 用 double 计算绝对时间的相位，避免 Unix 时间戳在 float32 中损失秒精度。
        angles = (times.double()[..., None] * (2 * math.pi / self.temporal_periods))
        angles = angles.remainder(2 * math.pi)
        co = angles.cos().to(x.dtype).unsqueeze(1)
        si = angles.sin().to(x.dtype).unsqueeze(1)
        a, b = x.chunk(2, dim=-1)
        return torch.cat((a * co - b * si, b * co + a * si), dim=-1)

    def project(self, x, token_type_ids=0):
        _check_input(x, self.d_model)
        B, L, D = x.shape

        Q = self.Wq(x, token_type_ids)
        K = self.Wk(x, token_type_ids)
        V = self.Wv(x, token_type_ids)
        G = self.Wg(x, token_type_ids)

        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        Q = self._head_norm(Q, self.Q_norm)
        K = self._head_norm(K, self.K_norm)
        return Q, K, V, G

    def forward(self, x, mask=None, token_type_ids=0, *, history_kv=None,
                query_times=None, history_times=None, return_kv=False):
        """返回注意力分支；return_kv=True 时额外返回当前输入的未重复 KV。

        mask: bool [B, Tq, history_length + Tq]，True 表示可见。
        KV 保留计算图，以便阶段损失回传至历史编码。
        """
        Q, K, V, G = self.project(x, token_type_ids)
        B, L, D = x.shape
        current_kv = (K, V)
        history_len = 0
        if history_kv is not None:
            hk, hv = history_kv
            if (hk.ndim != 4 or hk.shape != hv.shape or hk.shape[0] != B
                    or hk.shape[1] != self.num_kv_heads or hk.shape[3] != self.head_dim):
                raise ValueError("history KV must have shape [B, kv_heads, L, head_dim]")
            history_len = hk.shape[2]
            K = torch.cat((hk, K), dim=2)
            V = torch.cat((hv, V), dim=2)

        if (query_times is None) != (history_times is None):
            raise ValueError("query_times and history_times must be supplied together")
        if query_times is not None and history_kv is None:
            raise ValueError("time encoding requires history_kv")

        K = torch.repeat_interleave(K, self.groups, dim=1)
        V = torch.repeat_interleave(V, self.groups, dim=1)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        if query_times is not None:
            # 只替换 stage -> history 分数；阶段内部、历史自注意力不旋转。
            tq = self._temporal_rotate(Q, query_times)
            tk = self._temporal_rotate(K[:, :, :history_len, :], history_times)
            temporal_scores = tq @ tk.transpose(-2, -1) / math.sqrt(self.head_dim)
            scores = torch.cat((temporal_scores, scores[..., history_len:]), dim=-1)

        if mask is not None:
            if mask.dtype != torch.bool or mask.shape != (B, L, K.shape[2]):
                raise ValueError("mask must be boolean with shape [B, Tq, Tk]")
            scores = scores.masked_fill(
                ~mask.unsqueeze(1), float("-inf")
            )
            # 全遮蔽行先变为有限值，softmax 后再将其概率清零。
            scores = scores.masked_fill(~mask.any(-1)[:, None, :, None], 0.0)

        probabilities = torch.softmax(scores.float(), dim=-1).to(V.dtype)
        if mask is not None:
            probabilities = probabilities.masked_fill(~mask.unsqueeze(1), 0.0)
        attn_weight = probabilities @ V
        attn_weight = attn_weight.transpose(1, 2).contiguous().view(B, L, D)

        output = self.Wo(attn_weight * torch.sigmoid(G))
        return (output, current_kv) if return_kv else output


class SparseMoE(nn.Module):
    def __init__(self, cfg: OneTransV2Config):
        super().__init__()

        self.dim = cfg.dim
        self.hidden_dim = cfg.ff_dim
        self.num_experts = cfg.experts
        self.top_k = cfg.top_k
        self.routed_scale = cfg.routed_scale
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must be between 1 and experts")

        # 共享专家
        self.shared = SwiGLU(cfg)

        self.experts = nn.ModuleList([
            SwiGLU(cfg)
            for _ in range(self.num_experts)
        ])

        self.router = nn.Linear(self.dim, self.num_experts, bias=False)

    def forward(self, x):
        _check_input(x, self.dim)
        B, L, D = x.shape

        tokens = x.reshape(-1, self.dim)

        scores = self.router(tokens).float().sigmoid()

        weights, expert_ids = scores.topk(self.top_k, dim=-1)

        weights = weights / weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        weights = weights * self.routed_scale

        output = self.shared(tokens).float()

        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(
                expert_ids == expert_id
            )

            if token_ids.numel() == 0:
                continue

            expert_output = expert(tokens[token_ids]).float()
            expert_weights = weights[token_ids, slots].unsqueeze(-1)
            contribution = expert_output * expert_weights

            output = output.index_add(
                0, token_ids, contribution
            )

        return output.contiguous().view(B, L, D).to(x.dtype)


class MixedFFN(nn.Module):
    """根据固定语义类型选择 MoE；专家路由仍由各自的 SparseMoE 完成。"""

    def __init__(self, cfg: OneTransV2Config):
        super().__init__()
        self.dim = cfg.dim
        self.num_types = cfg.num_types
        self.ffns = nn.ModuleList([SparseMoE(cfg) for _ in range(self.num_types)])

    def forward(self, x, type_ids=0):
        _check_input(x, self.dim)
        _check_types(type_ids, x.shape[1], self.num_types)
        if isinstance(type_ids, int):
            return self.ffns[type_ids](x)
        if x.shape[1] == 0:
            return x
        return torch.cat([
            self.ffns[type_id](x[:, i:i + 1, :])
            for i, type_id in enumerate(type_ids)
        ], dim=1)


class OnetransBlock(nn.Module):
    """Apply the attention and MoE branches with pre-norm and scaled residuals."""

    def __init__(self, cfg: OneTransV2Config):
        super().__init__()
        if cfg.layers < 1:
            raise ValueError("layers must be positive for residual scaling")

        self.dim = cfg.dim
        self.gamma = 1 / math.sqrt(2 * cfg.layers)
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
        self.attention = GatedAttention(cfg)
        self.ffn = MixedFFN(cfg)

    def forward(self, x, attn_mask=None, token_type_ids=0, time_info=None, *,
                history_kv=None, return_kv=False):
        """time_info 可为含 query_times [B,T]、history_times [B,L] 的字典。

        默认返回 Tensor；history 路径通过 return_kv=True 获取本层缓存。
        未提供 mask 时是全可见注意力；history/stage 会构建各自的因果 mask。
        """
        token_type_ids = 0 if token_type_ids is None else token_type_ids
        times = {} if time_info is None else time_info
        attention_output, current_kv = self.attention(
            self.norm1(x), mask=attn_mask, token_type_ids=token_type_ids,
            history_kv=history_kv, query_times=times.get("query_times"),
            history_times=times.get("history_times"), return_kv=True,
        )

        # Scale the entire branch output, including the output projection bias.
        x = x + self.gamma * attention_output
        x = x + self.gamma * self.ffn(self.norm2(x), token_type_ids)
        return (x, current_kv) if return_kv else x

    def history(self, x, valid):
        """历史仅使用类型 0；返回输出和供当前层阶段 token 读取的 KV。"""
        return self(x, build_history_mask(valid), token_type_ids=0, return_kv=True)

    def stage(self, x, type_ids, cache, visible, request_times=None, history_times=None):
        """cache 的 batch 行需由调用者按曝光所属用户选好，且必须来自同一层。"""
        time_info = None
        if request_times is not None or history_times is not None:
            if request_times is None or history_times is None:
                raise ValueError("request_times and history_times must be supplied together")
            if request_times.shape != (x.shape[0],):
                raise ValueError("request_times must have shape [B]")
            time_info = {
                "query_times": request_times[:, None].expand(-1, x.shape[1]),
                "history_times": history_times,
            }
        return self(x, build_stage_mask(visible, x.shape[1]), type_ids, time_info,
                    history_kv=cache)
