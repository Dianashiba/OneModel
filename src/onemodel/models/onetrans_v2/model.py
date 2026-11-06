"""Compose OneTrans-V2 blocks and heads over already-tokenized features.

Raw feature/ID tokenization belongs to Inputs, outside this model. forward uses:
  history_tokens:    [B, L, D], including an always-valid start token at index 0
  history_valid:     bool [B, L]
  history_times:     [B, L], timestamps in seconds, including the start position
  owners:            long [E], user-window index for each exposure
  anchors:           long [E], last visible history index, including start
  times:             [E], request timestamps in seconds
  retrieval_context: [E, D], encoded retrieval-specific context features
  pre_tokens:        [E, 1, D]
  fine_tokens:       [E, cfg.fine_tokens, D]
  decisions:         long [E, 4], teacher-forcing decision labels
  sid_targets:       long [E, cfg.sid_levels], target semantic identifiers

forward is the joint teacher-forcing training path. Inference uses encode,
select_context, score, retrieval and retrieval_logits with predicted prefixes.
"""

import gin
import torch
from torch import Tensor, nn

from onemodel.data.schemas import Batch
from onemodel.models.onetrans_v2.config import OneTransV2Config
from onemodel.models.onetrans_v2.layers import OnetransBlock
from onemodel.models.outputs import ModelOutput
from onemodel.modules.normalize import RMSNorm


def select_context(context: dict, owners: Tensor) -> dict:
    """Expand shared user windows into exposure/beam rows without detaching KV."""
    if owners.ndim != 1 or owners.dtype != torch.long:
        raise ValueError("owners must be a long [E] tensor")
    if ((owners < 0) | (owners >= context["valid"].shape[0])).any():
        raise ValueError("owners must index an existing user window")
    return {
        "cache": [
            (k.index_select(0, owners), v.index_select(0, owners))
            for k, v in context["cache"]
        ],
        "valid": context["valid"].index_select(0, owners),
        "times": context["times"].index_select(0, owners),
    }


@gin.configurable
class OneTransV2(nn.Module):
    def __init__(self, cfg: OneTransV2Config) -> None:
        super().__init__()
        self.cfg = cfg.validate()
        self.blocks = nn.ModuleList([OnetransBlock(cfg) for _ in range(cfg.layers)])
        self.final_norm = RMSNorm(cfg.dim)

        self.decision_embeddings = nn.ModuleList([
            nn.Embedding(size, cfg.dim) for size in cfg.decision_sizes
        ])
        self.sid_embeddings = nn.ModuleList([
            nn.Embedding(cfg.codebook_size, cfg.dim) for _ in range(cfg.sid_levels)
        ])
        self.decision_heads = nn.ModuleList([
            nn.Linear(cfg.dim, size) for size in cfg.decision_sizes
        ])
        self.sid_heads = nn.ModuleList([
            nn.Linear(cfg.dim, cfg.codebook_size) for _ in range(cfg.sid_levels)
        ])
        self.pre_head = nn.Linear(cfg.dim, len(cfg.tasks))
        self.fine_head = nn.Sequential(
            nn.Linear(cfg.fine_tokens * cfg.dim, cfg.dim),
            nn.SiLU(),
            nn.Linear(cfg.dim, len(cfg.tasks)),
        )

    def encode(self, batch: Batch) -> dict:
        """Encode history once; each layer stores KV projected from its own input."""
        x = batch["history_tokens"]
        valid = batch["history_valid"]
        times = batch["history_times"]
        if x.ndim != 3 or x.shape[-1] != self.cfg.dim:
            raise ValueError("history_tokens must have shape [B, L, dim]")
        if valid.dtype != torch.bool or valid.shape != x.shape[:2]:
            raise ValueError("history_valid must be boolean with shape [B, L]")
        if times.shape != valid.shape:
            raise ValueError("history_times must have shape [B, L]")
        if x.shape[1] == 0 or not valid[:, 0].all():
            raise ValueError("history must include an always-valid start token at index 0")

        cache = []
        for block in self.blocks:
            x, kv = block.history(x, valid)
            cache.append(kv)
        return {"cache": cache, "valid": valid, "times": times}

    def stage(self, x: Tensor, type_ids, context: dict,
              anchors: Tensor, times: Tensor) -> Tensor:
        """Run one stage/exposure per row over its causal history prefix."""
        if x.ndim != 3 or x.shape[-1] != self.cfg.dim:
            raise ValueError("stage tokens must have shape [E, T, dim]")
        rows = x.shape[0]
        valid = context["valid"]
        if valid.shape[0] != rows:
            raise ValueError("select context by owners before calling stage")
        if anchors.shape != (rows,) or anchors.dtype != torch.long:
            raise ValueError("anchors must be a long [E] tensor")
        if times.shape != (rows,):
            raise ValueError("request times must have shape [E]")
        if ((anchors < 0) | (anchors >= valid.shape[1])).any():
            raise ValueError("anchors must index history, including the start token")
        if len(context["cache"]) != len(self.blocks):
            raise ValueError("context must contain one KV pair per block")

        positions = torch.arange(valid.shape[1], device=x.device)
        visible = valid & (positions[None, :] <= anchors[:, None])
        for block, cache in zip(self.blocks, context["cache"]):
            x = block.stage(x, type_ids, cache, visible, times, context["times"])
        return self.final_norm(x)

    def retrieval(self, ctx: Tensor, context: dict, anchors: Tensor, times: Tensor,
                  decisions: Tensor | None = None, sids: Tensor | None = None) -> Tensor:
        """Build CTX -> DTp -> DTd -> SID prefix and return hidden states.

        decisions may have 3 columns (parallel decisions) or 4 (also dependent).
        sids is an optional known/predicted prefix, available only after DTd.
        No ground-truth labels are read implicitly by this inference-capable API.
        """
        if ctx.ndim != 2 or ctx.shape[1] != self.cfg.dim:
            raise ValueError("retrieval context must have shape [E, dim]")
        tokens = [ctx]
        if decisions is not None:
            if (decisions.ndim != 2 or decisions.shape[0] != ctx.shape[0]
                    or decisions.shape[1] not in (3, 4) or decisions.dtype != torch.long):
                raise ValueError("decisions must be a long [E, 3] or [E, 4] tensor")
            # 三个并行决策合成一个 token，不是三个依次输入的位置。
            parallel = sum(self.decision_embeddings[i](decisions[:, i]) for i in range(3))
            tokens.append(parallel)
            if decisions.shape[1] == 4:
                tokens.append(self.decision_embeddings[3](decisions[:, 3]))
        if sids is not None:
            if decisions is None or decisions.shape[1] != 4:
                raise ValueError("SID prefixes require all four decisions")
            if (sids.ndim != 2 or sids.shape[0] != ctx.shape[0]
                    or sids.shape[1] > self.cfg.sid_levels or sids.dtype != torch.long):
                raise ValueError("sids must be a long [E, prefix_length <= sid_levels] tensor")
            tokens.extend(self.sid_embeddings[i](sids[:, i]) for i in range(sids.shape[1]))

        x = torch.stack(tokens, dim=1)
        return self.stage(x, self.cfg.retrieval_types[:x.shape[1]], context, anchors, times)

    def retrieval_logits(self, hidden: Tensor) -> dict[str, Tensor]:
        """Read next-token predictions at their preceding positions, as raw logits."""
        if (hidden.ndim != 3 or hidden.shape[-1] != self.cfg.dim
                or not 1 <= hidden.shape[1] <= len(self.cfg.retrieval_types)):
            raise ValueError("hidden must contain a valid retrieval prefix")
        # CTX predicts all three parallel decisions.
        logits = {f"decision_{i}": self.decision_heads[i](hidden[:, 0]) for i in range(3)}
        if hidden.shape[1] >= 2:
            # DTp predicts the dependent decision.
            logits["decision_3"] = self.decision_heads[3](hidden[:, 1])
        for i, head in enumerate(self.sid_heads):
            # DTd predicts SID0; SID(i-1) predicts SIDi.
            if hidden.shape[1] > 2 + i:
                logits[f"sid_{i}"] = head(hidden[:, 2 + i])
        return logits

    def score(self, tokens: Tensor, context: dict, anchors: Tensor,
              times: Tensor, stage: str = "fine") -> Tensor:
        """Score already-tokenized candidates; return [E, number_of_tasks] logits."""
        if stage == "pre":
            type_ids, head = [self.cfg.pre_type], self.pre_head
        elif stage == "fine":
            type_ids, head = self.cfg.fine_types, self.fine_head
        else:
            raise ValueError("stage must be 'pre' or 'fine'")
        if tokens.ndim != 3 or tokens.shape[1] != len(type_ids):
            raise ValueError(f"{stage} tokens must have shape [E, {len(type_ids)}, dim]")
        hidden = self.stage(tokens, type_ids, context, anchors, times)
        return head(hidden.flatten(1))

    def forward(self, batch: Batch) -> ModelOutput:
        """Joint teacher-forcing forward; loss calculation remains in objectives.py."""
        context = select_context(self.encode(batch), batch["owners"])
        anchors, times = batch["anchors"], batch["times"]
        rows = batch["owners"].shape[0]
        decisions, targets = batch["decisions"], batch["sid_targets"]
        if decisions.shape != (rows, 4) or decisions.dtype != torch.long:
            raise ValueError("training decisions must be a long [E, 4] tensor")
        if targets.shape != (rows, self.cfg.sid_levels) or targets.dtype != torch.long:
            raise ValueError("sid_targets must be a long [E, sid_levels] tensor")
        if ((targets < 0) | (targets >= self.cfg.codebook_size)).any():
            raise ValueError("sid_targets must index the SID codebook")

        # 最后一个 SID 是预测目标，不必输入；保留前缀的因果错位。
        hidden = self.retrieval(
            batch["retrieval_context"], context, anchors, times,
            decisions=decisions, sids=targets[:, :-1],
        )
        logits = self.retrieval_logits(hidden)
        logits["pre"] = self.score(batch["pre_tokens"], context, anchors, times, "pre")
        logits["fine"] = self.score(batch["fine_tokens"], context, anchors, times, "fine")
        return ModelOutput(
            logits=logits,
            states={"pre_tasks": tuple(self.cfg.tasks), "fine_tasks": tuple(self.cfg.tasks)},
        )
