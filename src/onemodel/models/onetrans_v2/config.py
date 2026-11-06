"""
OneTransV2Config
"""
from dataclasses import asdict, dataclass, field
import json

import gin


@gin.configurable
@dataclass
class OneTransV2Config:
    model: str = "onetrans_v2"
    dim: int = 64
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2
    ff_dim: int = 96
    experts: int = 4
    top_k: int = 2
    routed_scale: float = 1.0
    fine_tokens: int = 3
    # MixedLinear 的输出宽度；None 表示 dim，K/V 使用配置副本指定较小宽度。
    projection_dim: int | None = None
    # 时间戳单位为秒；周期范围沿用旧复现的工程设置，可按数据调整。
    temporal_min_period: float = 60.0
    temporal_max_period: float = 30 * 86400.0
    tasks: list = field(default_factory=lambda: ["click", "long_view", "like"])
    task_weights: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    # Public-video adaptation: click / like / long_view (parallel),
    # watch-ratio bin (dependent). They are NOT order/GMV/supply labels.
    decision_sizes: list = field(default_factory=lambda: [2, 2, 2, 4])
    sid_levels: int = 3
    codebook_size: int = 32
    sid_weights: list = field(default_factory=lambda: [1.0, 0.7, 0.5])
    recall_weight: float = 1.0
    rank_weight: float = 1.0
    pre_weight: float = 1.0
    kd_weight: float = 0.2
    kd_temperature: float = 1.0
    kd_momentum: float = 0.9999
    base_dim: int = 64
    base_layers: int = 2
    mup: bool = True
    history_limit: int = 128
    exposures_per_window: int = 8
    batch_size: int = 4
    epochs: int = 3
    warmup_epochs: int = 1
    lr: float = 0.001
    weight_decay: float = 0.01
    seed: int = 42
    eval_seed: int = 2026
    device: str = "cpu"
    grad_clip: float = 1.0
    beam_size: int = 32
    pre_keep: int = 16
    metric_ks: list = field(default_factory=lambda: [5, 10, 20])
    eval_requests: int = 100
    ranking_eval_limit: int = 0
    selection_metric: str = "ranking/macro_auc"
    threads: int = 4

    @property
    def retrieval_types(self):
        # 0 留给历史；随后为 CTX、DTp、DTd 和各 SID 位置。
        return list(range(1, 4 + self.sid_levels))

    @property
    def pre_type(self):
        return 4 + self.sid_levels

    @property
    def fine_types(self):
        return list(range(self.pre_type + 1, self.pre_type + 1 + self.fine_tokens))

    @property
    def num_types(self):
        return self.pre_type + 1 + self.fine_tokens

    def validate(self):
        if self.model not in {"onetrans_v2", "unir2"}:
            raise ValueError("model must be onetrans_v2 or unir2")
        if min(self.dim, self.heads, self.kv_heads, self.ff_dim, self.sid_levels, self.codebook_size,
               self.history_limit, self.threads) < 1:
            raise ValueError("dimensions and vocabulary sizes must be positive")
        if self.dim % self.heads or self.heads % self.kv_heads:
            raise ValueError("dim/heads and heads/kv_heads must be integers")
        if (self.dim // self.heads) % 2:
            raise ValueError("rotary head dimension must be even")
        if self.projection_dim is not None and self.projection_dim < 1:
            raise ValueError("projection_dim must be positive")
        if not 0 < self.temporal_min_period <= self.temporal_max_period:
            raise ValueError("invalid temporal period range")
        if len(self.decision_sizes) != 4 or any(s < 2 for s in self.decision_sizes):
            raise ValueError("use three parallel decisions and one dependent decision")
        if len(self.tasks) != len(self.task_weights):
            raise ValueError("one weight per task required")
        if "click" not in self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be unique and include click for the public recall protocol")
        if any(w < 0 for w in self.task_weights) or not any(self.task_weights):
            raise ValueError("task weights must be nonnegative with a positive total")
        if len(self.sid_weights) != self.sid_levels:
            raise ValueError("one weight per SID level required")
        if not 1 <= self.top_k <= self.experts:
            raise ValueError("invalid routed-expert top_k")
        if min(self.layers, self.fine_tokens, self.batch_size, self.epochs,
               self.exposures_per_window, self.beam_size, self.pre_keep) < 1:
            raise ValueError("counts must be positive")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must be smaller than epochs")
        if not 0 <= self.kd_momentum < 1 or self.kd_temperature <= 0:
            raise ValueError("invalid distillation EMA/temperature")
        if self.eval_requests < 0 or self.ranking_eval_limit < 0:
            raise ValueError("evaluation limits must be nonnegative")
        return self

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f)).validate()

    def to_dict(self):
        return asdict(self)
