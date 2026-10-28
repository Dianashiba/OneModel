"""Raw data semantics are shared; each model may define its own tensor batch layout."""

from dataclasses import dataclass, field
from typing import Any

Batch = dict[str, Any]


@dataclass
class HistoryEvent:
    item_id: int
    timestamp: int
    behavior: str
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class Exposure:
    user_id: str
    request_id: str
    request_timestamp: int
    item_id: int
    history: list[HistoryEvent] = field(default_factory=list)
    labels: dict[str, float] = field(default_factory=dict)
    label_mask: dict[str, bool] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    stage: str | None = None
    history_anchor: int | None = None


@dataclass
class UserWindow:
    user_id: str
    history: list[HistoryEvent]
    exposures: list[Exposure]
