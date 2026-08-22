"""Proof-bound distributed gradient integrity quorum controller."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from statistics import median
from typing import Iterable

EVIDENCE_STATE = "DETERMINISTIC_GRADIENT_QUORUM_MODEL"


class QuorumAction(str, Enum):
    COMMIT = "COMMIT"
    ISOLATE = "ISOLATE"
    ABORT = "ABORT"


@dataclass(frozen=True)
class GradientReport:
    rank: int
    step: int
    model_version: str
    grad_norm: float
    finite: bool = True
    overflow: bool = False
    heartbeat_age_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.rank < 0 or self.step < 0:
            raise ValueError("rank and step must be non-negative")
        if not self.model_version.strip():
            raise ValueError("model_version must be non-empty")
        if not isfinite(self.heartbeat_age_ms) or self.heartbeat_age_ms < 0:
            raise ValueError("heartbeat_age_ms must be finite and non-negative")


@dataclass(frozen=True)
class QuorumPolicy:
    world_size: int
    minimum_healthy_fraction: float = 0.75
    stale_heartbeat_ms: float = 5_000.0
    grad_outlier_factor: float = 8.0

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 < self.minimum_healthy_fraction <= 1:
            raise ValueError("minimum_healthy_fraction must be in (0, 1]")
        if not isfinite(self.stale_heartbeat_ms) or self.stale_heartbeat_ms <= 0:
            raise ValueError("stale_heartbeat_ms must be finite and positive")
        if not isfinite(self.grad_outlier_factor) or self.grad_outlier_factor <= 0:
            raise ValueError("grad_outlier_factor must be finite and positive")


@dataclass(frozen=True)
class QuorumDecision:
    step: int
    action: QuorumAction
    healthy_ranks: tuple[int, ...]
    isolated_ranks: tuple[int, ...]
    reasons: tuple[str, ...]
    healthy_fraction: float
    model_version: str
    evidence_state: str = EVIDENCE_STATE

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["action"] = self.action.value
        return result

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class GradientQuorumController:
    """Validate one distributed gradient step before commit/allreduce authority."""

    def __init__(self, policy: QuorumPolicy) -> None:
        self.policy = policy
        self.last_step = -1

    @staticmethod
    def _grad_limit(reports: tuple[GradientReport, ...], factor: float) -> float:
        values = sorted(
            report.grad_norm
            for report in reports
            if report.finite and isfinite(report.grad_norm) and report.grad_norm >= 0
        )
        if not values:
            return 0.0
        center = median(values)
        mad = median(abs(value - center) for value in values)
        if center == 0 and mad == 0:
            return 0.0
        return center + factor * mad if mad > 0 else center * factor

    def evaluate(self, step: int, reports: Iterable[GradientReport]) -> QuorumDecision:
        if step <= self.last_step:
            raise ValueError("step must be strictly increasing")
        rows = tuple(reports)
        if len(rows) != self.policy.world_size:
            raise ValueError(
                f"expected {self.policy.world_size} rank reports, received {len(rows)}"
            )
        ranks = [row.rank for row in rows]
        expected = set(range(self.policy.world_size))
        if len(ranks) != len(set(ranks)) or set(ranks) != expected:
            raise ValueError(f"rank set must exactly match {sorted(expected)}")
        if any(row.step != step for row in rows):
            raise ValueError("all reports must be bound to the evaluated step")

        versions = {row.model_version for row in rows}
        if len(versions) != 1:
            self.last_step = step
            return QuorumDecision(
                step=step,
                action=QuorumAction.ABORT,
                healthy_ranks=(),
                isolated_ranks=tuple(sorted(ranks)),
                reasons=("MODEL_VERSION_SPLIT_BRAIN",),
                healthy_fraction=0.0,
                model_version="DIVERGENT",
            )

        grad_limit = self._grad_limit(rows, self.policy.grad_outlier_factor)
        healthy: list[int] = []
        isolated: list[int] = []
        reasons: list[str] = []

        for row in rows:
            local: list[str] = []
            if not row.finite or not isfinite(row.grad_norm) or row.grad_norm < 0:
                local.append("NONFINITE_GRADIENT")
            elif grad_limit > 0 and row.grad_norm > grad_limit:
                local.append("GRADIENT_OUTLIER")
            if row.overflow:
                local.append("MIXED_PRECISION_OVERFLOW")
            if row.heartbeat_age_ms > self.policy.stale_heartbeat_ms:
                local.append("STALE_HEARTBEAT")
            if local:
                isolated.append(row.rank)
                reasons.extend(f"rank={row.rank}:{reason}" for reason in local)
            else:
                healthy.append(row.rank)

        healthy_fraction = len(healthy) / self.policy.world_size
        if healthy_fraction < self.policy.minimum_healthy_fraction:
            action = QuorumAction.ABORT
        elif isolated:
            action = QuorumAction.ISOLATE
        else:
            action = QuorumAction.COMMIT

        self.last_step = step
        return QuorumDecision(
            step=step,
            action=action,
            healthy_ranks=tuple(sorted(healthy)),
            isolated_ranks=tuple(sorted(isolated)),
            reasons=tuple(sorted(reasons)),
            healthy_fraction=healthy_fraction,
            model_version=next(iter(versions)),
        )
