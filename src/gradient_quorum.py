"""Gradient integrity quorum — finite-metric vote before allreduce commit."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping


def digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Commit(str, Enum):
    COMMIT = "COMMIT"
    ISOLATE = "ISOLATE"
    ABORT = "ABORT"


@dataclass(frozen=True)
class RankReport:
    rank: int
    grad_norm: float
    finite: bool


@dataclass(frozen=True)
class QuorumReceipt:
    decision: Commit
    poison_ranks: tuple[int, ...]
    healthy: int
    fingerprint: str


class GradientIntegrityQuorum:
    def __init__(self, min_healthy_ratio: float = 0.5):
        self.min_healthy_ratio = min_healthy_ratio

    def evaluate(self, reports: Mapping[int, RankReport]) -> QuorumReceipt:
        poison = tuple(sorted(r for r, rep in reports.items() if not rep.finite or not math.isfinite(rep.grad_norm)))
        healthy = len(reports) - len(poison)
        ratio = healthy / max(len(reports), 1)
        if healthy == 0:
            dec = Commit.ABORT
        elif poison and ratio >= self.min_healthy_ratio:
            dec = Commit.ISOLATE
        elif ratio < self.min_healthy_ratio:
            dec = Commit.ABORT
        else:
            dec = Commit.COMMIT
        body = {"d": dec.value, "poison": list(poison), "healthy": healthy}
        return QuorumReceipt(dec, poison, healthy, digest(body))
