#!/usr/bin/env python3
"""Execute deterministic gradient-quorum scenarios and emit a content-hashed receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quorum_controller import (  # noqa: E402
    GradientQuorumController,
    GradientReport,
    QuorumPolicy,
)


def report(rank: int, step: int, *, version: str = "model-a", grad: float = 1.0, finite: bool = True):
    return GradientReport(rank, step, version, grad, finite=finite)


def execute() -> dict[str, object]:
    clean = GradientQuorumController(QuorumPolicy(world_size=4)).evaluate(
        0, [report(rank, 0) for rank in range(4)]
    )

    isolate_rows = [report(rank, 0) for rank in range(4)]
    isolate_rows[3] = report(3, 0, grad=math.nan, finite=False)
    isolate = GradientQuorumController(QuorumPolicy(world_size=4)).evaluate(
        0, isolate_rows
    )

    split = GradientQuorumController(QuorumPolicy(world_size=2)).evaluate(
        0,
        [report(0, 0, version="model-a"), report(1, 0, version="model-b")],
    )

    payload = {
        "schema": "glaciereq.nvidia-gradient-quorum-scenarios.v1",
        "evidence_state": "DETERMINISTIC_GRADIENT_QUORUM_MODEL",
        "scenarios": {
            "clean": clean.as_dict(),
            "single_poison_rank": isolate.as_dict(),
            "model_version_split_brain": split.as_dict(),
        },
        "fingerprints": {
            "clean": clean.fingerprint,
            "single_poison_rank": isolate.fingerprint,
            "model_version_split_brain": split.fingerprint,
        },
        "claims_not_established": [
            "NVIDIA hardware execution",
            "CUDA or NCCL allreduce interception",
            "production gradient commit authority",
            "measured training throughput or convergence improvement"
        ],
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    payload = execute()
    expected = {
        "clean": "COMMIT",
        "single_poison_rank": "ISOLATE",
        "model_version_split_brain": "ABORT",
    }
    actual = {name: row["action"] for name, row in payload["scenarios"].items()}
    if actual != expected:
        raise SystemExit(f"unexpected quorum actions: {actual}")

    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()

    receipt = {
        "schema": "glaciereq.nvidia-gradient-quorum-receipt.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository": os.environ.get(
            "GITHUB_REPOSITORY", "GlacierEQ/nvidia-gradient-integrity-quorum"
        ),
        "commit": os.environ.get("GITHUB_SHA", "local"),
        "artifact": str(args.output),
        "artifact_sha256": digest,
        "verified_state": "QUORUM_SCENARIOS_EXECUTED",
        "actions": actual,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
