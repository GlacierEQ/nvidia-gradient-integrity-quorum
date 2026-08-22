#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_DIR=".verification-artifacts"
SCENARIOS="${ARTIFACT_DIR}/gradient-quorum-scenarios.json"
RECEIPT="${ARTIFACT_DIR}/gradient-quorum-scenarios.receipt.json"
mkdir -p "${ARTIFACT_DIR}"

python -m compileall -q src tests scripts
python -m unittest discover -s tests -v | tee "${ARTIFACT_DIR}/unittest.txt"
python scripts/quorum_probe.py \
  --output "${SCENARIOS}" \
  --receipt "${RECEIPT}" \
  | tee "${ARTIFACT_DIR}/quorum-probe.txt"

python - <<'PY'
import hashlib
import json
from pathlib import Path

scenario_path = Path('.verification-artifacts/gradient-quorum-scenarios.json')
receipt_path = Path('.verification-artifacts/gradient-quorum-scenarios.receipt.json')
scenario = json.loads(scenario_path.read_text(encoding='utf-8'))
receipt = json.loads(receipt_path.read_text(encoding='utf-8'))

actions = {name: row['action'] for name, row in scenario['scenarios'].items()}
assert actions == {
    'clean': 'COMMIT',
    'single_poison_rank': 'ISOLATE',
    'model_version_split_brain': 'ABORT',
}
assert scenario['evidence_state'] == 'DETERMINISTIC_GRADIENT_QUORUM_MODEL'
assert scenario['scenarios']['model_version_split_brain']['healthy_fraction'] == 0.0
actual = hashlib.sha256(scenario_path.read_bytes()).hexdigest()
assert receipt['artifact_sha256'] == actual
assert receipt['verified_state'] == 'QUORUM_SCENARIOS_EXECUTED'
print(json.dumps({
    'elite_core': 'PASS',
    'actions': actions,
    'artifact_sha256': actual,
}, indent=2))
PY
