from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PromotionLockoutRemovalTests(unittest.TestCase):
    def test_keyed_promotion_artifacts_and_machine_references_are_absent(self) -> None:
        self.assertFalse((ROOT / "src" / "promotion_authority.py").exists())
        self.assertFalse((ROOT / "machine" / "promotion_authority.json").exists())
        source_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "src").rglob("*.py")
        )
        machine_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "machine").rglob("*.json")
        )
        self.assertNotIn("GLACIEREQ_PROMOTION_SECRET", source_text)
        self.assertNotIn("LOCAL_OPERATOR_SECRET", source_text)
        self.assertNotIn("promotion_authority", machine_text)
        self.assertNotIn("AUTHORITY_BOUND", machine_text)

    def test_proof_receipt_remains_non_authorizing_evidence(self) -> None:
        self.assertTrue((ROOT / "machine" / "proof_receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
