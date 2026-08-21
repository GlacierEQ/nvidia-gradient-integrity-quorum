from __future__ import annotations
import hashlib, json, os, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from src.promotion_authority import (
    PROMOTION_SECRET_ENV, PromotionAuthority, promotion_secret_from_environment,
    verify_bound_grant,
)
ROOT = Path(__file__).resolve().parents[1]
class PromotionAuthTests(unittest.TestCase):
    def test_issue_verify(self):
        a = PromotionAuthority(b"test-secret", ttl_s=60)
        g = a.issue("GlacierEQ/x", "abc", "def", now=1000.0)
        ok, r = a.verify(g, now=1001.0)
        self.assertTrue(ok)
    def test_expired(self):
        a = PromotionAuthority(b"test-secret", ttl_s=10)
        g = a.issue("GlacierEQ/x", "abc", "def", now=1000.0)
        ok, r = a.verify(g, now=2000.0)
        self.assertFalse(ok)
        self.assertEqual(r, "GRANT_EXPIRED")
    def test_real_machine_grant_is_structurally_bound_to_proof_receipt(self):
        grant_path = ROOT / "machine" / "promotion_authority.json"
        proof_path = ROOT / "machine" / "proof_receipt.json"
        if not grant_path.is_file() or not proof_path.is_file():
            self.skipTest("receipts not yet bound")
        grant = json.loads(grant_path.read_text())
        proof = json.loads(proof_path.read_text())
        file_digest = hashlib.sha256(proof_path.read_bytes()).hexdigest()
        self.assertEqual(grant["proof_receipt_digest"], file_digest)
        self.assertEqual(grant["source_sha"], proof["source_sha"])
        self.assertEqual(grant["secret_ref"], f"environment:{PROMOTION_SECRET_ENV}")
        ok, reason = verify_bound_grant(grant, proof_path)
        self.assertFalse(ok)
        self.assertEqual(reason, "PROMOTION_SECRET_REQUIRED")

    def test_explicit_environment_secret_verifies_bound_grant(self):
        proof = {"source_sha": "abc"}
        with tempfile.TemporaryDirectory() as directory:
            proof_path = Path(directory) / "proof.json"
            proof_path.write_text(json.dumps(proof), encoding="utf-8")
            digest = hashlib.sha256(proof_path.read_bytes()).hexdigest()
            grant = PromotionAuthority(b"test-secret").issue(
                "GlacierEQ/nvidia-gradient-integrity-quorum", "abc", digest, now=1000.0
            )
            with patch.dict(os.environ, {PROMOTION_SECRET_ENV: "test-secret"}, clear=True):
                secret = promotion_secret_from_environment()
            ok, reason = verify_bound_grant(
                grant.__dict__, proof_path, secret=secret, now=1001.0
            )
        self.assertTrue(ok, reason)

    def test_missing_secret_fails_closed(self):
        ok, reason = verify_bound_grant({}, ROOT / "missing-proof.json")
        self.assertFalse(ok)
        self.assertEqual(reason, "PROMOTION_SECRET_REQUIRED")
if __name__ == "__main__":
    unittest.main()
