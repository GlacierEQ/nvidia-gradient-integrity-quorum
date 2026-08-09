from __future__ import annotations
import unittest
from src.gradient_quorum import Commit, GradientIntegrityQuorum, RankReport

class GQTests(unittest.TestCase):
    def test_isolate_poison(self):
        reps = {
            0: RankReport(0, 1.0, True),
            1: RankReport(1, float("nan"), False),
            2: RankReport(2, 1.2, True),
        }
        r = GradientIntegrityQuorum(0.5).evaluate(reps)
        self.assertEqual(r.decision, Commit.ISOLATE)
        self.assertEqual(r.poison_ranks, (1,))

    def test_commit_clean(self):
        reps = {0: RankReport(0, 1.0, True), 1: RankReport(1, 1.1, True)}
        r = GradientIntegrityQuorum().evaluate(reps)
        self.assertEqual(r.decision, Commit.COMMIT)

if __name__ == "__main__":
    unittest.main()
