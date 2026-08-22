import math
import unittest

from src.quorum_controller import (
    GradientQuorumController,
    GradientReport,
    QuorumAction,
    QuorumPolicy,
)


def report(rank, step=0, version="v1", grad=1.0, **kwargs):
    return GradientReport(rank, step, version, grad, **kwargs)


class QuorumControllerTests(unittest.TestCase):
    def test_clean_world_commits(self):
        quorum = GradientQuorumController(QuorumPolicy(world_size=4))
        decision = quorum.evaluate(0, [report(i) for i in range(4)])
        self.assertEqual(decision.action, QuorumAction.COMMIT)
        self.assertEqual(len(decision.fingerprint), 64)

    def test_one_nonfinite_isolates_with_quorum(self):
        quorum = GradientQuorumController(QuorumPolicy(world_size=4))
        rows = [report(i) for i in range(4)]
        rows[3] = report(3, grad=math.nan, finite=False)
        decision = quorum.evaluate(0, rows)
        self.assertEqual(decision.action, QuorumAction.ISOLATE)
        self.assertEqual(decision.isolated_ranks, (3,))

    def test_lost_quorum_aborts(self):
        quorum = GradientQuorumController(
            QuorumPolicy(world_size=4, minimum_healthy_fraction=0.75)
        )
        rows = [
            report(0),
            report(1),
            report(2, grad=math.nan, finite=False),
            report(3, grad=math.inf),
        ]
        self.assertEqual(quorum.evaluate(0, rows).action, QuorumAction.ABORT)

    def test_split_brain_model_version_aborts_every_rank(self):
        quorum = GradientQuorumController(QuorumPolicy(world_size=2))
        decision = quorum.evaluate(
            0, [report(0, version="a"), report(1, version="b")]
        )
        self.assertEqual(decision.action, QuorumAction.ABORT)
        self.assertEqual(decision.reasons, ("MODEL_VERSION_SPLIT_BRAIN",))
        self.assertEqual(decision.isolated_ranks, (0, 1))

    def test_outlier_overflow_and_stale_rank_is_isolated(self):
        quorum = GradientQuorumController(QuorumPolicy(world_size=5))
        rows = [report(i) for i in range(4)]
        rows.append(report(4, grad=1000, overflow=True, heartbeat_age_ms=6000))
        decision = quorum.evaluate(0, rows)
        self.assertEqual(decision.action, QuorumAction.ISOLATE)
        joined = " ".join(decision.reasons)
        self.assertIn("GRADIENT_OUTLIER", joined)
        self.assertIn("MIXED_PRECISION_OVERFLOW", joined)
        self.assertIn("STALE_HEARTBEAT", joined)

    def test_replay_and_wrong_step_fail_closed(self):
        quorum = GradientQuorumController(QuorumPolicy(world_size=2))
        with self.assertRaises(ValueError):
            quorum.evaluate(0, [report(0, step=0), report(1, step=1)])
        quorum.evaluate(1, [report(0, step=1), report(1, step=1)])
        with self.assertRaises(ValueError):
            quorum.evaluate(1, [report(0, step=1), report(1, step=1)])

    def test_missing_or_duplicate_rank_fails_closed(self):
        quorum = GradientQuorumController(QuorumPolicy(world_size=2))
        with self.assertRaises(ValueError):
            quorum.evaluate(0, [report(0)])
        with self.assertRaises(ValueError):
            quorum.evaluate(0, [report(0), report(0)])


if __name__ == "__main__":
    unittest.main()
