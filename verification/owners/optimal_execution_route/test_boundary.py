# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import json
import tempfile
import unittest
from pathlib import Path

from oracle import Oracle, QUESTION, score_wire
from qualification_gate import HERE, assess
from release_gate import source_lock_check, workflow_check


class SourceBoundaryTests(unittest.TestCase):
    def test_source_tool_lock_has_broker_wheel_and_mandatory_hashes(self):
        lock = (HERE / "requirements-source.lock").read_text()
        gate = (HERE / "verify.sh").read_text()
        source_lock_check(lock, gate)
        missing = lock.replace("--hash=sha256:b8bb0864c5a28024fac8a632c443c87c5aa6f215c0b126c449ae1a150412f31d", "")
        with self.assertRaises(ValueError):
            source_lock_check(missing, gate)
        with self.assertRaises(ValueError):
            source_lock_check(lock, gate.replace("--require-hashes", ""))

    def test_source_gate_does_not_claim_qualification(self):
        report = assess()
        self.assertFalse(report["registration_ready"])
        self.assertEqual(report["input_fields"], ["pair"])
        self.assertEqual(report["regression_state_groups"], 2)
        self.assertEqual(report["audit_pairs"], 0)

    def test_context_free_collision_retains_equal_lengths(self):
        policy = json.loads((HERE / "policy.json").read_text())
        a = Oracle(policy, json.loads((HERE / "evidence/STATE.json").read_text())["states"][0])
        b = Oracle(policy, json.loads((HERE / "evidence/STATE-OLDER.json").read_text())["states"][0])
        candidates = [json.dumps({"amount_out": o.best_amount}) for o in (a, b)]
        self.assertEqual(len(candidates[0]), len(candidates[1]))
        self.assertTrue(score_wire(QUESTION, candidates[0], a)["score_ppm"] > score_wire(QUESTION, candidates[1], a)["score_ppm"])
        self.assertTrue(score_wire(QUESTION, candidates[0], b)["score_ppm"] < score_wire(QUESTION, candidates[1], b)["score_ppm"])
        for body in candidates:
            self.assertEqual(score_wire(QUESTION, body)["status"], "abstain")

    def test_miner_supplied_reference_does_not_end_abstention(self):
        body = json.dumps({"amount_out": 1799488639, "reference_state": {"block_hash": "invented", "amount_out": 1799488639}})
        self.assertEqual(score_wire(QUESTION, body)["status"], "abstain")

    def test_workflow_refuses_paid_runner_and_overlong_job(self):
        import yaml
        path = HERE.parents[2] / ".github/workflows/evaluator-owner-optimal_execution_route.yml"
        base = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        with tempfile.TemporaryDirectory(prefix="route-source-workflow-") as directory:
            target = Path(directory) / "workflow.yml"
            for field, value in (("runs-on", "self-hosted"), ("timeout-minutes", "351")):
                changed = copy.deepcopy(base)
                changed["jobs"]["source"][field] = value
                target.write_text(yaml.safe_dump(changed))
                with self.assertRaises(ValueError):
                    workflow_check(target)


if __name__ == "__main__":
    unittest.main()
