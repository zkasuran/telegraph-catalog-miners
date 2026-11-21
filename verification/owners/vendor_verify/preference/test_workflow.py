# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Hostile controls for added remote resources and restricted uploads."""

import copy
import unittest

import workflow_contract as contract


class WorkflowContracts(unittest.TestCase):
    def test_projection_executes_source_validator(self):
        old = {
            "name": "old",
            "on": {
                "workflow_dispatch": {"inputs": {"owner_attempt": {"type": "string"}}}
            },
            "permissions": {"contents": "read"},
            "jobs": {"source_checks": {"steps": []}},
        }
        new = contract.extend_workflow(old)
        calls = []
        self.assertTrue(
            contract.validate_extended_workflow(new, lambda v: calls.append(v) or True)
        )
        self.assertEqual(calls[0]["jobs"], old["jobs"])
        self.assertEqual(calls[0]["on"], old["on"])

    def test_paid_timeout_parallel_privilege_upload_refused(self):
        old = {
            "on": {
                "workflow_dispatch": {"inputs": {"owner_attempt": {"type": "string"}}}
            },
            "permissions": {"contents": "read"},
            "jobs": {"source_checks": {"steps": []}},
        }
        original = contract.extend_workflow(old)
        mutations = [
            lambda v: v["jobs"]["preference_training"].update(
                {"runs-on": "self-hosted"}
            ),
            lambda v: v["jobs"]["preference_training"].update({"timeout-minutes": 351}),
            lambda v: v["jobs"].update({"another": {}}),
            lambda v: v.update({"permissions": {"contents": "write"}}),
            lambda v: v["jobs"]["preference_training"]["steps"][-1]["with"].update(
                {"path": "**/*"}
            ),
            lambda v: v["jobs"]["source_checks"].update({"if": "always()"}),
            lambda v: v["jobs"]["preference_training"]["steps"][-1]["with"].update(
                {
                    "path": "${{ runner.temp }}/vendor-general-preference/retained-training/train.jsonl"
                }
            ),
        ]
        for mutate in mutations:
            value = copy.deepcopy(original)
            mutate(value)
            with self.assertRaises(ValueError):
                contract.validate_extended_workflow(value, lambda v: True)

    def test_frozen_baseline_artifact_not_training_rows(self):
        paths = contract.REMOTE_JOB["steps"][-1]["with"]["path"].splitlines()
        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(
            any(path.endswith("frozen-training-baseline.tar.gz") for path in paths)
        )
        self.assertTrue(
            any(path.endswith("filtered-dataset-gate.json") for path in paths)
        )
        self.assertFalse(any(path.endswith(".jsonl") or "**" in path for path in paths))


if __name__ == "__main__":
    unittest.main()
