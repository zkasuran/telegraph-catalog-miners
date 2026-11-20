# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source packaging controls, no taxpayer or evaluation examples."""

import copy
import json
import unittest
from pathlib import Path

from check_public import WORKFLOW, validate_workflow
from ci import identity


class PublicTests(unittest.TestCase):
    def config(self):
        root = Path(__file__).resolve().parent.parents[2]
        return json.loads((root / WORKFLOW).read_text().split("\n", 1)[1])

    def test_actual_workflow_contract(self):
        self.assertTrue(validate_workflow(self.config()))

    def test_runner_and_timeout_refused(self):
        for key, value in (("runs-on", "self-hosted"), ("timeout-minutes", 351)):
            bad = copy.deepcopy(self.config())
            bad["jobs"]["source_checks"][key] = value
            with self.assertRaises(ValueError):
                validate_workflow(bad)

    def test_privilege_and_unpinned_action_refused(self):
        bad = self.config()
        bad["permissions"]["contents"] = "write"
        with self.assertRaises(ValueError):
            validate_workflow(bad)
        bad = self.config()
        bad["jobs"]["source_checks"]["steps"][0]["uses"] = "actions/checkout@main"
        with self.assertRaises(ValueError):
            validate_workflow(bad)

    def test_remote_identity_not_inferred(self):
        with self.assertRaises(ValueError):
            identity({"GITHUB_ACTIONS": "true"})

    def test_local_identity_not_called_remote(self):
        self.assertEqual(identity({})["mode"], "local_source_checks")
        self.assertIsNone(identity({})["run_id"])

    def test_synthetic_remote_identity_parser(self):
        # Artificial parser inputs only. Not a claim any such run occurred.
        env = {
            "GITHUB_ACTIONS": "true",
            "GITHUB_REPOSITORY": "zkasuran/telegraph-catalog-miners",
            "GITHUB_RUN_ID": "1",
            "GITHUB_SHA": "a" * 40,
            "OWNER_ATTEMPT": "b" * 32,
        }
        self.assertEqual(identity(env)["run_id"], 1)
        env["GITHUB_SHA"] = "not-a-commit"
        with self.assertRaises(ValueError):
            identity(env)


if __name__ == "__main__":
    unittest.main()
