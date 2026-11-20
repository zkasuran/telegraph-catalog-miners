# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import ast
import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import data
from policy import execution_gate, quality, remote_only
from shortcuts import evaluate, fit


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.rows = data.build(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_inventory_and_split_counts(self):
        self.assertEqual(data.check(self.rows)["source_rows"], 184)
        self.assertEqual({s: sum(r["split"] == s for r in self.rows) for s in data.PLAN}, {"train": 96, "validation": 24, "audit": 64})
        self.assertEqual(len({r["template_group"] for r in self.rows}), 46)
        self.assertFalse(any(r["template_group"] in ("design-6", "design-7") for r in self.rows))

    def test_source_tamper_refused(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["source"] += " contract Tampered {}"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            data.check(rows)

    def test_group_leak_refused(self):
        rows = copy.deepcopy(self.rows)
        audit_row = next(r for r in rows if r["split"] == "audit")
        audit_row["template_group"] = rows[0]["template_group"]
        with self.assertRaisesRegex(ValueError, "crosses split"):
            data.check(rows)

    def test_normalized_clone_refused(self):
        rows = copy.deepcopy(self.rows)
        audit_row = next(r for r in rows if r["split"] == "audit")
        audit_row["source"] = rows[0]["source"].replace("C1000", "C9999")
        audit_row["source_sha256"] = data.sha(audit_row["source"].encode())
        audit_row["prompt"] = audit_row["source"]
        with self.assertRaisesRegex(ValueError, "normalized source overlaps"):
            data.check(rows)

    def test_identical_preferences_refused(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["rejected"] = rows[0]["chosen"]
        with self.assertRaisesRegex(ValueError, "identical"):
            data.check(rows)

    def test_length_and_answer_only_balanced(self):
        frozen = fit([r for r in self.rows if r["split"] == "train"])
        measured = evaluate(self.rows, frozen)
        self.assertTrue(all(v == 0 for v in frozen["coefficients"].values()))
        for split in data.PLAN:
            self.assertEqual(measured[split]["length"], 0.5)
            self.assertEqual(measured[split]["answer_only"], 0.5)

    def test_literal_render(self):
        self.assertEqual(data.render(" a\n", "b\r\n"), "### Task:\n a\n\n\n### Answer:\nb\r\n")
        self.assertNotEqual(data.render(" a", "b"), data.render("a", "b"))

    def test_local_model_authority_refused(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Model loading"):
                remote_only()

    def test_remote_owner_identity_is_required(self):
        # Exercise only the pure authority contract, never model entry points.
        environment = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "zkasuran/telegraph-catalog-miners",
                       "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_OS": "Linux",
                       "GITHUB_WORKFLOW": "SMART_CONTRACT_AUDIT owner pipeline", "GITHUB_RUN_ID": "123",
                       "GITHUB_SHA": "a" * 40}
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(RuntimeError, "owner attempt"):
                remote_only()
            os.environ["OWNER_RUN_ID"] = "28ca47b08d8649b186e5a057059a3aaa"
            remote_only()

    def test_quality_gate_refutes_chance_and_shortcut(self):
        base = {"accuracy": 0.85, "untrained_accuracy": 0.70,
                "by_mechanism": {"access": 0.85, "arithmetic": 0.85},
                "by_label": {"vulnerable": 0.85, "fixed": 0.85}, "answer_only_accuracy": 0.5, "shortcut_accuracy": 0.5}
        good = {k: copy.deepcopy(base) for k in ("prose", "json", "report")}
        self.assertEqual(quality(good)["verdict"], "PASS")
        good["prose"]["accuracy"] = 0.5
        self.assertEqual(quality(good)["verdict"], "FAIL")
        good["prose"] = copy.deepcopy(base)
        good["report"]["answer_only_accuracy"] = 0.9
        self.assertEqual(quality(good)["verdict"], "FAIL")

    def test_execution_incomplete_refused(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            execution_gate(self.rows, {"outcome": "incomplete"}, "x")

    def test_frozen_coefficients_and_data_binding_parse(self):
        import json
        here = Path(__file__).parent
        binding = json.loads((here / "data-bindings.json").read_text())
        coefficients = json.loads((here / data.BASELINE_FILE).read_text())
        self.assertEqual(binding["family"], data.FAMILY)
        self.assertEqual(binding["owner_attempt"], data.ATTEMPT)
        self.assertEqual(coefficients["schema_version"], 1)
        self.assertEqual(coefficients["train_identity"], data.sha(json.dumps([r for r in self.rows if r["split"] == "train"], sort_keys=True).encode()))
        self.assertEqual(binding["inventory_sha256"], data.sha((Path(self.tmp.name) / "inventory.json").read_bytes()))

    def test_training_extension_preserves_heldout_and_original_rows(self):
        import json
        import hashlib
        original = [r for r in self.rows if r["split"] == "train" and not r["template_group"].startswith("new-training-")]
        encoded = lambda rows: b"".join((json.dumps(r, sort_keys=True) + "\n").encode() for r in rows)
        self.assertEqual(hashlib.sha256(encoded(original)).hexdigest(), "19b181ee63a1b0506116783392be0e4a49ba394b9375b14d1bd0e8e0556bb1fd")
        for split, digest in (("validation", "8c79f65c18479354acc40065036a7d1e7c26a6ddcb6cc2b41eb608266138aec4"), ("audit", "6cd7d2c0b11206df3c6ad04471cd53683c791f9393140e8c764101457cc31dd8")):
            self.assertEqual(hashlib.sha256(encoded([r for r in self.rows if r["split"] == split])).hexdigest(), digest)
        extension = [r for r in self.rows if r["template_group"].startswith("new-training-")]
        self.assertEqual(len(extension), 32)
        self.assertEqual({r["split"] for r in extension}, {"train"})
        self.assertEqual(len({r["template_group"] for r in extension}), 8)
        groups = {}
        for row in extension:
            groups.setdefault(row["sibling_group"], []).append(row)
        self.assertEqual(len(groups), 16)
        self.assertTrue(all({r["vulnerable"] for r in siblings} == {True, False} for siblings in groups.values()))

    def test_original_baseline_preserved_and_not_reused_for_new_training(self):
        import hashlib
        import json
        original_bytes = (Path(__file__).parent / "frozen-baselines.json").read_bytes()
        self.assertEqual(hashlib.sha256(original_bytes).hexdigest(), "656bf351d7c5f34f3d194615d72eb8692b84c8d2e9bbfc8ecf0ccdfe8809ccf2")
        train = [r for r in self.rows if r["split"] == "train"]
        self.assertNotEqual(json.loads(original_bytes)["train_identity"], data.sha(json.dumps(train, sort_keys=True).encode()))

    def test_syntax_all_owned_python(self):
        for path in Path(__file__).parent.glob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
            compile(path.read_text(), str(path), "exec")

    def test_repeat_data_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            data.build(Path(temp))
            for split in data.PLAN:
                self.assertEqual((Path(temp) / (split + ".jsonl")).read_bytes(), (Path(self.tmp.name) / (split + ".jsonl")).read_bytes())


if __name__ == "__main__":
    unittest.main()
