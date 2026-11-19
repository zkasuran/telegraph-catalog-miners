# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from broad_data import CHILD, DESCRIPTION, build, check_splits, make_pairs, source_controls, source_rows, task_from_prompt, typed_question

HERE = Path(__file__).resolve().parent


class BroadContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = source_rows()
        cls.splits = {name: make_pairs(records, name) for name, records in cls.sources.items()}

    def test_original_source_facts_not_model_captures(self):
        self.assertEqual({name: len(records) for name, records in self.sources.items()}, {"train": 74, "validation": 76})
        s18 = next(r for r in self.sources["train"] if r["id"] == "newton2008-st001-row4")
        self.assertEqual(s18["source_annotation"], "Authentic")
        self.assertEqual(s18["status"], "Inconclusive")

    def test_complete_grouped_development(self):
        self.assertTrue(check_splits(self.splits))
        self.assertEqual(len(self.splits["train"]), 1184)
        self.assertEqual(len(self.splits["validation"]), 1216)

    def test_literal_product_limit_not_framed_question_limit(self):
        product = "x" * 256
        expected = CHILD + " request. product (" + DESCRIPTION + "): " + product + "."
        self.assertEqual(typed_question(product), expected)
        self.assertEqual(len(expected.encode()), 340)
        with self.assertRaisesRegex(ValueError, "256-byte"):
            typed_question("é" * 129)
        self.assertEqual(len(typed_question("é" * 128).encode()), 340)

    def test_group_and_formulation_overlap_refused(self):
        for key in ("source_group", "entity_group", "time_group", "group_id", "formulation_group"):
            changed = copy.deepcopy(self.splits)
            changed["validation"][0][key] = changed["train"][0][key]
            with self.assertRaisesRegex(ValueError, "overlap"):
                check_splits(changed)

    def test_normalized_overlap_refused(self):
        changed = copy.deepcopy(self.splits)
        changed["validation"][0]["rejected"] = changed["train"][0]["rejected"].upper()
        with self.assertRaisesRegex(ValueError, "normalized overlap"):
            check_splits(changed)

    def test_byte_and_prefix_mutation_refused(self):
        changed = copy.deepcopy(self.splits)
        row = next(r for r in changed["train"] if r["view"] == "typed")
        row["prompt"] = row["prompt"].replace(" request.", " request!", 1)
        with self.assertRaisesRegex(ValueError, "literal"):
            check_splits(changed)

    def test_unknown_and_new_object_not_counterfeit(self):
        rows = [r for r in self.splits["train"] if r["kind"] == "scope" and r["format"] == "json"]
        for row in rows:
            task = task_from_prompt(row["prompt"])
            if task["requested"] != task["unit"]:
                self.assertEqual(json.loads(row["chosen"])["verdict"], "Inconclusive")
                self.assertEqual(json.loads(row["rejected"])["verdict"], "Inconclusive")
                self.assertEqual(row["label"], "Inconclusive")
        self.assertEqual({r["label"] for r in self.splits["validation"]}, {"Authentic", "Counterfeit", "Inconclusive"})

    def test_controls_are_measured_not_expected(self):
        observed = source_controls(self.splits["validation"])
        self.assertEqual(observed["json"]["byte_length"], .5)
        self.assertEqual(observed["prose"]["byte_length"], .5)
        self.assertLessEqual(max(v for m in observed.values() for v in m.values()), .65)
        reversed_rows = [{**r, "chosen": r["rejected"], "rejected": r["chosen"]} for r in self.splits["validation"]]
        self.assertLess(source_controls(reversed_rows)["json"]["source_status_copy"], observed["json"]["source_status_copy"])

    def test_configs_parse_and_keep_audit_closed(self):
        config = json.loads((HERE / "config.json").read_text())
        self.assertEqual(config["family"], "PRODUCT_AUTHENTICITY")
        self.assertEqual(config["max_tokens"], 512)
        self.assertLessEqual(config["timeout_minutes"], 350)
        self.assertFalse(config["registration_ready"])

    def test_standalone_data_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-W", "error", str(HERE / "broad_data.py"), "--output", directory], capture_output=True, timeout=20, check=True)
            report = json.loads(result.stdout)
            self.assertEqual(report["files"]["train"]["pairs"], 1184)
            self.assertFalse(report["registration_ready"])
            self.assertEqual(report["audit_pairs"], 0)

    def test_remote_entry_refuses_before_creating_model(self):
        result = subprocess.run([sys.executable, str(HERE / "remote_broad.py"), "--root", str(HERE / "never-created")],
                                env={"PATH": "/usr/bin:/bin", "GITHUB_ACTIONS": "false", "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"bounded owned standard public CPU", result.stderr)
        self.assertFalse((HERE / "never-created").exists())

    def test_build_never_claims_independent_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            report, _ = build(directory)
            self.assertTrue(report["formulation_groups_disjoint"])
            self.assertEqual(report["normalized_overlap"], 0)
            self.assertFalse(report["audit_available"])


if __name__ == "__main__":
    unittest.main()
