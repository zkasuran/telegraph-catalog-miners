# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import json
from pathlib import Path
import tempfile
import unittest
import os
import subprocess
from unittest.mock import patch

from data_contract import build, canonical, check_splits, make_pairs, render, source_rows
from runtime_contract import check_reply, remote_only, score_value

HERE = Path(__file__).resolve().parent


class DevelopmentContracts(unittest.TestCase):
    def test_primary_source_counts_and_uncertainty_preserved(self):
        rows = source_rows()
        self.assertEqual(len(rows["train"]), 74)
        self.assertEqual(len(rows["validation"]), 76)
        s18 = next(r for r in rows["train"] if r["id"] == "newton2008-st001-row4")
        self.assertEqual(s18["source_annotation"], "Authentic")
        self.assertEqual(s18["status"], "Inconclusive")
        metformin = [r for r in rows["train"] if r["id"].startswith("metformin")]
        self.assertEqual(sum(r["status"] == "Inconclusive" for r in metformin), 33)

    def test_grouped_data_and_literal_byte_contract(self):
        rows = source_rows()
        splits = {name: make_pairs(records, name) for name, records in rows.items()}
        self.assertTrue(check_splits(splits))
        self.assertEqual(len(splits["train"]), 592)
        self.assertEqual(len(splits["validation"]), 608)
        for group in splits.values():
            for row in group:
                self.assertLessEqual(len(row["prompt"].encode()), 256)
                self.assertEqual(len(row["chosen"].encode()), len(row["rejected"].encode()))
                self.assertNotIn(row["source_record_id"], row["prompt"])

    def test_source_and_entity_overlap_mutations_refused(self):
        rows = source_rows()
        original = {name: make_pairs(records, name) for name, records in rows.items()}
        for field in ("source_group", "entity_group", "time_group", "group_id"):
            altered = copy.deepcopy(original)
            altered["validation"][0][field] = altered["train"][0][field]
            with self.assertRaisesRegex(ValueError, "overlap"):
                check_splits(altered)

    def test_new_object_pair_is_not_a_counterfeit_label(self):
        rows = make_pairs(source_rows()["train"], "train")
        for row in rows:
            if row["kind"] == "scope" and row["format"] == "json" and json.loads(row["prompt"])["requested"] != json.loads(row["prompt"])["unit"]:
                self.assertEqual(json.loads(row["chosen"])["verdict"], "Inconclusive")
                self.assertEqual(json.loads(row["rejected"])["verdict"], "Inconclusive")
                self.assertIn(json.loads(row["prompt"])["unit"], json.loads(row["chosen"])["evidence"][0])

    def test_literal_render_and_normalized_comparison_are_distinct(self):
        self.assertEqual(render(" X \r\n", " Y "), "### Task:\n X \r\n\n\n### Answer:\n Y ")
        self.assertNotEqual(render(" X ", " Y "), render("X", "Y"))
        self.assertEqual(canonical(" ＧＥＮＵＩＮＥ\n"), canonical("genuine"))

    def test_frozen_data_report_never_claims_audit_or_ready(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            report, _ = build(directory)
            self.assertFalse(report["audit_available"])
            self.assertFalse(report["registration_ready"])
            self.assertEqual(report["audit_pairs"], 0)
            self.assertEqual(report["normalized_overlap"], 0)

    def test_local_neural_entrypoint_refuses_before_loading(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}):
            with self.assertRaises(RuntimeError):
                remote_only()
        result = subprocess.run([__import__("sys").executable, str(HERE / "remote_develop.py"), "--root", str(HERE / "must-not-exist")],
                                capture_output=True, text=True, timeout=5,
                                env={"PATH": os.environ["PATH"], "GITHUB_ACTIONS": "false", "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bounded owned standard public CPU", result.stderr)
        self.assertFalse((HERE / "must-not-exist").exists())

    def test_literal_bare_hash_and_finite_bits_controls(self):
        f, t, a = "1" * 64, "2" * 64, "3" * 64
        reply = {"foundation_hash": f, "tokenizer_hash": t, "adapter_hash": a, "results": [{"score": 0x3f800000}]}
        self.assertEqual(check_reply(reply, 1, f, t, a), [0x3f800000])
        self.assertEqual(score_value(0x3f800000), 1.0)
        for malformed in (0x7f800000, 0x7fc00000, True, -1):
            with self.assertRaises(ValueError):
                score_value(malformed)
        for key in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
            mutated = copy.deepcopy(reply)
            mutated[key] = "0x" + mutated[key]
            with self.assertRaisesRegex(ValueError, "literal"):
                check_reply(mutated, 1, f, t, a)


if __name__ == "__main__":
    unittest.main()
