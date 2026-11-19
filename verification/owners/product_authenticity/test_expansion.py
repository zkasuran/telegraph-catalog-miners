# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import product_evidence
import scorer
from source_inventory import all_records, readiness, sanada_records

ROOT = Path(__file__).resolve().parent


class ExpansionTests(unittest.TestCase):
    def test_complete_published_annotation_ranges(self):
        rows = all_records()
        self.assertEqual(len(rows), 110)
        additions = [r for r in rows.values() if r["record_id"].startswith("sanada")]
        self.assertEqual(len(additions), 76)
        self.assertEqual(sum(r["source_label"] == "Authentic" for r in additions), 22)
        self.assertEqual(sum(r["source_label"] == "Counterfeit" for r in additions), 54)
        self.assertEqual({r["sample"] for r in additions if r["sample"].startswith("GV")}, {"GV1", "GV2", "GV3", "GV4"})

    def test_no_withheld_spectra_or_extra_standard_ids(self):
        for record in all_records().values():
            if record["record_id"].startswith("sanada"):
                self.assertFalse(record["raw_spectra_released"])
                self.assertNotIn("spectrum", record)
        self.assertNotIn("sanada2020-standard", all_records())

    def test_preserved_source_false_negative_not_relabelled(self):
        record = all_records()["sanada2020-FV1"]
        self.assertEqual(record["source_label"], "Counterfeit")
        # Article says its PCA plot is close to the standard. Never turn that
        # analytical similarity into an invented genuine label.
        self.assertIn("manufacturer", record["source_paragraph"])

    def test_all_110_archive_answers_and_reversals(self):
        rows = all_records()
        for rid in rows:
            product = product_evidence.product_for(rid)
            answer = product_evidence.expectation(product, rows)
            self.assertEqual(product_evidence.score(product, json.dumps(answer), rows)["score"], 10000)
            reverse = copy.deepcopy(answer)
            reverse["authentic"] = not answer["authentic"]
            self.assertEqual(product_evidence.score(product, json.dumps(reverse), rows)["score"], 0)

    def test_same_source_entity_cohort_not_three_new_studies(self):
        report = readiness()
        self.assertEqual(report["source_studies"], 2)
        self.assertEqual(report["conservative_source_components"], 2)
        self.assertFalse(report["qualification_ready"])
        self.assertEqual(report["training_rows"], 0)
        self.assertEqual(report["audit_pairs"], 0)

    def test_source_grant_and_identifier_range_mutations_fail(self):
        xml = (ROOT / "sources/sanada2020.xml").read_bytes()
        for original, replacement in ((b"Creative Commons Attribution (CC BY)", b"unverified grant"), (b"GC9", b"GC8")):
            mutated = xml.replace(original, replacement)
            self.assertNotEqual(mutated, xml)
            with self.assertRaises(ValueError):
                sanada_records(mutated)

    def test_copied_identifier_new_listing_and_wrong_echo_refused(self):
        rows = all_records()
        product = "New listing claims same identifier as sanada2020-GC1"
        answer = product_evidence.expectation(product, rows)
        self.assertEqual(answer["verdict"], "Inconclusive")
        product = product_evidence.product_for("sanada2020-GC1")
        answer = product_evidence.expectation(product, rows)
        answer["product"] += "\n"
        self.assertEqual(product_evidence.score(product, json.dumps(answer), rows)["score"], 0)
        self.assertEqual(scorer.SCOPE, "Archived record only. This does not authenticate a new physical item.")

    def test_primary_public_cli_real_and_tampered(self):
        rows = all_records()
        product = product_evidence.product_for("sanada2020-FV1")
        answer = product_evidence.expectation(product, rows)
        for expected in (10000, 0):
            if expected == 0:
                answer["verdict"] = "Authentic"
                answer["authentic"] = True
            process = subprocess.run([sys.executable, str(ROOT / "product_evidence.py")],
                                     input=json.dumps({"product": product, "answer": json.dumps(answer)}),
                                     text=True, capture_output=True, check=True, timeout=5)
            self.assertEqual(json.loads(process.stdout)["score"], expected)


if __name__ == "__main__":
    unittest.main()
