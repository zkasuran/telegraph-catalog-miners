# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import audit_source_admission as gate

HERE = Path(__file__).resolve().parent


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = HERE / "sources"
        cls.text = (cls.sources / "nigeria-table-s2.txt").read_text()
        cls.rows = gate.nigeria_rows(cls.text)
        cls.book = json.loads((cls.sources / "bangladesh-workbook-summary.json").read_text())
        cls.grants = json.loads((cls.sources / "source-grants.json").read_text())

    def test_full_population_and_truth(self):
        self.assertEqual(len(self.rows), 260)
        self.assertEqual(sum(r["physical_authenticity"] == "Counterfeit" for r in self.rows), 4)
        self.assertEqual(sum(r["physical_authenticity"] == "Unknown" for r in self.rows), 256)
        self.assertEqual(sum(r["source_combined_quality"] == "probably falsified" for r in self.rows), 5)
        self.assertEqual(sum(r["source_combined_quality"] == "in specification" for r in self.rows), 194)

    def test_original_ids_not_table_ordinals(self):
        self.assertEqual(self.rows[195]["source_ordinal"], 196)
        self.assertEqual(self.rows[195]["record_id"], "QMN 197")
        self.assertEqual(self.rows[-1]["record_id"], "QMN 261")
        self.assertEqual(sum(r["split_batch_suffix"] is not None for r in self.rows), 6)

    def test_dashes_do_not_become_batches(self):
        self.assertEqual(self.rows[4]["batch"], "200203")
        self.assertEqual(self.rows[38]["batch"], "200303")
        self.assertEqual(self.rows[48]["batch"], "2OFA301")

    def test_wrapped_manufacturer_preserved(self):
        self.assertEqual(self.rows[4]["stated_manufacturer"], "Shanghai Juchen Import and Exports Co. Ltd.")
        self.assertEqual(self.rows[7]["stated_manufacturer"], "Jiangsu Pengyao Pharmaceutical Co. Ltd. (INC.)")
        self.assertEqual(self.rows[3]["stated_manufacturer"], "Scott\u2010Edil Advance Research Laboratories & Education Ltd.")

    def test_missing_row_refused(self):
        removed = "\n".join(line for line in self.text.splitlines() if "QMN 099" not in line)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            gate.nigeria_rows(removed)

    def test_normalized_duplicate_refused(self):
        with self.assertRaisesRegex(ValueError, "identity duplicate"):
            gate.nigeria_rows(self.text.replace("QMN 002", "QMN 001"))

    def test_quality_pass_not_genuine(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["physical_authenticity"] = "Authentic"
        with self.assertRaisesRegex(ValueError, "promoted"):
            gate.admission(rows, self.book, self.grants)

    def test_probably_falsified_not_confirmed(self):
        rows = copy.deepcopy(self.rows)
        row = next(r for r in rows if r["source_combined_quality"] == "probably falsified")
        row["physical_authenticity"] = "Counterfeit"
        with self.assertRaisesRegex(ValueError, "promoted"):
            gate.admission(rows, self.book, self.grants)

    def test_full_workbook_not_population(self):
        self.assertEqual([s["populated_rows"] for s in self.book["sheets"]], [17, 9, 8])
        self.assertEqual(self.book["specimen_codes_in_cells"], [])
        self.assertGreater(len(self.book["members"]), 3)
        report = gate.admission(self.rows, self.book, self.grants)
        self.assertEqual(report["bangladesh"]["aggregate_confirmations_admitted_as_individual_labels"], 0)
        self.assertEqual(report["certified_independent_audit_groups"], 0)
        self.assertEqual(report["audit_pairs_created"], 0)
        self.assertFalse(report["registration_ready"])

    def test_attribution_and_grants_required(self):
        self.assertEqual([len(g["creators"]) for g in self.grants], [10, 10])
        for grant in self.grants:
            self.assertTrue(grant["license_links"])
        grants = copy.deepcopy(self.grants)
        grants[0]["creators"] = []
        with self.assertRaisesRegex(ValueError, "attribution"):
            gate.admission(self.rows, self.book, grants)

    def test_noncommercial_original_refused(self):
        raw = b'<article><permissions><license>Creative Commons Attribution NonCommercial</license></permissions></article>'
        with self.assertRaisesRegex(ValueError, "noncommercial"):
            gate.source_grant(raw)

    def test_frozen_snapshot_tamper_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for name in ("SOURCE-MANIFEST.json", "nigeria-table-s2.txt", "bangladesh-workbook-summary.json", "source-grants.json"):
                (path / name).write_bytes((self.sources / name).read_bytes())
            (path / "nigeria-table-s2.txt").write_text(self.text.replace("in specification", "falsified", 1))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                gate.snapshot(path)

    def test_primary_standalone_cli(self):
        result = subprocess.run([sys.executable, "-W", "error", str(HERE / "audit_source_admission.py"), "--snapshot", str(self.sources)],
                                capture_output=True, text=True, timeout=20, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report["nigeria"]["released_specimens"], 260)
        self.assertEqual(report["status"], "AUDIT_ADMISSION_REFUSED")
        self.assertEqual(report["model_operations"], 0)


if __name__ == "__main__":
    unittest.main()
