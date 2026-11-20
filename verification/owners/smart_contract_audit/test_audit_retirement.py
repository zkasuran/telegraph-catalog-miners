# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import data
from sampling import fit_rational, weights


class RetirementChecks(unittest.TestCase):
    def test_exposed_audit_cannot_generate_new_rows(self):
        for template in range(8, 40):
            with self.assertRaisesRegex(ValueError, "retired exposed"):
                data.source_for(template, 5000, True)

    def test_current_training_and_validation_retained_but_audit_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = data.build(root)
            retained = [r for r in rows if r["split"] == "train" and not r["template_group"].startswith("fresh-train-")]
            raw = b"".join((json.dumps(r, sort_keys=True) + "\n").encode() for r in retained)
            self.assertEqual(data.sha(raw), "657badf477d3e67fce0311000688117a8c58d00add9ee79dc2f4d7782cac264a")
            self.assertEqual(data.sha((root / "validation.jsonl").read_bytes()), "8c79f65c18479354acc40065036a7d1e7c26a6ddcb6cc2b41eb608266138aec4")
            audit = [r for r in rows if r["split"] == "audit"]
            self.assertEqual(len(audit), 64)
            self.assertEqual(len({r["template_group"] for r in audit}), 32)
            self.assertTrue(all(r["template_group"].startswith("fresh-audit-") for r in audit))
            self.assertNotEqual(data.sha((root / "audit.jsonl").read_bytes()), data.RETIRED_AUDIT_SHA256)

    def test_frozen_rational_weights_equalize_groups_without_refitting(self):
        rows = [{"id": str(i), "source_sha256": "a" * 64, "split": "train", "response_view": "prose", "template_group": g}
                for g, n in (("many", 60), ("medium", 24), ("few", 6)) for i in range(n)]
        frozen = fit_rational(rows)
        actual = weights(rows, frozen)
        masses = {g: sum(Fraction(1, frozen["denominators"][g]) for r in rows if r["template_group"] == g) for g in frozen["coefficients"]}
        self.assertEqual(set(masses.values()), {Fraction(1)})
        self.assertTrue(all(0 < w <= 1 for w in actual))
        bad = copy.deepcopy(frozen)
        bad["denominators"]["few"] = 7
        with self.assertRaises(ValueError):
            weights(rows, bad)
        for split in ("validation", "audit"):
            changed = copy.deepcopy(rows)
            changed[0]["split"] = split
            with self.assertRaises(ValueError):
                fit_rational(changed)


if __name__ == "__main__":
    unittest.main()
