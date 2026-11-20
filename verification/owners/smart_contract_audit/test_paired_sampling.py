# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import json
import math
import unittest
from pathlib import Path
from unittest.mock import patch

from data import response_views
from paired_sampling import pairs, report
from sampling import fit


def fixture():
    rows = []
    for group, number in (("design-many", 4), ("design-few", 1)):
        for entity in range(number):
            for vulnerable in (True, False):
                rows.append({"id": group + str(entity), "sibling_group": group + str(entity), "template_group": group,
                             "split": "train", "vulnerable": vulnerable, "source_sha256": ("a" if vulnerable else "b") * 64,
                             "response_view": "prose", "facts": {"call": "same disputed call"},
                             "chosen": "yes" if vulnerable else "no", "rejected": "no" if vulnerable else "yes"})
    return rows


class PairedChecks(unittest.TestCase):
    def test_primary_complete_literal_siblings_without_mutation(self):
        rows = fixture()
        before = copy.deepcopy(rows)
        batches, coefficients = pairs(rows, fit(rows))
        self.assertEqual(len(batches), 5)
        self.assertEqual(coefficients.count(1), 4)
        self.assertEqual(coefficients.count(4), 1)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(len(rows))))
        for left, right in batches:
            self.assertEqual({rows[left]["vulnerable"], rows[right]["vulnerable"]}, {True, False})
            self.assertEqual(rows[left]["chosen"], rows[right]["rejected"])
        self.assertEqual(rows, before)

    def test_missing_sibling_same_label_and_literal_shortcut_are_refused(self):
        for defect in ("same label", "different answer", "different call"):
            rows = fixture()
            if defect == "same label":
                rows[1]["vulnerable"] = True
            elif defect == "different answer":
                rows[1]["chosen"] += " extra"
            else:
                rows[1]["facts"]["call"] = "another call"
            with self.assertRaises(ValueError):
                pairs(rows, fit(rows))
        rows = fixture()[:-1]
        with self.assertRaises(ValueError):
            pairs(rows, fit(rows))

    def test_audit_inputs_and_local_runtime_are_refused(self):
        rows = fixture()
        frozen = fit(rows)
        rows[0]["split"] = "audit"
        with self.assertRaises(ValueError):
            pairs(rows, frozen)
        from paired_sampling_remote import SiblingBatchSampler
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Model loading"):
                SiblingBatchSampler([], {}, 42)

    def test_answer_only_bias_cannot_reduce_opposed_pair_loss(self):
        def loss(difference):
            return (math.log1p(math.exp(-difference)) + math.log1p(math.exp(difference))) / 2
        for difference in (-10, -2, -0.5, 0.5, 2, 10):
            self.assertGreater(loss(difference), loss(0))
            self.assertAlmostEqual(loss(difference), loss(-difference))

    def test_actual_frozen_training_plan_is_complete(self):
        import data
        import tempfile
        with tempfile.TemporaryDirectory() as out:
            rows = data.build(Path(out))
        train = response_views([r for r in rows if r["split"] == "train"], "train")
        frozen = json.loads(Path(__file__).with_name(data.SAMPLING_FILE).read_text())
        actual = report(train, frozen)
        self.assertEqual(actual["training_views"], 432)
        self.assertEqual(actual["pair_batches"], 216)
        self.assertEqual(actual["training_groups"], 12)


if __name__ == "__main__":
    unittest.main()
