# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import unittest
from collections import Counter

from sampling import fit, weights


def fixture():
    return [{"id": str(i), "source_sha256": "a" * 64, "template_group": group,
             "response_view": view, "split": "train"}
            for group, count in (("many-variants", 4), ("few-variants", 1))
            for i in range(count) for view in ("prose", "json", "report")]


class SamplingChecks(unittest.TestCase):
    def test_primary_equal_group_mass_without_corpus_duplication_or_mutation(self):
        rows = fixture()
        before = copy.deepcopy(rows)
        frozen = fit(rows)
        actual = weights(rows, frozen)
        self.assertEqual(frozen["coefficients"], {"few-variants": 4, "many-variants": 1})
        mass = Counter()
        for row, weight in zip(rows, actual):
            mass[row["template_group"]] += weight
        self.assertEqual(set(mass.values()), {12})
        self.assertEqual(len(actual), len(rows))
        self.assertEqual(rows, before)

    def test_train_only_fit_refuses_validation_and_audit(self):
        for split in ("validation", "audit"):
            rows = fixture()
            rows[0]["split"] = split
            with self.assertRaisesRegex(ValueError, "training views only"):
                fit(rows)

    def test_frozen_coefficients_refuse_identity_drift_and_bad_weights(self):
        rows = fixture()
        frozen = fit(rows)
        changed = copy.deepcopy(rows)
        changed[0]["source_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "identity"):
            weights(changed, frozen)
        for value in (0, -1, True, 1.5, float("nan"), 17):
            bad = copy.deepcopy(frozen)
            bad["coefficients"]["few-variants"] = value
            with self.assertRaisesRegex(ValueError, "coefficients"):
                weights(rows, bad)
        bad = copy.deepcopy(frozen)
        bad["coefficients"]["few-variants"] = 1
        with self.assertRaisesRegex(ValueError, "equalize"):
            weights(rows, bad)

    def test_repeat_fit_is_exact_integer_bytes(self):
        import json
        rows = fixture()
        self.assertEqual(json.dumps(fit(rows), sort_keys=True), json.dumps(fit(rows), sort_keys=True))
        self.assertTrue(all(type(v) is int for v in fit(rows)["coefficients"].values()))

    def test_actual_torch_control_refuses_local_execution(self):
        from sampling_remote import check
        from unittest.mock import patch
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Model loading"):
                check([], {}, "unused", 42)


if __name__ == "__main__":
    unittest.main()
