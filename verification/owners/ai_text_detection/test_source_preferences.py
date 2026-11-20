# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only controls on real retained public training records. No audit rows."""
import copy
import json
import unittest

from common import HERE, pair, fit_baselines, baseline_report, validation_gate, FORMATS
from augmentation_contract import validate_restored
from development_contract import validate_locked, repetition_gate
from window_contract import expanded_training_pairs
from compact_contract import compact_training
from origin_contract import equal_byte_views
from supplement_contract import source_bundle, validate_generation
from source_preferences import preference_rows, preference_policy


def population():
    source_raw = (HERE / "sources.json").read_bytes()
    source = json.loads(source_raw)
    recovery = HERE / "recovery"
    original = validate_locked((recovery / "locked-development.jsonl").read_bytes(), source,
        json.loads((recovery / "locked-development-producer.json").read_text()),
        json.loads((recovery / "locked-development-lineage.json").read_text()))
    augmented = []
    for name, plan in (("locked-augmentation", "train-window-plan.json"),
                       ("locked-additional-augmentation", "additional-train-window-plan.json")):
        rows, _ = validate_restored((recovery / (name + ".jsonl")).read_bytes(),
            (recovery / (name + "-producer.json")).read_bytes(),
            json.loads((recovery / (name + "-lineage.json")).read_text()),
            (HERE / plan).read_bytes(), source, source_raw)
        augmented += rows
    train, _ = expanded_training_pairs(original, augmented, bare_group_count=6)
    train = compact_training(train)
    validation = [pair(r) for r in original if r["split"] == "validation"]
    _, extra, base = source_bundle()
    raw = (recovery / "locked-supplement.jsonl").read_bytes()
    report = json.loads((recovery / "locked-supplement-producer.json").read_text())
    views = validate_generation(raw, report, extra, base, equal_byte_views(train) + equal_byte_views(validation))
    return train, validation, views


class SupplementPreferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.validation, cls.views = population()

    def test_real_retained_population_and_original_preservation(self):
        before = copy.deepcopy(self.train)
        rows, binding = preference_rows(self.views, self.train, self.validation)
        self.assertEqual(self.train, before)
        self.assertEqual(binding["combined_rows"], 880)
        self.assertEqual(binding["independent_training_groups"], 192)
        self.assertEqual(binding["new_bare_rows"], 0)
        self.assertEqual(sum(r["style"] == "bare_label" for r in self.train + rows), 12)
        self.assertEqual(sum(r["label"] == "Human" for r in rows), 64)
        self.assertTrue(binding["repetition"]["passed"])
        self.assertEqual(binding["normalized_overlap"], 0)
        self.assertFalse(binding["audit_used"])

    def test_preference_source_role_identity_corruptions_are_refused(self):
        for key, value in (("split", "test"), ("year", 2016), ("human", True),
                           ("group_id", "f" * 64), ("text_id", "f" * 64), ("text_bytes", 31)):
            rows = copy.deepcopy(self.views)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                preference_rows(rows, self.train, self.validation)

    def test_preference_normalized_original_or_validation_overlap_is_refused(self):
        for old in (self.train[0], self.validation[0]):
            rows = copy.deepcopy(self.views)
            rows[0]["original_text_id"] = old["text_id"]
            with self.assertRaisesRegex(ValueError, "normalized"):
                preference_rows(rows, self.train, self.validation)

    def test_preference_group_population_and_origin_balance_are_required(self):
        with self.assertRaises(ValueError):
            preference_rows(self.views[:-1], self.train, self.validation)
        rows = copy.deepcopy(self.views)
        grouped = {}
        for r in rows:
            grouped.setdefault(r["view_id"], []).append(r)
        first = next(iter(grouped.values()))
        first[1]["human"] = first[0]["human"]
        with self.assertRaisesRegex(ValueError, "paired"):
            preference_rows(rows, self.train, self.validation)

    def test_preference_quotes_are_true_and_the_same_on_both_sides(self):
        rows, _ = preference_rows(self.views, self.train, self.validation)
        for row in rows:
            words = json.loads(row["prompt"])["text"].split()
            if row["style"] == "json":
                chosen, rejected = json.loads(row["chosen"]), json.loads(row["rejected"])
                self.assertEqual(chosen["cues"], rejected["cues"])
                self.assertEqual(chosen["cues"], ["Opening words: " + " ".join(words[:4]), "Ending words: " + " ".join(words[-4:])])
            else:
                self.assertEqual(row["chosen"].split(". Opening words: ", 1)[1], row["rejected"].split(". Opening words: ", 1)[1])
            self.assertEqual(row["split"], "train")
            self.assertIn(row["style"], ("json", "prose"))

    def test_preference_policy_cannot_change_audit_or_runtime(self):
        recipe = json.loads((HERE / "recipe.json").read_text())
        self.assertEqual(preference_policy(recipe)["new_preference_rows"], 128)
        for key, value in (("audit_used", True), ("new_bare_rows", 2), ("runtime_transform", True), ("pair_coefficient", 5),
                           ("pair_coefficient", True), ("new_bare_rows", False), ("original_phase_unchanged", 1)):
            bad = copy.deepcopy(recipe)
            bad["supplement_preferences"][key] = value
            with self.assertRaises(ValueError):
                preference_policy(bad)

    def test_original_strongest_shortcut_remains_a_required_comparison(self):
        rows, _ = preference_rows(self.views, self.train, self.validation)
        fitted = fit_baselines(self.train + rows)
        fitted["original_source_origin_length"] = fit_baselines(self.train)["source_origin_length"]
        before = copy.deepcopy(fitted)
        measured = baseline_report(self.validation, fitted)
        self.assertEqual(measured["original_source_origin_length"], 0.78125)
        metrics = {s: {"n_eval": 64, "pairwise_accuracy": 0.85, "val_loss": 0.2,
            "accuracy_by_label": {"Human": 0.85, "AI-generated": 0.85}} for s in FORMATS}
        baselines = {s: measured for s in FORMATS}
        self.assertFalse(validation_gate(metrics, baselines)["passed"])
        self.assertEqual(fitted, before)

    def test_combined_data_cannot_hide_repeated_negative_failure(self):
        bad = copy.deepcopy(self.train)
        for row in bad:
            row["rejected"] = "one repeated incorrect verdict"
        self.assertFalse(repetition_gate(bad)["passed"])
        with self.assertRaisesRegex(ValueError, "unchanged repetition"):
            preference_rows(self.views, bad, self.validation)


if __name__ == "__main__":
    unittest.main()
