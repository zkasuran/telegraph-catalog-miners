# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only development boundary and alignment controls. No model operations."""
import copy
import os
import unittest
from unittest.mock import patch

import development_probe as probe
from contracts import FORMATS, file_sha, strict_json, validate_ready
from source_data import group_id


def validation_fixture():
    result = []
    for label in probe.LABELS:
        for index in range(8):
            text = f"Synthetic unit fixture {label} text {index}"
            result.append({"text": text, "label": label, "group_id": group_id(text), "composition": 1})
    return result


class DevelopmentTests(unittest.TestCase):
    def test_frozen_validation_config_and_original_baseline(self):
        cfg = probe.config()
        manifest = strict_json((probe.HERE / "model/v3-source-manifest.json").read_bytes())
        self.assertEqual(cfg["validation_groups_sha256"], manifest["files"]["validation"]["groups_sha256"])
        self.assertEqual(cfg["baseline_sha256"], file_sha(probe.BASELINE))
        self.assertEqual(cfg["package_sha256"], file_sha(probe.PACKAGE))

    def test_audit_split_bool_counts_and_missing_hash_are_refused(self):
        for changed in ({"split": "audit"}, {"per_label_groups": True}, {"per_label_groups": 17}, {"package_sha256": ""}, {"max_tokens": 512}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                probe.check_config({**probe.config(), **changed})
        with self.assertRaises(ValueError):
            probe.check_config([])

    def test_primary_cases_are_balanced_deterministic_and_all_formats(self):
        first = probe.cases(validation_fixture(), 8)
        self.assertEqual(first, probe.cases(list(reversed(validation_fixture())), 8))
        self.assertEqual(set(first), set(probe.CONDITIONS))
        for formats in first.values():
            self.assertEqual(set(formats), set(FORMATS))
            for rows in formats.values():
                self.assertEqual(len(rows), 96)
                self.assertEqual({label: sum(row["label"] == label for row in rows) for label in probe.LABELS}, {label: 24 for label in probe.LABELS})

    def test_cross_label_prompt_control_preserves_answers_and_uses_another_group(self):
        cases = probe.cases(validation_fixture(), 8)
        for name in FORMATS:
            for aligned, changed in zip(cases["aligned"][name], cases["cross_label_prompt"][name]):
                self.assertNotEqual(changed["prompt_label"], aligned["label"])
                self.assertNotEqual(changed["prompt_group_id"], aligned["group_id"])
                self.assertNotEqual(changed["prompt"], aligned["prompt"])
                self.assertEqual((changed["chosen"], changed["rejected"]), (aligned["chosen"], aligned["rejected"]))

    def test_answer_only_control_keeps_excerpt_and_removes_only_task(self):
        cases = probe.cases(validation_fixture(), 8)
        for name in FORMATS:
            for aligned, changed in zip(cases["aligned"][name], cases["answer_only"][name]):
                self.assertEqual(changed["prompt"], "")
                self.assertEqual(changed["chosen"], aligned["chosen"])
                self.assertEqual(changed["rejected"], aligned["rejected"])

    def test_missing_label_duplicate_source_and_unbounded_selection_are_refused(self):
        for rows in (validation_fixture()[:-8], [*validation_fixture()[:-1], validation_fixture()[-2]]):
            with self.assertRaises(ValueError):
                probe.cases(rows, 8)
        with self.assertRaises(ValueError):
            probe.cases(validation_fixture(), True)

    def test_literal_primary_task_answer_render_is_not_normalized(self):
        row = {"prompt": " raw task\n", "chosen": "Positive", "rejected": "Negative"}
        self.assertEqual(probe.render(row, "chosen"), "### Task:\n raw task\n\n\n### Answer:\nPositive")

    def test_actual_finite_score_contract_ties_and_missing_scores(self):
        rows = probe.cases(validation_fixture(), 8)["aligned"]["bare_label"]
        coefficient = strict_json(probe.BASELINE.read_bytes())
        tied = probe.measure(rows, [0, 0] * len(rows), coefficient)
        self.assertEqual(tied["ties"], len(rows))
        self.assertEqual(tied["pair_preference_rate"], 0)
        won = probe.measure(rows, [1065353216, 0] * len(rows), coefficient)
        self.assertEqual(won["pair_preference_rate"], 1)
        self.assertTrue(won["shortcuts"])
        for bits in ([0], [0x7f800000, 0] * len(rows), [True, 0] * len(rows)):
            with self.assertRaises(ValueError):
                probe.measure(rows, bits, coefficient)

    def test_remote_neural_path_refuses_locally_before_source_or_model(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false", "OWNER_SOURCE_ONLY": "1"}), patch.object(probe, "source") as source:
            with self.assertRaises(RuntimeError):
                probe.remote(probe.HERE)
            source.assert_not_called()

    def test_development_measurement_is_never_registration_ready(self):
        actual = strict_json((probe.HERE / "model/endpoint-registration-ready.json").read_bytes())
        changed = copy.deepcopy(actual)
        changed["status"] = "DEVELOPMENT_DIAGNOSTIC"
        with self.assertRaises(ValueError):
            validate_ready(changed)

    def diagnostic_fixture(self):
        # Schema fixture only. No model was executed and this is not exported.
        cfg = probe.config()
        origin = strict_json((probe.HERE / "model/endpoint-registration-ready.json").read_bytes())
        metrics = {condition: {name: {"pair_preference_rate": 0.0, "pairs": 96, "ties": 0,
                                      "per_label": {label: 0.0 for label in probe.LABELS}, "shortcuts": {"length": .5}}
                              for name in FORMATS} for condition in probe.CONDITIONS}
        return {"schema_version": 1, "status": "DEVELOPMENT_DIAGNOSTIC", "family": probe.FAMILY, "owner": probe.OWNER,
                "run_id": 1, "commit": "a" * 40, "model_fitted": False, "candidate_selected": False, "audit_rows_scored": 0,
                "independent_groups": 32, "package_sha256": cfg["package_sha256"], "package_keccak": cfg["package_keccak"],
                "config_sha256": file_sha(probe.CONFIG), "foundation_sha256": origin["foundation_sha256"],
                "engine_sha256": origin["engine_sha256"], "tokenizer_sha256": origin["tokenizer_sha256"],
                "cases_sha256": cfg["cases_sha256"], "score_bits_sha256": "b" * 64,
                "source": {"split": "validation", "validation_groups_sha256": cfg["validation_groups_sha256"],
                           "baseline_sha256": cfg["baseline_sha256"], "normalized_overlap": 0, "grants_verified": True, "audit_rows_scored": 0},
                "metrics": metrics, "runtime": {**copy.deepcopy(origin["runtime"]), "literal_wire_passed": True}}

    def test_diagnostic_contract_accepts_honest_low_measurements_not_quality_claim(self):
        fixture = self.diagnostic_fixture()
        self.assertEqual(probe.validate_diagnostic(fixture), fixture)
        with self.assertRaises(ValueError):
            validate_ready(fixture)

    def test_diagnostic_cannot_hide_selection_audit_use_bad_runtime_or_labels(self):
        fixture = self.diagnostic_fixture()
        for key, value in (("candidate_selected", True), ("audit_rows_scored", 1), ("audit_rows_scored", False),
                           ("package_keccak", "0x" + fixture["package_keccak"]), ("cases_sha256", "a" * 64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.validate_diagnostic({**fixture, key: value})
        for mutation in ("missing_label", "nonfinite", "missing_condition", "runtime", "ties"):
            changed = copy.deepcopy(fixture)
            if mutation == "missing_label":
                changed["metrics"]["aligned"]["json"]["per_label"].pop("Mixed")
            elif mutation == "nonfinite":
                changed["metrics"]["aligned"]["json"]["pair_preference_rate"] = float("nan")
            elif mutation == "missing_condition":
                changed["metrics"].pop("answer_only")
            elif mutation == "runtime":
                changed["runtime"]["full_parser_passed"] = False
            else:
                changed["metrics"]["aligned"]["json"]["ties"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                probe.validate_diagnostic(changed)


if __name__ == "__main__":
    unittest.main()
