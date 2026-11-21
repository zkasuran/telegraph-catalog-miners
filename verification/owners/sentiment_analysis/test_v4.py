# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only no-copy supervision, quarantine and guarded runtime controls."""
from collections import Counter
import copy
import hashlib
import os
import unittest
from unittest.mock import patch

from contracts import FORMATS, file_sha, strict_json
from source_data import group_id
import v3_data
import v3_remote
import v4_data as data
import v4_remote as runtime


def fixture(n=12):
    return [{"text": f"Synthetic unit source {label} sample number {index}", "label": label,
             "group_id": group_id(f"Synthetic unit source {label} sample number {index}")}
            for label in data.LABELS for index in range(n)]


class V4Tests(unittest.TestCase):
    def test_actual_frozen_source_and_warm_model_config(self):
        cfg = runtime.config()
        self.assertEqual(cfg["source_version"], 4)
        self.assertEqual(cfg["warm_package_sha256"], file_sha(runtime.WARM))
        self.assertEqual(cfg["baseline_sha256"], file_sha(data.BASELINE))
        self.assertEqual(cfg["hyperparameters"], strict_json((runtime.HERE / "v3-config.json").read_bytes())["hyperparameters"])

    def test_every_exposed_v2_v3_audit_identity_is_excluded(self):
        old, new = strict_json(v3_data.QUARANTINE.read_bytes()), strict_json(data.QUARANTINE.read_bytes())
        for key in ("sources", "entities", "threads", "source_groups", "constructed_groups"):
            self.assertLess(set(old[key]), set(new[key]))
        self.assertEqual(len(new["sources"]), 448)
        self.assertNotIn("text", new)

    def test_all_styles_and_conditions_have_balanced_contrasts(self):
        for style in range(3):
            rows = data.pairs(fixture(), style)
            self.assertEqual(len(rows), 288)
            self.assertEqual(Counter(row["condition"] for row in rows), {"task_only": 144, "untrusted_excerpt": 144})
            self.assertEqual(Counter(row["label"] for row in rows), {label: 72 for label in data.LABELS})

    def test_task_only_answers_do_not_copy_source_text(self):
        groups = fixture()
        by_id = {row["group_id"]: row for row in groups}
        for style in range(3):
            for row in data.pairs(groups, style):
                if row["condition"] == "task_only":
                    self.assertNotIn(by_id[row["group_id"]]["text"], row["chosen"])
                    self.assertNotIn(by_id[row["group_id"]]["text"], row["rejected"])

    def test_untrusted_excerpt_class_is_independent_of_task_class(self):
        groups = fixture()
        by_id = {row["group_id"]: row for row in groups}
        rows = [row for row in data.pairs(groups, 0) if row["condition"] == "untrusted_excerpt"]
        counts = Counter((row["label"], by_id[row["support_group_id"]]["label"]) for row in rows)
        self.assertEqual(set(counts.values()), {9})
        for row in rows:
            self.assertNotEqual(row["group_id"], row["support_group_id"])
            self.assertEqual(strict_json(row["chosen"])["reason"], strict_json(row["rejected"])["reason"])

    def test_source_grouping_policy_restores_original_globals_on_failure(self):
        old = v3_data.COUNTS, v3_data.QUARANTINE
        with self.assertRaises(RuntimeError):
            with data.grouping_policy():
                self.assertEqual(v3_data.COUNTS, data.COUNTS)
                raise RuntimeError("synthetic interrupted source gate")
        self.assertEqual((v3_data.COUNTS, v3_data.QUARANTINE), old)

    def test_training_policy_restores_original_pipeline_on_failure(self):
        old = v3_remote.config, v3_remote.pairs, v3_remote.WARM
        with self.assertRaises(RuntimeError):
            with runtime.training_policy():
                self.assertEqual(v3_remote.WARM, runtime.WARM)
                raise RuntimeError("synthetic interrupted training policy, no model loaded")
        self.assertEqual((v3_remote.config, v3_remote.pairs, v3_remote.WARM), old)

    def test_all_neural_entrypoints_refuse_locally_before_source(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false", "OWNER_SOURCE_ONLY": "1"}), patch.object(runtime, "source") as source:
            for function in (runtime.train, runtime.audit, runtime.evaluate, runtime.checkpoint):
                with self.assertRaises(RuntimeError):
                    if function == runtime.evaluate:
                        function(None, None, None, None)
                    elif function == runtime.checkpoint:
                        function(None, None, None, None, None, None, None)
                    else:
                        function(runtime.HERE)
            source.assert_not_called()

    def test_missing_label_or_condition_cannot_select_candidate(self):
        formats = {name: {"accuracy": .9, "per_label": {label: .9 for label in data.LABELS}} for name in FORMATS}
        valid = {"subsets": {condition: copy.deepcopy(formats) for condition in runtime.CONDITIONS}, "pairwise_accuracy": .9, "val_loss": .2}
        self.assertEqual(runtime.selection_key(valid)[:2], (.9, .9))
        for changed in ("label", "condition", "nan"):
            value = copy.deepcopy(valid)
            if changed == "label":
                value["subsets"]["untrusted_excerpt"]["json"]["per_label"].pop("Mixed")
            elif changed == "condition":
                value["subsets"].pop("untrusted_excerpt")
            else:
                value["subsets"]["task_only"]["json"]["accuracy"] = float("nan")
            with self.assertRaises(ValueError):
                runtime.selection_key(value)

    def test_worst_untrusted_label_beats_favorable_average_selection(self):
        def metrics(worst, overall):
            formats = {name: {"accuracy": overall, "per_label": {label: overall for label in data.LABELS}} for name in FORMATS}
            result = {"subsets": {condition: copy.deepcopy(formats) for condition in runtime.CONDITIONS}, "pairwise_accuracy": overall, "val_loss": .1}
            result["subsets"]["untrusted_excerpt"]["json"]["per_label"]["Negative"] = worst
            return result
        self.assertGreater(runtime.selection_key(metrics(.8, .85)), runtime.selection_key(metrics(.5, .99)))

    def quality_fixture(self):
        part = {"accuracy": .9, "per_label": {label: .9 for label in data.LABELS}, "untrained_accuracy": .5,
                "shortcut_accuracy": .5, "shortcuts": {"length": .5, "reference_answer_only": .5}}
        return {name: {**copy.deepcopy(part), "subsets": {condition: copy.deepcopy(part) for condition in runtime.CONDITIONS}} for name in FORMATS}

    def test_good_average_cannot_hide_failed_condition_or_neural_shortcut(self):
        self.assertTrue(runtime.quality_passed(self.quality_fixture()))
        for changed in ("condition_label", "condition_accuracy", "shortcut", "missing_reference"):
            values = self.quality_fixture()
            if changed == "condition_label":
                values["json"]["subsets"]["untrusted_excerpt"]["per_label"]["Negative"] = .5
            elif changed == "condition_accuracy":
                values["prose"]["subsets"]["task_only"]["accuracy"] = .7
            elif changed == "shortcut":
                values["json"]["shortcut_accuracy"] = .85
                values["json"]["shortcuts"]["reference_answer_only"] = .85
            else:
                values["json"]["shortcuts"].pop("reference_answer_only")
            with self.subTest(changed=changed):
                self.assertFalse(runtime.quality_passed(values))

    def test_missing_format_never_passes_quality(self):
        values = self.quality_fixture()
        values.pop("bare_label")
        self.assertFalse(runtime.quality_passed(values))

    def test_actual_development_proof_bytes_and_result_grants_are_retained(self):
        root = runtime.HERE / "diagnostics"
        notice = strict_json((root / "RESULT-NOTICE.json").read_bytes())
        for name, expected in notice["assets"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
        report = strict_json((root / "development-report.json").read_bytes())
        readback = strict_json((root / "task-label-readback.json").read_bytes())
        self.assertEqual(report["status"], "DEVELOPMENT_DIAGNOSTIC")
        self.assertEqual(readback["score_bits_verified"], 1728)
        self.assertEqual(readback["metrics_recomputed"]["cross_label_prompt"]["json"]["actual_prompt_label_preferred"]["wins"], 18)

    def test_training_structure_refuses_repetition_without_lowering_primary_limits(self):
        repeated = [{"chosen": "Positive", "rejected": "Negative"} for _ in range(200)]
        self.assertFalse(data.training_structure(repeated)["structural_passed"])
        groups = fixture(144)
        rows = data.pairs(groups, training=True)
        result = data.training_structure(rows)
        self.assertTrue(result["structural_passed"])
        self.assertGreaterEqual(result["metrics"]["unique_rejected_frac"], .8)
        self.assertLessEqual(result["metrics"]["top_rejected_share"], .02)
        self.assertLessEqual(result["metrics"]["eval_negative_leak"], .1)

    def test_training_reference_contexts_are_real_source_text_not_nonce_padding(self):
        groups = fixture(144)
        ids = {group["group_id"] for group in groups}
        for row in data.pairs(groups, training=True):
            self.assertEqual(len(row["support_group_ids"]), 2)
            self.assertTrue(set(row["support_group_ids"]) <= ids)
            self.assertNotIn(row["group_id"], row["support_group_ids"])
            self.assertNotIn(row["group_id"], row["chosen"])
            if row["chosen"].startswith("{"):
                chosen, rejected = strict_json(row["chosen"]), strict_json(row["rejected"])
                self.assertEqual(chosen["reason"], rejected["reason"])
                self.assertIn("Synthetic unit source", chosen["reason"])


if __name__ == "__main__":
    unittest.main()
