# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Authored metric callbacks and guard controls. No neural model or audit data."""
import copy
import sys
import unittest
from unittest.mock import patch

from contracts import FAMILY, FORMATS, LABELS
from quality_selection import attach_selection, check_experiments, compile_selector, gate_slack, require_selection_trace
from quality_selection_remote import selector_adapter


def formats(accuracy=.8, entailment=.8, contradiction=.8):
    return {style: {"accuracy": accuracy, "per_label": dict(zip(LABELS, (entailment, contradiction))),
            "fit_split": "validation", "original_pairs": 30, "untrained_accuracy": .5, "shortcut_accuracy": .5}
            for style in FORMATS}


class QualitySelectionTests(unittest.TestCase):
    def test_primary_selector_prefers_passing_labels_not_larger_aggregate(self):
        collapsed = formats(5/6, 1., 2/3)
        balanced = formats()
        self.assertGreater(min(m["accuracy"] for m in collapsed.values()), .8)
        self.assertLess(gate_slack(collapsed, 30)["score"], gate_slack(balanced, 30)["score"])
        self.assertEqual(gate_slack(balanced, 30)["score"], 0)

    def test_metric_attachment_never_relabels_native_accuracy_or_loss(self):
        measured = {"pairwise_accuracy": 5/6, "val_loss": .123, "n_eval": 60}
        actual, selection = attach_selection(measured, formats(5/6, 1., 2/3), 30)
        self.assertEqual(actual["pairwise_accuracy"], measured["pairwise_accuracy"])
        self.assertEqual(actual["val_loss"], measured["val_loss"])
        self.assertNotIn("selection_gate_slack", measured)
        self.assertEqual(actual["selection_gate_slack"], selection["score"])

    def test_missing_label_or_format_refuses_selection(self):
        changed = formats()
        del changed["prose"]["per_label"]["Contradiction"]
        with self.assertRaises(ValueError):
            gate_slack(changed, 30)
        with self.assertRaises(ValueError):
            gate_slack({"json": formats()["json"]}, 30)

    def test_audit_train_or_wrong_population_refuses_selection(self):
        for split in ("audit", "train"):
            changed = formats()
            changed["prose"]["fit_split"] = split
            with self.assertRaises(ValueError):
                gate_slack(changed, 30)
        with self.assertRaises(ValueError):
            gate_slack(formats(), 31)

    def test_nonfinite_metrics_cannot_improve_a_selector(self):
        for value in (float("nan"), float("inf"), True, -1):
            changed = formats()
            changed["json"]["accuracy"] = value
            with self.assertRaises(ValueError):
                gate_slack(changed, 30)
        with self.assertRaises(ValueError):
            attach_selection({"pairwise_accuracy": .8, "val_loss": float("nan")}, formats(), 30)

    def test_untrained_and_strongest_shortcut_remain_constraints(self):
        changed = formats()
        changed["json"]["untrained_accuracy"] = .8
        self.assertEqual(gate_slack(changed, 30)["weakest_constraint"], "json:untrained")
        changed = formats()
        changed["prose"]["shortcut_accuracy"] = .8
        self.assertEqual(gate_slack(changed, 30)["weakest_constraint"], "prose:shortcut")

    def test_source_transform_refuses_unpinned_source(self):
        with self.assertRaisesRegex(ValueError, "primary trainer source"):
            compile_selector(b"authored not the primary trainer")
        workflow = {"on": {"workflow_dispatch": {"inputs": {"objective": {"type": "choice", "default": "independent",
            "options": ["independent", "paired-worst-format", "quality-slack", "paired-quality-slack"]}}}}}
        check_experiments(workflow)
        changed = copy.deepcopy(workflow)
        changed["on"]["workflow_dispatch"]["inputs"].update({"authored-" + str(i): {} for i in range(10)})
        with self.assertRaisesRegex(ValueError, "ten-input"):
            check_experiments(changed)

    def test_neural_selector_guard_precedes_source_or_trainer_access(self):
        before = set(sys.modules)
        with patch.dict("os.environ", {"OWNER_SOURCE_ONLY": "1"}):
            with self.assertRaisesRegex(RuntimeError, "bounded public GitHub"):
                with selector_adapter(object()):
                    self.fail("local neural selector activated")
        self.assertFalse({"torch", "numpy", "transformers", "peft"} & (set(sys.modules) - before))

    def test_independent_audit_requires_bound_actual_selection_records(self):
        require_selection_trace({}, None, None)
        freeze = {"validation_selector": "quality-slack", "family": FAMILY, "run_id": 1,
                  "commit": "a" * 40, "source_manifest_sha256": "b" * 64, "validation_selection_trace_sha256": "c" * 64}
        trace = {"schema_version": 1, "selector": "quality-slack", "family": FAMILY, "run_id": 1,
                 "commit": "a" * 40, "source_manifest_sha256": "b" * 64, "audit_used_for_selection": False,
                 "records": [{"epoch_index": 0, "formats": formats(), "selection": gate_slack(formats(), 30)}]}
        require_selection_trace(freeze, trace, "c" * 64)  # Authored metadata, not a model result.
        for key, value in (("records", []), ("audit_used_for_selection", True), ("run_id", 2)):
            with self.assertRaises(ValueError):
                require_selection_trace(freeze, {**trace, key: value}, "c" * 64)
        changed = copy.deepcopy(trace)
        changed["records"][0]["selection"]["score"] = 1
        with self.assertRaises(ValueError):
            require_selection_trace(freeze, changed, "c" * 64)
        with self.assertRaises(ValueError):
            require_selection_trace(freeze, trace, "d" * 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
