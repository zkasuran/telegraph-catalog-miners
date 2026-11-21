# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source contracts and local model-refusal controls for the opt-in executor."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import unittest

from support import HERE, load_inputs, training_representations, validate_development_workflow, validate_rows
from core import answer_value
import train


class DevelopmentControls(unittest.TestCase):
    def inputs(self):
        return load_inputs()

    def test_real_bound_source_inputs_smoke(self):
        spec, rows, report = self.inputs()
        self.assertEqual((len(rows["train"]), len(rows["validation"])), (192, 32))
        self.assertIs(report["independent_audit"], False)
        self.assertEqual(report["negative_tmax"]["train"], 45)
        self.assertEqual(report["negative_tmax"]["validation"], 18)
        self.assertTrue(validate_rows(rows, spec))

    def test_rotating_views_preserve_every_source_question_once(self):
        spec, rows, _ = self.inputs()
        represented = training_representations(rows["train"])
        self.assertEqual({name: sum(row["answer_view"] == name for row in represented) for name in ("json", "celsius", "bare")},
                         {"json": 64, "celsius": 64, "bare": 64})
        self.assertEqual(len({row["group"] for row in represented}), 192)
        for original, row in zip(rows["train"], represented):
            self.assertEqual(original["source_evidence"], row["source_evidence"])
            for side in ("chosen", "rejected"):
                source_value = json.loads(original[side], parse_float=Decimal)["temp_max_c"]
                value = answer_value(row[side], spec["station_names"][row["station"]], row["day"])
                self.assertEqual(source_value, value)

    def test_source_flag_cell_value_and_capture_digest_tampering_refused(self):
        spec, rows, _ = self.inputs()
        for field, value in (("flag", "W"), ("cell", "  123  0"), ("quality", "X"), ("sha256", "bad")):
            altered = deepcopy(rows)
            altered["train"][0]["source_evidence"][field] = value
            with self.assertRaises(ValueError):
                validate_rows(altered, spec)

    def test_literal_prompt_date_and_answer_tampering_refused(self):
        spec, rows, _ = self.inputs()
        for field, value in (("prompt", "a source-injected stand-in"), ("day", "2026-12-31"), ("chosen", "{\"temp_max_c\":0}")):
            altered = deepcopy(rows)
            altered["train"][0][field] = value
            with self.assertRaises(ValueError):
                validate_rows(altered, spec)

    def test_original_nontraining_entities_and_audit_split_refused(self):
        spec, rows, _ = self.inputs()
        altered = deepcopy(rows)
        altered["train"][0]["station"] = "USW00094728"
        with self.assertRaises(ValueError):
            validate_rows(altered, spec)
        with self.assertRaises(ValueError):
            validate_rows({**rows, "audit": []}, spec)

    def test_unbalanced_offsets_and_nonfinite_margin_refused(self):
        spec, rows, _ = self.inputs()
        for value in (float("nan"), True, 1):
            altered = deepcopy(rows)
            altered["train"][0]["margin"] = value
            with self.assertRaises(ValueError):
                validate_rows(altered, spec)

    def test_local_neural_execution_and_baseline_fit_refused(self):
        with self.assertRaises(RuntimeError):
            train.run(Path("not-a-model-run"))
        with self.assertRaises(RuntimeError):
            train.frozen_controls({"train": [], "validation": []})

    def test_development_flag_and_original_audit_separation_controls(self):
        workflow = json.loads((HERE.parent / "workflow.json").read_text())
        self.assertTrue(validate_development_workflow(workflow))
        altered = deepcopy(workflow)
        altered["on"]["workflow_dispatch"]["inputs"]["development_only"]["default"] = True
        with self.assertRaises(ValueError):
            validate_development_workflow(altered)
        altered = deepcopy(workflow)
        altered["jobs"]["audit"]["if"] = "${{ inputs.mode == 'experiment' }}"
        with self.assertRaises(ValueError):
            validate_development_workflow(altered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
