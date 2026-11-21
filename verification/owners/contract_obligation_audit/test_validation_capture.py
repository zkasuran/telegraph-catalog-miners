# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Authored capture controls only, with no model loading or inference."""
import copy
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from contracts import FAMILY, FORMATS, FOUNDATION
from executor import ReferenceExecutor
from test_development_diagnostics import rows, wires
from validation_capture_remote import capture, capture_batches, summarize, validation_vectors, verify_freeze


class ValidationCaptureTests(unittest.TestCase):
    def test_primary_literal_capture_and_label_hypothesis_report(self):
        originals = rows("validation")
        vectors = validation_vectors(originals, lambda text: [1, 2])
        returned = wires([1, 0, 0, 0])["json"]
        score = Mock(side_effect=[copy.deepcopy(returned), copy.deepcopy(returned)])
        responses = capture_batches(score, vectors, "a" * 64, 20, clock=lambda: 1)
        result = summarize(originals, responses, "a" * 64)
        self.assertEqual(score.call_count, 2)
        self.assertEqual(score.call_args.kwargs["timeout"], 19)
        self.assertEqual(result["formats"]["json"]["per_hypothesis"]["authored-hypothesis"]["ties"], 1)
        self.assertFalse(result["independent_audit"])
        self.assertFalse(result["new_inference_performed"])

    def test_capture_guard_refuses_local_model_work_before_reading_files(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false", "OWNER_SOURCE_ONLY": "1"}, clear=True):
            with self.assertRaises(RuntimeError):
                capture(Path("authored-no-model-or-files"))

    def test_actual_freeze_identity_and_audit_selection_controls(self):
        source = {"files": {"validation": {"sha256": "a" * 64}}}
        freeze = {"schema_version": 1, "family": FAMILY, "run_id": 1, "commit": "b" * 40,
                  "audit_used_for_selection": False, "source_manifest_sha256": "c" * 64,
                  "candidate_sha256": "d" * 64, "untrained_sha256": "e" * 64}
        verify_freeze(freeze, source, "c" * 64, "a" * 64, 1, "b" * 40)
        for key, value in (("run_id", 2), ("commit", "f" * 40), ("audit_used_for_selection", True),
                           ("source_manifest_sha256", "f" * 64), ("candidate_sha256", "0x" + "d" * 64)):
            with self.assertRaises(ValueError):
                verify_freeze({**freeze, key: value}, source, "c" * 64, "a" * 64, 1, "b" * 40)

    def test_audit_rows_and_truncated_vectors_are_refused(self):
        with self.assertRaises(ValueError):
            validation_vectors(rows("audit"), lambda text: [1])
        with self.assertRaises(ValueError):
            validation_vectors(rows("validation"), lambda text: [1] * 513)

    def test_literal_prefix_count_nonfinite_and_deadline_controls(self):
        vectors = {style: [[1], [2], [3], [4]] for style in FORMATS}
        for mutation in ("prefix", "count", "nonfinite"):
            response = wires([1, 0, 1, 0])["json"]
            if mutation == "prefix":
                response["adapter_hash"] = "0x" + "a" * 64
            elif mutation == "count":
                response["results"].pop()
            else:
                response["results"][0]["score"] = 0x7F800000
            with self.assertRaises(ValueError):
                capture_batches(Mock(return_value=response), vectors, "a" * 64, 20, clock=lambda: 1)
        score = Mock()
        with self.assertRaises(TimeoutError):
            capture_batches(score, vectors, "a" * 64, 1, clock=lambda: 1)
        score.assert_not_called()

    def test_reference_timeout_invalid_controls_precede_process_access(self):
        backend = ReferenceExecutor.__new__(ReferenceExecutor)
        for timeout in (0, -1, 601, True, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                backend.score({}, timeout=timeout)


if __name__ == "__main__":
    unittest.main()
