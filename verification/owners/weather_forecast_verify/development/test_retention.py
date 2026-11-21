# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Constructed JSON fixtures verify byte retention, never model fitting or inference."""
import json
from pathlib import Path
import tempfile
import unittest

from support import HERE
from retention import FILES, licensed_controls, retain_fitted_records, validate_controls, verify_retained_records


class RetentionControls(unittest.TestCase):
    def controls(self):
        return {"fit_split": "train", "integer_coefficients": {"length": 1, "answer_only_numeric": -1,
                "answer_only_digits": 1, "overlap": -1}, "answer_only_token_weights": {"fixture": 2}}

    def terms(self):
        return {name: (HERE.parent / filename).read_text() for name, filename in
                (("sand", "LICENSE"), ("kit_mit", "KIT-LICENSE"), ("apache_2_0", "UPSTREAM-APACHE-2.0.txt"))}

    def setup_files(self, directory):
        root = Path(directory)
        context = root / "development-context"
        (context / "data").mkdir(parents=True)
        controls = licensed_controls(self.controls(), self.terms())
        raw = (json.dumps(controls, indent=2, sort_keys=True) + "\n").encode()
        (context / "data/frozen-controls.json").write_bytes(raw)
        (context / "data/data-report.json").write_text('{"independent_audit":false,"fixture_only":true}\n')
        return root, context, raw

    def test_old_nested_snapshot_is_outside_literal_root_json_contract(self):
        workflow = json.loads((HERE.parent / "workflow.json").read_text())
        upload = next(step for step in workflow["jobs"]["train"]["steps"] if step.get("name") == "Retain licensed failed or successful checkpoint")
        self.assertIn("${{ runner.temp }}/weather-result/*.json", upload["with"]["path"].splitlines())
        with tempfile.TemporaryDirectory() as directory:
            root, context, _ = self.setup_files(directory)
            self.assertTrue((context / "data/frozen-controls.json").is_file())
            self.assertFalse((root / "frozen-development-controls.json").exists())

    def test_primary_exact_bytes_and_in_bytes_terms_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root, context, raw = self.setup_files(directory)
            identity = {"family": "WEATHER_FORECAST_VERIFY", "run_id": 7, "commit": "a" * 40}
            binding = retain_fitted_records(root, context, identity)
            self.assertEqual((root / "frozen-development-controls.json").read_bytes(), raw)
            self.assertEqual(verify_retained_records(root), binding)
            self.assertIs(binding["fitting_performed_by_retention"], False)
            self.assertIs(binding["registration_ready"], False)

    def test_same_byte_retry_preserves_and_changed_source_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root, context, _ = self.setup_files(directory)
            identity = {"family": "WEATHER_FORECAST_VERIFY", "run_id": 7, "commit": "a" * 40}
            first = retain_fitted_records(root, context, identity)
            self.assertEqual(retain_fitted_records(root, context, identity), first)
            source = context / "data/frozen-controls.json"
            source.write_bytes(source.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                retain_fitted_records(root, context, identity)

    def test_fit_scope_coefficient_types_and_unlicensed_bytes_refused(self):
        for field, value in (("fit_split", "validation"), ("integer_coefficients", {"length": True}),
                             ("answer_only_token_weights", {"fixture": 1.0})):
            controls = self.controls()
            controls[field] = value
            with self.assertRaises(ValueError):
                validate_controls(controls)
        with tempfile.TemporaryDirectory() as directory:
            root, context, _ = self.setup_files(directory)
            (context / "data/frozen-controls.json").write_text(json.dumps(self.controls()))
            with self.assertRaises(ValueError):
                retain_fitted_records(root, context, {"family": "WEATHER_FORECAST_VERIFY", "run_id": 7, "commit": "a" * 40})
            self.assertFalse((root / "development-artifact-binding.json").exists())

    def test_bound_artifact_tamper_and_escape_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root, context, _ = self.setup_files(directory)
            retain_fitted_records(root, context, {"family": "WEATHER_FORECAST_VERIFY", "run_id": 7, "commit": "a" * 40})
            (root / "development-source-report.json").write_text("{}")
            with self.assertRaises(ValueError):
                verify_retained_records(root)
            (root / "development-source-report.json").write_bytes((context / "data/data-report.json").read_bytes())
            binding = json.loads((root / "development-artifact-binding.json").read_text())
            binding["files"][0]["path"] = "../outside.json"
            (root / "development-artifact-binding.json").write_text(json.dumps(binding))
            with self.assertRaises(ValueError):
                verify_retained_records(root)

    def test_incomplete_or_duplicate_retention_inventory_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root, context, _ = self.setup_files(directory)
            retain_fitted_records(root, context, {"family": "WEATHER_FORECAST_VERIFY", "run_id": 7, "commit": "a" * 40})
            original = json.loads((root / "development-artifact-binding.json").read_text())
            for records in ([], original["files"][:1], [original["files"][0], original["files"][0]]):
                binding = {**original, "files": records}
                (root / "development-artifact-binding.json").write_text(json.dumps(binding))
                with self.assertRaises(ValueError):
                    verify_retained_records(root)

    def test_nonfinite_and_false_run_identity_do_not_produce_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root, context, _ = self.setup_files(directory)
            for identity in ({"family": "WEATHER_FORECAST_VERIFY", "run_id": True, "commit": "a" * 40},
                             {"family": "OTHER", "run_id": 7, "commit": "a" * 40}):
                with self.assertRaises(ValueError):
                    retain_fitted_records(root, context, identity)
            (context / "data/data-report.json").write_text('{"independent_audit":false,"value":NaN}')
            with self.assertRaises(ValueError):
                retain_fitted_records(root, context, {"family": "WEATHER_FORECAST_VERIFY", "run_id": 7, "commit": "a" * 40})
            self.assertFalse((root / "development-artifact-binding.json").exists())

    def test_workflow_explicit_retention_paths_are_all_declared_json_roots(self):
        workflow = json.loads((HERE.parent / "workflow.json").read_text())
        upload = next(step for step in workflow["jobs"]["train"]["steps"] if step.get("name") == "Retain licensed failed or successful checkpoint")
        paths = upload["with"]["path"].splitlines()
        for name in set(FILES.values()) | {"development-artifact-binding.json"}:
            self.assertIn("${{ runner.temp }}/weather-result/" + name, paths)


if __name__ == "__main__":
    unittest.main(verbosity=2)
