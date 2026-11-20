# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic ZIP/layout controls only, never neural or source preferences."""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from artifact_layout import stage, validate_names


class ArtifactLayoutTests(unittest.TestCase):
    def test_train_literal_root_and_nested_failure_control(self):
        names = ["candidate.json", "foundation.json", "frozen-baselines.json", "KEYWORD_EXTRACTION.safetensors", "KEYWORD_EXTRACTION_INITIAL.safetensors", "KEYWORD_EXTRACTION.safetensors.manifest.json"]
        self.assertTrue(validate_names(names, "train", True))
        with self.assertRaises(ValueError):
            validate_names(["_temp/keyword-pilot/" + name for name in names], "train", True)
        with self.assertRaises(ValueError):
            validate_names([n for n in names if n != "KEYWORD_EXTRACTION_INITIAL.safetensors"], "train", True)

    def test_real_zip_root_names_and_bounded_metadata_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "keyword-audit"
            root.mkdir()
            (root / "registration-ready.json").write_text('{"synthetic_schema_control_only":true}')
            (root / "audit-result.json").write_text('{"synthetic_schema_control_only":true}')
            dependency = base / "audit-dependencies.json"
            dependency.write_text('{"synthetic_dependency_control_only":true}')
            stage(root, "audit", dependency)
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w") as archive:
                for path in root.iterdir():
                    archive.write(path, arcname=path.relative_to(root).as_posix())
            with zipfile.ZipFile(io.BytesIO(output.getvalue())) as archive:
                self.assertTrue(validate_names(archive.namelist(), "audit", True))
                self.assertEqual(json.loads(archive.read("registration-ready.json")), {"synthetic_schema_control_only": True})
            self.assertTrue((root / "dependency-audit.json").is_file())

    def test_private_or_traversing_member_never_staged(self):
        for name in ("forward-key.json", "../registration-ready.json", ".hidden.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_names([name], "audit")


if __name__ == "__main__":
    unittest.main()
