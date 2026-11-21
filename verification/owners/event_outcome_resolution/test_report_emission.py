# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only placeholder containers. No fixture is a model or real audit."""
import copy
from contextlib import redirect_stdout
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import report
from common import FOUNDATION, KIT, sha
from test_contracts import synthetic_ready_fixture


class ReportEmission(unittest.TestCase):
    def fixture(self, directory):
        raw = b"source-only placeholder, not a neural package"
        name = report.FAMILY + ".safetensors"
        (directory / name).write_bytes(raw)
        return {"package": {"file": name, "sha256": sha(raw), "bytes": len(raw)}}, raw

    def test_verified_package_primary_path(self):
        with tempfile.TemporaryDirectory() as temp:
            candidate, raw = self.fixture(Path(temp))
            self.assertEqual(report.verified_package_bytes(Path(temp), candidate), raw)

    def test_tampering_never_calls_ready_emitter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, _ = self.fixture(root)
            (root / candidate["package"]["file"]).write_bytes(b"changed")
            output = root / "registration-ready.json"
            with patch.object(report, "combine") as combine, self.assertRaises(ValueError):
                report.emit_verified_result(candidate, {}, root, output)
            combine.assert_not_called()
            self.assertFalse(output.exists())

    def test_path_and_count_identity_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, _ = self.fixture(root)
            for field, value in (("file", "../other.safetensors"), ("file", "OTHER.safetensors"),
                                 ("bytes", True), ("bytes", 1.0), ("sha256", "0" * 64)):
                bad = copy.deepcopy(candidate)
                bad["package"][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    report.verified_package_bytes(root, bad)

    def test_output_uses_verified_buffer_not_changed_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "candidate"
            output_dir = root / "result"
            source.mkdir()
            output_dir.mkdir()
            candidate, original = self.fixture(source)
            def combine(*args):
                (source / candidate["package"]["file"]).write_bytes(b"late changed")
                return {"synthetic_control": True}
            with patch.object(report, "combine", combine):
                result = report.emit_verified_result(candidate, {}, source, output_dir / "registration-ready.json")
            self.assertTrue(result["synthetic_control"])
            self.assertEqual((output_dir / candidate["package"]["file"]).read_bytes(), original)

    def test_actual_verification_bypass_mutation_is_caught(self):
        source = inspect.getsource(report.emit_verified_result)
        marker = "raw = verified_package_bytes(directory, candidate)"
        self.assertIn(marker, source)
        namespace = dict(report.__dict__)
        emitter = Mock(return_value={"synthetic_control": True})
        namespace["combine"] = emitter
        exec(compile(source.replace(marker, 'raw = b"unverified"'), "mutant_report_emission", "exec"), namespace)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, _ = self.fixture(root)
            (root / candidate["package"]["file"]).write_bytes(b"changed")
            namespace["emit_verified_result"](candidate, {}, root, root / "registration-ready.json")
            with self.assertRaises(AssertionError):
                emitter.assert_not_called()

    def test_symlink_package_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            candidate, raw = self.fixture(root)
            package = root / candidate["package"]["file"]
            package.unlink()
            other = root / "placeholder.txt"
            other.write_bytes(raw)
            package.symlink_to(other)
            with self.assertRaises(ValueError):
                report.verified_package_bytes(root, candidate)

    def test_real_report_cli_with_synthetic_evidence_and_material_wait(self):
        # Drive main and combine, not a stand-in emitter. Metrics, vectors and
        # package are synthetic protocol controls kept in a disposable directory.
        for contributors in (1, 5):
            with self.subTest(contributors=contributors), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, audits, result = (root / name for name in ("candidate", "audits", "result"))
                source.mkdir()
                audits.mkdir()
                candidate, raw = self.fixture(source)
                source_hash = sha((report.HERE / "frozen/frozen.json").read_bytes())
                candidate.update(family=report.FAMILY, scope="parent_only", claimed_intents=[report.FAMILY],
                                 source_manifest_sha256=source_hash, audit_read=False, run_id=123, commit="a" * 40)
                (source / "candidate.json").write_text(json.dumps(candidate))
                control = synthetic_ready_fixture()
                for name in report.FORMATS:
                    measured = {"family": report.FAMILY, "format": name, "scope": "parent_only",
                                "claimed_intents": [report.FAMILY], "passed": True, "run_id": 123,
                                "commit": "a" * 40, "package_sha256": sha(raw), "source_manifest_sha256": source_hash,
                                "foundation": FOUNDATION, "kit_commit": KIT,
                                "quality_engine": "pinned reference WASM, no native Torch audit",
                                "package_keccak": "c" * 64, "runtime": control["runtime"],
                                "executor_sha256": "d" * 64, "independent_groups": 39, "audit_pairs": 384,
                                "metrics": control["quality"]["formats"][name], "runtime_versions": {},
                                "class_snapshot_sha256": "e" * 64}
                    (audits / ("audit-" + name + ".json")).write_text(json.dumps(measured))
                status = {"intent_id": report.FAMILY, "eligible_submitters": contributors, "min_submitters": 3,
                          "active_snapshot": {"corpus_version": "f" * 64, "submitters": contributors,
                                              "rows": 150, "qualification": 78, "ranking": 72}}
                argv = ["report.py", "--candidate-dir", str(source), "--audits", str(audits), "--out", str(result)]
                with patch.object(report, "remote_only"), patch.object(report, "SCOPE", "parent_only"), \
                        patch.object(report, "public_json", lambda _: (status, json.dumps(status).encode())), \
                        patch.dict(os.environ, {"GITHUB_RUN_ID": "123", "GITHUB_SHA": "a" * 40}), patch("sys.argv", argv):
                    if contributors == 1:
                        with self.assertRaisesRegex(ValueError, "parent material readiness"):
                            report.main()
                        condition = json.loads((result / "material-condition.json").read_text())
                        self.assertFalse(condition["registration_ready"])
                        self.assertFalse((result / "registration-ready.json").exists())
                    else:
                        with redirect_stdout(io.StringIO()):
                            report.main()
                        bindings = json.loads((result / "result-bindings.json").read_text())
                        self.assertEqual(bindings["registration_ready_sha256"], sha((result / "registration-ready.json").read_bytes()))
                self.assertEqual((result / candidate["package"]["file"]).read_bytes(), raw)
                self.assertEqual((result / "LICENSE").read_bytes(), (report.HERE / "LICENSE").read_bytes())


if __name__ == "__main__":
    unittest.main()
