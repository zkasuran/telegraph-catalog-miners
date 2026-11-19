# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import io
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import sentiment_recover as recovery
from data import group_id, pair
from runtime_all_formats import check_response, format_rows, heldout_verdict, metrics


class RecoveryTests(unittest.TestCase):
    def metadata(self):
        run = {"id": recovery.RUN, "head_sha": recovery.COMMIT, "status": "completed",
               "conclusion": "failure", "repository": {"full_name": recovery.REPO},
               "head_repository": {"full_name": recovery.REPO}}
        artifact = {"id": recovery.ARTIFACT, "name": "training-" + recovery.FAMILY,
                    "size_in_bytes": recovery.ZIP_BYTES, "expired": False,
                    "digest": "sha256:" + recovery.ZIP_SHA,
                    "workflow_run": {"id": recovery.RUN, "head_sha": recovery.COMMIT},
                    "archive_download_url": f"https://api.github.com/repos/{recovery.REPO}/actions/artifacts/{recovery.ARTIFACT}/zip"}
        return run, artifact

    def originals(self):
        rows = []
        for n in range(256):
            text = f"Source review number {n}"
            rows.append(pair(recovery.FAMILY, {"text": text, "label": "Negative" if n < 128 else "Positive",
                        "group_id": group_id(text), "source_id": str(n), "domain": "yelp"}))
        return rows

    def test_exact_artifact_metadata_passes(self):
        recovery.check_metadata(*self.metadata())

    def test_wrong_run_commit_hash_artifact_and_expiry_refuse(self):
        for side, key, value in ((0, "id", 1), (0, "head_sha", "0"*40),
                                 (1, "id", 1), (1, "digest", "sha256:" + "0"*64),
                                 (1, "expired", True), (1, "size_in_bytes", 1)):
            with self.subTest(key=key):
                objects = list(self.metadata())
                objects[side][key] = value
                with self.assertRaises(ValueError):
                    recovery.check_metadata(*objects)
        for key, value in (("id", 1), ("head_sha", "0"*40)):
            run, artifact = self.metadata()
            artifact["workflow_run"][key] = value
            with self.assertRaises(ValueError):
                recovery.check_metadata(run, artifact)

    def test_wrong_archive_bytes_refuse_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "ZIP hash"):
                recovery.unpack_original(b"invalid", Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_exact_archive_member_hashes_are_checked_before_any_write(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("_temp/telegraph-training/result.json", b"fixture-report")
            for n in range(8):
                archive.writestr(f"unused/{n}.json", b"{}")
        raw = buffer.getvalue()
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(
                recovery, ZIP_BYTES=len(raw), ZIP_SHA=recovery.digest(raw),
                FILES={"result.json": recovery.digest(b"fixture-report")}):
            root = Path(tmp)
            recovery.unpack_original(raw, root)
            self.assertEqual((root / "result.json").read_bytes(), b"fixture-report")
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(
                recovery, ZIP_BYTES=len(raw), ZIP_SHA=recovery.digest(raw),
                FILES={"result.json": "0"*64}):
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                recovery.unpack_original(raw, Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_validator_refusal_keeps_stderr_and_quality_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "class-snapshot.json").write_text("{}")
            (root / "foundation.json").write_text("{}")

            def refused(arguments, stdout, stderr, timeout, check):
                stderr.write(b"packagecheck: FAILED: class_mismatch\n")
                return subprocess_result

            subprocess_result = type("Result", (), {"returncode": 1})()
            with patch.object(recovery, "original_report", return_value={}), \
                    patch.object(recovery, "capture_snapshot", return_value="0"*64), \
                    patch.object(recovery.subprocess, "run", side_effect=refused), \
                    self.assertRaisesRegex(ValueError, "persisted stderr"):
                recovery.validate(root)
            self.assertIn(b"class_mismatch", (root / "package-check.stderr.txt").read_bytes())
            status = json.loads((root / "package-check-status.json").read_text())
            self.assertEqual(status["returncode"], 1)
            self.assertEqual(status["quality_verdict"], "PASS")
            self.assertFalse((root / "package-manifest.json").exists())

    def test_token_cannot_reach_non_github_host(self):
        with self.assertRaisesRegex(ValueError, "only reach"):
            recovery.https_read("https://example.com", token="ephemeral-fixture")
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            recovery.https_read("http://example.com")

    def test_no_redirect_handler_never_forwards_credentials(self):
        self.assertIsNone(recovery.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.com"))

    def test_local_runtime_refused(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(RuntimeError):
            recovery.remote_only()

    def test_fixed_original_source_hashes(self):
        recovery.check_source_inputs()

    def test_all_formats_use_the_same_originals(self):
        views = format_rows(self.originals())
        self.assertEqual(set(views), set(recovery.FORMATS))
        for style, name in enumerate(recovery.FORMATS):
            self.assertEqual(len(views[name]), 256)
            self.assertTrue(all(r["style"] == style for r in views[name]))
            self.assertEqual([r["group_id"] for r in views[name]], [r["group_id"] for r in self.originals()])
        self.assertEqual(views["bare_label"][0]["chosen"], "Negative")

    def test_missing_or_duplicate_original_refused(self):
        for rows in (self.originals()[:-1], [self.originals()[0]] * 256):
            with self.assertRaises(ValueError):
                format_rows(rows)

    def test_repeat_bits_and_nonfinite_scores_refused(self):
        rows = format_rows(self.originals())["json"]
        bits = [0x3f800000, 0] * 256
        good = metrics(rows, bits, bits)
        self.assertEqual(good["correct"], 256)
        self.assertEqual(good["repeat_bit_vectors"], 512)
        for invalid in (0x7fc00000, 0x7f800000, True, -1, 1 << 32):
            changed = [invalid] + bits[1:]
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                metrics(rows, changed, changed)
        with self.assertRaisesRegex(ValueError, "bits changed"):
            metrics(rows, bits, [0] + bits[1:])
        with self.assertRaisesRegex(ValueError, "missing format"):
            metrics(rows, bits[:-1], bits)

    def test_runtime_response_identity_and_count(self):
        manifest = {"keccak256": "0x" + "a"*64}
        response = {"adapter_hash": manifest["keccak256"],
                    "foundation_hash": recovery.FOUNDATION["foundation_lora_fp32.onnx"],
                    "tokenizer_hash": recovery.FOUNDATION["tokenizer.json"], "results": [{"score": 0}]}
        self.assertEqual(check_response(response, 1, manifest), [0])
        for key in ("adapter_hash", "foundation_hash", "tokenizer_hash"):
            with self.assertRaisesRegex(ValueError, "identity"):
                check_response({**response, key: "wrong"}, 1, manifest)
        with self.assertRaisesRegex(ValueError, "result count"):
            check_response(response, 2, manifest)

    def test_missing_format_label_and_quality_thresholds_fail(self):
        rows = format_rows(self.originals())["json"]
        bits = [0x3f800000, 0] * 256
        formats = {name: metrics(rows, bits, bits) for name in recovery.FORMATS}
        report = {"untrained": {name: {"pairwise_accuracy": 0.5} for name in recovery.FORMATS}}
        self.assertEqual(heldout_verdict(formats, report)["verdict"], "PASS")
        missing = copy.deepcopy(formats)
        missing.pop("bare_label")
        self.assertEqual(heldout_verdict(missing, report)["verdict"], "FAIL")
        for key, value in (("pairwise_accuracy", 0.79), ("repeat_bit_vectors", 8),
                           ("pairwise_accuracy", float("nan")), ("accuracy_by_label", {"Positive": 1.0}),
                           ("accuracy_by_label", {"Positive": 1.0, "Negative": 0.69})):
            mutated = copy.deepcopy(formats)
            mutated["bare_label"][key] = value
            self.assertEqual(heldout_verdict(mutated, report)["verdict"], "FAIL")

    def test_ties_are_failures(self):
        rows = format_rows(self.originals())["json"]
        bits = [struct.unpack("<I", struct.pack("<f", 1.0))[0]] * 512
        self.assertEqual(metrics(rows, bits, bits)["correct"], 0)

    def test_original_quality_missing_format_or_identity_refused(self):
        report = {"family": recovery.FAMILY, "run_id": str(recovery.RUN), "commit": recovery.COMMIT,
                  "verdict": "PASS", "dataset_gate": {"ok": True},
                  "formats": {name: {"n_eval": 256, "pairwise_accuracy": 0.9,
                              "accuracy_by_label": {"Positive": 0.9, "Negative": 0.9},
                              "length_baseline": 0.5, "beats_length_baseline": True} for name in recovery.FORMATS},
                  "untrained": {name: {"n_eval": 256, "pairwise_accuracy": 0.5} for name in recovery.FORMATS}}
        recovery.check_quality(report)
        for key, value in (("run_id", "1"), ("commit", "0"*40), ("formats", {})):
            with self.assertRaises(ValueError):
                recovery.check_quality({**report, key: value})
        report["formats"]["json"]["pairwise_accuracy"] = float("inf")
        with self.assertRaises(ValueError):
            recovery.check_quality(report)


if __name__ == "__main__":
    unittest.main()
