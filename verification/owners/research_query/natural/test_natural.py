# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Original preference contracts, no neural inference or exposed test data."""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data import HERE, read_splits, source_helper
from runtime_check import check_response, RELEASE_HASHES


class NaturalContracts(unittest.TestCase):
    def test_primary_original_rank_and_group_smoke(self):
        report, splits = read_splits(HERE / "fixtures")
        self.assertEqual(len(splits["train"]), 512)
        self.assertEqual(len({row["group_id"] for row in splits["audit"]}), 64)
        self.assertTrue(all(row["chosen_rank"] == 0 and row["rejected_rank"] > 0 for rows in splits.values() for row in rows))
        self.assertEqual(report["normalized_cross_split_overlap"], 0)

    def test_source_annotation_privacy_and_synthetic_controls(self):
        helper = source_helper()
        row = {"lang": "en", "synthetic": False, "deleted": False, "review_result": True, "review_count": 1, "text": "Explain a technical concept.", "labels": [{"name": "pii", "value": 0}]}
        self.assertTrue(helper.eligible(row))
        for key, value in (("synthetic", True), ("deleted", True), ("review_result", False), ("review_count", 0)):
            self.assertFalse(helper.eligible({**row, key: value}))
        self.assertFalse(helper.eligible({**row, "labels": [{"name": "pii", "value": 1}]}))

    def test_normalized_source_overlap_control_can_fail(self):
        _, splits = read_splits(HERE / "fixtures")
        helper = source_helper()
        bad = copy.deepcopy(splits)
        bad["audit"][0]["prompt"] = "  " + bad["train"][0]["prompt"].upper() + "  "
        with self.assertRaises(ValueError):
            helper.check_splits(bad)

    def test_actual_file_hash_failure(self):
        with tempfile.TemporaryDirectory(prefix="natural-source-") as directory:
            target = Path(directory)
            for name in ("train.jsonl", "validation.jsonl", "audit.jsonl", "data-report.json"):
                shutil.copyfile(HERE / "fixtures" / name, target / name)
            with (target / "audit.jsonl").open("ab") as output:
                output.write(b" ")
            with self.assertRaises(ValueError):
                read_splits(target)

    def test_literal_wire_echo_not_normalized(self):
        manifest = {"keccak256": "0x" + "a" * 64}
        response = {"foundation_hash": RELEASE_HASHES["foundation_lora_fp32.onnx"], "tokenizer_hash": RELEASE_HASHES["tokenizer.json"], "adapter_hash": "a" * 64, "results": [{"score": 0}]}
        self.assertEqual(check_response(response, 1, manifest), [0])
        with self.assertRaises(ValueError):
            check_response({**response, "adapter_hash": "0x" + "a" * 64}, 1, manifest)

    def test_remote_entrypoint_refuses_local_execution(self):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("GITHUB_")}
        result = subprocess.run([sys.executable, str(HERE / "run.py"), "--root", str(HERE / "never-created")], env=environment, capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((HERE / "never-created").exists())


if __name__ == "__main__":
    unittest.main()
