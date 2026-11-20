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

from data import HERE, check_confidence, check_normalized_aliases, read_splits, source_helper
from runtime_check import check_response, RELEASE_HASHES


class NaturalContracts(unittest.TestCase):
    def test_frozen_training_improvement_and_retained_baseline(self):
        contract = json.loads((HERE / "contract.json").read_text())
        frozen = json.loads((HERE / "fixtures/frozen-controls.json").read_text())
        report = json.loads((HERE / "fixtures/data-report.json").read_text())
        self.assertEqual(frozen["fit_split"], "train")
        self.assertEqual(frozen["training_sha256"], report["files"]["train"]["sha256"])
        self.assertEqual(frozen["previous_baseline"]["training_sha256"], "10f90e26c7b016dd919ccddbd8b7bcd6fdbf609d6b00e986aa9a43ae328fe4dd")
        self.assertEqual(frozen["previous_baseline"]["training_rows"], 768)
        self.assertEqual(frozen["previous_baseline"]["previous_baseline"]["training_rows"], 512)
        self.assertEqual(contract["model_recipe"], {"lr": .00002, "lora_dropout": .1, "lora_rank": 16, "lora_alpha": 32, "weight_decay": .1})
        self.assertTrue(all(type(value) is int for value in frozen["coefficients"].values()))
        self.assertEqual(report["curriculum_source"]["main_training_groups"], 760)
        self.assertEqual(report["curriculum_source"]["warmup_groups"], 332)
        self.assertEqual(frozen["warmup_sha256"], report["files"]["warmup"]["sha256"])
        self.assertEqual(report["files"]["audit"]["sha256"], "4734c2c4d83f8a66eb72af146bbf0af737c7db4dee4e32b29e049f2c50a11143")

    def test_bridge_rejects_changed_root_dependency(self):
        with tempfile.TemporaryDirectory(prefix="natural-bridge-hash-") as directory:
            root = Path(directory)
            natural = root / "natural"
            natural.mkdir()
            shutil.copyfile(HERE / "bridge.py", natural / "bridge.py")
            for name in ("source.py", "common.py", "contract.json"):
                shutil.copyfile(HERE.parent / name, root / name)
            with (root / "source.py").open("ab") as output:
                output.write(b"\n")
            result = subprocess.run([sys.executable, "-I", "-W", "error", str(natural / "bridge.py"), "--source-smoke"],
                                    cwd=directory, capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"root preparation source identity changed: source.py", result.stderr)

    def test_standalone_bridge_import_and_model_refusal(self):
        # The actual runner starts natural/run.py, not the root verify.sh with
        # its extra PYTHONPATH. Exercise that literal standalone import context.
        code = """import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import data
natural_data = data
original_path = list(sys.path)
import bridge
assert sys.modules['data'] is natural_data
assert sys.path == original_path
assert bridge.FOUNDATION['tokenizer.json'] == '67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456'
try:
    bridge.prepare_reference(Path(sys.argv[2]))
except RuntimeError as error:
    assert 'bounded public CPU GitHub dispatch' in str(error)
else:
    raise AssertionError('local foundation preparation was accepted')
assert 'torch' not in sys.modules
assert not Path(sys.argv[2]).exists()
print('STANDALONE BRIDGE AND LOCAL MODEL REFUSAL PASS')
"""
        environment = {key: value for key, value in os.environ.items() if not key.startswith("GITHUB_") and key != "PYTHONPATH"}
        with tempfile.TemporaryDirectory(prefix="natural-bridge-") as directory:
            result = subprocess.run([sys.executable, "-I", "-W", "error", "-c", code, str(HERE), str(Path(directory) / "never-created")],
                                    cwd=directory, env=environment, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertIn(b"STANDALONE BRIDGE AND LOCAL MODEL REFUSAL PASS", result.stdout)

    def test_primary_original_rank_and_group_smoke(self):
        report, splits = read_splits(HERE / "fixtures")
        self.assertEqual(len(splits["train"]), 760)
        self.assertEqual(len({row["group_id"] for row in splits["train"]}), 760)
        self.assertEqual(len(splits["warmup"]), 332)
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

    def test_within_training_alias_is_not_an_independent_component(self):
        _, splits = read_splits(HERE / "fixtures")
        check_normalized_aliases(splits)
        bad = copy.deepcopy(splits)
        bad["train"][1]["prompt"] = "  " + bad["train"][0]["prompt"].upper() + "  "
        self.assertNotEqual(bad["train"][0]["group_id"], bad["train"][1]["group_id"])
        with self.assertRaises(ValueError):
            check_normalized_aliases(bad)

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

    def test_confidence_weights_are_training_only_and_can_fail(self):
        from training_policy import confidence_weight, training_stage
        _, splits = read_splits(HERE / "fixtures")
        self.assertEqual([training_stage(epoch) for epoch in range(1, 5)], ["warmup", "train", "train", "train"])
        for name in ("train", "warmup"):
            check_confidence(splits[name])
            weights = [confidence_weight(row) for row in splits[name]]
            self.assertAlmostEqual(sum(weights), len(splits[name]), places=9)
            bad = copy.deepcopy(splits[name])
            bad[0]["confidence_units"] = 7
            with self.assertRaises(ValueError):
                check_confidence(bad)
            with self.assertRaises(ValueError):
                confidence_weight(bad[0])
            bad = copy.deepcopy(splits[name])
            bad[0]["confidence_normalizer"] += 1
            with self.assertRaises(ValueError):
                check_confidence(bad)
        for name in ("validation", "audit"):
            self.assertTrue(all("confidence_units" not in row for row in splits[name]))
        with self.assertRaises(ValueError):
            training_stage(5)


if __name__ == "__main__":
    unittest.main()
