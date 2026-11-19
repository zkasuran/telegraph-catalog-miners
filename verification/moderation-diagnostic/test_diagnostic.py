# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Static and fake controls only. No Torch, tokenizer or model loading."""
import ast
import copy
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import core
from core import (
    CONTRACT,
    FORMATS,
    LABELS,
    audit_select,
    choose,
    isolated_rng,
    quality,
    schedule,
    sha,
    summary,
)

HERE = Path(__file__).resolve().parent


class State(int):
    def clone(self):
        return State(self)


class FakeTorch:
    def __init__(self, restore=True):
        self.state = 17
        self.restore = restore
        self.random = self
    def get_rng_state(self):
        return State(self.state)
    def equal(self, a, b):
        return a == b
    @contextmanager
    def fork_rng(self, devices):
        if devices != []:
            raise AssertionError("CPU-only fork required")
        before = self.state
        try:
            yield
        finally:
            if self.restore:
                self.state = before


class FakeNumpy:
    def __init__(self):
        self.random = self
        self.state = (11, 23)
    def get_state(self):
        return self.state
    def set_state(self, value):
        self.state = value


def rows(n=100):
    return [{"label": label, "chosen": "chosen", "rejected": "reject"} for label in LABELS for _ in range(n)]


def metrics(success=90):
    margins = [1.0] * success + [-1.0] * (100 - success)
    return summary(rows(), margins + margins)


class ContractTests(unittest.TestCase):
    def test_source_bindings(self):
        core.bindings()

    def test_remote_guard_precedes_framework_import(self):
        import model_ops
        import remote_controls
        with patch.dict(os.environ, {}, clear=True):
            for call in (lambda: model_ops.runtime(HERE, 3), lambda: model_ops.tokenizer(HERE), remote_controls.controls):
                with self.assertRaises(RuntimeError):
                    call()
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_schedule_exact_warmup_and_step_counts(self):
        c3, h6 = schedule(3), schedule(6)
        self.assertEqual((c3["optimizer_calls"], c3["warmup_calls"]), (144, 8))
        self.assertEqual((h6["optimizer_calls"], h6["warmup_calls"]), (288, 17))
        self.assertEqual(c3["optimizer_lr_before_call"][0], 0)
        self.assertAlmostEqual(c3["optimizer_lr_before_call"][-1], 1e-4 / 136)
        self.assertNotEqual(c3["optimizer_lr_before_call"][1], h6["optimizer_lr_before_call"][1])
        self.assertEqual(h6["terminal_lr"], 0)

    def test_accumulation_tail_is_refused(self):
        with self.assertRaises(ValueError):
            schedule(3, n=770)

    def test_fake_rng_restores_every_stream_and_exception(self):
        torch, np = FakeTorch(), FakeNumpy()
        before = random.getstate()
        with self.assertRaisesRegex(ValueError, "probe"), isolated_rng(torch, np):
            torch.state += 2  # Fake DataLoader iteration.
            np.state = (33, 9)
            random.random()
            raise ValueError("probe")
        self.assertEqual(torch.state, 17)
        self.assertEqual(np.state, (11, 23))
        self.assertEqual(random.getstate(), before)

    def test_fake_unguarded_control_fails(self):
        torch = FakeTorch(restore=False)
        with self.assertRaisesRegex(RuntimeError, "Torch RNG"), isolated_rng(torch, FakeNumpy()):
            torch.state += 1

    def test_summary_counts_ties_and_errors(self):
        result = summary(rows(2), [1.0, 0.0, -1.0, 2.0])
        self.assertEqual((result["n"], result["correct"], result["strict_errors"], result["ties"]), (4, 2, 1, 1))
        self.assertEqual(result["by_label"]["Flagged"]["ties"], 1)
        self.assertEqual(result["by_label"]["Safe"]["strict_errors"], 1)
        self.assertTrue(math.isfinite(result["bt_loss"]))

    def test_nonfinite_scores_and_missing_labels_fail(self):
        for margins in ([float("nan")] * 4, [float("inf")] * 4, [1.0]):
            with self.assertRaises(ValueError):
                summary(rows(2), margins)
        with self.assertRaises(ValueError):
            summary(rows(1)[:1], [1.0])

    def test_every_quality_gate_has_a_negative_control(self):
        good = {f: metrics() for f in FORMATS}
        baseline = {f: metrics(50) for f in FORMATS}
        self.assertEqual(quality(good, baseline, True), [])
        mutations = []
        bad = copy.deepcopy(good)
        del bad["bare_label"]
        mutations.append((bad, baseline, True))
        bad = copy.deepcopy(good)
        bad["prose"] = metrics(69)
        mutations.append((bad, baseline, True))
        bad = copy.deepcopy(good)
        bad["json"]["length_baseline"] = 0.85
        mutations.append((bad, baseline, True))
        bad = copy.deepcopy(baseline)
        bad["json"] = metrics(86)
        mutations.append((good, bad, True))
        mutations.append((good, baseline, False))
        bad = copy.deepcopy(good)
        bad["json"]["accuracy"] = float("nan")
        mutations.append((bad, baseline, True))
        for args in mutations:
            self.assertTrue(quality(*args))

    def test_failed_label_not_hidden_by_overall(self):
        good = {f: metrics() for f in FORMATS}
        good["json"] = summary(rows(), [1.0] * 69 + [-1.0] * 31 + [1.0] * 100)
        self.assertGreater(good["json"]["accuracy"], 0.8)
        self.assertTrue(quality(good, {f: metrics(50) for f in FORMATS}, True))

    def test_inconsistent_denominator_fails(self):
        good = {f: metrics() for f in FORMATS}
        good["json"]["correct"] -= 1
        self.assertTrue(quality(good, {f: metrics(50) for f in FORMATS}, True))

    def test_selector_uses_worst_group_and_earlier_epoch(self):
        records = [{"arm": "H6", "epoch": 1, "formats": {f: metrics(92) for f in FORMATS}},
                   {"arm": "C3", "epoch": 0, "formats": {f: metrics(90) for f in FORMATS}}]
        baseline = {f: metrics(50) for f in FORMATS}
        self.assertEqual(choose(records, baseline, True)["arm"], "H6")
        records[0]["formats"]["prose"] = metrics(69)
        self.assertEqual(choose(records, baseline, True)["arm"], "C3")
        records[0]["formats"] = copy.deepcopy(records[1]["formats"])
        self.assertEqual(choose(records, baseline, True)["epoch"], 0)
        self.assertIsNone(choose(records, baseline, False))

    def test_audit_excludes_groups_and_conflicts_before_selection(self):
        source = [{"group_id": sha(str(i).encode()), "label": label} for i, label in enumerate(["Safe"] * 6 + ["Flagged"] * 6)]
        excluded = {source[0]["group_id"]}
        source.append({**source[1], "label": "Flagged"})
        selected, conflicts = audit_select(source, excluded, per_label=2)
        self.assertEqual(conflicts, 1)
        self.assertEqual(len(selected), 4)
        self.assertFalse(excluded.intersection(r["group_id"] for r in selected))
        self.assertNotIn(source[1]["group_id"], {r["group_id"] for r in selected})
        self.assertEqual(selected, audit_select(list(reversed(source)), excluded, per_label=2)[0])

    def test_audit_does_not_replace_insufficient_pool(self):
        with self.assertRaises(ValueError):
            audit_select([], set())

    def test_campaign_exclusion_schema(self):
        self.assertEqual(core.validate_exclusions(["a" * 64]), {"a" * 64})
        for value in ({}, ["not-a-hash"], ["a" * 64, "a" * 64], [7]):
            with self.assertRaises(ValueError):
                core.validate_exclusions(value)

    def test_old_test_and_audit_access_denied_in_subprocess(self):
        script = "from core import deny_measurement_reads; from pathlib import Path; import sys; root=Path(sys.argv[1]); deny_measurement_reads(root); open(root/sys.argv[2])"
        for name in ("audit-private/audit.jsonl", "originals/test.jsonl"):
            result = subprocess.run([sys.executable, "-c", script, str(HERE), name], cwd=HERE, capture_output=True, timeout=10, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"PermissionError", result.stderr)

    def test_audit_is_unopened_when_candidate_freeze_is_absent(self):
        from stages import final
        with tempfile.TemporaryDirectory(prefix=".freeze-control-", dir=HERE) as directory:
            root = Path(directory)
            (root / "evidence").mkdir()
            (root / "audit-private").mkdir()
            (root / "audit-private" / "audit.jsonl").write_text("not an audit fixture\n")
            with self.assertRaises(FileNotFoundError):
                final(root)
            self.assertFalse((root / "evidence" / "audit-access.json").exists())

    def test_changed_shared_source_identity_fails(self):
        changed = dict(CONTRACT["source_bindings"])
        changed["data.py"] = "0" * 64
        with patch.dict(CONTRACT, {"source_bindings": changed}), self.assertRaisesRegex(ValueError, "binding changed"):
            core.bindings()

    def test_no_model_import_at_module_scope(self):
        for path in HERE.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                    self.assertFalse(any(n.split(".")[0] in {"torch", "transformers", "peft", "tokenizers", "numpy"} for n in names), path.name)

    def test_overlay_has_private_namespace_and_no_training_rewrite(self):
        # Fake source anchors exercise the overlay without loading the real trainer.
        source = "def train():\n    optim.step()\n    sched.step()\n    avg_loss = 1.0\n    val = evaluate(model, tok, val_pairs, hp, train_negatives)\n"
        namespace = {"marker": object()}
        with patch.dict(CONTRACT, {"trainer_sha256": sha(source.encode())}):
            function, text = core.overlay(source, namespace)
            self.assertEqual(namespace.keys(), {"marker"})
            self.assertIsNot(function.__globals__, namespace)
            self.assertIn("diagnostic_step(optim, params, gstep)", text)
            self.assertIn("val = evaluate(model, tok, val_pairs, hp, train_negatives)", text)
            self.assertIn("sched.step()", text)
            with self.assertRaises(ValueError):
                core.overlay(source + "\n", namespace)
        broken = source.replace("optim.step()", "optim.zero_grad()")
        with patch.dict(CONTRACT, {"trainer_sha256": sha(broken.encode())}), self.assertRaisesRegex(ValueError, "anchors"):
            core.overlay(broken, namespace)

    def test_preservation_keeps_tensor_payload_and_upstream_grants(self):
        _, _, module = core.load_shared()
        header = json.dumps({"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        raw = struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
        licences = json.loads((core.TRAINING / "licences.json").read_text())
        stamped = module.stamp(raw, licences)
        size = struct.unpack("<Q", stamped[:8])[0]
        self.assertEqual(stamped[8 + size:], raw[8 + len(header):])
        metadata = json.loads(stamped[8:8 + size])["__metadata__"]
        self.assertEqual(metadata["x-terms-apache_2_0"], licences["apache_2_0"])
        self.assertEqual(metadata["x-terms-kit_mit"], licences["kit_mit"])

    def test_incremental_phase_failure_is_loud(self):
        import time

        from run_remote import phase
        with tempfile.TemporaryDirectory(prefix=".source-control-", dir=HERE) as directory:
            root = Path(directory)
            (root / "evidence").mkdir()
            with self.assertRaisesRegex(RuntimeError, "phase failed"):
                phase(root, "fake-refusal", [sys.executable, "-c", "raise SystemExit(7)"], 5, time.monotonic() + 300)
            report = json.loads((root / "evidence" / "phase.json").read_text())
            self.assertEqual(report["exit_code"], 7)
            self.assertTrue((root / "evidence" / "fake-refusal.log").is_file())


if __name__ == "__main__":
    unittest.main()
