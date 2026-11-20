# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only failures and contract fixtures, not model quality evidence."""
import base64
import copy
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audit_policy import check_source_quarantine, development_gate
from baseline import load_controls
from common import DEFAULT_OWNER, lease_for, validate_lease
from curriculum import check_tokens, source_check, warmup_rows
from data import HERE, VIEWS, check_splits, prepare, read_splits, view
from diagnostic import training_controls
from frozen_candidate import original_evidence, read_frozen, verify_frozen_package
from licensing import preserve_artifact, tensor_parts, verify_checkpoint_metadata
from metrics import all_formats, pair_metrics, quality
from preserve import preserve
from primary import anchors
from proxy import validate_request
from reciprocal import policy, schedules, selected_rows
from report import completed_checks, validate_ready
from role_bootstrap import admit, warmup_plan
from role_bootstrap import warmup_rows as bootstrap_rows
from runtime_check import RELEASE_HASHES, compile_tools, model_remote_only
from serve import parse_settings
from static_check import check_curriculum_training
from training_policy import EPOCHS, development_selection
from wire import decode_bits, input_root, validate_response


def fixture_rows():
    return [{"group_id": "fixture", "target": "A", "chosen": "Long fixture title", "rejected": "Short", "prompt": "Study A"},
            {"group_id": "fixture", "target": "B", "chosen": "Short", "rejected": "Long fixture title", "prompt": "Study B"}]


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((HERE / "data/manifest.json").read_text())
        cls.splits = {name: [json.loads(line) for line in (HERE / "data" / (name + ".jsonl")).read_text().splitlines()] for name in ("train", "validation", "audit")}

    def test_primary_path_source_smoke(self):
        with tempfile.TemporaryDirectory(prefix="summary-source-test-") as directory:
            report = prepare(Path(directory))
            replay, splits = read_splits(Path(directory))
            self.assertEqual(report, replay)
            self.assertEqual([len(splits[n]) for n in ("train", "validation", "audit")], [256, 64, 128])
            self.assertEqual(report["boundary"]["normalized_overlap"], 0)
            self.assertFalse(report["training_allowed"])

    def test_real_complete_reciprocal_sources(self):
        result = check_splits(self.manifest, self.splits)
        self.assertEqual(result["independent_groups"], 224)
        for rows in self.splits.values():
            for first, second in zip(rows[::2], rows[1::2]):
                self.assertEqual((first["chosen"], first["rejected"]), (second["rejected"], second["chosen"]))

    def test_source_mismatch_controls(self):
        for attack in ("shortened", "grant", "anchor", "identity", "time", "overlap"):
            with self.subTest(attack=attack):
                manifest, splits = copy.deepcopy(self.manifest), copy.deepcopy(self.splits)
                group = manifest["splits"]["train"][0]
                if attack == "shortened":
                    group["pairs"][0]["prompt"] = "Shortened reference"
                elif attack == "grant":
                    group["documents"][0]["rights_clause"] = ""
                elif attack == "anchor":
                    group["pairs"][0]["exclusive_title_anchors"] = ["unsupported-fictional-anchor"]
                elif attack == "identity":
                    group["source_entity_ids"] = []
                elif attack == "time":
                    group["known_at"] = "9999-12-31"
                else:
                    manifest["splits"]["validation"][0]["source_entity_ids"] += group["source_entity_ids"]
                if attack in ("shortened", "anchor"):
                    splits["train"][0] = copy.deepcopy(group["pairs"][0])
                with self.assertRaises(ValueError):
                    check_splits(manifest, splits)

    def test_exclusive_anchor_control(self):
        one = {"title": "Cell response signaling", "abstract": "Cell signaling activates growth."}
        two = {"title": "Cell response mechanism discovery", "abstract": "Cell mechanism discovery controls division."}
        self.assertEqual(anchors(one, two), ["signaling"])
        self.assertEqual(anchors(two, one), ["discovery", "mechanism"])
        self.assertEqual(anchors(one, one), [])

    def test_all_format_rendering_contract(self):
        row = fixture_rows()[0]
        self.assertEqual(view(row, "json")["chosen"], '{"title":"Long fixture title"}')
        self.assertEqual(view(row, "prose")["rejected"], "Title: Short")
        with self.assertRaises(ValueError):
            view(row, "unknown")


class WireTests(unittest.TestCase):
    def setUp(self):
        self.row = {"keccak": "a" * 64}
        self.request = {"batch": [{"tokens": [50281, 22, 50282]}], "adapter_hash": self.row["keccak"],
                        "foundation_hash": RELEASE_HASHES["foundation_lora_fp32.onnx"], "tokenizer_hash": RELEASE_HASHES["tokenizer.json"]}
        self.request["input_root"] = input_root(self.request["batch"])
        self.response = {k: v for k, v in self.request.items() if k != "batch"}
        self.response["results"] = [{"score": 1065353216}]

    def test_literal_bare_hash_primary_path(self):
        self.assertEqual(validate_response(self.response, self.request), [1065353216])
        self.assertEqual(validate_request(self.request, self.row), self.request)

    def test_identity_and_input_root_controls(self):
        for key in ("adapter_hash", "foundation_hash", "tokenizer_hash", "input_root"):
            for malformed in ("0x" + self.response[key], self.response[key].upper(), "f" * 64, None):
                with self.subTest(key=key, value=malformed):
                    bad = {**self.response, key: malformed}
                    with self.assertRaises(ValueError):
                        validate_response(bad, self.request)

    def test_result_and_nonfinite_controls(self):
        for malformed in (True, -1, 1 << 32, 0x7f800000, 0xff800000, 0x7fc00000):
            with self.assertRaises(ValueError):
                decode_bits(malformed)
        for results in ([], [{"score": 1065353216}, {"score": 1065353216}], [{"error": "refused"}], [{"score": 1, "error": "bad"}]):
            with self.assertRaises(ValueError):
                validate_response({**self.response, "results": results}, self.request)

    def test_invalid_tokens_refused_by_bounded_front_end(self):
        for tokens in ([], [-1], [True], [50368], [1] * 513):
            with self.subTest(tokens=str(tokens[:2])):
                request = {**self.request, "batch": [{"tokens": tokens}]}
                request.pop("input_root")
                with self.assertRaises(ValueError):
                    validate_request(request, self.row)


class ModelContractTests(unittest.TestCase):
    def test_actual_nine_epoch_ready_checks_and_stale_epoch_control(self):
        freeze, report, primary = original_evidence()
        self.assertTrue(completed_checks(report, primary)["passed"])
        self.assertEqual(freeze["training_run_id"], 44796102138)
        for count in (3, 8, 10):
            bad = copy.deepcopy(report)
            bad["epochs_completed"] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                completed_checks(bad, primary)
        bad = copy.deepcopy(report)
        bad["selection_split"] = "audit"
        with self.assertRaises(ValueError):
            completed_checks(bad, primary)

    def test_frozen_package_and_original_grants_tamper_controls(self):
        freeze = read_frozen()
        source = HERE / "frozen/TEXT_SUMMARIZATION.safetensors"
        self.assertEqual(verify_frozen_package(source)["package_sha256"], freeze["package_sha256"])
        with tempfile.TemporaryDirectory(prefix="summary-frozen-test-") as directory:
            raw = source.read_bytes()
            bad = Path(directory) / "mutated.safetensors"
            bad.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with self.assertRaises(ValueError):
                verify_frozen_package(bad)
            descriptor = json.loads((HERE / "frozen-candidate.json").read_text())
            descriptor["training_disabled"] = False
            (Path(directory) / "frozen-candidate.json").write_text(json.dumps(descriptor))
            with patch("frozen_candidate.HERE", Path(directory)), self.assertRaises(ValueError):
                read_frozen()

    def test_frozen_candidate_refuses_retraining_before_work(self):
        import remote
        with patch("remote.remote_only"), self.assertRaisesRegex(ValueError, "candidate is frozen"):
            remote.experiment(Path("fixture-must-not-be-created"))

    def test_recovery_workflow_installs_before_engine_and_never_fits(self):
        workflow = json.loads((HERE.parents[2] / ".github/workflows/evaluator-owner-text_summarization.yml").read_text().split("\n", 1)[1])
        self.assertEqual(workflow["on"]["workflow_dispatch"]["inputs"]["mode"]["options"], ["recover", "serve"])
        steps = workflow["jobs"]["owner"]["steps"]
        names = [s.get("name") for s in steps]
        self.assertLess(names.index("Install hash-locked public CPU dependencies"), names.index("Byte-preserving readiness recovery and fresh engine vectors"))
        recovery = next(s for s in steps if s.get("name") == "Byte-preserving readiness recovery and fresh engine vectors")
        self.assertIn("recover.py", recovery["run"])
        self.assertNotIn("remote.py", recovery["run"])
        self.assertLessEqual(recovery["timeout-minutes"], 85)

    def test_original_task_bootstrap_exact_rows_and_repeats(self):
        manifest = json.loads((HERE / "data/manifest.json").read_text())
        train = [r for g in manifest["splits"]["train"] for r in g["pairs"]]
        rows = bootstrap_rows(manifest, train)
        self.assertEqual(rows, train[:8])
        plan = warmup_plan(rows)
        self.assertEqual(plan, schedules(rows)["stochastic_batch"][:256])
        self.assertEqual(len(plan), 256)
        self.assertEqual([sum(i in batch for batch in plan) for i in range(8)], [64] * 8)
        for name in ("validation", "audit"):
            with self.subTest(split=name), self.assertRaises(ValueError):
                bootstrap_rows(manifest, [r for g in manifest["splits"][name] for r in g["pairs"]])
        good = pair_metrics(rows, [2., 1.] * 8)
        weak = pair_metrics(rows, [2., 1., 0., 1.] * 4)
        self.assertTrue(admit(good))
        self.assertFalse(admit(weak))
        self.assertFalse(admit({**good, "independent_groups": 3}))
        self.assertFalse(admit({**good, "per_label": {"A": 1., "B": .5}}))

    def test_matched_original_task_batches(self):
        manifest = json.loads((HERE / "data/manifest.json").read_text())
        train = [r for g in manifest["splits"]["train"] for r in g["pairs"]]
        rows = selected_rows(manifest, train)
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows, train[:8])
        plans = schedules(rows)
        self.assertEqual(plans, schedules(rows))
        for batches in plans.values():
            self.assertEqual(len(batches), 512)
            self.assertTrue(all(len(batch) == len(set(batch)) == 2 for batch in batches))
            self.assertEqual([sum(i in batch for batch in batches) for i in range(8)], [128] * 8)
        self.assertTrue(all(rows[a]["group_id"] == rows[b]["group_id"] for a, b in plans["reciprocal_batch"]))
        self.assertTrue(all({rows[a]["target"], rows[b]["target"]} == {"A", "B"} for a, b in plans["reciprocal_batch"]))
        self.assertTrue(any(rows[a]["group_id"] != rows[b]["group_id"] for a, b in plans["stochastic_batch"]))

    def test_batch_control_source_and_policy_negatives(self):
        manifest = json.loads((HERE / "data/manifest.json").read_text())
        train = [r for g in manifest["splits"]["train"] for r in g["pairs"]]
        for name in ("validation", "audit"):
            with self.subTest(split=name), self.assertRaises(ValueError):
                selected_rows(manifest, [r for g in manifest["splits"][name] for r in g["pairs"]])
        rows = selected_rows(manifest, train)
        for field, value in (("chosen", "unsupported title"), ("target", "A"), ("source_id", "wrong-doi")):
            bad = copy.deepcopy(rows)
            bad[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                schedules(bad)
        with self.assertRaises(ValueError):
            schedules(rows[:-1])
        with tempfile.TemporaryDirectory(prefix="summary-policy-test-") as directory:
            config = policy()
            config["audit_scored"] = True
            (Path(directory) / "reciprocal-policy.json").write_text(json.dumps(config))
            with patch("reciprocal.HERE", Path(directory)), self.assertRaises(ValueError):
                policy()

    def test_warmup_preserves_full_sidecars_and_tensor_bytes(self):
        header = json.dumps({"fixture": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        payload = struct.pack("<f", 1.)
        with tempfile.TemporaryDirectory(prefix="summary-warmup-grants-") as directory:
            root = Path(directory)
            (root / "warmup").mkdir()
            path = root / "warmup/fixture.safetensors"
            path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
            preserve(root)
            self.assertEqual(tensor_parts(path.read_bytes())[1], payload)
            verify_checkpoint_metadata(path)
            for name in ("LICENSE", "SAND-1.0.txt", "NOTICE", "CC-BY-4.0.txt", "KIT-LICENSE", "UPSTREAM-APACHE-2.0.txt"):
                self.assertEqual((root / "warmup" / name).read_bytes(), (HERE / name).read_bytes())

    def test_curriculum_complete_exact_training_sources(self):
        manifest = json.loads((HERE / "data/manifest.json").read_text())
        train = [r for g in manifest["splits"]["train"] for r in g["pairs"]]
        before = copy.deepcopy(train)
        rows = warmup_rows(manifest, train)
        self.assertEqual(train, before)
        self.assertEqual(len(rows), 256)
        self.assertEqual(source_check()["training_pairs"], 256)
        documents = {d["doi"]: d for g in manifest["splits"]["train"] for d in g["documents"]}
        excluded = {d["doi"] for name in ("validation", "audit") for g in manifest["splits"][name] for d in g["documents"]}
        for original, row in zip(train, rows):
            self.assertFalse({row["source_id"], row["donor_id"]} & excluded)
            self.assertTrue(row["prompt"].endswith(documents[row["source_id"]]["abstract"]))
            self.assertEqual({k: v for k, v in row.items() if k not in ("prompt", "training_view")},
                             {k: v for k, v in original.items() if k != "prompt"})

    def test_curriculum_rejects_source_split_and_title_mutations(self):
        manifest = json.loads((HERE / "data/manifest.json").read_text())
        train = [r for g in manifest["splits"]["train"] for r in g["pairs"]]
        for name in ("validation", "audit"):
            with self.subTest(split=name), self.assertRaises(ValueError):
                warmup_rows(manifest, [r for g in manifest["splits"][name] for r in g["pairs"]])
        for attack in ("missing", "duplicate", "title", "source"):
            changed = copy.deepcopy(train)
            if attack == "missing":
                changed.pop()
            elif attack == "duplicate":
                changed[-1] = changed[0]
            elif attack == "title":
                changed[0]["chosen"] = "Unverified title"
            else:
                changed[0]["source_id"] = "unverified-source"
            with self.subTest(attack=attack), self.assertRaises(ValueError):
                warmup_rows(manifest, changed)
        changed = copy.deepcopy(manifest)
        changed["splits"]["train"][0]["documents"][0]["abstract"] = "Cropped."
        with self.assertRaises(ValueError):
            warmup_rows(changed, train)

    def test_curriculum_token_boundaries_and_actual_training_mutations(self):
        class FixtureTokenizer:
            def encode(self, text):
                return type("Encoding", (), {"ids": list(range(len(text)))})()
        def render(prompt, answer):
            return prompt + answer
        report = check_tokens([{"prompt": "x" * 255, "chosen": "y", "rejected": "z"}], FixtureTokenizer(), render)
        self.assertEqual(report, {"complete_warmup_inputs": 2, "maximum_warmup_tokens": 256})
        for prompt in ("x" * 256, ""):
            answer = "y" if prompt else ""
            with self.assertRaises(ValueError):
                check_tokens([{"prompt": prompt, "chosen": answer, "rejected": answer}], FixtureTokenizer(), render)
        source = (HERE / "remote.py").read_text()
        check_curriculum_training(source)
        for old, new in (("warmup_rows(manifest_source, splits[\"train\"])", "warmup_rows(manifest_source, splits[\"audit\"])"),
                         ("lr=hp.lr, weight_decay=hp.weight_decay", "lr=WARMUP_LR, weight_decay=hp.weight_decay"),
                         ("warm_steps != WARMUP_STEPS", "warm_steps < 1"),
                         ("if not admit(bootstrap_metrics):", "if False:"),
                         ("from role_bootstrap import", "from curriculum import"),
                         ('root / "warmup"', 'root / "checkpoint"')):
            self.assertIn(old, source)
            with self.subTest(mutation=old), self.assertRaises(ValueError):
                check_curriculum_training(source.replace(old, new))

    def test_training_diagnostic_excludes_validation_and_audit(self):
        manifest = json.loads((HERE / "data/manifest.json").read_text())
        train = [row for g in manifest["splits"]["train"] for row in g["pairs"]]
        controls = training_controls(manifest, train)
        excluded = {d["doi"] for name in ("validation", "audit") for g in manifest["splits"][name] for d in g["documents"]}
        for rows in controls.values():
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({r["group_id"] for r in rows}), 4)
            self.assertFalse({r["source_id"] for r in rows} & excluded)
        self.assertEqual([(r["chosen"], r["rejected"]) for r in controls["original_two_reference"]],
                         [(r["chosen"], r["rejected"]) for r in controls["single_requested_reference_control"]])
    def test_development_gate_keeps_clean_audit_closed(self):
        rows = [dict(row, group_id="fixture-" + str(i)) for i in range(32) for row in fixture_rows()]
        good = pair_metrics(rows, [2., 1.] * 64)
        weak = pair_metrics(rows, [2., 1., 0., 1.] * 32)
        trained = {style: good for style in VIEWS}
        untrained = {style: weak for style in VIEWS}
        controls = {style: {"fixture_shortcut": .5} for style in VIEWS}
        self.assertTrue(development_gate(trained, untrained, controls)["audit_allowed"])
        self.assertFalse(development_gate({style: weak for style in VIEWS}, untrained, controls)["audit_allowed"])
        self.assertFalse(development_gate(trained, trained, controls)["audit_allowed"])
        missing = dict(trained)
        missing.pop("json")
        self.assertFalse(development_gate(missing, untrained, controls)["audit_allowed"])

    def test_exposed_original_audit_is_excluded(self):
        self.assertEqual(check_source_quarantine()["source_entity_overlap"], 0)
        exposure = json.loads((HERE / "exposed-audit.json").read_text())
        manifest = json.loads((HERE / "data/manifest.json").read_text())
        manifest["splits"]["audit"][0]["source_entity_ids"].append(exposure["source_entity_ids"][0])
        splits = {name: [r for g in groups for r in g["pairs"]] for name, groups in manifest["splits"].items()}
        with self.assertRaises(ValueError):
            check_splits(manifest, splits)

    def test_compile_processes_are_bounded_without_skipping_checks(self):
        with patch("runtime_check.subprocess.run") as execute:
            compile_tools(Path("fixture-output"), Path("fixture-kit"))
        self.assertEqual(execute.call_count, 3)
        for call in execute.call_args_list:
            self.assertIn("-p=1", call.args[0])
            self.assertEqual(call.kwargs["env"]["GOMAXPROCS"], "2")
            self.assertTrue(call.kwargs["check"])
        self.assertEqual(execute.call_args_list[0].args[0][1], "test")
        self.assertEqual([call.args[0][1] for call in execute.call_args_list[1:]], ["build", "build"])
        gate = (HERE / "verify.sh").read_text()
        self.assertIn("export GOMAXPROCS=2\n", gate)
        self.assertIn('go -C "$KIT" test -p=1 -mod=readonly -v', gate)

    def test_immutable_baseline_tamper_refused_without_refit(self):
        controls = json.loads((HERE / "data/frozen-count-baselines.json").read_text())
        self.assertEqual(controls["fit_split"], "train")
        self.assertEqual(controls["training_pairs"], 256)
        class NoTokenizer:
            def encode(self, text):
                raise AssertionError("baseline identity failure must precede tokenization")
        with self.assertRaises(ValueError):
            load_controls([], NoTokenizer())

    def test_development_loss_only_and_ties(self):
        first, changed = development_selection(None, 1, {"val_loss": 1.0, "audit_accuracy": 0})
        self.assertTrue(changed)
        self.assertEqual(development_selection(first, 2, {"val_loss": 1.0, "audit_accuracy": 1}), (first, False))
        best, changed = development_selection(first, 3, {"val_loss": .9, "pairwise_accuracy": 0})
        self.assertEqual(best["epoch"], 3)
        self.assertTrue(changed)
        with self.assertRaises(ValueError):
            development_selection(best, EPOCHS + 1, {"val_loss": 0})

    def test_quality_failure_controls(self):
        good = pair_metrics(fixture_rows(), [2., 1., 2., 1.])
        weak = pair_metrics(fixture_rows(), [2., 1., 0., 1.])
        controls = {"fixture_shortcut": .5}
        self.assertTrue(quality(good, weak, controls)["passed"])
        self.assertFalse(quality(weak, weak, controls)["passed"])
        self.assertFalse(quality(good, good, controls)["passed"])
        self.assertFalse(quality(good, weak, {"fixture_shortcut": .95})["passed"])
        with self.assertRaises(ValueError):
            all_formats({"raw": good}, {"raw": weak}, {"raw": controls})
        self.assertTrue(all_formats({s: good for s in VIEWS}, {s: weak for s in VIEWS}, {s: controls for s in VIEWS})["passed"])

    def test_no_local_neural_entry_points(self):
        for name in ("remote.py", "serve.py", "diagnostic.py", "reciprocal.py", "recover.py"):
            completed = subprocess.run([sys.executable, str(HERE / name), "--help"], env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, timeout=10, check=False)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(b"runtime scoring is limited", completed.stderr)
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}), self.assertRaises(RuntimeError):
            model_remote_only()
        self.assertNotIn("torch", sys.modules)

    def test_licence_stamp_preserves_exact_tensor_payload(self):
        header = json.dumps({"fixture": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        payload = struct.pack("<f", 1.0)
        raw = struct.pack("<Q", len(header)) + header + payload
        with tempfile.TemporaryDirectory(prefix="summary-stamp-test-") as directory:
            path = Path(directory) / "fixture.safetensors"
            path.write_bytes(raw)
            preserve_artifact(path)
            verify_checkpoint_metadata(path)
            self.assertEqual(tensor_parts(path.read_bytes())[1], payload)
            stamped = path.read_bytes()
            preserve_artifact(path)
            self.assertEqual(stamped, path.read_bytes())


class HandoffTests(unittest.TestCase):
    def fixture_ready(self):
        # Synthetic validator fixture only. Never stored as a real remote report.
        return {"schema_version": 1, "status": "REGISTRATION_READY", "family": "TEXT_SUMMARIZATION",
                "run_id": 123, "commit": "b" * 40, "serving_commit": "b" * 40,
                "package_sha256": "a" * 64, "package_keccak": "c" * 64,
                "foundation_sha256": RELEASE_HASHES["foundation_lora_fp32.onnx"],
                "engine_sha256": RELEASE_HASHES["foundation_engine.wasm"], "tokenizer_sha256": RELEASE_HASHES["tokenizer.json"],
                "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                           "source_manifest_sha256": "d" * 64, "independent_groups": 64},
                "quality": {"passed": True, "audit_pairs": 128, "all_supported_formats_tested": True,
                            "formats": {style: {"accuracy": .9, "per_label": {"A": .9, "B": .9}, "untrained_accuracy": .5, "shortcut_accuracy": .5} for style in VIEWS}},
                "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                            "negative_identity_controls": 3, "vectors": [{"tokens": [50281, 22 + i, 50282], "score_bits": 1065353216} for i in range(8)]}}

    def test_readiness_rejects_missing_or_failed_real_gates(self):
        self.assertEqual(validate_ready(self.fixture_ready())["run_id"], 123)
        for section, field, value in (("source", "normalized_overlap", 1), ("source", "audit_used_for_selection", True),
                                      ("source", "grants_verified", False), ("source", "independent_groups", 31),
                                      ("quality", "audit_pairs", 63), ("quality", "all_supported_formats_tested", False),
                                      ("runtime", "reference_engine_passed", False), ("runtime", "repeat_bits_identical", False),
                                      ("runtime", "full_parser_passed", False), ("runtime", "negative_identity_controls", 2)):
            bad = self.fixture_ready()
            bad[section][field] = value
            with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                validate_ready(bad)
        for field in ("package_keccak", "package_sha256"):
            bad = self.fixture_ready()
            bad[field] = "0x" + bad[field]
            with self.assertRaises(ValueError):
                validate_ready(bad)
        bad = self.fixture_ready()
        bad["quality"]["formats"].pop("json")
        with self.assertRaises(ValueError):
            validate_ready(bad)
        bad = self.fixture_ready()
        bad["quality"]["formats"]["prose"]["accuracy"] = .79
        with self.assertRaises(ValueError):
            validate_ready(bad)

    def test_actual_lease_contract_scope_and_tamper(self):
        row = {"family": "TEXT_SUMMARIZATION", "owner": DEFAULT_OWNER, "slot": 1, "forward_port": 48601,
               "runnable_url": "https://tga1.zkasuran.dev/_summary_fixture/runnable", "keccak": "a" * 64, "sha256": "b" * 64,
               "existing_registration_id": 0, **{name + "_proof": {"sha256": "c" * 64} for name in ("candidate", "runtime", "source")}}
        blob = struct.pack(">I", 11) + b"ssh-ed25519" + struct.pack(">I", 32) + bytes(range(32))
        key = "ssh-ed25519 " + base64.b64encode(blob).decode()
        value = lease_for(row, key, "summary-fixture", "123", "b" * 40)
        validate_lease(value, row)
        self.assertIn('permitlisten="127.0.0.1:48601"', value["authorized_keys_line"])
        self.assertIn('command="/bin/false"', value["authorized_keys_line"])
        self.assertEqual({value[name + "_proof_sha256"] for name in ("candidate", "runtime", "source")}, {"c" * 64})
        for field, replacement in (("forward_port", 48602), ("package_sha256", "d" * 64), ("runtime_proof_sha256", "d" * 64)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_lease({**value, field: replacement}, row)

    def test_closed_serving_identity(self):
        root = "https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/" + "b" * 40 + "/verification/owners/text_summarization/"
        value = {"package_url": root + "fixture.safetensors", "report_url": root + "fixture.json",
                 "package_sha256": "a" * 64, "package_keccak": "c" * 64, "report_sha256": "d" * 64,
                 "slot": 1, "forward_port": 48601, "runnable_url": "https://tga1.zkasuran.dev/_summary_fixture/runnable"}
        self.assertEqual(parse_settings(value, "b" * 40), value)
        for field, replacement in (("slot", 7), ("forward_port", 48602), ("package_keccak", "0x" + "c" * 64),
                                   ("package_url", "https://example.com/private"), ("runnable_url", "https://example.com/new")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_settings({**value, field: replacement}, "b" * 40)


if __name__ == "__main__":
    unittest.main()
