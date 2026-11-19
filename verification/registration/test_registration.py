# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only contracts and local HTTP fixtures. No model or SSH execution."""
import argparse
import base64
import copy
import http.client
import io
import json
import socket
import struct
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import common
import prepare as registration_prepare
import proxy
import session
from common import (
    ANCHORS,
    DEFAULT_OWNER,
    FOUNDATION,
    KIT,
    REPO,
    check_classes,
    check_proofs,
    check_run,
    fetch_hash,
    json_bytes,
    lease_for,
    load_inventory,
    sha,
    strict_json,
    validate_lease,
    validate_row,
)
from proxy import (
    BoundedServer,
    identity,
    input_root,
    validate_request,
    validate_response,
)

FAMILY = "SENTIMENT_ANALYSIS"
CLOCK = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)


def candidate():
    anchor = ANCHORS[FAMILY]
    row = {key: anchor[key] for key in ("class", "file", "sha256", "keccak", "slot", "forward_port",
                                       "runnable_url", "verified_run_id", "verified_commit")}
    row["family"] = FAMILY
    prefix = f"https://raw.githubusercontent.com/{REPO}/" + anchor["verified_commit"] + "/verification/registration/proofs/"
    row["url"] = prefix + row["file"]
    for name in ("candidate", "runtime", "source"):
        row[name + "_proof"] = {"url": prefix + name + ".json", "sha256": anchor[name + "_sha256"]}
    return validate_row(row)


def proofs(row):
    formats = {name: {"n_eval": 256, "pairwise_accuracy": 0.875,
                      "accuracy_by_label": {"Negative": 0.85, "Positive": 0.90}, "repeat_bit_vectors": 512}
               for name in ("json", "prose", "bare_label")}
    recovery = {"run_id": row["verified_run_id"], "commit": row["verified_commit"]}
    candidate_proof = {"schema_version": 1, "status": "CANDIDATE_PENDING_CONTROLLER_REGISTRATION_TEST",
                       "family": row["family"], "child": row["child"],
                       "package": {"file": row["file"], "sha256": row["sha256"],
                                   "keccak256": "0x" + row["keccak"], "bytes": 6798100},
                       "recovery_identity": recovery, "kit_revision": KIT, "runtime_inputs": FOUNDATION,
                       "retrained": False, "originals": 256, "formats": formats,
                       "training_identity": {"repository": REPO, "run_id": 44266324976,
                                             "commit": "a3a6f86c1bd831b41de8d99d12b0620c3a287f6a",
                                             "quality_report_sha256": "4ffc26072a347b9af879afbd3c17fbecc1c1115bf6fe141943088577dff23422"},
                       "evidence_sha256": {"runtime-result.json": row["runtime_proof"]["sha256"],
                                           "data-binding.json": row["source_proof"]["sha256"]}}
    runtime = {"family": row["family"], "verdict": "PASS", "failures": [], "recovery_identity": recovery,
               "package_sha256": row["sha256"], "adapter_hash": "0x" + row["keccak"],
               "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
               "engine_hash": FOUNDATION["foundation_engine.wasm"], "tokenizer_hash": FOUNDATION["tokenizer.json"],
               "originals": 256, "repeat_bit_vectors": 1536, "formats": formats}
    source = {"data_report_sha256": "417f9901e5bfc7ddbff1b3cf016f973649f9c50ebf0410f2391583e80d214451",
              "normalized_source_overlap": 0, "files": {name: {"pairs": n, "labels": {"Positive": n//2, "Negative": n//2}}
              for name, n in (("train", 768), ("validation", 128), ("test", 256))}}
    return candidate_proof, runtime, source


def class_snapshot(row):
    return {"schema": 1, "fetched_at": CLOCK.isoformat(),
            "family": {"url": common.NODE + "/api/wasm?resolve=" + row["family"],
                       "raw": json.dumps({"resolve": {"intent": row["family"], "class": row["class"], "served_by": "GENERIC"}})},
            "child": {"url": common.NODE + "/engine/v1/intents/" + row["child"],
                      "raw": json.dumps({"intent_id": row["child"], "intent_name": row["child"], "class": row["class"],
                                         "evaluation": {"kind": row["class"]},
                                         "request": {"served_by": row["family"]}, "response": {"served_by": row["family"]},
                                         "onchain": {"served_by": row["family"], "ready": True}})}}


def key():
    # Fixed public fixture bytes only. No private key is generated or read.
    blob = struct.pack(">I", 11) + b"ssh-ed25519" + struct.pack(">I", 32) + b"p"*32
    return "ssh-ed25519 " + base64.b64encode(blob).decode() + " source-only-fixture"


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.row = candidate()

    def test_protocol_type_failures_keep_exact_value_error_contract(self):
        request = {"batch": [{"tokens": [1]}], **identity(self.row)}
        response = {"results": [None], **identity(self.row)}
        calls = (
            (common.utc, (None,)),
            (proxy.normalize_hash, (None,)),
            (proxy.validate_response, (response, request, self.row)),
            (session.find_decision, (None, self.row, CLOCK, CLOCK)),
        )
        for function, arguments in calls:
            with self.subTest(function=function.__name__), self.assertRaises(ValueError) as error:
                function(*arguments)
            self.assertIs(type(error.exception), ValueError)

    def test_entrypoints_cleanup_even_for_unexpected_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "candidate-registration"
            root.mkdir()
            args = argparse.Namespace(inventory=root / "candidates.json", family=FAMILY,
                                      batch="source-only", existing=0)
            with patch.object(registration_prepare, "remote_only"), \
                    patch.dict("os.environ", {"RUNNER_TEMP": tmp}), \
                    patch.object(registration_prepare, "load_inventory", return_value=self.row), \
                    patch.object(registration_prepare.argparse.ArgumentParser, "parse_args", return_value=args), \
                    patch.object(registration_prepare, "prepare", side_effect=AssertionError("source-only fixture")), \
                    patch.object(Path, "unlink", autospec=True) as unlink, self.assertRaises(AssertionError):
                registration_prepare.main()
            unlink.assert_any_call(root / "forward-key", missing_ok=True)
            unlink.assert_any_call(root / "forward-key.pub", missing_ok=True)
            with patch.object(session, "remote_only"), patch.dict("os.environ", {"RUNNER_TEMP": tmp}), \
                    patch.object(session.argparse.ArgumentParser, "parse_args", return_value=argparse.Namespace(cleanup=False)), \
                    patch.object(session.signal, "signal"), \
                    patch.object(session, "run", side_effect=AssertionError("source-only fixture")), \
                    patch.object(session, "cleanup") as cleanup, self.assertRaises(AssertionError):
                session.main()
            cleanup.assert_called_once_with(root)

    def test_exact_candidate_is_admitted(self):
        self.assertEqual(self.row["forward_port"], 48601)
        self.assertEqual(self.row["owner"], DEFAULT_OWNER)
        self.assertEqual(self.row["keccak"], ANCHORS[FAMILY]["keccak"])

    def test_wrong_candidate_hash_class_run_and_routing_are_refused(self):
        mutations = {"sha256": "0"*64, "keccak": "0"*64, "class": "hybrid", "verified_run_id": "1",
                     "verified_commit": "0"*40, "file": "OTHER.safetensors", "slot": 2,
                     "forward_port": 48602, "runnable_url": "https://tga1.zkasuran.dev/_other/runnable"}
        for name, value in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_row({**self.row, name: value})

    def test_candidate_types_and_owner_must_be_valid(self):
        for name, value in (("slot", True), ("owner", "other"), ("existing_registration_id", -1),
                            ("forward_port", "48601"), ("child", "OTHER_GENERAL"), ("family", "OTHER")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_row({**self.row, name: value})

    def test_mutable_foreign_and_credential_urls_refused(self):
        for url in (f"https://raw.githubusercontent.com/{REPO}/master/pkg.safetensors",
                    "http://example.com/file", "https://user:password@raw.githubusercontent.com/file",
                    self.row["url"] + "?token=fixture", self.row["url"].replace("/proofs/", "/../")):
            with self.assertRaises(ValueError):
                validate_row({**self.row, "url": url})

    def test_wrong_proof_hash_or_extra_registry_fields_refused(self):
        changed = copy.deepcopy(self.row)
        changed["runtime_proof"]["sha256"] = "0"*64
        with self.assertRaises(ValueError):
            validate_row(changed)
        with self.assertRaises(ValueError):
            validate_row({**self.row, "private_key": "fixture"})

    def test_duplicate_registry_identity_and_wrong_existing_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.json"
            path.write_bytes(json_bytes({"schema_version": 1, "candidates": [self.row, self.row]}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_inventory(path, FAMILY)
            path.write_bytes(json_bytes({"schema_version": 1, "candidates": [self.row]}))
            self.assertEqual(load_inventory(path, FAMILY, 77)["existing_registration_id"], 77)
            with self.assertRaises(ValueError):
                load_inventory(path, FAMILY, True)

    def test_recovery_run_must_be_public_success_at_exact_commit(self):
        value = {"id": int(self.row["verified_run_id"]), "head_sha": self.row["verified_commit"],
                 "path": ".github/workflows/evaluator-sentiment-recover.yml", "status": "completed", "conclusion": "success",
                 "repository": {"full_name": REPO, "private": False}, "head_repository": {"full_name": REPO}}
        check_run(value, self.row)
        for name, change in (("id", 1), ("head_sha", "0"*40), ("conclusion", "failure"), ("status", "in_progress")):
            with self.assertRaises(ValueError):
                check_run({**value, name: change}, self.row)
        with self.assertRaises(ValueError):
            check_run({**value, "repository": {"full_name": REPO, "private": True}}, self.row)

    def test_source_and_runtime_proof_identity_and_quality(self):
        values = proofs(self.row)
        check_proofs(self.row, *values)
        mutations = ((0, "family", "OTHER"), (0, "retrained", True),
                     (0, "recovery_identity", {"run_id": "1", "commit": self.row["verified_commit"]}),
                     (1, "adapter_hash", "0x" + "0"*64), (1, "repeat_bit_vectors", 8),
                     (1, "formats", {}), (2, "normalized_source_overlap", 1))
        for index, name, change in mutations:
            changed = copy.deepcopy(values)
            changed[index][name] = change
            with self.subTest(name=name), self.assertRaises(ValueError):
                check_proofs(self.row, *changed)
        changed = copy.deepcopy(values)
        changed[1]["formats"]["json"]["pairwise_accuracy"] = float("nan")
        changed[0]["formats"] = changed[1]["formats"]
        with self.assertRaises(ValueError):
            check_proofs(self.row, *changed)

    def test_full_live_class_agreement(self):
        value = class_snapshot(self.row)
        check_classes(value, self.row, CLOCK)
        for section in ("request", "response", "onchain"):
            changed = copy.deepcopy(value)
            child = json.loads(changed["child"]["raw"])
            child[section]["served_by"] = "GENERIC"
            changed["child"]["raw"] = json.dumps(child)
            with self.assertRaises(ValueError):
                check_classes(changed, self.row, CLOCK)
        for field, change in (("class", "hybrid"), ("intent_id", "OTHER_GENERAL")):
            changed = copy.deepcopy(value)
            child = json.loads(changed["child"]["raw"])
            child[field] = change
            changed["child"]["raw"] = json.dumps(child)
            with self.assertRaises(ValueError):
                check_classes(changed, self.row, CLOCK)

    def test_class_snapshot_stale_missing_unknown_or_wrong_url(self):
        value = class_snapshot(self.row)
        for timestamp in (CLOCK - timedelta(minutes=16), CLOCK + timedelta(minutes=2)):
            with self.assertRaises(ValueError):
                check_classes({**value, "fetched_at": timestamp.isoformat()}, self.row, CLOCK)
        with self.assertRaises(ValueError):
            check_classes({**value, "extra": "other"}, self.row, CLOCK)
        changed = copy.deepcopy(value)
        changed["family"]["url"] = "https://example.com"
        with self.assertRaises(ValueError):
            check_classes(changed, self.row, CLOCK)

    def test_remote_only_guard_precedes_runtime_or_key_creation(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(RuntimeError):
            common.remote_only()

    def test_download_wrong_hash_leaves_no_accepted_file(self):
        class Response(io.BytesIO):
            status = 200

            def geturl(self):
                return self_url

        self_url = self.row["url"]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "package"
            with patch.object(common.urllib.request, "urlopen", return_value=Response(b"source-fixture")), \
                    self.assertRaisesRegex(ValueError, "hash mismatch"):
                fetch_hash(self_url, target, "0"*64, 1024)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name("package.partial").exists())
            with patch.object(common.urllib.request, "urlopen", return_value=Response(b"source-fixture")):
                fetch_hash(self_url, target, sha(b"source-fixture"), 1024)
            self.assertEqual(target.read_bytes(), b"source-fixture")

    def test_public_key_and_exact_restricted_lease(self):
        lease = lease_for(self.row, key(), "sentiment-candidate", "55555", "a"*40, CLOCK)
        validate_lease(lease, self.row, CLOCK)
        line = lease["authorized_keys_line"]
        for restriction in ('permitlisten="127.0.0.1:48601"', 'permitopen="registration-denied.invalid:1"', 'command="/bin/false"', "restrict,port-forwarding", "expiry-time="):
            self.assertIn(restriction, line)
        self.assertNotIn('permitopen="none"', line)
        self.assertEqual(lease["required_server_policy"]["AllowTcpForwarding"], "remote")
        self.assertEqual(lease["required_server_policy"]["AllowStreamLocalForwarding"], "no")
        self.assertNotIn("private", json.dumps(lease))
        for name, value in (("public_key", "ssh-rsa fixture"), ("forward_port", 48602),
                            ("runnable_url", "https://example.com"), ("adapter_hash", "0"*64)):
            with self.assertRaises(ValueError):
                validate_lease({**lease, name: value}, self.row, CLOCK)
        with self.assertRaises(ValueError):
            validate_lease(lease, self.row, CLOCK + timedelta(hours=5))

    def test_invalid_public_key_wire_shapes_refused(self):
        for value in ("private-fixture", "ssh-ed25519 AAAA", key().replace("ssh-ed25519", "ssh-rsa"), key() + "\nextra"):
            with self.assertRaises(ValueError):
                common.public_key(value)

    def test_ssh_targets_proxy_only_and_disables_other_identities(self):
        command = session.ssh_command(Path("/tmp/candidate-registration"), self.row, 12345)
        self.assertEqual(command[-1], "telegraph-verify@98.95.110.88")
        self.assertIn("127.0.0.1:48601:127.0.0.1:12345", command)
        for option in ("IdentityAgent=none", "IdentitiesOnly=yes", "StrictHostKeyChecking=yes", "ForwardAgent=no"):
            self.assertIn(option, command)

    def test_cleanup_refuses_reused_or_unrelated_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "candidate-registration"
            root.mkdir()
            record = root / "forward-process.json"
            record.write_bytes(json_bytes({"pid": 1234567, "start": "owned", "role": "forward"}))
            with patch.object(session, "remote_only"), patch.dict("os.environ", {"RUNNER_TEMP": tmp}), \
                    patch.object(session, "process_start", return_value="reused"), patch.object(session.os, "killpg") as signal_process:
                session.cleanup(root)
                signal_process.assert_not_called()
            record.write_bytes(json_bytes({"pid": 1234567, "start": "owned", "role": "forward"}))
            original_read = Path.read_bytes

            def read(path):
                return b"other-process\0" if str(path).startswith("/proc/") else original_read(path)

            with patch.object(session, "remote_only"), patch.dict("os.environ", {"RUNNER_TEMP": tmp}), \
                    patch.object(session, "process_start", return_value="owned"), patch.object(Path, "read_bytes", read), \
                    patch.object(session.os, "killpg") as signal_process:
                with self.assertRaisesRegex(ValueError, "outside this lease"):
                    session.cleanup(root)
                signal_process.assert_not_called()


class WireTests(unittest.TestCase):
    def setUp(self):
        self.row = candidate()
        self.request = session.fixed_request(self.row)

    def test_input_root_matches_pinned_canonical_encoding(self):
        batch = [{"tokens": [1, 2]}, {"tokens": [1, 2]}, {"tokens": [3]}]
        first = sha(b'{"input_ids":[1,2],"attention_mask":[1,1]}')
        second = sha(b'{"input_ids":[3],"attention_mask":[1]}')
        expected = sha(f"0:{first}\n1:{first}\n2:{second}".encode())
        self.assertEqual(input_root(batch), expected)
        self.assertEqual(input_root([]), "")
        self.assertNotEqual(input_root(batch), input_root(batch[::-1]))

    def test_valid_request_and_boundary_tokens(self):
        validate_request(self.request, self.row)
        upper = {**self.request, "input_root": self.request["input_root"].upper()}
        self.assertEqual(validate_request(upper, self.row)["input_root"], self.request["input_root"])
        large = {"batch": [{"tokens": [50367]*512}]*100, **identity(self.row)}
        self.assertEqual(len(validate_request(large, self.row)["batch"]), 100)

    def test_invalid_inputs_and_identities_refused(self):
        for tokens in ([], [-1], [50368], [True], [1.5], ["1"], None, [1]*513):
            value = {"batch": [{"tokens": tokens}], **identity(self.row)}
            with self.subTest(tokens_type=type(tokens).__name__), self.assertRaises(ValueError):
                validate_request(value, self.row)
        for body in ({}, {"batch": [1], **identity(self.row)},
                     {"batch": [{"text": "not tokens"}], **identity(self.row)},
                     {**self.request, "input_root": "0"*64},
                     {"batch": [{"tokens": [1]}]*101, **identity(self.row)}):
            with self.assertRaises(ValueError):
                validate_request(body, self.row)
        for field in identity(self.row):
            with self.assertRaises(ValueError):
                validate_request({**self.request, field: "0"*64}, self.row)

    def response(self):
        return {**identity(self.row), "input_root": self.request["input_root"], "results": [{"score": 0x3f800000}]*4}

    def test_equal_score_bits_are_legitimate_positional_results(self):
        self.assertEqual(len(validate_response(self.response(), self.request, self.row)["results"]), 4)

    def test_missing_extra_duplicate_nonfinite_and_conflicting_results_refused(self):
        value = self.response()
        for results in (None, [], value["results"] + [value["results"][0]], [{}]*4,
                        [{"score": float("nan")}]*4, [{"score": 0x7fc00000}]*4,
                        [{"score": 0x7f800000}]*4, [{"score": True}]*4,
                        [{"score": 1, "error": "conflict"}]*4, [{"error": "source-fixture"}]*4):
            with self.assertRaises(ValueError):
                validate_response({**value, "results": results}, self.request, self.row)
        for raw in (b'{"results":[],"results":[]}', b'{"score":1,"score":2}', b'{"score":NaN}'):
            with self.assertRaises(ValueError):
                strict_json(raw)

    def test_response_identity_and_input_root_cannot_be_echoed_wrong(self):
        value = self.response()
        for field in (*identity(self.row), "input_root"):
            with self.assertRaises(ValueError):
                validate_response({**value, field: "0"*64}, self.request, self.row)

    def test_source_only_preflight_requires_repeats_and_real_refusal(self):
        class Fixture:
            def __init__(self, changing=False, accepts_invalid=False):
                self.calls = 0
                self.changing = changing
                self.accepts_invalid = accepts_invalid

            def score(self, request):
                self.calls += 1
                invalid = any(not item["tokens"] or item["tokens"][0] < 0 for item in request["batch"])
                item = {"error": "source-only refusal"} if invalid and not self.accepts_invalid else {"score": 0x3f800000 + int(self.changing and self.calls > 1)}
                return {**identity(candidate()), "input_root": request["input_root"], "results": [item] * len(request["batch"])}

        session.preflight(Fixture(), self.row)
        for fixture in (Fixture(changing=True), Fixture(accepts_invalid=True)):
            with self.assertRaises(ValueError):
                session.preflight(fixture, self.row)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.row = candidate()
        self.entry = {"intent_id": FAMILY, "author_address": self.row["owner"], "wasm_hash": self.row["keccak"],
                      "registration_id": 101, "registered_at": (CLOCK + timedelta(seconds=5)).isoformat(),
                      "activation_status": "active", "eval": {"private_fixture": "never copied"}, "rejection_reason": "never copied"}

    def listing(self, entries):
        return {"intents": {FAMILY: {"entries": entries}}}

    def test_new_exact_registration_decision_is_sanitized(self):
        decision = session.find_decision(self.listing([self.entry]), self.row, CLOCK, CLOCK + timedelta(minutes=1))
        self.assertEqual(decision["registration_id"], 101)
        self.assertEqual(set(decision), {"family", "owner", "adapter_hash", "registration_id", "registered_at", "status"})

    def test_stale_wrong_family_owner_hash_future_and_pending_are_ignored(self):
        changes = ({"registered_at": (CLOCK - timedelta(seconds=1)).isoformat()}, {"intent_id": "OTHER"},
                   {"author_address": "0x" + "0"*40}, {"wasm_hash": "0"*64},
                   {"registered_at": (CLOCK + timedelta(days=1)).isoformat()}, {"activation_status": "pending"})
        for change in changes:
            self.assertIsNone(session.find_decision(self.listing([{**self.entry, **change}]), self.row, CLOCK, CLOCK + timedelta(minutes=1)))

    def test_existing_id_matches_only_that_id_even_for_newer_entries(self):
        row = {**self.row, "existing_registration_id": 101}
        old = {**self.entry, "registered_at": (CLOCK - timedelta(days=1)).isoformat()}
        self.assertEqual(session.find_decision(self.listing([old]), row, CLOCK, CLOCK)["registration_id"], 101)
        self.assertIsNone(session.find_decision(self.listing([{**self.entry, "registration_id": 102}]), row, CLOCK, CLOCK + timedelta(minutes=1)))

    def test_duplicate_or_multiple_new_registration_decisions_refused(self):
        for entries in ([self.entry, self.entry], [self.entry, {**self.entry, "registration_id": 102}]):
            with self.assertRaises(session.AmbiguousDecision):
                session.find_decision(self.listing(entries), self.row, CLOCK, CLOCK + timedelta(minutes=1))

    def test_all_terminal_decisions_stop_without_copying_diagnostics(self):
        for status in ("active", "rejected", "superseded", "deregistered"):
            decision = session.find_decision(self.listing([{**self.entry, "activation_status": status}]),
                                             self.row, CLOCK, CLOCK + timedelta(minutes=1))
            self.assertEqual(decision["status"], status)
            self.assertNotIn("never copied", json.dumps(decision))


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.row = candidate()

        class FixtureBackend:
            def __init__(self):
                self.calls = 0
                self.bad = False
                self.block = False
                self.entered = threading.Event()
                self.release = threading.Event()

            def score(self, request):
                self.calls += 1
                self.entered.set()
                if self.block:
                    self.release.wait(timeout=3)
                bits = 0x7fc00000 if self.bad else 0x3f800000
                return {**identity(candidate()), **({"input_root": request["input_root"]} if "input_root" in request else {}),
                        "results": [{"score": bits} for _ in request["batch"]]}

        self.fixture = FixtureBackend()
        self.server = BoundedServer(("127.0.0.1", 0), self.row, self.fixture)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.fixture.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def post(self, value, path="/runnable", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request("POST", path, json_bytes(value), headers or {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, strict_json(response.read())
        finally:
            connection.close()

    def test_primary_path_real_http_contract_and_no_input_logging(self):
        value = session.fixed_request(self.row)
        output = io.StringIO()
        with redirect_stderr(output):
            code, response = self.post(value)
        self.assertEqual(code, 200)
        self.assertEqual(len(response["results"]), 4)
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(self.post(value, "/_s2sent/runnable")[0], 200)

    def test_wrong_paths_and_invalid_inputs_never_reach_executor(self):
        self.assertEqual(self.post(session.fixed_request(self.row), "/other")[0], 404)
        value = {"batch": [{"tokens": [-1]}], **identity(self.row)}
        code, result = self.post(value)
        self.assertEqual(code, 400)
        self.assertEqual(self.fixture.calls, 0)
        self.assertEqual(result, {"error": "request refused"})

    def test_startup_checks_real_proxy_refusals_before_forwarding(self):
        session.proxy_refusals(self.server.server_port, self.row)
        self.assertEqual(self.fixture.calls, 0)

    def test_ambiguous_length_transfer_encoding_and_oversize_headers(self):
        for headers in (
                "Content-Length: 2\r\nContent-Length: 2\r\n",
                "Content-Length: 2\r\nTransfer-Encoding: chunked\r\n",
                "Content-Length: 9999999\r\n",
                "X-Filler: " + "x" * (common.MAX_HEADERS + 1) + "\r\nContent-Length: 2\r\n"):
            with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=3) as connection:
                message = "POST /runnable HTTP/1.1\r\nHost: fixture\r\nContent-Type: application/json\r\n" + headers + "\r\n{}"
                connection.sendall(message.encode())
                response = connection.recv(1024)
                self.assertNotIn(b" 200 ", response.split(b"\r\n", 1)[0])
        self.assertEqual(self.fixture.calls, 0)

    def test_nonfinite_backend_sets_fatal_and_never_publishes_a_score(self):
        self.fixture.bad = True
        code, value = self.post(session.fixed_request(self.row))
        self.assertEqual(code, 502)
        self.assertEqual(value, {"error": "request refused"})
        self.assertTrue(self.server.fatal.is_set())

    def test_no_inference_queue_under_concurrent_requests(self):
        self.fixture.block = True
        results = []
        request = session.fixed_request(self.row)
        worker = threading.Thread(target=lambda: results.append(self.post(request)), daemon=True)
        worker.start()
        self.assertTrue(self.fixture.entered.wait(timeout=2))
        self.assertEqual(self.post(request)[0], 503)
        self.assertEqual(self.fixture.calls, 1)
        self.fixture.release.set()
        worker.join(timeout=3)
        self.assertEqual(results[0][0], 200)

    def test_slow_body_has_absolute_read_deadline(self):
        with patch.object(proxy, "READ_TIMEOUT", 0.15), socket.create_connection(
                ("127.0.0.1", self.server.server_port), timeout=2) as connection:
            connection.sendall(b"POST /runnable HTTP/1.1\r\nHost: fixture\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n{")
            time.sleep(0.3)
            try:
                response = connection.recv(1024)
            except (ConnectionResetError, TimeoutError):
                response = b""
            self.assertNotIn(b" 200 ", response)
        self.assertEqual(self.fixture.calls, 0)

    def test_connection_limit_refuses_before_creating_another_worker(self):
        self.server.connections = threading.BoundedSemaphore(1)
        self.server.connections.acquire()
        try:
            # Admission rejects before reading headers/body, so observe the
            # actual socket response before a client races to write a body.
            with socket.create_connection(("127.0.0.1", self.server.server_port), timeout=2) as connection:
                response = connection.recv(1024)
                self.assertTrue(response.startswith(b"HTTP/1.1 503 "))
            self.assertEqual(self.fixture.calls, 0)
        finally:
            self.server.connections.release()


if __name__ == "__main__":
    unittest.main()
