# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal dependency and wire failure controls. All score replies are synthetic."""
import ast
import base64
import io
import json
import os
import re
import struct
import time
import types
import unittest
import zipfile
from datetime import datetime, timezone
from unittest.mock import patch

from contracts import (
    FAMILY,
    FOUNDATION,
    HERE,
    PACKAGE_KECCAK,
    PACKAGE_SHA,
    SOURCE_SHA,
    helper_scope,
    metrics,
    read_asset,
    remote_only,
    rows,
    verify_assets,
)
from wire import request_ok, response_ok


def score_function(scope):
    code = next(c for c in scope["server"].__wrapped__.__code__.co_consts if isinstance(c, types.CodeType) and c.co_name == "score")
    closed = {"deadline": time.monotonic() + 60, "logpath": types.SimpleNamespace(stat=lambda: types.SimpleNamespace(st_size=0)),
              "manifest": {"keccak256": PACKAGE_KECCAK}, "port": 1}
    def cell(value):
        return (lambda: value).__closure__[0]
    return types.FunctionType(code, scope, argdefs=(None,), closure=tuple(cell(closed[name]) for name in code.co_freevars))


class ContractsTests(unittest.TestCase):
    def test_public_lease_filename_matches_literal_activation_wire(self):
        tree = ast.parse((HERE / "serve.py").read_text())
        names = [node.args[0].right.value for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "write" and len(node.args) == 2
                 and isinstance(node.args[1], ast.Name) and node.args[1].id == "lease"
                 and isinstance(node.args[0], ast.BinOp) and isinstance(node.args[0].op, ast.Div)
                 and isinstance(node.args[0].right, ast.Constant)]
        # The actual activation consumer requires exactly this literal basename.
        self.assertEqual(names, ["lease.json"])
        workflow = (HERE.parents[2] / ".github/workflows/evaluator-owner-language_translation.yml").read_text()
        self.assertIn('"$RUNNER_TEMP/translation-serve/lease.json"', workflow)
        self.assertIn('${{ runner.temp }}/translation-serve/lease.json', workflow)
        self.assertNotIn("public-lease.json", workflow)
        selected = [node for node in ast.parse(read_asset("reference-common.py.txt").decode()).body
                    if isinstance(node, ast.FunctionDef) and node.name in
                    {"now", "utc", "public_key", "lease_for", "validate_lease"}]
        scope = {"base64": base64, "re": re, "struct": struct, "datetime": datetime,
                 "timezone": timezone, "WINDOW": 5 * 3600, "COMMIT": re.compile(r"[a-f0-9]{40}")}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "pinned-public-lease-contract", "exec"), scope)  # noqa: S102
        row = {"family": FAMILY, "owner": "0x8b224783fe5b3c52b7db0cb9b1754f8812b75287",
               "slot": 4, "forward_port": 48604, "runnable_url": "https://tga1.zkasuran.dev/_synthetic/runnable",
               "sha256": PACKAGE_SHA, "keccak": PACKAGE_KECCAK, "existing_registration_id": 0}
        for kind in ("candidate", "runtime", "source"):
            row[kind + "_proof"] = {"sha256": "a" * 64}
        # Synthetic public wire bytes only. No key, wallet or model is created.
        wire = struct.pack(">I", 11) + b"ssh-ed25519" + struct.pack(">I", 32) + b"\0" * 32
        key = "ssh-ed25519 " + base64.b64encode(wire).decode()
        lease = scope["lease_for"](row, key, "synthetic-control", "1234", "b" * 40)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(names[0], json.dumps(lease).encode())
        with zipfile.ZipFile(io.BytesIO(archive.getvalue())) as incoming:
            items = incoming.infolist()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].filename, "lease.json")
            self.assertLessEqual(items[0].file_size, 8192)
            scope["validate_lease"](json.loads(incoming.read(items[0])), row)

    def test_real_counterparty_wire_is_bare_and_never_truncates(self):
        request = {"batch": [{"tokens": [50281, 50282]}], "adapter_hash": PACKAGE_KECCAK,
                   "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"], "tokenizer_hash": FOUNDATION["tokenizer.json"]}
        self.assertIs(request_ok(request), request)
        for tokens in ([1] * 257, [-1], [True], [50368]):
            with self.assertRaises(ValueError):
                request_ok({**request, "batch": [{"tokens": tokens}]})
        with self.assertRaises(ValueError):
            request_ok({**request, "adapter_hash": "0x" + PACKAGE_KECCAK})
        response = {**{k:v for k,v in request.items() if k != "batch"}, "results": [{"score": 1065353216}]}
        self.assertIs(response_ok(response, 1), response)
        with self.assertRaises(ValueError):
            response_ok({**response, "adapter_hash": "0x" + PACKAGE_KECCAK}, 1)
    def test_immutable_assets_source_and_native_binding(self):
        source, native = verify_assets()
        self.assertEqual(native["source_manifest_sha256"], SOURCE_SHA)
        self.assertEqual(source["group_counts"], {"train": 128, "validation": 32, "test": 64})

    def test_literal_missing_json_and_repaired_score_wire(self):
        scope = helper_scope({})
        score = score_function(scope)
        del scope["json"]
        with self.assertRaisesRegex(NameError, "json"):
            score([[50281, 50282]])
        scope["json"] = json
        reply = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"], "tokenizer_hash": FOUNDATION["tokenizer.json"],
                 "adapter_hash": PACKAGE_KECCAK, "results": [{"score": 1065353216}]}
        sent = []
        class Response:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, size):
                return json.dumps(reply).encode()
        def response(request, timeout):
            sent.append(request.data)
            return Response()
        with patch("urllib.request.urlopen", side_effect=response):
            self.assertEqual(score([[50281, 50282]]), [1065353216])
            for key in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
                with self.assertRaises(ValueError):
                    score([[50281, 50282]], {key: "0" * 64})
            reply["results"] = [{"score": 2143289344}]
            with self.assertRaises(ValueError):
                score([[50281, 50282]])
        expected = {"batch": [{"tokens": [50281, 50282]}], "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
                    "tokenizer_hash": FOUNDATION["tokenizer.json"], "adapter_hash": PACKAGE_KECCAK}
        self.assertEqual(sent[0], json.dumps(expected).encode())

    def test_audit_has_complete_original_reciprocal_roles(self):
        for style in ("raw", "json", "prose"):
            items = rows(style)
            self.assertEqual(len(items), 256)
            self.assertEqual(len({item["group_id"] for item in items}), 64)
            self.assertEqual(sorted(item["chosen"] for item in items), sorted(item["rejected"] for item in items))

    def test_nonfinite_and_partial_score_coverage_refused(self):
        with self.assertRaises(ValueError):
            metrics([], [])
        with self.assertRaises(ValueError):
            metrics(rows("raw"), [2143289344] * 512)

    def test_neural_entry_guard_denies_local_execution(self):
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}), self.assertRaises(RuntimeError):
            remote_only()

    def test_source_rights_are_not_relabelled_cc0(self):
        source = json.loads(read_asset("source-manifest.json"))
        self.assertTrue(all(item["license"] == "CC-BY-2.0-FR" and item["credited_contributor"] for item in source["originals"].values()))


if __name__ == "__main__":
    unittest.main()
