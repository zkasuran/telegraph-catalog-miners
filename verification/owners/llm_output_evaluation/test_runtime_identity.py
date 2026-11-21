# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only execution-path controls. No model or real engine is loaded.

The actual nested runtime-audit function runs with synthetic socket/process/HTTP
doubles. Its literal requests are observed, not a normalized replacement. These
controls are not neural scores or independent audit measurements.
"""
import ast
import copy
from contextlib import nullcontext
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request

from common import FOUNDATION
from proxy import identity as wire_identity, input_root


HERE = Path(__file__).resolve().parent
SYNTHETIC_PACKAGE_HASH = "b" * 64
SYNTHETIC_SCORE_BITS = 0x3f000000


class RuntimeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.bodies = []
        self.bad_echo = False
        self.process = SimpleNamespace(poll=lambda: None, terminate=self.closed,
                                       wait=lambda timeout: 0, kill=self.closed)
        self.process_closed = False

    def closed(self):
        self.process_closed = True

    def execute(self, root, restore_shadow=False):
        source = (HERE / "runtime_audit.py").read_bytes()
        outer = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef) and node.name == "run_audit")
        actual = copy.deepcopy(next(node for node in outer.body
                                    if isinstance(node, ast.FunctionDef) and node.name == "execute"))
        if restore_shadow:
            class RestoreShadow(ast.NodeTransformer):
                def visit_Name(self, node):
                    if node.id == "wire_identity":
                        return ast.copy_location(ast.Name(id="identity", ctx=node.ctx), node)
                    return node
            actual = RestoreShadow().visit(actual)
            ast.fix_missing_locations(actual)
        run_source = (HERE / "remote.py").read_bytes()
        run_function = next(node for node in ast.parse(run_source).body
                            if isinstance(node, ast.FunctionDef) and node.name == "identity")
        run_scope = {"FAMILY": "LLM_OUTPUT_EVALUATION", "digest": lambda raw: hashlib.sha256(raw).hexdigest(),
                     "os": SimpleNamespace(environ={"OWNER_ATTEMPT": "synthetic-control",
                                                    "GITHUB_RUN_ID": "1", "GITHUB_SHA": "0" * 40,
                                                    "GITHUB_REPOSITORY": "synthetic-control"})}
        exec(compile(ast.Module(body=[run_function], type_ignores=[]),
                     "actual run metadata callback", "exec"), run_scope)

        def opened(request, timeout):
            body = json.loads(request.data)
            self.bodies.append(body)
            row = {"keccak": SYNTHETIC_PACKAGE_HASH}
            wanted = wire_identity(row)
            if any(body.get(key) != value for key, value in wanted.items()):
                raw = io.BytesIO(b'{"error":"request refused"}')
                raise urllib.error.HTTPError(request.full_url, 400, "synthetic refusal", {}, raw)
            value = {**wanted, "results": [{"score": SYNTHETIC_SCORE_BITS} for item in body["batch"]]}
            if self.bad_echo:
                value["adapter_hash"] = "0" * 64
            response = SimpleNamespace(read=lambda size: json.dumps(value).encode())
            return nullcontext(response)

        probe = SimpleNamespace(bind=lambda address: None, getsockname=lambda: ("127.0.0.1", 12345))
        fake_socket = SimpleNamespace(socket=lambda: nullcontext(probe),
                                      create_connection=lambda address, timeout: nullcontext())
        fake_thread = SimpleNamespace(start=lambda: None, join=lambda timeout: None)
        fake_server = SimpleNamespace(server_port=12346, serve_forever=lambda: None,
                                      shutdown=lambda: None, server_close=lambda: None)
        fake_http = SimpleNamespace(request=SimpleNamespace(Request=urllib.request.Request, urlopen=opened),
                                    error=SimpleNamespace(HTTPError=urllib.error.HTTPError))
        scope = {"socket": fake_socket, "root": root, "time": time, "json": json,
                 "subprocess": SimpleNamespace(Popen=lambda *args, **kwargs: self.process,
                                                TimeoutExpired=TimeoutError),
                 "threading": SimpleNamespace(Thread=lambda **kwargs: fake_thread),
                 "BoundedServer": lambda *args: fake_server, "Backend": lambda port: None,
                 "identity": run_scope["identity"], "wire_identity": wire_identity,
                 "input_root": input_root, "helper": SimpleNamespace(FOUNDATION=FOUNDATION),
                 "expected": SYNTHETIC_PACKAGE_HASH, "executable": root / "not-a-real-executor",
                 "package": root / "not-a-model-package", "env": {}, "urllib": fake_http,
                 "decode": lambda bits: bits}
        exec(compile(ast.Module(body=[actual], type_ignores=[]),
                     "actual nested audit execution path with source-only service doubles", "exec"), scope)
        return scope["execute"]

    def test_unfixed_shadowing_reproduces_wrong_callback_input(self):
        with tempfile.TemporaryDirectory(prefix="llm-identity-control-") as temporary:
            execute = self.execute(Path(temporary), restore_shadow=True)
            with self.assertRaises(TypeError):
                execute([[50281, 22, 50282]], "synthetic-unfixed")
        self.assertEqual(self.bodies, [])
        self.assertTrue(self.process_closed)

    def test_actual_wire_controls_and_literal_echoes(self):
        vectors = [[50281, 22, 50282], [50281, 23, 50282]]
        with tempfile.TemporaryDirectory(prefix="llm-identity-control-") as temporary:
            values, negatives = self.execute(Path(temporary))(vectors, "synthetic-fixed")
        self.assertEqual(values, [SYNTHETIC_SCORE_BITS] * len(vectors))
        self.assertEqual(negatives, 3)
        self.assertEqual(len(self.bodies), 4)
        self.assertEqual(self.bodies[-1]["batch"], [{"tokens": ids} for ids in vectors])
        self.assertEqual(self.bodies[-1]["adapter_hash"], SYNTHETIC_PACKAGE_HASH)
        self.assertFalse(self.bodies[-1]["adapter_hash"].startswith("0x"))
        self.assertEqual(self.bodies[-1]["foundation_hash"], FOUNDATION["foundation_lora_fp32.onnx"])
        self.assertEqual(self.bodies[-1]["tokenizer_hash"], FOUNDATION["tokenizer.json"])
        self.assertTrue(self.process_closed)

    def test_wrong_literal_wire_echo_is_refused(self):
        self.bad_echo = True
        with tempfile.TemporaryDirectory(prefix="llm-identity-control-") as temporary:
            with self.assertRaisesRegex(ValueError, "literal runtime identity/count mismatch"):
                self.execute(Path(temporary))([[50281, 22, 50282]], "synthetic-bad-echo")
        self.assertTrue(self.process_closed)

    def test_run_metadata_callback_is_preserved(self):
        outer = next(node for node in ast.parse((HERE / "runtime_audit.py").read_bytes()).body
                     if isinstance(node, ast.FunctionDef) and node.name == "run_audit")
        output = next(node.value for node in outer.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "output" for target in node.targets))
        metadata = [value for key, value in zip(output.keys, output.values) if key is None]
        self.assertEqual(len(metadata), 1)
        self.assertEqual(ast.unparse(metadata[0]), "identity(root)")

    def test_source_controls_have_no_neural_imports(self):
        self.assertFalse({"torch", "transformers", "peft", "onnxruntime", "tensorflow"} & set(sys.modules))


if __name__ == "__main__":
    unittest.main()
