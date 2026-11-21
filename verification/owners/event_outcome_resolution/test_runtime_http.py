# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Real loopback HTTP with a labelled synthetic backend, never neural inference."""
from contextlib import contextmanager
import ast
import http.client
import inspect
from pathlib import Path
import textwrap
import threading
import unittest
from urllib.parse import urlsplit

from common import json_bytes, strict_json
import proxy
from proxy import BoundedServer, identity, input_root, routes_for
from runtime import executor_row, negative_controls


class SyntheticBackend:
    """Wire fixtures only. A returned bit pattern is not an engine measurement."""
    def __init__(self, row):
        self.row = row
        self.calls = 0

    def score(self, request):
        self.calls += 1
        return {**identity(self.row), **({"input_root": request["input_root"]} if request.get("input_root") else {}),
                "results": [{"score": 0x3f800000} if item["tokens"] and all(t >= 0 for t in item["tokens"])
                            else {"error": "synthetic refusal"} for item in request["batch"]]}


@contextmanager
def http_fixture(row, handler=None):
    backend = SyntheticBackend(row)
    server = BoundedServer(("127.0.0.1", 0), row, backend)
    if handler is not None:
        server.RequestHandlerClass = handler
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server, backend
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if thread.is_alive():
            raise RuntimeError("source HTTP fixture failed to stop")


def post(server, path, request):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=2)
    try:
        connection.request("POST", path, json_bytes(request), {"Content-Type": "application/json"})
        reply = connection.getresponse()
        return reply.status, strict_json(reply.read(4096))
    finally:
        connection.close()


def request_for(row):
    value = {"batch": [{"tokens": [50281, 22, 50282]}], **identity(row)}
    value["input_root"] = input_root(value["batch"])
    return value


class RuntimeHTTP(unittest.TestCase):
    def test_actual_audit_row_without_public_route_primary_path(self):
        # runtime.executor originally supplies exactly this row, not a full lease.
        row = {"keccak": "c" * 64}
        request = request_for(row)
        with http_fixture(row) as (server, backend):
            status, response = post(server, "/runnable", request)
            self.assertEqual(status, 200)
            self.assertEqual(response, {**identity(row), "input_root": request["input_root"], "results": [{"score": 0x3f800000}]})
            self.assertEqual(backend.calls, 1)

    def test_reserved_alias_and_literal_root_both_work(self):
        url = "https://tga1.zkasuran.dev/_event_control/runnable"
        row = executor_row({"keccak256": "0x" + "c" * 64}, url)
        with http_fixture(row) as (server, backend):
            for path in ("/runnable", "/_event_control/runnable"):
                status, response = post(server, path, request_for(row))
                self.assertEqual(status, 200)
                self.assertEqual(response["results"], [{"score": 0x3f800000}])
            self.assertEqual(backend.calls, 2)

    def test_unconfigured_alias_and_query_variants_refused(self):
        row = executor_row({"keccak256": "0x" + "c" * 64})
        with http_fixture(row) as (server, backend):
            for path in ("/_event_control/runnable", "/runnable?x=1", "/runnable/", "/runnable%2f"):
                self.assertEqual(post(server, path, request_for(row)), (404, {"error": "request refused"}))
            self.assertEqual(backend.calls, 0)

    def test_invalid_reserved_urls_fail_before_server_or_executor(self):
        for value in (None, True, "", "https://example.com/_event/runnable", "http://tga1.zkasuran.dev/_event/runnable",
                      "https://tga1.zkasuran.dev/_event/runnable?x=1", "https://user@tga1.zkasuran.dev/_event/runnable",
                      "https://tga1.zkasuran.dev/_event/runnable#fragment"):
            with self.subTest(url=value), self.assertRaises(ValueError):
                routes_for({"keccak": "c" * 64, "runnable_url": value})
        with self.assertRaises(ValueError):
            executor_row({"keccak256": "0x" + "c" * 64}, "https://example.com/x")

    def test_configured_routes_are_immutable_after_server_start(self):
        row = {"keccak": "c" * 64, "runnable_url": "https://tga1.zkasuran.dev/_event_control/runnable"}
        with http_fixture(row) as (server, _):
            row["runnable_url"] = "https://tga1.zkasuran.dev/_changed/runnable"
            self.assertIsInstance(server.paths, frozenset)
            self.assertEqual(post(server, "/_changed/runnable", request_for(row))[0], 404)
            self.assertEqual(post(server, "/_event_control/runnable", request_for(row))[0], 200)

    def test_actual_identity_negative_controls_with_synthetic_backend(self):
        row = executor_row({"keccak256": "0x" + "c" * 64})
        with http_fixture(row) as (server, backend):
            self.assertEqual(negative_controls(server.server_address[1], row, backend), 3)
            # Only the two synthetic invalid-token backend probes were called.
            self.assertEqual(backend.calls, 2)

    def test_serving_passes_reserved_url_to_real_executor_call(self):
        tree = ast.parse(Path(__file__).with_name("serving.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "executor"]
        self.assertEqual(len(calls), 1)
        binding = next(key.value for key in calls[0].keywords if key.arg == "runnable_url")
        self.assertEqual(ast.unparse(binding), "row['runnable_url']")

    def test_actual_original_route_mutation_disconnects_primary_path(self):
        source = textwrap.dedent(inspect.getsource(proxy.Handler.do_POST))
        marker = "if self.path not in self.server.paths:"
        self.assertIn(marker, source)
        old = 'if self.path not in ("/runnable", urlsplit(self.server.row["runnable_url"]).path):'
        namespace = {**proxy.__dict__, "urlsplit": urlsplit}
        exec(compile(source.replace(marker, old, 1), "original_route_mutation", "exec"), namespace)
        mutant = type("OriginalRouteMutant", (proxy.Handler,), {"do_POST": namespace["do_POST"]})
        row = {"keccak": "c" * 64}
        with http_fixture(row, mutant) as (server, backend):
            with self.assertRaises(http.client.RemoteDisconnected):
                post(server, "/runnable", request_for(row))
            self.assertEqual(backend.calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
