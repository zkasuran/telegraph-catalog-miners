# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Literal public wire around the identical pinned reference executor."""
import importlib
import json
import sys
import threading
import urllib.error
import urllib.request

from contracts import HERE, bits, check_reply, identity, input_root


def proxy_module():
    sys.path.insert(0, str(HERE / "support/registration"))
    return importlib.import_module("proxy")


class RuntimeBridge:
    def __init__(self, backend, package):
        self.backend, self.package = backend, package

    def score(self, request):
        p = proxy_module()
        request = p.validate_request(request, {"keccak": self.package})
        raw = self.backend.score(request)
        if not isinstance(raw, dict):
            raise ValueError("actual engine returned a non-object wire response")
        expected = identity(self.package)
        # The internal pinned engine can use its documented 0x package prefix.
        # Only these two exact forms enter the bridge. The public echo is bare.
        if raw.get("adapter_hash") not in (self.package, "0x" + self.package) or any(raw.get(k) != v for k, v in expected.items() if k != "adapter_hash"):
            raise ValueError("actual reference-engine identity mismatch")
        results = raw.get("results")
        if not isinstance(results, list) or len(results) != len(request["batch"]):
            raise ValueError("actual engine result count mismatch")
        safe = []
        for row in results:
            if not isinstance(row, dict) or set(row) != {"score"}:
                raise ValueError("actual engine refused the valid vector")
            bits(row["score"])
            safe.append({"score": row["score"]})
        if request.get("input_root") and raw.get("input_root") != request["input_root"]:
            raise ValueError("actual engine input root mismatch")
        return {"results": safe, **expected, **({"input_root": request["input_root"]} if request.get("input_root") else {})}


def request_http(url, request):
    call = urllib.request.Request(url, data=json.dumps(request, separators=(",", ":")).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(call, timeout=180) as response:
        raw = response.read((1 << 20) + 1)
        if response.status != 200 or len(raw) > 1 << 20:
            raise ValueError("bounded literal wire response refused")
    return json.loads(raw)


def live_controls(bridge, package):
    """Actual public-path positive, repeat, identity and token controls."""
    p = proxy_module()
    server = p.BoundedServer(("127.0.0.1", 0), {"keccak": package, "runnable_url": "http://127.0.0.1/runnable"}, bridge)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = "http://127.0.0.1:" + str(server.server_address[1]) + "/runnable"
    try:
        batch = [{"tokens": [50281] + [42 + i % 100 for i in range(n - 2)] + [50282]} for n in (4, 8, 16, 32, 64, 96, 128, 256)]
        request = {"batch": batch, **identity(package), "input_root": input_root(batch)}
        first = check_reply(request_http(url, request), request, package)
        repeated = check_reply(request_http(url, request), request, package)
        if first != repeated:
            raise ValueError("public reference vectors did not repeat bits")
        for key in identity(package):
            bad = {**request, key: "f" * 64 if request[key] != "f" * 64 else "e" * 64}
            try:
                request_http(url, bad)
            except urllib.error.HTTPError as error:
                if error.code != 400:
                    raise ValueError("wrong wire identity did not refuse") from None
            else:
                raise ValueError("wrong wire identity accepted")
        for tokens in ([], [-1], [True]):
            bad = {"batch": [{"tokens": tokens}], **identity(package)}
            try:
                request_http(url, bad)
            except urllib.error.HTTPError as error:
                if error.code != 400:
                    raise ValueError("invalid tokens did not refuse") from None
            else:
                raise ValueError("invalid token vector accepted")
        return [{"tokens": row["tokens"], "score_bits": score} for row, score in zip(batch, first, strict=True)], 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
