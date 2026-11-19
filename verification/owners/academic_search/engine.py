# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Real pinned reference executor, remote-only and literal-wire checked."""
import json
import os
import socket
import subprocess
import time
import urllib.request
from contextlib import contextmanager

from bootstrap import MODEL_FILES, file_sha
from scorer import remote_only
from wire import FOUNDATION, ENGINE, TOKENIZER, input_root, request, response


def package_identity(root, package):
    done = subprocess.run([str(root / "digest"), str(package)], capture_output=True, timeout=30, check=True)
    return json.loads(done.stdout)


@contextmanager
def server(root, package, name, max_seconds=10000, request_timeout=180):
    remote_only()
    foundation = root / "foundation"
    for file, expected in MODEL_FILES.items():
        if file_sha(foundation / file) != expected:
            raise ValueError("actual reference input hash mismatch: " + file)
    identity = package_identity(root, package)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "FOUNDATION_MODEL_PATH": str(foundation / "foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": FOUNDATION, "FOUNDATION_ENGINE_WASM_PATH": str(foundation / "foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": ENGINE, "ADAPTER_SCORE_TIMEOUT": "180s"}
    logpath = root / ("engine-" + name + ".log")
    with logpath.open("w") as log:
        process = subprocess.Popen([str(root / "adapterexec"), "--package", str(package), "--listen", "127.0.0.1:" + str(port)],
                                   env=env, stdout=log, stderr=log)
        try:
            started = time.monotonic()
            while time.monotonic() - started < 240:
                if process.poll() is not None:
                    raise RuntimeError("reference executor exited before readiness")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(.5)
            else:
                raise TimeoutError("reference executor readiness deadline")
            deadline = time.monotonic() + max_seconds

            def post(payload):
                if time.monotonic() > deadline or logpath.stat().st_size > 16 << 20:
                    raise TimeoutError("reference execution time/log ceiling")
                wire = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
                req = urllib.request.Request("http://127.0.0.1:" + str(port) + "/runnable", data=wire, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=min(request_timeout, max(1, deadline - time.monotonic()))) as reply:
                    raw = reply.read((1 << 20) + 1)
                    if reply.status != 200 or len(raw) > 1 << 20:
                        raise ValueError("reference executor response status/size")
                return raw

            def score(batch, override=None):
                sent = request(batch, identity["keccak"], input_root(batch))
                sent.update(override or {})
                return response(post(sent), sent)

            yield score, post, identity
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)


def score_all(score, vectors):
    # Serial engine scoring within bounded four-item HTTP batches.
    return [bit for i in range(0, len(vectors), 4) for bit in score(vectors[i:i + 4])]
