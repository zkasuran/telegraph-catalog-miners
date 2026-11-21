# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""One frozen-format audit on the actual reference engine, never native Torch."""
import argparse
import json
import os
import socket
import struct
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from assets import CHILD, FAMILY, HERE, SCOPE, capture_classes, prepare, remote_only
from common import DEFAULT_OWNER, FOUNDATION, KIT, sha, strict_json, write_json
from contracts import FORMATS
from freeze import token_counts
from groups import components
from proxy import Backend, BoundedServer, identity, input_root, validate_response
from remote import read_split
from scorer import normalized, render, response
from parent_scope import claimed_intents, verify_scope


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def compile_runtime(root, kit):
    for executable, source in (("adapterexec", "./cmd/adapterexec"), ("stagecheck", str(HERE / "stagecheck.go"))):
        subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / executable), source],
                       cwd=kit, check=True, timeout=600)
    subprocess.run(["go", "test", "-mod=readonly", "-count=1", str(HERE / "stagecheck.go"), str(HERE / "stagecheck_test.go")],
                   cwd=kit, check=True, timeout=180)


def parse_package(root, path, expected):
    if sha(path.read_bytes()) != expected:
        raise ValueError("exported package bytes differ from freeze")
    capture_classes(root / "class-snapshot.json")
    command = [str(root / "stagecheck"), "--intent", FAMILY, "--child", CHILD, "--expected-class", "judgment",
               "--scope", SCOPE,
               "--package", str(path), "--sha256", expected, "--foundation", str(root / "foundation.json"),
               "--foundation-sha256", sha((root / "foundation.json").read_bytes()),
               "--class-snapshot", str(root / "class-snapshot.json"),
               "--snapshot-sha256", sha((root / "class-snapshot.json").read_bytes()), "--out", str(root)]
    result = subprocess.run(command, capture_output=True, timeout=60, check=True)
    (root / (path.stem + "-parser.stdout")).write_bytes(result.stdout)
    parsed = strict_json(result.stdout)
    if (parsed["sha256"] != expected or parsed["manifest"]["verification"] != {"stage1": True, "loaded": False} or
            parsed["manifest"]["class"] != "judgment"):
        raise ValueError("actual full parser result mismatch")
    return parsed


@contextmanager
def executor(root, package, parsed):
    row = {"keccak": parsed["keccak256"].removeprefix("0x")}
    port = free_port()
    env = {**os.environ, "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"]}
    with (root / (package.stem + "-engine.log")).open("wb") as log:
        process = subprocess.Popen([str(root / "adapterexec"), "--package", str(package), "--listen", f"127.0.0.1:{port}"],
                                   env=env, stdout=log, stderr=log)
        server = None
        thread = None
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise ValueError("reference engine stopped during load")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise ValueError("reference load deadline")
            backend = Backend(port, timeout=600)
            server = BoundedServer(("127.0.0.1", 0), row, backend)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            yield row, backend, server.server_address[1]
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
            if thread is not None:
                thread.join(timeout=5)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def score_vectors(backend, row, vectors):
    request = {"batch": [{"tokens": ids} for ids in vectors], **identity(row)}
    request["input_root"] = input_root(request["batch"])
    reply = validate_response(backend.score(request), request, row)
    return [item["score"] for item in reply["results"]]


def negative_controls(port, row, backend):
    import http.client
    from common import json_bytes
    valid = {"batch": [{"tokens": [50281, 22, 23, 50282]}], **identity(row)}
    for key in identity(row):
        wrong = {**valid, key: "0"*64}
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            connection.request("POST", "/runnable", json_bytes(wrong), {"Content-Type": "application/json"})
            reply = connection.getresponse()
            if reply.status != 400 or strict_json(reply.read(4096)) != {"error": "request refused"}:
                raise ValueError("actual proxy failed negative identity control")
        finally:
            connection.close()
    for tokens in ([], [-1]):
        request = {"batch": [{"tokens": tokens}], **identity(row)}
        result = validate_response(backend.score(request), request, row, errors=True)
        if result["results"] != [{"error": "refused"}]:
            raise ValueError("actual reference engine accepted invalid tokens")
    return 3


def shortcuts(rows, baseline):
    def acc(feature):
        return sum((feature(r["chosen"]) > feature(r["rejected"])) +
                   0.5*(feature(r["chosen"]) == feature(r["rejected"])) for r in rows)/len(rows)
    weights = baseline["coefficients"]
    unigram = acc(lambda a: sum(weights.get(k, 0)*v for k, v in token_counts(a).items()))
    length = acc(lambda a: len(a)*baseline["length_direction"])
    utf8 = acc(lambda a: len(a.encode())*baseline["length_direction"])
    overlap = sum((len(set(normalized(r["prompt"]).split()) & set(normalized(r["chosen"]).split())) >
                   len(set(normalized(r["prompt"]).split()) & set(normalized(r["rejected"]).split()))) +
                   0.5*(len(set(normalized(r["prompt"]).split()) & set(normalized(r["chosen"]).split())) ==
                        len(set(normalized(r["prompt"]).split()) & set(normalized(r["rejected"]).split()))) for r in rows)/len(rows)
    return {"answer_unigram": unigram, "length": length, "utf8_length": utf8, "token_overlap": overlap}


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", required=True, choices=FORMATS)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(os.environ["RUNNER_TEMP"]) / ("event-runtime-" + args.format)
    root.mkdir(exist_ok=True)
    out = root / ("audit-" + args.format + ".json")
    if out.exists():
        raise ValueError("refuse repeated audit on the same frozen package")
    candidate = strict_json((args.candidate_dir / "candidate.json").read_bytes())
    verify_scope(candidate, SCOPE)
    if candidate["run_id"] != int(os.environ["GITHUB_RUN_ID"]) or candidate["commit"] != os.environ["GITHUB_SHA"] or candidate["audit_read"] is not False:
        raise ValueError("candidate run/commit/audit boundary mismatch")
    frozen = strict_json((HERE / "frozen/frozen.json").read_bytes())
    if sha((HERE / "frozen/frozen.json").read_bytes()) != candidate["source_manifest_sha256"]:
        raise ValueError("candidate source freeze mismatch")
    audit = read_split("audit", frozen)
    independent = components(audit)
    if independent < 32:
        raise ValueError("fewer than 32 independent source/entity groups")
    rows = [{**r, "chosen": response(r["prompt"], r["label"], args.format),
             "rejected": response(r["prompt"], r["wrong_label"], args.format)} for r in audit]
    _, _, kit = prepare(root)
    compile_runtime(root, kit)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(max_length=256)
    vectors = [tok.encode(render(r["prompt"], r[side])).ids for r in rows for side in ("chosen", "rejected")]
    # Verify no answer distinction disappeared under the runtime's 256-token cap.
    tok.no_truncation()
    if any(len(tok.encode(render(r["prompt"], r[side])).ids) > 256 for r in rows for side in ("chosen", "rejected")):
        raise ValueError("audit distinction truncates")
    outputs = {}
    fixed = [[50281] + [22+(i % 100) for i in range(n-2)] + [50282] for n in (4, 8, 16, 32, 64, 128, 192, 256)]
    actual_vectors = []
    controls = 0
    parsed_trained = None
    for name, key in (("trained", "package"), ("untrained", "untrained_package")):
        spec = candidate[key]
        package = root / spec["file"]
        package.write_bytes((args.candidate_dir / spec["file"]).read_bytes())
        parsed = parse_package(root, package, spec["sha256"])
        scores = []
        with executor(root, package, parsed) as (row, backend, port):
            for start in range(0, len(vectors), 8):
                scores += score_vectors(backend, row, vectors[start:start+8])
            if name == "trained":
                bits = score_vectors(backend, row, fixed)
                if bits != score_vectors(backend, row, fixed):
                    raise ValueError("reference engine repeated bits differ")
                actual_vectors = [{"tokens": v, "score_bits": b} for v, b in zip(fixed, bits)]
                controls = negative_controls(port, row, backend)
                parsed_trained = parsed
        values = [struct.unpack("<f", struct.pack("<I", b))[0] for b in scores]
        wins = [values[2*i] > values[2*i+1] for i in range(len(rows))]
        outputs[name] = {"accuracy": sum(wins)/len(wins), "per_label": {
            label: sum(w for r, w in zip(rows, wins) if r["label"] == label)/sum(r["label"] == label for r in rows)
            for label in ("yes", "no", "unresolved")}}
    baseline_raw = (HERE / "frozen/baseline.json").read_bytes()
    if sha(baseline_raw) != frozen["baseline_sha256"]:
        raise ValueError("immutable baseline changed")
    probes = shortcuts(rows, strict_json(baseline_raw))
    metrics = {**outputs["trained"], "untrained_accuracy": outputs["untrained"]["accuracy"],
               "shortcut_accuracy": max(probes.values()), "shortcut_probes": probes}
    passed = (metrics["accuracy"] >= 0.80 and min(metrics["per_label"].values()) >= 0.70 and
              metrics["accuracy"] + 1e-9 >= metrics["untrained_accuracy"] + 0.05 and
              metrics["accuracy"] + 1e-9 >= metrics["shortcut_accuracy"] + 0.10)
    report = {"schema_version": 1, "family": FAMILY, "format": args.format,
              "scope": SCOPE, "claimed_intents": claimed_intents(SCOPE),
              "run_id": candidate["run_id"], "commit": candidate["commit"],
              "package_sha256": candidate["package"]["sha256"],
              "package_keccak": parsed_trained["keccak256"].removeprefix("0x"),
              "source_manifest_sha256": candidate["source_manifest_sha256"],
              "independent_groups": independent, "audit_pairs": len(rows), "metrics": metrics, "passed": passed,
              "runtime": {"full_parser_passed": True, "reference_engine_passed": True,
                          "repeat_bits_identical": True, "negative_identity_controls": controls, "vectors": actual_vectors},
              "foundation": FOUNDATION, "kit_commit": KIT,
              "executor_sha256": sha((root / "adapterexec").read_bytes()),
              "runtime_versions": {"go": subprocess.run(["go", "version"], capture_output=True, check=True, text=True, timeout=10).stdout.strip(),
                                   "python": __import__("sys").version, "platform": __import__("platform").platform()},
              "class_snapshot_sha256": sha((root / "class-snapshot.json").read_bytes()),
              "candidate_report_sha256": sha((args.candidate_dir / "candidate.json").read_bytes()),
              "owner": DEFAULT_OWNER, "quality_engine": "pinned reference WASM, no native Torch audit"}
    write_json(out, report)
    print(json.dumps({"format": args.format, "passed": passed, "metrics": metrics}))
    if not passed:
        raise ValueError("genuine source audit quality gate failed; preserve report, do not retune on audit")


if __name__ == "__main__":
    main()
