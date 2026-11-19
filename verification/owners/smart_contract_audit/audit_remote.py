# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Independent job: one frozen candidate, exact real executor wire and audit only."""
import argparse
import json
import math
import os
import socket
import struct
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from data import FAMILY, ATTEMPT, answers, render, sha, write
from policy import execution_gate, quality, remote_only
from remote_prepare import bound_data, prepare
from contracts import finite_bits, response_bits


def decode(bits):
    if isinstance(bits, bool) or not isinstance(bits, int) or not 0 <= bits <= 0xffffffff:
        raise ValueError("bad score bits")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        raise ValueError("nonfinite score")
    return value


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    root = parser.parse_args().root
    if (root / "audit-result.json").exists():
        raise ValueError("audit already measured; no adaptive repeat")
    manifest = json.loads((root / "candidate-manifest.json").read_text())
    if manifest["run_id"] != os.environ["GITHUB_RUN_ID"] or manifest["commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("candidate belongs to another run or commit")
    if manifest["owner_attempt"] != os.environ["OWNER_RUN_ID"]:
        raise ValueError("candidate belongs to another owner attempt")
    rows, binding = bound_data(root)
    execution_gate(rows, json.loads((root / "evm-report.json").read_text()), binding["inventory_sha256"])
    audit = [r for r in rows if r["split"] == "audit"]
    foundation, _, _, hashes, _ = prepare(root, training=False)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(max_length=384)
    for name, key in (("adapterexec", "executor_sha256"), ("wire-root", "wire_root_sha256")):
        if sha((root / name).read_bytes()) != manifest[key]:
            raise ValueError("runtime binary identity mismatch")
        (root / name).chmod(0o700)
    report = {"schema_version": 1, "family": FAMILY, "owner_attempt": os.environ["OWNER_RUN_ID"],
              "run_id": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"],
              "data": binding, "candidate_manifest_sha256": sha((root / "candidate-manifest.json").read_bytes()),
              "runtime_sha256": manifest["executor_sha256"], "foundation": hashes,
              "node_identity": "pinned Telegraph node-compatible kit " + manifest["kit"]["revision"],
              "live_node_qualification": "not attempted", "audit_designs": len({r["template_group"] for r in audit}),
              "audit_source_rows": len(audit), "measurements": {}, "verdict": "INCOMPLETE"}
    write(root / "audit-result.json", report)  # durable before starting model inference
    vectors = {}
    maximum = 0
    for style, name in enumerate(("prose", "json", "report")):
        full, ablated = [], []
        for row in audit:
            for answer in answers(row["facts"], row["vulnerable"], style):
                tok.no_truncation()
                ids = tok.encode(render(row["prompt"], answer)).ids
                if len(ids) > 384:
                    raise ValueError("complete audit source or answer exceeds scored input")
                maximum = max(maximum, len(ids))
                tok.enable_truncation(max_length=384)
                full.append(ids)
                ablated.append(tok.encode(render("", answer)).ids)
        vectors[name], vectors[name + "_answer_only"] = full, ablated
    for candidate in ("untrained", "candidate"):
        package = root / (candidate + ".safetensors")
        if sha(package.read_bytes()) != manifest["packages"][candidate]["sha256"]:
            raise ValueError("frozen package identity changed")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {**os.environ, "FOUNDATION_MODEL_PATH": str(foundation / "foundation_lora_fp32.onnx"),
               "FOUNDATION_MODEL_HASH": hashes["foundation_lora_fp32.onnx"],
               "FOUNDATION_ENGINE_WASM_PATH": str(foundation / "foundation_engine.wasm"),
               "FOUNDATION_ENGINE_WASM_HASH": hashes["foundation_engine.wasm"]}
        with (root / (candidate + "-runtime.log")).open("w") as log:
            process = subprocess.Popen([str(root / "adapterexec"), "--package", str(package), "--listen", f"127.0.0.1:{port}"],
                                       env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise ValueError("executor failed to load")
                    try:
                        with socket.create_connection(("127.0.0.1", port), timeout=1):
                            break
                    except OSError:
                        time.sleep(1)
                else:
                    raise ValueError("executor readiness timeout")
                endpoint = f"http://127.0.0.1:{port}/runnable"
                try:
                    urlopen(Request(endpoint, data=b'{"tokens":[1,2]}', headers={"Content-Type": "application/json"}), timeout=30)
                    raise ValueError("no-batch request was not refused")
                except HTTPError as error:
                    if error.code != 400:
                        raise

                def score(batch):
                    actual = subprocess.run([str(root / "wire-root"), str(package)], input=json.dumps(batch),
                                            text=True, capture_output=True, check=True, timeout=30)
                    wire = json.loads(actual.stdout)
                    request = {"batch": [{"tokens": ids} for ids in batch],
                               "foundation_hash": hashes["foundation_lora_fp32.onnx"],
                               "tokenizer_hash": hashes["tokenizer.json"], **wire}
                    with urlopen(Request(endpoint, data=json.dumps(request).encode(), headers={"Content-Type": "application/json"}), timeout=180) as response:
                        raw = response.read((1 << 20) + 1)
                    if len(raw) > 1 << 20:
                        raise ValueError("response ceiling")
                    body = json.loads(raw)
                    return response_bits(body, request)

                negative_controls = []
                if candidate == "candidate":
                    actual = subprocess.run([str(root / "wire-root"), str(package)], input=json.dumps(vectors["prose"][:1]),
                                            text=True, capture_output=True, check=True, timeout=30)
                    wire = json.loads(actual.stdout)
                    for field in ("foundation_hash", "adapter_hash", "tokenizer_hash"):
                        request = {"batch": [{"tokens": vectors["prose"][0]}], "foundation_hash": hashes["foundation_lora_fp32.onnx"], "tokenizer_hash": hashes["tokenizer.json"], **wire}
                        request[field] = "0x" + request[field]
                        with urlopen(Request(endpoint, data=json.dumps(request).encode(), headers={"Content-Type": "application/json"}), timeout=180) as response:
                            body = json.loads(response.read(1 << 20))
                        try:
                            response_bits(body, request)
                        except ValueError:
                            negative_controls.append({"field": field, "observed_echo": body.get(field), "rejected_by_literal_client_contract": True})
                        else:
                            raise ValueError("wrong identity passed literal client contract")
                    report["negative_identity_controls"] = negative_controls

                measurements = {}
                for name, batch in vectors.items():
                    if candidate == "untrained" and name.endswith("_answer_only"):
                        continue
                    bits = []
                    for index in range(0, len(batch), 8):
                        bits.extend(score(batch[index:index + 8]))
                    if score(batch[:8]) != bits[:8]:
                        raise ValueError("same vectors produced different FP32 bits")
                    wins = [decode(bits[2*i]) > decode(bits[2*i+1]) for i in range(len(audit))]
                    measurements[name] = {"bits": bits, "accuracy": sum(wins) / len(wins),
                        "by_mechanism": {m: sum(w for r, w in zip(audit, wins) if r["facts"]["mechanism"] == m) / sum(r["facts"]["mechanism"] == m for r in audit) for m in ("access_control", "arithmetic")},
                        "by_label": {str(v): sum(w for r, w in zip(audit, wins) if r["vulnerable"] == v) / sum(r["vulnerable"] == v for r in audit) for v in (True, False)},
                        "repeat_vectors": 8}
                    report["measurements"][candidate] = measurements
                    write(root / "audit-result.json", report)
                if candidate == "candidate":
                    recorded = measurements["prose"]["bits"][:8]
                    report["vectors"] = [{"tokens": ids, "score_bits": bits} for ids, bits in zip(vectors["prose"][:8], recorded)]
                    for value in report["vectors"]:
                        finite_bits(value["score_bits"])
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    metrics = {}
    shortcuts = json.loads((root / "shortcut-report.json").read_text())["audit"]
    for name in ("prose", "json", "report"):
        m = report["measurements"]["candidate"][name]
        metrics[name] = {**m, "untrained_accuracy": report["measurements"]["untrained"][name]["accuracy"],
                         "answer_only_accuracy": report["measurements"]["candidate"][name + "_answer_only"]["accuracy"],
                         "shortcut_accuracy": max(shortcuts["length"], shortcuts["source_answer_overlap"], shortcuts["answer_only"], report["measurements"]["candidate"][name + "_answer_only"]["accuracy"])}
    report.update(quality(metrics))
    report["limits"] = ["Bounded original fixture groups are not universal contract proof", "Same author constructs source and claims; no human review", "No canonical qualification or runnable public endpoint"]
    write(root / "audit-result.json", report)
    print(json.dumps({"verdict": report["verdict"], "failures": report["failures"], "audit_source_rows": len(audit)}))
    if report["verdict"] != "PASS":
        raise ValueError("independent exported-runtime quality gate failed")
    grants = json.loads((root / "grant-report.json").read_text())
    if grants["grants_verified"] is not True or grants["source_manifest_sha256"] != manifest["source_manifest_sha256"]:
        raise ValueError("source grants not bound to candidate")
    from data import check
    checked = check(rows)
    actual = subprocess.run([str(root / "wire-root"), str(root / "candidate.safetensors")], input="[]",
                            text=True, capture_output=True, check=True, timeout=30)
    package_keccak = json.loads(actual.stdout)["adapter_hash"]
    ready = {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY,
             "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
             "package_sha256": sha((root / "candidate.safetensors").read_bytes()), "package_keccak": package_keccak,
             "foundation_sha256": hashes["foundation_lora_fp32.onnx"], "engine_sha256": hashes["foundation_engine.wasm"],
             "tokenizer_sha256": hashes["tokenizer.json"], "serving_commit": os.environ["GITHUB_SHA"],
             "source": {"normalized_overlap": checked["normalized_cross_split_duplicates"], "audit_used_for_selection": False,
                        "grants_verified": grants["grants_verified"], "source_manifest_sha256": grants["source_manifest_sha256"],
                        "independent_groups": len({r["template_group"] for r in rows}), "audit_groups": report["audit_designs"]},
             "quality": {"passed": report["verdict"] == "PASS", "audit_pairs": len(audit), "all_supported_formats_tested": set(metrics) == {"prose", "json", "report"},
                         "formats": {name: {"accuracy": m["accuracy"], "per_label": {**m["by_label"], **m["by_mechanism"]}, "untrained_accuracy": m["untrained_accuracy"], "shortcut_accuracy": m["shortcut_accuracy"]} for name, m in metrics.items()}},
             "runtime": {"full_parser_passed": manifest["full_parser_passed"], "reference_engine_passed": True,
                         "repeat_bits_identical": True, "negative_identity_controls": len(report["negative_identity_controls"]),
                         "vectors": report["vectors"], "audit_max_tokens": maximum},
             "evidence": {"audit_report_sha256": sha((root / "audit-result.json").read_bytes()), "executor_sha256": manifest["executor_sha256"],
                          "candidate_manifest_sha256": sha((root / "candidate-manifest.json").read_bytes())}}
    if ready["source"]["independent_groups"] < 32 or ready["quality"]["audit_pairs"] < 64 or len(ready["runtime"]["vectors"]) < 8:
        raise ValueError("registration evidence count gate")
    write(root / "registration-ready.json", ready)


if __name__ == "__main__":
    main()
