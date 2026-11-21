# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""One remote-only audit of a frozen keyword candidate in the primary runtime."""
import argparse
import importlib.metadata
import json
import math
import os
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from gates import shortcut_report
from prepare_data import MAX_TOKENS, answer
from ready_contract import validate_ready
from remote import approved_data, remote_only, verify_sources
from scorer import render, score
from source import fetch as source_fetch
from source import sha, write
from wire import bits_value, check_response

HERE = Path(__file__).resolve().parent
FORMATS = ("plain", "array", "object")


def views(originals):
    return {name: [{**r, "chosen": answer(r["chosen_term"], name), "rejected": answer(r["rejected_term"], name)}
                   for r in originals] for name in FORMATS}


def metrics(rows, bits):
    if not rows or len(bits) != 2 * len(rows):
        raise ValueError("missing audit pairs or scores")
    values = [bits_value(value) for value in bits]
    wins = [values[2*i] > values[2*i+1] for i in range(len(rows))]
    grouped = {}
    strata = {}
    labels = {}
    for r, win in zip(rows, wins):
        grouped.setdefault(r["group_id"], []).append(int(win))
        direction = "longer" if len(r["chosen"]) > len(r["rejected"]) else "shorter" if len(r["chosen"]) < len(r["rejected"]) else "tie"
        strata.setdefault(direction, []).append(int(win))
        labels.setdefault(r["label_kind"], []).append(int(win))
    # Each connected source component has one vote regardless of row count.
    rates = [sum(v) / len(v) for v in grouped.values()]
    return {"pairs": len(rows), "groups": len(grouped), "correct": sum(wins),
            "pairwise_accuracy": sum(wins) / len(wins), "group_accuracy": sum(rates) / len(rates),
            "group_mean_95pct_bound": [max(0.0, sum(rates) / len(rates) - math.sqrt(math.log(40) / (2 * len(rates)))),
                                       min(1.0, sum(rates) / len(rates) + math.sqrt(math.log(40) / (2 * len(rates))))],
            "uncertainty_scope": "Hoeffding bound for bounded component means under independent grouped sampling, not a population or factual guarantee",
            "group_results": {k: {"correct": sum(v), "n": len(v)} for k, v in grouped.items()},
            "strata": {k: {"correct": sum(v), "n": len(v), "accuracy": sum(v) / len(v)} for k, v in strata.items()},
            "labels": {k: {"correct": sum(v), "n": len(v), "accuracy": sum(v) / len(v)} for k, v in labels.items()},
            "tie_policy": "reward ties are losses"}


def verdict(trained, initial, shortcuts, config):
    failures = []
    if set(trained) != set(FORMATS) or set(initial) != set(FORMATS):
        return {"verdict": "FAIL", "failures": ["missing exact audit format"]}
    strongest = max(shortcuts.values())
    if strongest > config["maximum_data_shortcut"]:
        failures.append("independent audit shortcut exceeds unchanged 65% data gate")
    for name in FORMATS:
        m = trained[name]
        if m["pairs"] < 64 or m["groups"] < 32 or m["pairs"] != initial[name]["pairs"] or m["groups"] != initial[name]["groups"]:
            failures.append(name + " has insufficient or mismatched independent audit samples")
        values = [m["pairwise_accuracy"], m["group_accuracy"], initial[name]["pairwise_accuracy"], *shortcuts.values(),
                  *(s["accuracy"] for s in m["strata"].values()), *(s["accuracy"] for s in m["labels"].values())]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            failures.append(name + " invalid audit or baseline metric")
            continue
        if m["pairwise_accuracy"] < config["minimum_accuracy"] or m["group_accuracy"] < config["minimum_accuracy"]:
            failures.append(name + " below independent group/pair accuracy gate")
        if any(s["accuracy"] < config["minimum_stratum_accuracy"] for s in m["strata"].values()):
            failures.append(name + " below length-direction stratum gate")
        if not m["labels"] or any(s["accuracy"] < config["minimum_stratum_accuracy"] for s in m["labels"].values()):
            failures.append(name + " below observed source label-kind gate")
        if m["pairwise_accuracy"] < initial[name]["pairwise_accuracy"] + config["improvement_over_initial"]:
            failures.append(name + " did not beat seed-matched initialization")
        if m["pairwise_accuracy"] < strongest + config["improvement_over_shortcut"]:
            failures.append(name + " did not beat strongest frozen shortcut")
    return {"verdict": "FAIL" if failures else "PASS", "failures": failures, "strongest_frozen_shortcut": strongest}


def identity_controls(exchange, batch, identity, honest_bits):
    """Probe actual identity echoes. The primary server returns its identity, not ours."""
    observed = {}
    for field in ("foundation_hash", "adapter_hash", "tokenizer_hash", "input_root"):
        wrong = {**identity, field: "0x" + identity[field]}
        reply = exchange(batch, wrong)
        if check_response(reply, identity, len(batch)) != honest_bits:
            raise ValueError("identity probe changed actual scores or executor identity")
        try:
            check_response(reply, wrong, len(batch))
        except ValueError:
            observed[field] = {"supplied": wrong[field], "returned": reply[field],
                               "validator_refused": True, "score_bits_unchanged": True}
        else:
            raise ValueError("actual identity probe was accepted")
    return observed


def registration_report(audit, manifest, source_manifest_sha256, source_verification, vectors, assets):
    """Combine measured audit/parser/runtime evidence. Never substitute expected results."""
    if audit["verdict"] != "PASS" or not source_verification["verified_originals"]:
        raise ValueError("passing measured audit and verified source bytes required")
    verification = manifest["manifest"]["verification"]
    runtime = audit["runtime"]
    if runtime.get("token_control", {}).get("negative_tokens_refused") is not True:
        raise ValueError("actual reference-engine negative token refusal required")
    controls = audit["failure_controls"]
    if set(controls) != {"foundation_hash", "adapter_hash", "tokenizer_hash", "input_root"} or any(
        c.get("validator_refused") is not True or c.get("score_bits_unchanged") is not True or
        c.get("supplied") != "0x" + c.get("returned", "") for c in controls.values()
    ):
        raise ValueError("actual complete literal identity controls required")
    formats = {}
    for name in FORMATS:
        metric = audit["formats"]["candidate"][name]
        formats[name] = {"accuracy": metric["pairwise_accuracy"], "group_accuracy": metric["group_accuracy"],
                         "per_label": {k: v["accuracy"] for k, v in metric["labels"].items()},
                         "label_counts": {k: v["n"] for k, v in metric["labels"].items()},
                         "untrained_accuracy": audit["formats"]["initial"][name]["pairwise_accuracy"],
                         "shortcut_accuracy": max(audit["format_shortcuts"][name].values())}
    measured = audit["formats"]["candidate"]["array"]
    report = {"schema_version": 1, "status": "REGISTRATION_READY", "family": "KEYWORD_EXTRACTION",
              "run_id": int(audit["audit_run_id"]), "commit": audit["audit_commit"], "serving_commit": audit["audit_commit"],
              "package_sha256": manifest["sha256"], "package_keccak": manifest["runtime_adapter_hash"],
              "foundation_sha256": assets["foundation_lora_fp32.onnx"], "engine_sha256": assets["foundation_engine.wasm"],
              "tokenizer_sha256": assets["tokenizer.json"],
              "source": {"normalized_overlap": audit["normalized_overlap"], "audit_used_for_selection": False,
                         "grants_verified": True, "source_manifest_sha256": source_manifest_sha256,
                         "source_verification_data_sha256": sha(json.dumps(source_verification, sort_keys=True).encode()),
                         "independent_groups": measured["groups"]},
              "quality": {"passed": True, "audit_pairs": measured["pairs"], "all_supported_formats_tested": True,
                          "formats": formats, "label_scope": "Observed source label_kind values measure pairwise preference accuracy, not absolute relevance labels."},
              "runtime": {"full_parser_passed": verification["stage1"], "reference_engine_passed": runtime["reference_engine_passed"],
                          "repeat_bits_identical": runtime["repeat_bits_identical"],
                          "negative_identity_controls": len(audit["failure_controls"]),
                          "negative_tokens_refused": runtime["token_control"]["negative_tokens_refused"],
                          "vectors": [{"tokens": tokens, "score_bits": bits} for tokens, bits in zip(vectors, audit["fixed_vector_bits"], strict=True)],
                          "executor_sha256": audit["executor_sha256"]},
              "audit_result_data_sha256": sha(json.dumps(audit, sort_keys=True).encode()),
              "scope": "Licensed scientific-abstract centrality preferences, AI-assisted. Independent node qualification remains required."}
    if manifest["sha256"] != audit["candidate"]["package_sha256"]:
        raise ValueError("parser/audit package identity mismatch")
    validate_ready(report)
    if len(json.dumps(report).encode()) > 1 << 20:
        raise ValueError("registration-ready report byte ceiling exceeded")
    return report


def bound_candidate(bundle, data_report, recovery=None):
    candidate = json.loads((bundle / "candidate.json").read_text())
    if candidate["family"] != "KEYWORD_EXTRACTION" or candidate["data"] != data_report:
        raise ValueError("candidate/data identity mismatch")
    if candidate["config_sha256"] != sha((HERE / "config.json").read_bytes()):
        raise ValueError("candidate/audit literal config byte identity mismatch")
    for name, field in (("KEYWORD_EXTRACTION.safetensors", "package_sha256"),
                        ("KEYWORD_EXTRACTION_INITIAL.safetensors", "initial_package_sha256"),
                        ("frozen-baselines.json", "frozen_baselines_sha256"),
                        ("foundation.json", "foundation_spec_sha256")):
        path = bundle / name
        if path.stat().st_size > 64 << 20 or sha(path.read_bytes()) != candidate[field]:
            raise ValueError("frozen candidate artifact identity mismatch: " + name)
    if recovery is not None:
        from recovery_contract import validate_candidate
        validate_candidate(candidate, data_report, recovery, sha((HERE / "config.json").read_bytes()))
    elif candidate["commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("audit must use the candidate's exact source commit")
    return candidate


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--recovery-contract", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if (args.root / "audit-started.json").exists():
        raise ValueError("audit already started for this root; inspect existing evidence, no lottery retry")
    config = json.loads((HERE / "config.json").read_text())
    data_report, splits = approved_data(args.data)
    recovery = None
    if args.recovery_contract:
        raw = args.recovery_contract.read_bytes()
        if len(raw) > 1 << 20:
            raise ValueError("recovery contract byte ceiling")
        from recovery_contract import validate_contract
        recovery = validate_contract(json.loads(raw))
        if sha((args.data / "approved-manifest.json").read_bytes()) != recovery["data_manifest_sha256"]:
            raise ValueError("recovery source manifest bytes changed")
    candidate = bound_candidate(args.bundle, data_report, recovery)
    if recovery is not None:
        write(args.root / "integration-recovery.json", {"contract": recovery, "audit_commit": os.environ["GITHUB_SHA"],
              "audit_run_id": os.environ["GITHUB_RUN_ID"], "candidate_changed": False,
              "scope": "Continue the exact frozen candidate after documented runtime timeout. No model selection or retraining."})
    write(args.root / "audit-started.json", {"package_sha256": candidate["package_sha256"], "commit": candidate["commit"],
          "audit_run_id": os.environ["GITHUB_RUN_ID"], "started_at": datetime.now(timezone.utc).isoformat()})
    source_verification = verify_sources(args.data, {"test": splits["test"]})
    write(args.root / "source-verification.json", source_verification)
    import native as standard
    foundation = args.root / "foundation"
    foundation.mkdir(exist_ok=True)
    for name, expected in standard.FOUNDATION.items():
        standard.fetch(standard.RELEASE + name, foundation / name, expected)
    archive = args.root / "kit.tar.gz"
    spec = standard.SOURCES["kit"]
    standard.fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + spec["revision"], archive, spec["sha256"], 32 << 20)
    with tarfile.open(archive) as tar:
        if sum(m.size for m in tar.getmembers()) > 128 << 20:
            raise ValueError("kit extraction ceiling")
        tar.extractall(args.root, filter="data")
    kit = args.root / ("telegraph-wasm-baseline-" + spec["revision"])
    programs = {"executor": ["./cmd/adapterexec"], "checker": [str(HERE / "packagecheck.go")], "input-root": [str(HERE / "input_root.go")]}
    for name, source in programs.items():
        subprocess.run(["go", "build", "-trimpath", "-o", str(args.root / name), *source], cwd=kit, check=True, timeout=600)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    row_views = views(splits["test"])
    vectors = {}
    for name, rows in row_views.items():
        vectors[name] = [tokenizer.encode(render(r["prompt"], r[side])).ids for r in rows for side in ("chosen", "rejected")]
        if any(not ids or len(ids) > MAX_TOKENS for ids in vectors[name]):
            raise ValueError("complete audit format exceeds literal token cap")
    frozen = json.loads((args.bundle / "frozen-baselines.json").read_text())
    format_shortcuts = {}
    for name, rows in row_views.items():
        shortcuts = shortcut_report(rows, frozen)
        lexical_wins = [0.5 if score(r["source"]["text"], r["chosen"])["score_milli"] == score(r["source"]["text"], r["rejected"])["score_milli"] else
                       float(score(r["source"]["text"], r["chosen"])["score_milli"] > score(r["source"]["text"], r["rejected"])["score_milli"]) for r in rows]
        shortcuts["fixed_lexical_diagnostic"] = sum(lexical_wins) / len(lexical_wins)
        format_shortcuts[name] = shortcuts
    shortcuts = {name + ":" + key: value for name, metrics in format_shortcuts.items() for key, value in metrics.items()}
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "FOUNDATION_MODEL_PATH": str(foundation / "foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": standard.FOUNDATION["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(foundation / "foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": standard.FOUNDATION["foundation_engine.wasm"]}
    raw_scores, measured, failure_controls, executable = {}, {}, {}, args.root / "executor"
    token_control = {}
    started = time.monotonic()
    for label, package_name in (("candidate", "KEYWORD_EXTRACTION.safetensors"), ("initial", "KEYWORD_EXTRACTION_INITIAL.safetensors")):
        package = args.bundle / package_name
        url = "https://devnode.telegraphprotocol.com/engine/v1/intents/KEYWORD_EXTRACTION_GENERAL"
        family_url = "https://devnode.telegraphprotocol.com/api/wasm?resolve=KEYWORD_EXTRACTION"
        write(args.root / "class-snapshot.json", {"at": datetime.now(timezone.utc).isoformat(), "url": url,
              "child": json.loads(source_fetch(url, 2 << 20)), "family_url": family_url,
              "parent": json.loads(source_fetch(family_url, 2 << 20))})
        checked = subprocess.run([str(args.root / "checker"), "--snapshot", str(args.root / "class-snapshot.json"), "--package", str(package),
                                  "--foundation", str(args.bundle / "foundation.json"), "--out", str(args.root)], capture_output=True, text=True, check=True, timeout=60)
        manifest = json.loads(checked.stdout)
        write(args.root / (label + "-manifest.json"), manifest)
        expected = {"foundation_hash": standard.FOUNDATION["foundation_lora_fp32.onnx"], "adapter_hash": manifest["runtime_adapter_hash"],
                    "tokenizer_hash": standard.FOUNDATION["tokenizer.json"]}
        streams = []
        # A fresh executor process repeats exactly the same fixed audit inputs.
        for repetition in range(2):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            with (args.root / f"{label}-{repetition}.log").open("w") as log:
                server = subprocess.Popen([str(executable), "--package", str(package), "--listen", f"127.0.0.1:{port}"], env=env, stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 180
                    while True:
                        if server.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError("primary runtime readiness failed")
                        try:
                            with socket.create_connection(("127.0.0.1", port), timeout=1):
                                break
                        except OSError:
                            time.sleep(1)
                    def exchange(batch, identity, port=port):
                        request = urllib.request.Request(f"http://127.0.0.1:{port}/runnable", data=json.dumps({"batch": [{"tokens": ids} for ids in batch], **identity}).encode(),
                                                         headers={"Content-Type": "application/json"})
                        with urllib.request.urlopen(request, timeout=180) as response:
                            raw = response.read((1 << 20) + 1)
                            if response.status != 200 or len(raw) > 1 << 20:
                                raise ValueError("runtime response ceiling or status")
                        return json.loads(raw)
                    def score_batch(batch, expected=expected, label=label, repetition=repetition):
                        root = subprocess.run([str(args.root / "input-root")], input=json.dumps(batch), capture_output=True, text=True, check=True, timeout=15).stdout.strip()
                        identity = {**expected, "input_root": root}
                        reply = exchange(batch, identity)
                        bits = check_response(reply, identity, len(batch))
                        if label == "candidate" and repetition == 0 and not failure_controls:
                            failure_controls.update(identity_controls(exchange, batch, identity, bits))
                        return bits
                    if label == "candidate" and repetition == 0:
                        refused = exchange([[-1]], expected)
                        failed_rows = refused.get("results")
                        if not isinstance(failed_rows, list) or len(failed_rows) != 1 or not isinstance(failed_rows[0], dict) or "score" in failed_rows[0] or not failed_rows[0].get("error"):
                            raise ValueError("actual executor accepted negative tokens")
                        for field, value in expected.items():
                            if refused.get(field, "").casefold() != value.casefold():
                                raise ValueError("token refusal came from a different actual identity")
                        token_control = {"negative_tokens_refused": True, "actual_reply": refused}
                    stream = {}
                    for name in FORMATS:
                        bits = []
                        for start in range(0, len(vectors[name]), 8):
                            bits.extend(score_batch(vectors[name][start:start + 8]))
                        stream[name] = bits
                    streams.append(stream)
                finally:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=10)
        if streams[0] != streams[1]:
            raise ValueError("fresh-process FP32 bits differ")
        raw_scores[label] = streams[0]
        measured[label] = {name: metrics(row_views[name], streams[0][name]) for name in FORMATS}
        write(args.root / "audit-score-bits.json", raw_scores)
    result = verdict(measured["candidate"], measured["initial"], shortcuts, config)
    result_report = {"family": "KEYWORD_EXTRACTION", **result, "candidate": candidate,
          "formats": measured, "frozen_shortcuts": shortcuts, "failure_controls": failure_controls,
          "format_shortcuts": format_shortcuts,
          "runtime": {"reference_engine_passed": True, "repeat_bits_identical": True, "token_control": token_control},
          "normalized_overlap": data_report["normalized_cross_split_overlap"],
          "fixed_vector_bits": raw_scores["candidate"]["array"][:8],
          "audit_run_id": os.environ["GITHUB_RUN_ID"], "audit_commit": os.environ["GITHUB_SHA"], "node_url": "https://devnode.telegraphprotocol.com",
          "foundation": standard.FOUNDATION, "executor_sha256": sha(executable.read_bytes()), "kit_revision": spec["revision"],
          "live_class_snapshot_sha256": sha((args.root / "class-snapshot.json").read_bytes()),
          "token_vectors_sha256": sha(json.dumps(vectors, sort_keys=True).encode()), "score_bits_sha256": sha((args.root / "audit-score-bits.json").read_bytes()),
          "versions": {"python": sys.version, "tokenizers": importlib.metadata.version("tokenizers")},
          "elapsed_seconds": time.monotonic() - started, "source_bound_quality_only": True, "node_qualification": "not attempted"}
    write(args.root / "audit-result.json", result_report)
    if result["verdict"] == "PASS":
        assets = {name: sha((foundation / name).read_bytes()) for name in standard.FOUNDATION}
        ready = registration_report(result_report, json.loads((args.root / "candidate-manifest.json").read_text()),
                                    sha((args.data / "approved-manifest.json").read_bytes()), source_verification,
                                    vectors["array"][:8], assets)
        write(args.root / "registration-ready.json", ready)
    print(json.dumps(result))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
