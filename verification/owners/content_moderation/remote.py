# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Frozen H9 fitting and actual reference-engine audit. Never executed locally."""
import argparse
import gc
import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from baselines import fit as fit_baselines, scores as shortcut_scores
from contracts import CHILD, EXPERIMENT, FAMILY, FORMATS, FOUNDATION, HERE, bits, check_reply, eligibility, identity, input_root, remote_only, sha, write


def support():
    sys.path.insert(0, str(HERE / "support" / "moderation-diagnostic"))
    return {name: importlib.import_module(name) for name in ("core", "model_ops", "stages", "fit", "freeze_data", "matched", "preserve_all")}


def read(path, ceiling=2 << 20):
    raw = path.read_bytes()
    if len(raw) > ceiling:
        raise ValueError("bounded JSON input required")
    return json.loads(raw)


def capture_class(root):
    from contracts import check_class
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    value = {"schema": 1, "fetched_at": datetime.now(timezone.utc).isoformat()}
    for name, url in (("family", "https://devnode.telegraphprotocol.com/api/wasm?resolve=" + FAMILY),
                      ("child", "https://devnode.telegraphprotocol.com/engine/v1/intents/" + CHILD)):
        request = urllib.request.Request(url, headers={"User-Agent": "CONTENT_MODERATION-owner/1.0", "Cache-Control": "no-cache"})
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=25) as response:
            raw = response.read((1 << 20) + 1)
            if response.status != 200 or len(raw) > 1 << 20:
                raise ValueError("class source status/size refused")
        value[name] = {"url": url, "raw": raw.decode()}
    check_class(value)
    write(root / "evidence/class-snapshot.json", value)
    return value


def compile_tools(root):
    kit = root / ("telegraph-wasm-baseline-1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd")
    files = HERE / "support/registration"
    subprocess.run(["go", "test", "-mod=readonly", "-count=1", str(files / "stagecheck.go"), str(files / "stagecheck_test.go")], cwd=kit, check=True, timeout=180)
    for name, source in (("stagecheck", str(files / "stagecheck.go")), ("adapterexec", "./cmd/adapterexec")):
        subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / name), source], cwd=kit, check=True, timeout=600)


def execute_phase(root, name, argv, maximum, deadline):
    remaining = deadline - time.monotonic() - 120
    if remaining <= 0:
        raise TimeoutError("preservation reserve exhausted")
    write(root / "evidence/phase.json", {"phase": name, "state": "running", "cap_seconds": min(maximum, remaining)})
    with (root / "evidence" / (name + ".log")).open("wb") as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=min(maximum, remaining))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=10)
            raise TimeoutError("phase timed out: " + name) from None
    write(root / "evidence/phase.json", {"phase": name, "exit_code": code, "state": "complete" if code == 0 else "failed"})
    if code:
        raise RuntimeError("phase failed: " + name)


def prepare(root):
    remote_only()
    m = support()
    _, original, _ = m["core"].load_shared()
    licences = read(HERE / "support/training/licences.json")
    if original.SOURCES[FAMILY]["licence"] != "CC0-1.0" or "Civil Comments" not in licences["notice"] or "Permission is hereby granted" not in licences["kit_mit"] or "Grant of Copyright License" not in licences["apache_2_0"]:
        raise ValueError("retained primary source grants missing")
    original.prepare(root)
    compile_tools(root)
    capture_class(root)  # Before any foundation model fitting.
    (root / "originals").mkdir()
    extra = json.loads(os.environ.get("CAMPAIGN_EXCLUSIONS_JSON", "[]"))
    source = m["freeze_data"].freeze(root, extra)
    splits = m["model_ops"].read_development(root)
    coefficients = fit_baselines(splits["train"])
    write(root / "evidence/baseline-coefficients.json", coefficients)
    if max(shortcut_scores(splits["train"], coefficients).values()) > 0.65:
        raise ValueError("training strongest-shortcut source gate failed")
    write(root / "evidence/source-manifest.json", {"sources": source, "coefficient_sha256": sha((root / "evidence/baseline-coefficients.json").read_bytes()),
          "scope": EXPERIMENT["scope"], "fitted_preferences": "Programmatically constructed from original public annotations. Not human-reviewed preferences.",
          "support_sha256": read(HERE / "SUPPORT.json")})


def select(root):
    m = support()
    m["core"].deny_measurement_reads(root)
    if read(root / "evidence/dataset-gate.json")["ok"] is not True or read(root / "evidence/H9-dataset-gate.json")["ok"] is not True:
        raise ValueError("mandatory upstream data gates missing")
    match = m["matched"].matched_report(root)
    restore = read(root / "evidence/H9-restoration.json")
    if not restore["parameter_bytes_equal"] or not restore["validation_replay_equal"]:
        raise ValueError("snapshot restoration evidence missing")
    coefficients = read(root / "evidence/baseline-coefficients.json")
    source = read(root / "evidence/source-manifest.json")
    if sha((root / "evidence/baseline-coefficients.json").read_bytes()) != source["coefficient_sha256"]:
        raise ValueError("frozen train-fitted baseline changed")
    validation = m["model_ops"].views(m["model_ops"].read_development(root)["validation"])
    shortcuts = {name: max(shortcut_scores(rows, coefficients).values()) for name, rows in validation.items()}
    untrained = read(root / "evidence/untrained-validation.json")
    records = read(root / "evidence/H9-validation-history.json")
    if not records or len(records) > 9:
        raise ValueError("bounded actual H9 trajectory required")
    transcript = [{"epoch": r["epoch"], "failures": eligibility(r["formats"], untrained["formats"], shortcuts), "key": m["core"].rank_key(r)} for r in records]
    eligible = [r for r in records if not eligibility(r["formats"], untrained["formats"], shortcuts)]
    write(root / "evidence/selection.json", {"transcript": transcript, "shortcuts": shortcuts, "matching": match, "selection_uses_audit": False})
    if not eligible:
        write(root / "evidence/result.json", {"status": "NO_VALIDATION_CANDIDATE", "audit_opened": False, "historical_comparison_valid": False})
        raise RuntimeError("no H9 validation-eligible snapshot; audit remains sealed")
    candidate = min(eligible, key=m["core"].rank_key)
    for row in (candidate, untrained):
        path = root / "evidence" / (row["snapshot"] + ".safetensors")
        if sha(path.read_bytes()) != row["checkpoint_sha256"]:
            raise ValueError("frozen checkpoint identity changed")
    write(root / "evidence/candidate-freeze.json", {"candidate": candidate, "untrained": untrained,
          "source_manifest_sha256": sha((root / "evidence/source-manifest.json").read_bytes()),
          "selection_sha256": sha((root / "evidence/selection.json").read_bytes()),
          "experiment_sha256": sha((HERE / "experiment.json").read_bytes())})


def export_package(root, name, record):
    remote_only()
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, target_dims, tensor_names, write_package
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(root / "base", local_files_only=True)
    directory = root / "adapters" / record["snapshot"]
    if sha((directory / "adapter_model.safetensors").read_bytes()) != record["adapter_sha256"] or sha((directory / "adapter_config.json").read_bytes()) != record["config_sha256"]:
        raise ValueError("selected adapter/config changed")
    tensors, parameters = read_peft(directory, cfg)
    training = {"family": FAMILY, "children": [{"intent": CHILD,
                "dataset_version": sha((root / "evidence/source-manifest.json").read_bytes()),
                "pairs": 128, "accuracy": record["formats"]["json"]["accuracy"], "held_out": False}],
                "selection_scope": "Validation used for selection, not independent audit. Final reference-engine audit is reported separately.",
                "source_scope": EXPERIMENT["scope"], "preference_authorship": "AI-assisted code constructs preferences from original public source labels."}
    meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=FOUNDATION["foundation_lora_fp32.onnx"],
                    foundation_version="modernbert-base-lora16-v1", rank=parameters["rank"], alpha=parameters["alpha"],
                    target_modules=parameters["target_modules"], max_tokens=256, training=training)
    licences = read(HERE / "support/training/licences.json")
    for key, original in (("x-upstream-notice", "notice"), ("x-upstream-apache-2.0", "apache_2_0"), ("x-upstream-mit", "kit_mit"), ("x-licence", "sand")):
        meta[key] = licences[original]
    package = root / "evidence" / name
    write_package(package, meta, {n: tensors[n] for n in tensor_names(cfg.num_hidden_layers)})
    spec = {"foundation_hash": meta["foundation_hash"], "foundation_version": meta["foundation_version"],
            "tokenizer_hash": meta["tokenizer_hash"], "base_model": str(root / "base"), "r_max": 16,
            "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
            "targets": {k: list(v) for k, v in target_dims(cfg).items()}}
    write(root / "evidence/foundation.json", spec)
    capture_class(root)
    command = [str(root / "stagecheck"), "--intent", FAMILY, "--child", CHILD, "--expected-class", "judgment",
               "--package", str(package), "--sha256", sha(package.read_bytes()),
               "--foundation", str(root / "evidence/foundation.json"), "--foundation-sha256", sha((root / "evidence/foundation.json").read_bytes()),
               "--class-snapshot", str(root / "evidence/class-snapshot.json"), "--snapshot-sha256", sha((root / "evidence/class-snapshot.json").read_bytes()), "--out", str(root / "evidence")]
    checked = subprocess.run(command, capture_output=True, timeout=60, check=False)
    (root / "evidence" / (name + ".parser.stdout")).write_bytes(checked.stdout[:1 << 20])
    (root / "evidence" / (name + ".parser.stderr")).write_bytes(checked.stderr[:1 << 20])
    if checked.returncode or len(checked.stdout) > 1 << 20 or len(checked.stderr) > 1 << 20:
        raise ValueError("full package parser refused the exact package")
    manifest = json.loads(checked.stdout)
    if manifest["sha256"] != sha(package.read_bytes()) or manifest["manifest"]["verification"] != {"stage1": True, "loaded": False}:
        raise ValueError("parser result identity mismatch")
    write(root / "evidence" / (name + ".parser.json"), manifest)
    return package, manifest


def engine_audit(root, package, manifest, vectors):
    remote_only()
    sys.path.insert(0, str(HERE / "support/registration"))
    proxy = importlib.import_module("proxy")
    row = {"keccak": manifest["keccak256"].removeprefix("0x")}
    with socket.socket() as port_probe:
        port_probe.bind(("127.0.0.1", 0))
        port = port_probe.getsockname()[1]
    foundation = root / "foundation"
    env = {"PATH": os.environ["PATH"], "HOME": str(root), "FOUNDATION_POOL_SIZE": "1",
           "FOUNDATION_MODEL_PATH": str(foundation / "foundation_lora_fp32.onnx"), "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(foundation / "foundation_engine.wasm"), "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"]}
    from transport import RuntimeBridge
    outputs, fixed = {}, []
    with (root / "evidence" / (package.name + ".runtime.log")).open("wb") as log:
        server = subprocess.Popen([str(root / "adapterexec"), "--package", str(package), "--listen", "127.0.0.1:" + str(port)], env=env, stdout=log, stderr=log)
        try:
            end = time.monotonic() + 180
            while time.monotonic() < end:
                if server.poll() is not None:
                    raise RuntimeError("reference engine exited during loading")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("reference engine readiness ceiling")
            bridge = RuntimeBridge(proxy.Backend(port), row["keccak"])
            for name, tokens in vectors.items():
                streams = []
                for _ in range(2):
                    stream = []
                    for start in range(0, len(tokens), 8):
                        if (root / "evidence" / (package.name + ".runtime.log")).stat().st_size > 16 << 20:
                            raise ValueError("reference runtime log ceiling exceeded")
                        request = {"batch": [{"tokens": ids} for ids in tokens[start:start + 8]], **identity(row["keccak"])}
                        request["input_root"] = input_root(request["batch"])
                        stream.extend(check_reply(bridge.score(request), request, row["keccak"]))
                    streams.append(stream)
                if streams[0] != streams[1]:
                    raise ValueError("actual reference-engine score bits did not repeat")
                outputs[name] = streams[0]
                write(root / "evidence" / (package.name + ".score-bits.json"), outputs)
            if package.name == FAMILY + ".safetensors":
                from transport import live_controls
                fixed, negative = live_controls(bridge, row["keccak"])
            else:
                negative = 0
            return outputs, fixed, negative
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)


def qualify(root):
    remote_only()
    m = support()
    frozen = read(root / "evidence/candidate-freeze.json")
    if frozen["experiment_sha256"] != sha((HERE / "experiment.json").read_bytes()) or frozen["source_manifest_sha256"] != sha((root / "evidence/source-manifest.json").read_bytes()) or frozen["selection_sha256"] != sha((root / "evidence/selection.json").read_bytes()):
        raise ValueError("frozen experiment/source/selection changed")
    m["matched"].matched_report(root)
    torch, np, trainer, hp = m["model_ops"].runtime(root, 9)
    tok, node = m["model_ops"].tokenizer(root)
    m["model_ops"].seed(torch, np, hp)
    model, tok = trainer.build_model(str(root / "base"), hp)
    m["model_ops"].reload_snapshot(root, frozen["candidate"]["snapshot"], model)
    with m["core"].isolated_rng(torch, np):
        validation = m["model_ops"].score_views(trainer, model, tok, m["model_ops"].views(m["model_ops"].read_development(root)["validation"]), hp)
    if validation != frozen["candidate"]["formats"]:
        raise ValueError("selected validation margins did not replay")
    del model
    gc.collect()
    trained_package, trained_manifest = export_package(root, FAMILY + ".safetensors", frozen["candidate"])
    untrained_package, untrained_manifest = export_package(root, "UNTRAINED.safetensors", frozen["untrained"])
    with (root / "evidence/audit-access.json").open("x") as marker:
        marker.write(json.dumps({"frozen_candidate_sha256": frozen["candidate"]["checkpoint_sha256"], "measurement": "One final joint reference-engine audit, no audit-driven selection"}) + "\n")
    raw = (root / "audit-private/audit.jsonl").read_bytes()
    if sha(raw) != EXPERIMENT["audit_sha256"]:
        raise ValueError("sealed audit changed")
    data, _, _ = m["core"].load_shared()
    originals = data.decode_jsonl(raw)
    if len(originals) != 256 or len({r["group_id"] for r in originals}) != 256:
        raise ValueError("independent original audit count changed")
    splits = m["model_ops"].read_development(root)
    data.check_splits({**splits, "audit": originals})
    views = m["model_ops"].views(originals)
    m["model_ops"].tokenize(trainer, tok, node, list(views.values()), root / "evidence/audit-tokenizer-check.json")
    coefficients = read(root / "evidence/baseline-coefficients.json")
    source = read(root / "evidence/source-manifest.json")
    if sha((root / "evidence/baseline-coefficients.json").read_bytes()) != source["coefficient_sha256"]:
        raise ValueError("train-fitted coefficients changed")
    shortcuts = {name: max(shortcut_scores(rows, coefficients).values()) for name, rows in views.items()}
    token_vectors = {name: [node.encode(trainer.render(r["prompt"], r[side])).ids for r in rows for side in ("chosen", "rejected")] for name, rows in views.items()}
    trained_bits, vectors, negative = engine_audit(root, trained_package, trained_manifest, token_vectors)
    untrained_bits, _, _ = engine_audit(root, untrained_package, untrained_manifest, token_vectors)
    def metrics(streams):
        return {name: m["core"].summary(rows, [bits(streams[name][2*i]) - bits(streams[name][2*i+1]) for i in range(len(rows))]) for name, rows in views.items()}
    trained, initial = metrics(trained_bits), metrics(untrained_bits)
    failures = eligibility(trained, initial, shortcuts)
    report = {"schema_version": 1, "status": "REGISTRATION_READY" if not failures else "AUDIT_FAILED", "family": FAMILY,
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "serving_commit": os.environ["GITHUB_SHA"],
              "package_sha256": sha(trained_package.read_bytes()), "package_keccak": trained_manifest["keccak256"].removeprefix("0x"),
              "package_bytes": trained_package.stat().st_size, "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"],
              "engine_sha256": FOUNDATION["foundation_engine.wasm"], "tokenizer_sha256": FOUNDATION["tokenizer.json"],
              "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                         "source_manifest_sha256": frozen["source_manifest_sha256"], "independent_groups": len(originals), "scope": EXPERIMENT["scope"]},
              "quality": {"passed": not failures, "audit_pairs": len(originals), "all_supported_formats_tested": True,
                          "formats": {name: {"accuracy": trained[name]["accuracy"], "per_label": {label: cell["accuracy"] for label, cell in trained[name]["by_label"].items()},
                                             "untrained_accuracy": initial[name]["accuracy"], "shortcut_accuracy": shortcuts[name]} for name in FORMATS}},
              "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                          "negative_identity_controls": negative, "vectors": vectors}, "failures": failures,
              "evidence": {"selected_checkpoint": frozen["candidate"], "historical_comparison_valid": False}}
    write(root / "evidence/registration-ready.json", report)
    if failures:
        raise ValueError("frozen actual-reference audit failed; no audit tuning permitted")


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("run", "prepare", "select", "qualify"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.phase != "run":
        {"prepare": prepare, "select": select, "qualify": qualify}[args.phase](root)
        return
    if root.exists():
        raise ValueError("fresh owner run directory required; no audit rerun")
    (root / "evidence").mkdir(parents=True)
    write(root / "evidence/run-binding.json", {"family": FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
          "experiment_sha256": sha((HERE / "experiment.json").read_bytes()), "support": read(HERE / "SUPPORT.json")})
    deadline = time.monotonic() + 17400
    try:
        command = [sys.executable, str(HERE / "remote.py")]
        execute_phase(root, "prepare", [*command, "prepare", "--root", str(root)], 1800, deadline)
        path = HERE / "support/moderation-diagnostic"
        execute_phase(root, "initial", [sys.executable, str(path / "stages.py"), "initial", "--root", str(root)], 1200, deadline)
        execute_phase(root, "fit-H9", [sys.executable, str(path / "fit.py"), "--arm", "H9", "--root", str(root)], 12000, deadline)
        execute_phase(root, "select", [*command, "select", "--root", str(root)], 120, deadline)
        execute_phase(root, "qualify", [*command, "qualify", "--root", str(root)], 5400, deadline)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        write(root / "evidence/operational-failure.json", {"status": "FAILED", "error_type": type(error).__name__, "release_gate": "INCOMPLETE"})
        raise
    finally:
        support()["preserve_all"].preserve(root)


if __name__ == "__main__":
    main()
