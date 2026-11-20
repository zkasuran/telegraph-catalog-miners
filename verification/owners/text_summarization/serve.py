# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only exact reference executor with a restricted public lease."""
import argparse
import json
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from common import (
    DEFAULT_OWNER,
    HOST,
    HOST_KEY,
    immutable_url,
    lease_for,
    validate_lease,
)
from data import FAMILY, fetch, write_json
from licensing import verify_checkpoint_metadata
from proxy import Backend, BoundedServer, identity, input_root
from reference import RELEASE, prepare_kit
from report import validate_ready
from runtime_check import (
    RELEASE_HASHES,
    compile_tools,
    model_remote_only,
    validate_package,
)
from wire import validate_response


def parse_settings(value, commit):
    required = {"package_url", "package_sha256", "package_keccak", "report_url", "report_sha256", "slot", "forward_port", "runnable_url"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("closed serving settings required")
    for name in ("package_url", "report_url"):
        immutable_url(value[name])
    for name in ("package_sha256", "package_keccak", "report_sha256"):
        if not isinstance(value[name], str) or not re.fullmatch(r"[a-f0-9]{64}", value[name]):
            raise ValueError("literal serving package and report hashes required")
    if type(value["slot"]) is not int or not 1 <= value["slot"] <= 6 or value["forward_port"] != 48600 + value["slot"]:
        raise ValueError("controller-reserved free endpoint required")
    if not isinstance(commit, str) or not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("serving source commit required")
    if not isinstance(value["runnable_url"], str) or not re.fullmatch(r"https://tga1\.zkasuran\.dev/_[A-Za-z0-9_-]{1,64}/runnable", value["runnable_url"]):
        raise ValueError("unique controller-reserved existing proxy route required")
    return value


def setup(root, settings):
    model_remote_only()
    root.mkdir(parents=True, exist_ok=True)
    settings = parse_settings(settings, os.environ["GITHUB_SHA"])
    report_path = root / "registration-ready.json"
    fetch(settings["report_url"], report_path, settings["report_sha256"], 256 << 10)
    report = validate_ready(json.loads(report_path.read_text()))
    if (report["serving_commit"] != os.environ["GITHUB_SHA"] or report["package_sha256"] != settings["package_sha256"]
            or report["package_keccak"] != settings["package_keccak"]):
        raise ValueError("serving commit and audited candidate bytes mismatch")
    package = root / (FAMILY + ".safetensors")
    fetch(settings["package_url"], package, settings["package_sha256"], 64 << 20)
    verify_checkpoint_metadata(package)
    kit = prepare_kit(root)
    compile_tools(root, kit)
    foundation = root / "foundation"
    foundation.mkdir(exist_ok=True)
    for name, expected in RELEASE_HASHES.items():
        fetch(RELEASE + name, foundation / name, expected)
    # Header/layout for the full parser comes from the exact pinned foundation
    # config, never from an expected template or an altered candidate.
    pins = json.loads((Path(__file__).parent / "reference-pins.json").read_text())["base"]
    config_path = root / "config.json"
    fetch(f"https://huggingface.co/{pins['id']}/resolve/{pins['revision']}/config.json", config_path, pins["files"]["config.json"], 1 << 20)
    config = json.loads(config_path.read_text())
    from float_runtime.lora_format import target_dims
    cfg = SimpleNamespace(**config)
    write_json(root / "foundation.json", {"foundation_hash": RELEASE_HASHES["foundation_lora_fp32.onnx"],
               "foundation_version": "modernbert-base-lora16-v1", "tokenizer_hash": RELEASE_HASHES["tokenizer.json"],
               "r_max": 16, "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
               "targets": {k: list(v) for k, v in target_dims(cfg).items()}})
    checked = validate_package(root)
    if checked["keccak256"].removeprefix("0x") != report["package_keccak"]:
        raise ValueError("actual served package Keccak differs from the audit")
    (root / "known_hosts").write_text(HOST_KEY)
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "summary-forward", "-f", str(root / "forward-key")],
                   check=True, timeout=15)
    row = {"family": FAMILY, "owner": DEFAULT_OWNER, "keccak": report["package_keccak"], "sha256": report["package_sha256"],
           "slot": settings["slot"], "forward_port": settings["forward_port"], "runnable_url": settings["runnable_url"],
           "existing_registration_id": 0,
           **{name + "_proof": {"sha256": settings["report_sha256"]} for name in ("candidate", "runtime", "source")}}
    lease = lease_for(row, (root / "forward-key.pub").read_text().strip(), "summary-" + os.environ["GITHUB_RUN_ID"],
                      os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
    validate_lease(lease, row)
    write_json(root / "row.json", row)
    write_json(root / "lease.json", lease)
    write_json(root / "serve-report.json", report)
    return row, report


def run(root):
    model_remote_only()
    row = json.loads((root / "row.json").read_text())
    report = validate_ready(json.loads((root / "serve-report.json").read_text()))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        executor_port = probe.getsockname()[1]
    env = {**os.environ, "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"),
           "FOUNDATION_MODEL_HASH": RELEASE_HASHES["foundation_lora_fp32.onnx"],
           "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"),
           "FOUNDATION_ENGINE_WASM_HASH": RELEASE_HASHES["foundation_engine.wasm"], "ADAPTER_SCORE_TIMEOUT": "180s"}
    processes = []
    server = None
    with (root / "serve.log").open("wb") as log:
        try:
            child = subprocess.Popen([str(root / "adapterexec"), "--package", str(root / (FAMILY + ".safetensors")),
                                      "--listen", f"127.0.0.1:{executor_port}"], env=env, stdout=log, stderr=log)
            processes.append(child)
            backend = Backend(executor_port, timeout=180)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise ValueError("serving reference executor failed")
                try:
                    with socket.create_connection(("127.0.0.1", executor_port), timeout=1):
                        break
                except OSError:
                    time.sleep(.5)
            else:
                raise TimeoutError("serving executor startup ceiling")
            for vector in report["runtime"]["vectors"]:
                batch = [{"tokens": vector["tokens"]}]
                request = {"batch": batch, **identity(row), "input_root": input_root(batch)}
                if validate_response(backend.score(request), request) != [vector["score_bits"]]:
                    raise ValueError("served reference bits differ from the audited package")
            server = BoundedServer(("127.0.0.1", 0), row, backend)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            arguments = ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(root / "forward-key"),
                         "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "BatchMode=yes",
                         "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes", "-o", "GlobalKnownHostsFile=/dev/null",
                         "-o", "UserKnownHostsFile=" + str(root / "known_hosts"), "-o", "ExitOnForwardFailure=yes",
                         "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no",
                         "-o", "ForwardX11=no", "-o", "PermitLocalCommand=no", "-o", "RequestTTY=no",
                         "-R", f"127.0.0.1:{row['forward_port']}:127.0.0.1:{server.server_port}", "telegraph-verify@" + HOST]
            tunnel = subprocess.Popen(arguments, stdout=log, stderr=log)
            processes.append(tunnel)
            # The adapter activates the uploaded public lease. A denied initial
            # tunnel is retried after activation without generating a new key.
            expires = time.monotonic() + 5 * 3600 - 60
            while time.monotonic() < expires:
                if child.poll() is not None or server.fatal.is_set() or log.tell() > 16 << 20:
                    raise ValueError("serving runtime stopped or diagnostic ceiling reached")
                if tunnel.poll() is not None:
                    time.sleep(10)
                    tunnel = subprocess.Popen(arguments, stdout=log, stderr=log)
                    processes.append(tunnel)
                time.sleep(1)
        finally:
            if server is not None:
                server.shutdown()
                server.server_close()
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)


if __name__ == "__main__":
    model_remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--settings")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.root.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("serving storage must be inside the ephemeral runner")
    if args.run:
        run(args.root)
    else:
        setup(args.root, json.loads(args.settings))
