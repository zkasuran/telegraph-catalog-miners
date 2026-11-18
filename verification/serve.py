#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Expose the reference executor only for a bounded registration test."""
import datetime as dt
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

OWNER = "0x8b224783fe5b3c52b7db0cb9b1754f8812b75287"
NODE = "https://devnode.telegraphprotocol.com"

def read(url, body=None, timeout=30):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), headers={"User-Agent": "Telegraph-evaluator-integration/1.0", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read((1 << 20) + 1)
        if len(raw) > 1 << 20:
            raise RuntimeError("response exceeds the byte ceiling")
        return json.loads(raw)

def find_decision(entries, row, start):
    for entry in entries:
        if entry.get("author_address", "").lower() != OWNER or entry.get("wasm_hash", "").removeprefix("0x") != row["keccak"]:
            continue
        registered = dt.datetime.fromisoformat(entry["registered_at"].replace("Z", "+00:00"))
        if registered < start:
            continue
        if entry.get("activation_status") in ("active", "rejected", "superseded", "deregistered"):
            return {"family": row["family"], "status": entry["activation_status"], "registration_id": entry["registration_id"], "adapter_hash": row["keccak"]}
    return None

def main():
    root = Path(os.environ["RUNNER_TEMP"]) / "telegraph-evaluator"
    state = json.loads((root / "state.json").read_text())
    lease = json.loads((root / "lease.json").read_text())
    row = state["row"]
    environment = dict(os.environ, FOUNDATION_MODEL_PATH=str(Path(state["model"]) / "foundation_lora_fp32.onnx"),
        FOUNDATION_MODEL_HASH="0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
        FOUNDATION_ENGINE_WASM_PATH=str(Path(state["model"]) / "foundation_engine.wasm"),
        FOUNDATION_ENGINE_WASM_HASH="77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de", FOUNDATION_POOL_SIZE="1")
    start = dt.datetime.now(dt.timezone.utc)
    deadline = time.monotonic() + 5 * 3600
    executor = None
    original_executor = None
    forward = None
    verdict = {"family": row["family"], "status": "not_started"}
    with (root / "executor.log").open("w") as executor_log, (root / "forward.log").open("w") as forward_log:
        try:
            executor = subprocess.Popen([state["executable"], "--package", state["package"], "--listen", "127.0.0.1:8800"], env=environment, stdout=executor_log, stderr=executor_log)
            original_executor = subprocess.Popen([state["executable"], "--package", state["original_package"], "--listen", "127.0.0.1:8810"], env=environment, stdout=executor_log, stderr=executor_log)
            probe = {"batch": [], "foundation_hash": environment["FOUNDATION_MODEL_HASH"], "adapter_hash": row["keccak"], "tokenizer_hash": "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"}
            for _ in range(90):
                if executor.poll() is not None:
                    raise RuntimeError("reference executor exited while loading")
                try:
                    result = read("http://127.0.0.1:8800/runnable", probe)
                    if result.get("adapter_hash") != row["keccak"]:
                        raise RuntimeError("loaded adapter hash mismatch")
                    break
                except (OSError, ValueError):
                    time.sleep(2)
            else:
                raise RuntimeError("reference executor did not become ready")
            original_probe={**probe,"adapter_hash":row["origin_keccak"]}
            for _ in range(90):
                if original_executor.poll() is not None:
                    raise RuntimeError("original reference executor exited")
                try:
                    read("http://127.0.0.1:8810/runnable",original_probe)
                    break
                except (OSError,ValueError):
                    time.sleep(2)
            else:
                raise RuntimeError("original reference executor did not become ready")
            vectors=[{"tokens":[50281]+[22+(i%100) for i in range(n-2)]+[50282]} for n in (4,16,64,128)]
            old_scores=read("http://127.0.0.1:8810/runnable",{**original_probe,"batch":vectors},timeout=1200)
            new_scores=read("http://127.0.0.1:8800/runnable",{**probe,"batch":vectors},timeout=1200)
            result_list=new_scores.get("results",[])
            if len(result_list)!=4 or old_scores.get("results")!=result_list or any(type(r.get("score")) is not int or not 0<=r["score"]<(1<<32) or r["score"]&0x7f800000==0x7f800000 for r in result_list):
                raise RuntimeError("metadata recovery did not reproduce original score bits")
            (root/"reproduction.json").write_text(json.dumps({"family":row["family"],"vectors":4,"score_bits_identical":True,"tensor_payload_sha256":row["tensor_payload_sha256"]}))
            original_executor.terminate()
            original_executor.wait(timeout=15)
            print("Original score bits reproduced on four fixed vectors",flush=True)
            lease_deadline = time.monotonic() + 600
            command = ["ssh", "-N", "-i", state["private_key"], "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes",
                       "-o", "UserKnownHostsFile=" + state["known_hosts"], "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
                       "-R", f"127.0.0.1:{row['forward_port']}:127.0.0.1:8800", "ubuntu@98.95.110.88"]
            while time.monotonic() < lease_deadline:
                forward = subprocess.Popen(command, stdout=forward_log, stderr=forward_log)
                time.sleep(5)
                if forward.poll() is None:
                    break
                time.sleep(10)
            else:
                raise RuntimeError("restricted remote-forward lease was not authorized")
            for _ in range(30):
                try:
                    public = read(lease["runnable_url"], probe, timeout=60)
                    break
                except (OSError, ValueError):
                    time.sleep(5)
            else:
                raise RuntimeError("public endpoint did not become ready")
            if any(public.get(k) != probe[k] for k in ("foundation_hash", "adapter_hash", "tokenizer_hash")):
                raise RuntimeError("public executor identity mismatch")
            print("PUBLIC_EXECUTOR_READY " + row["family"], flush=True)
            while time.monotonic() < deadline:
                if executor.poll() is not None or forward.poll() is not None:
                    raise RuntimeError("reference executor or restricted forward stopped")
                try:
                    listing = read(NODE + "/api/wasm?intent=" + row["family"] + "&poll=" + str(time.time_ns()))
                    entries = (listing.get("intents", {}).get(row["family"]) or {}).get("entries", [])
                    decided = find_decision(entries, row, start)
                    if decided:
                        verdict = decided
                        print("NODE_DECISION " + json.dumps(verdict), flush=True)
                        if verdict["status"] != "active":
                            raise RuntimeError("node did not activate the candidate: " + verdict["status"])
                        return
                except (OSError, ValueError, KeyError):
                    print("Node decision unavailable; bounded verification continues", flush=True)
                time.sleep(30)
            verdict = {"family": row["family"], "status": "verification_window_ended"}
            raise RuntimeError("five-hour registration test window ended")
        finally:
            (root / "verdict.json").write_text(json.dumps(verdict, indent=2))
            for process in (forward, executor, original_executor):
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()

if __name__ == "__main__":
    main()
