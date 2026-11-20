# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded deterministic owner-side lease continuation using only the installed action adapter.

No neural imports, signer/provider credentials, publication or registration sends.
All remote writes pass through the existing adapter's fixed gates.
"""
import argparse
import fcntl
import hashlib
import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from common import validate_lease

FAMILY = "SENTIMENT_ANALYSIS"
REPO = "zkasuran/telegraph-catalog-miners"
MAX_SECONDS = 86400
MAX_GENERATIONS = 6
RENEW_BEFORE = 900


def action_key(action, payload):
    return hashlib.sha256(json.dumps({"family": FAMILY, "action": action, "payload": payload},
                                    sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def due(state, now):
    if now >= state["guard_deadline"] or state["generation"] >= MAX_GENERATIONS:
        return "bounded_stop"
    if state.get("node_status") in ("rejected", "superseded", "deregistered"):
        return "terminal_stop"
    if state.get("pending_action"):
        return "resolve_action"
    if state.get("pending_run_id"):
        return "inspect_public_lease"
    if now >= state["lease_expires_epoch"] - RENEW_BEFORE:
        return "dispatch_continuation"
    return "observe"


def read(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 256 << 10:
        raise ValueError("missing, unsafe or oversized owned guard state")
    return json.loads(path.read_text())


def write(path, value):
    raw = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if len(raw.encode()) > 256 << 10:
        raise ValueError("guard state ceiling")
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(raw)
    temporary.chmod(0o600)
    temporary.replace(path)


def command(args, workdir, timeout=60):
    result = subprocess.run(args, cwd=workdir, capture_output=True, text=True, check=True, timeout=timeout)
    if len(result.stdout.encode()) > 1 << 20:
        raise ValueError("guard command output ceiling")
    return result.stdout


def gh_json(path, workdir):
    if not path.startswith("repos/" + REPO + "/actions/"):
        raise ValueError("guard GitHub read outside fixed owned repository")
    return json.loads(command(["gh", "api", path], workdir, 30))


def refresh_proof(args, config, state):
    if args.proof is None or not args.proof.exists():
        return
    proof = read(args.proof)
    if (proof.get("family") != FAMILY or proof.get("run_id") != state["owner_attempt"] or
            proof.get("registration_id") != config["registration_id"] or proof.get("package_keccak") != config["package_keccak"]):
        raise ValueError("guard cannot update a different owner, model or registration proof")
    # Only controlled endpoint metadata may change. Model/report/data bindings are immutable.
    for artifact in proof["artifacts"]:
        if args.endpoint and artifact["path"] == str(args.endpoint.relative_to(args.workdir)):
            endpoint = read(args.endpoint)
            if endpoint.get("package_keccak") != config["package_keccak"] or endpoint.get("serving_run_id") != state["serving_run_id"]:
                raise ValueError("continued endpoint receipt does not match this exact model/run")
            artifact["sha256"] = hashlib.sha256(args.endpoint.read_bytes()).hexdigest()
    proof["serving_run_id"] = state["serving_run_id"]
    proof["continuity"] = {"pid": state.get("pid"), "generation": state["generation"],
                           "lease_expires_epoch": state["lease_expires_epoch"], "status": state["status"],
                           "guard_deadline": state["guard_deadline"], "activated_run_ids": state.get("activated_run_ids", [])}
    if state.get("last_node"):
        proof["continuity"]["last_node"] = state["last_node"]
    write(args.proof, proof)


def exact_node(config):
    url = "https://devnode.telegraphprotocol.com/api/wasm?intent=" + FAMILY + "&lease_guard=" + str(time.time_ns())
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0 Telegraph-owned-lease-guard", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=20) as response:
        raw = response.read((1 << 20) + 1)
        if response.status != 200 or len(raw) > 1 << 20:
            raise ValueError("bounded node metadata read failed")
    family = json.loads(raw).get("intents", {}).get(FAMILY) or {}
    entry = next((row for row in family.get("entries", []) if row.get("registration_id") == config["registration_id"]), None)
    if entry is None:
        return {"status": "awaiting_safe_index", "canonical": False}
    if entry.get("author_address", "").lower() != config["owner"] or entry.get("wasm_hash", "").removeprefix("0x") != config["package_keccak"]:
        raise ValueError("guard node record belongs to a different owner/package")
    champion = family.get("champion") or {}
    return {"status": entry.get("activation_status"), "stage": entry.get("stage"),
            "reason": entry.get("rejection_reason") or "",
            "canonical": champion.get("registration_id") == config["registration_id"] and
                         entry.get("activation_status") == "active" and entry.get("stage") == "canonical"}


def submit(args, state, action, payload):
    if action not in ("dispatch", "activate_endpoint", "reconcile"):
        raise ValueError("guard cannot publish, register, sign or bypass a gate")
    key = action_key(action, payload)
    request_path = args.state.parent / "lease-guard-action.json"
    request = {"schema_version": 1, "family": FAMILY, "run_id": state["owner_attempt"], "action": action, "payload": payload}
    # Persist the exact action boundary before invoking the installed credential-owning adapter.
    state["pending_action"] = {"key": key, "action": action, "payload": payload, "started_at": time.time()}
    write(args.state, state)
    write(request_path, request)
    command(["python3", str(args.adapter), "--family", FAMILY, "--request", str(request_path.relative_to(args.workdir)), "--wait", "0"],
            args.workdir, 40)


def advance(args, config, state):
    attempt = read(args.attempt)
    if attempt["family"] != FAMILY or attempt["run_id"] != state["owner_attempt"]:
        state["status"] = "owner_attempt_changed_stop"
        return False
    now = time.time()
    state.update(heartbeat_at=now, pid=__import__("os").getpid())
    action = due(state, now)
    if action in ("bounded_stop", "terminal_stop"):
        state["status"] = action
        return False
    if action == "resolve_action":
        pending = state["pending_action"]
        path = args.adapter.parent / FAMILY / "operations" / (pending["key"] + ".json")
        if not path.exists():
            if now - pending["started_at"] > 1800:
                raise ValueError("guard action not acknowledged; no resend allowed")
            return True
        operation = read(path)
        if operation["status"] in ("queued", "running", "publishing", "dispatching", "binding_execution_ref"):
            return True
        if operation["status"] != "succeeded":
            # Unknown dispatch sends are reconciled, never resubmitted. Explicit gate failures stop.
            if pending["action"] == "dispatch" and operation.get("dispatch_started_at") and not state.get("reconcile_sent"):
                state["original_dispatch_key"] = pending["key"]
                state["reconcile_sent"] = True
                submit(args, state, "reconcile", {"operation_key": pending["key"]})
                return True
            raise ValueError("installed guard action failed; preserve operation and do not circumvent its gate")
        result = operation["result"]
        if pending["action"] in ("dispatch", "reconcile"):
            if type(result.get("run_id")) is not int or result.get("commit") != config["commit"]:
                raise ValueError("unknown or wrong reconciled dispatch; no second send")
            state["pending_run_id"] = result["run_id"]
        else:
            if result.get("serving_run_id") != state["pending_run_id"] or result.get("package_keccak") != config["package_keccak"]:
                raise ValueError("continuation activated a different run/package")
            state.update(serving_run_id=state.pop("pending_run_id"),
                         lease_expires_epoch=datetime.fromisoformat(result["expires_at"]).timestamp(),
                         generation=state["generation"] + 1, status="exact_package_continued")
            state.setdefault("activated_run_ids", []).append(state["serving_run_id"])
            refresh_proof(args, config, state)
        state["pending_action"] = None
        return True
    if action == "inspect_public_lease":
        run_id = state["pending_run_id"]
        run = gh_json(f"repos/{REPO}/actions/runs/{run_id}", args.workdir)
        if run.get("head_sha") != config["commit"] or run.get("status") == "completed":
            raise ValueError("continuation runner ended or changed source before activation")
        listing = gh_json(f"repos/{REPO}/actions/runs/{run_id}/artifacts?per_page=100", args.workdir)
        artifacts = [item for item in listing.get("artifacts", []) if item.get("name") == f"lease-{FAMILY}-{run_id}"]
        if not artifacts:
            return True
        if len(artifacts) != 1 or artifacts[0].get("expired") or artifacts[0].get("size_in_bytes", 0) > 65536:
            raise ValueError("untrusted continuation public lease artifact")
        directory = args.state.parent / ("lease-guard-run-" + str(run_id))
        if not directory.exists():
            command(["gh", "run", "download", str(run_id), "--repo", REPO, "--name", f"lease-{FAMILY}-{run_id}", "--dir", str(directory)], args.workdir)
        lease = read(directory / "lease.json")
        row = {"family": FAMILY, "owner": config["owner"], "slot": config["slot"], "forward_port": config["forward_port"],
               "runnable_url": config["runnable_url"], "keccak": config["package_keccak"], "sha256": config["package_sha256"],
               "existing_registration_id": lease["existing_registration_id"]}
        for kind in ("candidate", "runtime", "source"):
            row[kind + "_proof"] = {"sha256": config["report_sha256"]}
        validate_lease(lease, row)
        if lease["run_id"] != str(run_id) or lease["commit"] != config["commit"]:
            raise ValueError("continuation lease identity mismatch")
        submit(args, state, "activate_endpoint", {"package_keccak": config["package_keccak"], "package_sha256": config["package_sha256"],
               "serving_run_id": run_id, "serving_commit": config["commit"], "lease_artifact_id": artifacts[0]["id"],
               "candidate_report_sha256": config["report_sha256"]})
        return True
    if action == "dispatch_continuation":
        running = gh_json(f"repos/{REPO}/actions/runs?status=in_progress&per_page=100", args.workdir)
        queued = gh_json(f"repos/{REPO}/actions/runs?status=queued&per_page=100", args.workdir)
        if sum("evaluator-owner-" in run.get("path", "") for run in running.get("workflow_runs", []) + queued.get("workflow_runs", [])) >= 6:
            state["status"] = "waiting_for_verified_free_cpu_capacity"
            return True
        inputs = dict(config["inputs"])
        inputs["lease_epoch"] = "continuation-" + str(state["generation"] + 1)
        inputs["registration_id"] = str(config["registration_id"])
        submit(args, state, "dispatch", {"workflow": config["workflow"], "commit": config["commit"],
                                        "publication_key": config["publication_key"], "inputs": inputs})
        return True
    try:
        node = exact_node(config)
        state.update(node_status=node["status"], last_node=node)
    except OSError:
        state["node_observation"] = "temporarily_unavailable; exact bounded lease maintained"
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--endpoint", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.workdir = args.workdir.resolve()
    for name in ("attempt", "config", "state", "proof", "endpoint"):
        path = getattr(args, name)
        if path is None:
            continue
        if not path.resolve().is_relative_to(args.workdir) or path.is_symlink():
            raise ValueError("guard state must stay in its private owner checkout")
        setattr(args, name, path.resolve())
    if args.adapter.name != "actions.py" or not args.adapter.is_file():
        raise ValueError("installed automatic adapter required")
    config = read(args.config)
    if (config.get("family") != FAMILY or config.get("max_generations") != MAX_GENERATIONS or
            config.get("owner") != "0x8b224783fe5b3c52b7db0cb9b1754f8812b75287" or
            config.get("workflow") != ".github/workflows/evaluator-owner-sentiment_analysis.yml" or
            not re.fullmatch(r"[a-f0-9]{40}", config.get("commit", "")) or
            any(not re.fullmatch(r"[a-f0-9]{64}", config.get(key, "")) for key in
                ("package_keccak", "package_sha256", "report_sha256", "publication_key"))):
        raise ValueError("guard is restricted to one bounded exact family")
    with args.state.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(args.state)
        if state["guard_deadline"] - state["guard_started_at"] > MAX_SECONDS:
            raise ValueError("guard must be bounded to one day")
        while True:
            try:
                keep = advance(args, config, state)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                state.update(status="blocked", error_type=type(error).__name__, error=str(error)[:256])
                keep = False
            write(args.state, state)
            if not keep or args.once:
                return
            time.sleep(30)


if __name__ == "__main__":
    main()
