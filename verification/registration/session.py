# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""One real executor, bounded reverse forwarding and exact public node decisions."""
import argparse
import fcntl
import hashlib
import http.client
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from common import (
    FOUNDATION,
    HOST,
    HOST_KEY,
    MAX_BODY,
    NODE,
    NoRedirect,
    json_bytes,
    now,
    public_json,
    remote_only,
    sha,
    strict_json,
    utc,
    validate_lease,
    validate_row,
    write_json,
)
from prepare import capture_classes, root_path
from proxy import (
    Backend,
    BoundedServer,
    identity,
    input_root,
    normalize_hash,
    validate_response,
)


class AmbiguousDecision(ValueError):
    pass


def find_decision(listing, row, created, clock=None):
    clock = clock or now()
    if not isinstance(listing, dict) or not isinstance(listing.get("intents"), dict):
        # Polling treats every malformed node envelope as the same ValueError.
        raise ValueError("invalid node decision envelope")  # noqa: TRY004
    family = listing["intents"].get(row["family"])
    if family is None:
        return None
    if not isinstance(family, dict) or not isinstance(family.get("entries"), list) or len(family["entries"]) > 512:
        raise ValueError("invalid node decision list")
    matches, seen = [], set()
    for entry in family["entries"]:
        if not isinstance(entry, dict) or entry.get("intent_id") != row["family"]:
            continue
        if not isinstance(entry.get("author_address"), str) or entry["author_address"].lower() != row["owner"]:
            continue
        try:
            if normalize_hash(entry.get("wasm_hash")) != row["keccak"]:
                continue
            registered = utc(entry.get("registered_at"))
        except ValueError:
            continue
        identifier = entry.get("registration_id")
        if type(identifier) is not int or identifier < 1 or registered > clock:
            continue
        existing = row["existing_registration_id"]
        if (existing and identifier != existing) or (not existing and registered < created):
            continue
        if identifier in seen:
            raise AmbiguousDecision("duplicate node registration identity")
        seen.add(identifier)
        matches.append(entry)
    if len(matches) > 1:
        raise AmbiguousDecision("multiple new registrations match this lease")
    if not matches or matches[0].get("activation_status") not in ("active", "rejected", "superseded", "deregistered"):
        return None
    match = matches[0]
    # Copy no evaluation data, reason, pool root, score or node diagnostic.
    return {"family": row["family"], "owner": row["owner"], "adapter_hash": row["keccak"],
            "registration_id": match["registration_id"], "registered_at": match["registered_at"],
            "status": match["activation_status"]}


def fixed_request(row):
    batch = [{"tokens": [50281] + [22 + (i % 100) for i in range(n - 2)] + [50282]}
             for n in (4, 16, 64, 128)]
    return {"batch": batch, **identity(row), "input_root": input_root(batch)}


def preflight(backend, row):
    request = fixed_request(row)
    first = validate_response(backend.score(request), request, row)
    repeated = validate_response(backend.score(request), request, row)
    if first != repeated:
        raise ValueError("fixed-vector repeated score bits changed")
    for tokens in ([], [-1]):
        invalid = {"batch": [{"tokens": tokens}], **identity(row)}
        invalid["input_root"] = input_root(invalid["batch"])
        refusal = validate_response(backend.score(invalid), invalid, row, errors=True)
        if refusal["results"] != [{"error": "refused"}]:
            raise ValueError("real executor accepted invalid tokens")
    return first


def ssh_command(root, row, port):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("invalid local proxy port")
    return ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(root / "forward-key"),
            "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes",
            "-o", "UserKnownHostsFile=" + str(root / "known_hosts"),
            "-o", "GlobalKnownHostsFile=/dev/null",
            "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20",
            "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
            "-o", "PermitLocalCommand=no", "-o", "RequestTTY=no", "-o", "LogLevel=ERROR",
            "-R", f"127.0.0.1:{row['forward_port']}:127.0.0.1:{port}", "telegraph-verify@" + HOST]


def proxy_refusals(port, row):
    for tokens in ([], [-1]):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            body = {"batch": [{"tokens": tokens}], **identity(row)}
            connection.request("POST", "/runnable", json_bytes(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            raw = response.read(MAX_BODY + 1)
            if response.status != 400 or len(raw) > MAX_BODY or strict_json(raw) != {"error": "request refused"}:
                raise ValueError("bounded proxy accepted invalid tokens")
        finally:
            connection.close()


def process_start(pid):
    # Fields after the final ')' start at field 3; field 22 is process start time.
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def spawn(arguments, root, role, environment=None):
    process = subprocess.Popen(arguments, env=environment, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        token = process_start(process.pid)
    except FileNotFoundError:
        process.wait(timeout=5)
        return process
    record = root / (role + "-process.json")
    write_json(record, {"pid": process.pid, "start": token, "role": role})
    return process


def terminate(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def cleanup(root):
    remote_only()
    root = root_path(root)
    for role in ("forward", "executor"):
        record = root / (role + "-process.json")
        if not record.exists():
            continue
        value = strict_json(record.read_bytes())
        pid = value.get("pid")
        if type(pid) is not int or pid <= 1 or value.get("role") != role:
            raise ValueError("invalid cleanup process identity")
        try:
            if process_start(pid) != value["start"]:
                continue
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            expected = str(root / ("forward-key" if role == "forward" else "adapterexec")).encode()
            if expected not in command:
                raise ValueError("cleanup process is outside this lease")
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if process_start(pid) != value["start"]:
                    break
                time.sleep(0.2)
            else:
                os.killpg(pid, signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError):
            pass
        finally:
            record.unlink(missing_ok=True)
    for name in ("forward-key", "forward-key.pub"):
        (root / name).unlink(missing_ok=True)


def public_score(url, request, row):
    if url != row["runnable_url"]:
        raise ValueError("public probe URL mismatch")
    message = urllib.request.Request(url, data=json_bytes(request),
                                    headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.build_opener(NoRedirect()).open(message, timeout=180) as response:
        raw = response.read(MAX_BODY + 1)
        if response.status != 200 or len(raw) > MAX_BODY:
            raise ValueError("public endpoint response exceeds bounds")
    return validate_response(strict_json(raw), request, row)


def run(root):
    remote_only()
    root = root_path(root)
    state = strict_json((root / "state.json").read_bytes())
    row = validate_row(state["row"])
    lease = strict_json((root / "lease.json").read_bytes())
    created = validate_lease(lease, row)
    if lease["run_id"] != os.environ["GITHUB_RUN_ID"] or lease["commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("lease belongs to another workflow run")
    for name, wanted in (("root", root), ("package", root / row["file"]), ("executable", root / "adapterexec"),
                         ("foundation", root / "foundation"), ("key", root / "forward-key"), ("known_hosts", root / "known_hosts")):
        if state[name] != str(wanted):
            raise ValueError("private state path mismatch")
    if (root / "forward-key").stat().st_mode & 0o777 != 0o600:
        raise ValueError("ephemeral key permissions mismatch")
    if (root / "known_hosts").read_text() != HOST_KEY:
        raise ValueError("pinned forwarding host key changed")
    if sha((root / row["file"]).read_bytes()) != row["sha256"]:
        raise ValueError("prepared package hash changed")
    for filename, expected in FOUNDATION.items():
        h = hashlib.sha256()
        with (root / "foundation" / filename).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise ValueError("prepared foundation identity changed")
    remaining = (utc(lease["expires_at"]) - now()).total_seconds()
    deadline = time.monotonic() + remaining
    wall_timer = threading.Timer(remaining, lambda: os.kill(os.getpid(), signal.SIGTERM))
    wall_timer.daemon = True
    wall_timer.start()
    executor, forward, proxy, thread = None, None, None, None
    verdict = {"family": row["family"], "owner": row["owner"], "adapter_hash": row["keccak"], "status": "runtime_failed"}

    def poll_decision():
        data, _ = public_json(NODE + "/api/wasm?intent=" + row["family"] + "&poll=" + str(time.time_ns()))
        return find_decision(data, row, created)

    def finish_decision(value):
        nonlocal verdict
        verdict = value
        print("NODE_DECISION " + value["status"], flush=True)
        return 0 if value["status"] == "active" else 1

    try:
        try:
            decision = poll_decision()
        except (OSError, ValueError) as error:
            if isinstance(error, AmbiguousDecision):
                raise
            decision = None
        if decision:
            return finish_decision(decision)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        environment = {"PATH": os.environ["PATH"], "HOME": str(root), "FOUNDATION_POOL_SIZE": "1",
                       "FOUNDATION_MODEL_PATH": str(root / "foundation/foundation_lora_fp32.onnx"),
                       "FOUNDATION_MODEL_HASH": FOUNDATION["foundation_lora_fp32.onnx"],
                       "FOUNDATION_ENGINE_WASM_PATH": str(root / "foundation/foundation_engine.wasm"),
                       "FOUNDATION_ENGINE_WASM_HASH": FOUNDATION["foundation_engine.wasm"]}
        executor = spawn([str(root / "adapterexec"), "--package", str(root / row["file"]),
                          "--listen", f"127.0.0.1:{port}"], root, "executor", environment)
        ready = min(deadline, time.monotonic() + 180)
        while time.monotonic() < ready:
            if executor.poll() is not None:
                raise RuntimeError("real executor exited during foundation load")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("real executor load timeout")
        backend = Backend(port)
        baseline = preflight(backend, row)
        capture_classes(root, row, "live-class-snapshot.json")
        proxy = BoundedServer(("127.0.0.1", 0), row, backend)
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        proxy_refusals(proxy.server_port, row)
        proof = {"family": row["family"], "package_sha256": row["sha256"], "adapter_hash": row["keccak"],
                 "fixed_vectors": 4, "identical_repeated_bits": True, "real_invalid_token_refusal": True,
                 "proxy_invalid_token_refusal": True, "public_identity_and_bits_match": False,
                 "live_class_snapshot_sha256": sha((root / "live-class-snapshot.json").read_bytes()),
                 "private_pool_access": False, "fixed_score_bits_sha256": sha(json_bytes(baseline["results"]))}
        write_json(root / "preflight.json", proof)
        command = ssh_command(root, row, proxy.server_port)
        authorization = min(deadline, time.monotonic() + 600)
        while time.monotonic() < authorization:
            forward = spawn(command, root, "forward", {"PATH": os.environ["PATH"], "HOME": str(root)})
            time.sleep(3)
            if forward.poll() is None:
                break
            terminate(forward)
            time.sleep(5)
        else:
            raise RuntimeError("restricted forwarding lease was not authorized")
        public_deadline = min(deadline, time.monotonic() + 600)
        while time.monotonic() < public_deadline:
            if forward.poll() is not None or executor.poll() is not None or proxy.fatal.is_set():
                raise RuntimeError("integration process stopped before public readiness")
            try:
                public = public_score(row["runnable_url"], fixed_request(row), row)
                if public != baseline:
                    raise ValueError("public score bits differ from the real local executor")
                break
            except (OSError, ValueError):
                time.sleep(5)
        else:
            raise RuntimeError("public endpoint identity did not become ready")
        proof["public_identity_and_bits_match"] = True
        write_json(root / "preflight.json", proof)
        print("PUBLIC_EXECUTOR_READY " + row["family"], flush=True)
        next_poll = 0
        while time.monotonic() < deadline:
            if executor.poll() is not None or forward.poll() is not None or proxy.fatal.is_set():
                raise RuntimeError("bounded runtime or forwarding failed")
            if time.monotonic() >= next_poll:
                try:
                    decision = poll_decision()
                except (OSError, ValueError) as error:
                    if isinstance(error, AmbiguousDecision):
                        raise
                    decision = None
                if decision:
                    return finish_decision(decision)
                next_poll = time.monotonic() + 30
            time.sleep(1)
        verdict["status"] = "verification_window_ended"
        return 1
    finally:
        wall_timer.cancel()
        write_json(root / "verdict.json", verdict)
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        terminate(forward)
        terminate(executor)
        if thread is not None:
            thread.join(timeout=5)
        cleanup(root)


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    root = root_path(Path(os.environ["RUNNER_TEMP"]) / "candidate-registration")
    if args.cleanup:
        if root.is_dir():
            cleanup(root)
        return
    with (root / "session.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def stop(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            result = run(root)
        except (OSError, ValueError, TypeError, LookupError, AttributeError, RuntimeError,
                http.client.HTTPException, subprocess.SubprocessError, KeyboardInterrupt):
            raise SystemExit("Bounded candidate integration stopped") from None
        finally:
            # Idempotent cleanup must run even when an unexpected error escapes.
            cleanup(root)
        raise SystemExit(result)


if __name__ == "__main__":
    main()
