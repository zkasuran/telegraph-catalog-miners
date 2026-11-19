# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Prepare one verified package and public forwarding lease on a free runner."""
import argparse
import http.client
import json
import os
import subprocess
import tarfile
from pathlib import Path

from common import (
    BASE_CONFIG_SHA,
    BASE_CONFIG_URL,
    FOUNDATION,
    FOUNDATION_RELEASE,
    HOST_KEY,
    KIT,
    KIT_SHA,
    MAX_BODY,
    VOCAB_SIZE,
    check_classes,
    check_proofs,
    check_run,
    fetch_hash,
    lease_for,
    load_inventory,
    now,
    public_json,
    remote_only,
    sha,
    strict_json,
    write_json,
)

HERE = Path(__file__).resolve().parent


def capture_classes(root, row, filename="class-snapshot.json"):
    if filename not in ("class-snapshot.json", "live-class-snapshot.json"):
        raise ValueError("invalid private class snapshot path")
    envelope = {"schema": 1}
    for name, url in (("family", "https://devnode.telegraphprotocol.com/api/wasm?resolve=" + row["family"]),
                      ("child", "https://devnode.telegraphprotocol.com/engine/v1/intents/" + row["child"])):
        _, raw = public_json(url)
        envelope[name] = {"url": url, "raw": raw.decode()}
    envelope["fetched_at"] = now().isoformat()
    check_classes(envelope, row)
    write_json(root / filename, envelope)
    return envelope


def root_path(value):
    root = value.resolve()
    temp = Path(os.environ["RUNNER_TEMP"]).resolve()
    if root.parent != temp or root.name != "candidate-registration":
        raise ValueError("runner-owned candidate-registration directory required")
    return root


def prepare(root, row, batch):
    remote_only()
    root = root_path(root)
    root.mkdir(mode=0o700, exist_ok=True)
    root.chmod(0o700)
    if (root / "state.json").exists() or (root / "forward-key").exists():
        raise ValueError("candidate run directory already holds a lease")
    # Anonymous source GETs only. No provider or signing credential is used.
    metadata, _ = public_json("https://api.github.com/repos/zkasuran/telegraph-catalog-miners/actions/runs/" + row["verified_run_id"])
    check_run(metadata, row)
    proofs = {}
    for name in ("candidate", "runtime", "source"):
        path = root / (name + "-proof.json")
        ref = row[name + "_proof"]
        fetch_hash(ref["url"], path, ref["sha256"], MAX_BODY)
        proofs[name] = strict_json(path.read_bytes())
    check_proofs(row, proofs["candidate"], proofs["runtime"], proofs["source"])
    package = root / row["file"]
    fetch_hash(row["url"], package, row["sha256"], 64 << 20)
    foundation = root / "foundation"
    foundation.mkdir(exist_ok=True)
    for name, digest in FOUNDATION.items():
        fetch_hash(FOUNDATION_RELEASE + name, foundation / name, digest, 1 << 30)
    fetch_hash(BASE_CONFIG_URL, root / "base-config.json", BASE_CONFIG_SHA, MAX_BODY)
    config = strict_json((root / "base-config.json").read_bytes())
    if config["vocab_size"] != VOCAB_SIZE:
        raise ValueError("pinned vocabulary boundary mismatch")
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    # The public recovery proof binds this exact foundation layout.
    spec = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
            "tokenizer_hash": FOUNDATION["tokenizer.json"], "foundation_version": "modernbert-base-lora16-v1",
            "r_max": 16, "num_layers": config["num_hidden_layers"], "hidden_size": hidden,
            "targets": {"attn.Wqkv": [hidden, 3*hidden], "attn.Wo": [hidden, hidden],
                        "mlp.Wi": [hidden, 2*intermediate], "mlp.Wo": [intermediate, hidden]}}
    # Match the original producer's serialization for its declared-layout binding.
    (root / "foundation.json").write_text(json.dumps(spec, indent=2) + "\n")
    expected_spec = proofs["candidate"]["evidence_sha256"]["foundation.json"]
    if sha((root / "foundation.json").read_bytes()) != expected_spec:
        raise ValueError("verified foundation layout proof mismatch")
    archive = root / "kit.tar.gz"
    fetch_hash("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + KIT,
               archive, KIT_SHA, 32 << 20)
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if len(members) > 5000 or sum(m.size for m in members) > 128 << 20:
            raise ValueError("reference kit member or byte ceiling exceeded")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + KIT)
    for executable, source in (("adapterexec", "./cmd/adapterexec"), ("stagecheck", str(HERE / "stagecheck.go"))):
        subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / executable), source],
                       cwd=kit, check=True, timeout=600)
    subprocess.run(["go", "test", "-mod=readonly", "-count=1", "-v", str(HERE / "stagecheck.go"),
                    str(HERE / "stagecheck_test.go")], cwd=kit, check=True, timeout=180)
    capture_classes(root, row)
    command = [str(root / "stagecheck"), "--intent", row["family"], "--child", row["child"],
               "--expected-class", row["class"], "--package", str(package), "--sha256", row["sha256"],
               "--foundation", str(root / "foundation.json"), "--foundation-sha256", expected_spec,
               "--class-snapshot", str(root / "class-snapshot.json"),
               "--snapshot-sha256", sha((root / "class-snapshot.json").read_bytes()), "--out", str(root)]
    with (root / "stagecheck.stdout").open("wb") as out, (root / "stagecheck.stderr").open("wb") as err:
        checked = subprocess.run(command, stdout=out, stderr=err, timeout=60, check=False)
    if checked.returncode or (root / "stagecheck.stdout").stat().st_size > MAX_BODY:
        raise ValueError("full Stage 1 candidate validation failed")
    manifest = strict_json((root / "stagecheck.stdout").read_bytes())
    if (manifest["sha256"] != row["sha256"] or manifest["keccak256"] != "0x" + row["keccak"] or
            manifest["manifest"]["class"] != row["class"] or
            manifest["manifest"]["verification"] != {"stage1": True, "loaded": False}):
        raise ValueError("checked package identity mismatch")
    write_json(root / "static-manifest.json", manifest)
    key = root / "forward-key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key),
                    "-C", "candidate-" + batch], check=True, timeout=15,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    key.chmod(0o600)
    (root / "known_hosts").write_text(HOST_KEY)
    lease = lease_for(row, key.with_suffix(".pub").read_text().strip(), batch,
                      os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_SHA"])
    write_json(root / "lease.json", lease)
    write_json(root / "state.json", {"schema_version": 1, "row": row, "root": str(root),
               "package": str(package), "executable": str(root / "adapterexec"),
               "foundation": str(foundation), "key": str(key), "known_hosts": str(root / "known_hosts")})
    print("CANDIDATE_STATIC_READY " + row["family"], flush=True)


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=HERE / "candidates.json")
    parser.add_argument("--family", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--existing", type=int, default=0)
    args = parser.parse_args()
    root = root_path(Path(os.environ["RUNNER_TEMP"]) / "candidate-registration")
    row = load_inventory(args.inventory, args.family, args.existing)
    prepared = False
    try:
        prepare(root, row, args.batch)
        prepared = True
    except (OSError, ValueError, TypeError, LookupError, AttributeError, RuntimeError,
            http.client.HTTPException, subprocess.SubprocessError, tarfile.TarError):
        # Private diagnostics are not published. Only a safe failure category is.
        if root.is_dir():
            write_json(root / "verdict.json", {"family": row["family"], "status": "preparation_failed"})
        raise SystemExit("Candidate preparation failed") from None
    finally:
        # Cleanup also runs for cancellation or an unexpected programming error.
        if not prepared and root.is_dir():
            for name in ("forward-key", "forward-key.pub"):
                (root / name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
