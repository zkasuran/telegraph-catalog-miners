# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Exercise the real pinned parser. No foundation loading or model imports."""
import argparse
import copy
import json
import struct
import subprocess
from pathlib import Path

from data import digest, write_json
from sentiment_recover import CHILD, FAMILY, FOUNDATION, PACKAGE_SHA, capture_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checker", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    raw = args.package.read_bytes()
    if digest(raw) != PACKAGE_SHA:
        raise ValueError("static proof requires the exact original package")
    capture_snapshot(root)
    snapshot = json.loads((root / "class-snapshot.json").read_bytes())
    spec = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
            "tokenizer_hash": FOUNDATION["tokenizer.json"], "foundation_version": "modernbert-base-lora16-v1",
            "r_max": 16, "num_layers": 22, "hidden_size": 768,
            "targets": {"attn.Wqkv": [768, 2304], "attn.Wo": [768, 768],
                        "mlp.Wi": [768, 2304], "mlp.Wo": [1152, 768]}}
    write_json(root / "foundation.json", spec)
    command = [str(args.checker.resolve()), "--intent", FAMILY, "--child", CHILD,
               "--class-snapshot", str(root / "class-snapshot.json"),
               "--snapshot-sha256", digest((root / "class-snapshot.json").read_bytes()),
               "--package", str(args.package.resolve()), "--sha256", PACKAGE_SHA,
               "--foundation", str(root / "foundation.json"),
               "--foundation-sha256", digest((root / "foundation.json").read_bytes()), "--out", str(root)]
    results = []

    def call(name, arguments, expected=None):
        result = subprocess.run(arguments, capture_output=True, timeout=60, check=False)
        (root / (name + ".stderr.txt")).write_bytes(result.stderr)
        (root / (name + ".stdout.txt")).write_bytes(result.stdout)
        if expected is None:
            if result.returncode:
                raise ValueError(name + " positive control failed")
            return json.loads(result.stdout)
        if result.returncode == 0 or expected.encode() not in result.stderr or result.stdout or b"panic:" in result.stderr:
            raise ValueError(name + " refusal control failed")
        results.append({"name": name, "returncode": result.returncode, "required_diagnostic": expected})

    positive = call("positive", command)
    if positive["manifest"]["verification"] != {"stage1": True, "loaded": False} or positive["sha256"] != PACKAGE_SHA:
        raise ValueError("static parser result changed the load boundary or byte identity")
    call("preflight", command + ["--preflight"])
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8+size])
    payload = raw[8+size:]

    def negative(name, expected, contents=None, snap=None, extra=None, expected_hash=None):
        arguments = command.copy()
        if contents is not None:
            path = root / (name + ".safetensors")
            path.write_bytes(contents)
            arguments[arguments.index("--package")+1] = str(path)
            arguments[arguments.index("--sha256")+1] = expected_hash or digest(contents)
        if snap is not None:
            path = root / (name + ".json")
            write_json(path, snap)
            arguments[arguments.index("--class-snapshot")+1] = str(path)
            arguments[arguments.index("--snapshot-sha256")+1] = digest(path.read_bytes())
        call(name, arguments + (extra or []), expected)

    for name, mutate, expected in (
        ("class-mismatch", lambda h: h["__metadata__"].update({"class": "hybrid"}), "class_mismatch"),
        ("bad-abi", lambda h: h["__metadata__"].update({"abi": "lora-v9"}), "bad_abi"),
        ("missing-class", lambda h: h["__metadata__"].pop("class"), "missing_metadata"),
        ("bad-dtype", lambda h: h["head.bias"].update({"dtype": "I32"}), "bad_dtype"),
        ("bad-shape", lambda h: h["head.bias"].update({"shape": [2]}), "bad_shape"),
        ("missing-tensor", lambda h: h.pop("head.bias"), "bad_shape"),
        ("bad-offsets", lambda h: h["head.bias"].update({"data_offsets": [4, 8]}), "bad_offsets"),
        ("bad-rank", lambda h: h["__metadata__"].update({"rank": "17"}), "rank_too_large"),
    ):
        changed = copy.deepcopy(header)
        mutate(changed)
        encoded = json.dumps(changed, separators=(",", ":")).encode()
        negative(name, expected, struct.pack("<Q", len(encoded)) + encoded + payload)
    negative("corrupt", "bad_container", b"\0"*8 + raw[8:])
    negative("truncated", "truncated", raw[:-4])
    negative("trailing", "trailing_bytes", raw + b"\0"*4)
    for name, bits in (("nan", 0x7fc00000), ("infinity", 0x7f800000)):
        negative(name, "non_finite", raw[:8+size] + struct.pack("<I", bits) + payload[4:])
    negative("wrong-hash", "SHA-256 mismatch", raw, expected_hash="0"*64)
    for name, section, key, value, expected in (
        ("wrong-source-class", "family", "class", "deterministic", "source class is absent"),
        ("wrong-child-class", "child", "class", "hybrid", "agreement failed"),
        ("wrong-child", "child", "intent_id", "OTHER_GENERAL", "agreement failed"),
    ):
        changed = copy.deepcopy(snapshot)
        obj = json.loads(changed[section]["raw"])
        (obj["resolve"] if section == "family" else obj)[key] = value
        changed[section]["raw"] = json.dumps(obj)
        negative(name, expected, snap=changed)
    negative("stale", "snapshot time", snap={**snapshot, "fetched_at": "2020-01-01T00:00:00Z"})
    negative("wrong-url", "source URL mismatch", snap={**snapshot, "family": {**snapshot["family"], "url": "https://example.com"}})
    for section in ("request", "response", "onchain"):
        changed = copy.deepcopy(snapshot)
        obj = json.loads(changed["child"]["raw"])
        obj[section]["served_by"] = "GENERIC"
        changed["child"]["raw"] = json.dumps(obj)
        negative("wrong-" + section + "-family", "schema family agreement", snap=changed)
    changed = copy.deepcopy(snapshot)
    obj = json.loads(changed["child"]["raw"])
    obj["onchain"]["ready"] = False
    changed["child"]["raw"] = json.dumps(obj)
    negative("not-ready", "readiness failed", snap=changed)
    negative("override", "flag provided but not defined", extra=["--class", "judgment"])
    negative("model-flag", "flag provided but not defined", extra=["--foundation-onnx", "absent"])
    if digest(args.package.read_bytes()) != PACKAGE_SHA:
        raise ValueError("original package changed")
    write_json(root / "STATIC-RESULT.json", {"positive_controls": 2, "negative_controls": results,
               "package_sha256": PACKAGE_SHA, "keccak256": positive["keccak256"],
               "loaded": False, "stage1": True, "foundation_scope": "declared layout only"})
    print(f"Static parser proof: 2 positive controls and {len(results)} expected refusals. No foundation load.")


if __name__ == "__main__":
    main()
