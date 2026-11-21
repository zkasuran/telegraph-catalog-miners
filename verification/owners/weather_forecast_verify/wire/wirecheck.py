# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Standalone source-only check of the pinned real caller on constructed replies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PIN = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
SOURCE_HASHES = {
    "pkg/wasm/remote/remote.go": "9d8939b86181449d2c92a5c7f1a720ca34aee27b5ae491c84246c4e2eb2c871f",
    "pkg/wasm/runtime/model_score.go": "42de40bb764baea33b654782a4c69cd830f5265aea107c592cce51dac7c73f01",
}
CONTROL_NAMES = {
    "finite_exact_bits", "wrong_foundation_hash", "wrong_adapter_hash", "wrong_tokenizer_hash",
    "wrong_input_root", "nan_bits", "positive_inf_bits", "negative_inf_bits", "overflow_bits",
    "negative_bits", "decimal_score", "boolean_score", "wrong_result_count", "temporary_503", "module_422",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix():
    rng = random.Random(4266)
    batches = [[[rng.randint(0, 50282) for _ in range(rng.randint(0, 32))] for _ in range(size)]
               for size in (0, 1, 2, 9, 10, 11, 16, 64, 100)]
    return batches + [[[0], [50281, 50282], [], [-1]], [[9223372036854775807], [-9223372036854775808]]]


def literal_root(batch):
    # Matches the unchanged family protocol and the primary client, including lexical indices.
    leaves = []
    for index, tokens in enumerate(batch):
        if any(type(token) is not int or not -(1 << 63) <= token < 1 << 63 for token in tokens):
            raise ValueError("literal int64 token required")
        raw = ('{"input_ids":[' + ','.join(str(token) for token in tokens)
               + '],"attention_mask":[' + ','.join('1' for _ in tokens) + ']}').encode()
        leaves.append(str(index) + ':' + hashlib.sha256(raw).hexdigest())
    return hashlib.sha256('\n'.join(sorted(leaves)).encode()).hexdigest() if leaves else ""


def validate_output(value, batches):
    if not isinstance(value, dict) or set(value) != {"input_roots", "controls", "neural_operations", "replies"}:
        raise ValueError("bounded exact wire report required")
    if value["neural_operations"] is not False or value["replies"] != "constructed HTTP fixtures, not engine scores":
        raise ValueError("source-only constructed-fixture identity required")
    controls = value["controls"]
    if (not isinstance(controls, list) or len(controls) != len(CONTROL_NAMES)
            or any(not isinstance(c, dict) or set(c) != {"name", "passed"}
                   or not isinstance(c["name"], str) or c["passed"] is not True for c in controls)):
        raise ValueError("complete actual passing controls required")
    names = [c["name"] for c in controls]
    if len(set(names)) != len(names) or set(names) != CONTROL_NAMES:
        raise ValueError("missing, duplicate or unsupported wire controls")
    if value["input_roots"] != [literal_root(batch) for batch in batches]:
        raise ValueError("primary caller differs from literal input-root contract")
    return True


def execute(kit, compiler, out):
    kit, out = kit.resolve(), out.resolve()
    executable = shutil.which(compiler)
    if executable is None:
        raise ValueError("Go 1.25.7 compiler is required for this source-only check")
    compiler_path = Path(executable).resolve()
    out.mkdir(parents=True, exist_ok=True)
    home = out / "home"
    home.mkdir(exist_ok=True)
    environment = {"PATH": str(compiler_path.parent) + os.pathsep + os.defpath, "HOME": str(home),
                   "LANG": "C.UTF-8", "GOTOOLCHAIN": "local", "CGO_ENABLED": "0",
                   "GOPROXY": "https://proxy.golang.org", "GOSUMDB": "sum.golang.org",
                   "GOPATH": str(out / "gopath"), "GOCACHE": str(out / "gocache")}
    commands = []
    with (out / "source-wire.log").open("x") as log:
        def command(arguments, timeout=300, data=None, expected=0):
            result = subprocess.run(arguments, cwd=kit, env=environment, input=data,
                                    capture_output=True, text=True, timeout=timeout)
            log.write("COMMAND " + json.dumps(arguments) + "\n" + result.stdout + result.stderr + "\n")
            log.flush()
            commands.append({"program": Path(arguments[0]).name, "returncode": result.returncode})
            if result.returncode != expected or len(result.stdout.encode()) > 1 << 20:
                raise ValueError("source-wire command failed or exceeded bound")
            return result.stdout
        if command(["git", "rev-parse", "HEAD"], 15).strip() != PIN:
            raise ValueError("primary caller kit revision changed")
        if command(["git", "status", "--porcelain", "--untracked-files=all"], 15).strip():
            raise ValueError("primary source checkout contains unreviewed changes")
        if any(digest(kit / name) != expected for name, expected in SOURCE_HASHES.items()):
            raise ValueError("primary source bytes changed")
        version = command([str(compiler_path), "version"], 15).strip()
        if version.split()[2:3] != ["go1.25.7"]:
            raise ValueError("pinned source-only compiler version required")
        source = HERE / "wire_contract.go"
        if command([str(compiler_path.parent / "gofmt"), "-l", str(source)], 15).strip():
            raise ValueError("wire probe formatting differs")
        command([str(compiler_path), "vet", "-mod=readonly", str(source)])
        program = out / "wire-contract"
        command([str(compiler_path), "build", "-mod=readonly", "-trimpath", "-o", str(program), str(source)])
        batches = matrix()
        observed = json.loads(command([str(program)], 60, json.dumps(batches)))
        validate_output(observed, batches)
        command([str(program)], 10, '[[["not-an-integer"]]]', expected=1)
    report = {"schema_version": 1, "stage": "source_wire", "passed": True,
              "source_commit": PIN, "source_sha256": SOURCE_HASHES,
              "compiler": version, "probe_sha256": digest(source), "executable_sha256": digest(program),
              "commands": commands, "primary_caller_controls": observed["controls"],
              "hash_matrix_batches": len(batches), "all_input_roots_match": True,
              "invalid_token_exit": 1, "neural_operations": False, "reference_engine_executed": False,
              "replies": observed["replies"], "log_sha256": digest(out / "source-wire.log")}
    with (out / "source-wire.json").open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--go", default="go")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(execute(args.kit, args.go, args.out), sort_keys=True))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print("source-wire check failed: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
