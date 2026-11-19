# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Retain controller helper bytes and reuse their checked primary contracts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from source_probe import write

HERE = Path(__file__).resolve().parent


def capture(packagecheck, runtime_check):
    sources = {
        "packagecheck.go.txt": packagecheck,
        "runtime-check.py.txt": runtime_check,
    }
    target = HERE / "controller"
    target.mkdir(exist_ok=True)
    pins = {}
    for name, source in sources.items():
        raw = source.read_bytes()
        destination = target / name
        if destination.exists() and destination.read_bytes() != raw:
            raise ValueError("controller helper changed after capture")
        destination.write_bytes(raw)
        pins[name] = hashlib.sha256(raw).hexdigest()
    write(HERE / "controller-pins.json", {"sha256": pins,
        "policy": "Controller source files are retained unchanged. Only a generated build copy specializes the Go family constant to LANGUAGE_TRANSLATION. Upstream Stage 1 parser and reference executor are unchanged."})
    print(json.dumps(pins))


def source(name):
    path = HERE / "controller" / name
    expected = json.loads((HERE / "controller-pins.json").read_text())["sha256"][name]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("controller helper bytes changed")
    return raw.decode()


def package_source():
    text = source("packagecheck.go.txt")
    old = 'const familyID = "SENTIMENT_ANALYSIS"'
    if text.count(old) != 1:
        raise ValueError("controller family constant shape changed")
    return text.replace(old, 'const familyID = "LANGUAGE_TRANSLATION"', 1)


def runtime_functions(env):
    tree = ast.parse(source("runtime-check.py.txt"))
    wanted = {"decode_bits", "check_response", "server"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise ValueError("controller reference runtime functions missing")
    # Only these three function ASTs from the same SHA-256-verified bytes run.
    # The helper's module imports and main entry point are never executed.
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "unchanged-controller-runtime", "exec"), env)  # noqa: S102
    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--packagecheck", required=True, type=Path)
    parser.add_argument("--runtime-check", required=True, type=Path)
    args = parser.parse_args()
    capture(args.packagecheck, args.runtime_check)
