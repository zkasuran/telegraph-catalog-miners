# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Standalone local release gate, no neural imports or mutable fitting."""
import ast
import json
import sys

from contracts import HERE, verify_assets


def main():
    source, native = verify_assets()
    for name in ("LICENSE", "NOTICE", "DATA-SOURCES.md", "SECURITY.md", "verify.sh"):
        if not (HERE / name).is_file() or not (HERE / name).stat().st_size:
            raise ValueError("missing release asset: " + name)
    for path in HERE.glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    if {"torch", "transformers", "peft", "onnxruntime", "tensorflow"}.intersection(sys.modules):
        raise RuntimeError("local source gate imported neural code")
    print(json.dumps({"source_groups": {key: len(value) for key, value in source["splits"].items()},
                      "native_run_id": native["run_id"], "immutable_package": True, "model_execution": False}))


if __name__ == "__main__":
    main()
