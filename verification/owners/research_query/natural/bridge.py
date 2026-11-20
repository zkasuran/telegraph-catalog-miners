# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Reuse the verified public foundation preparation without local neural work."""
import importlib.util
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def checked_module(name, filename):
    """Load the exact root source without changing natural/ import resolution."""
    path = ROOT / filename
    pins = json.loads((ROOT / "contract.json").read_text())["owned_source_sha256"]
    if filename not in pins or hashlib.sha256(path.read_bytes()).hexdigest() != pins[filename]:
        raise ValueError("root preparation source identity changed: " + filename)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


# common.py's source import is a root sibling, not natural/data.py. Bind only
# that checked dependency during module execution, then restore the caller's
# namespace. Do not rely on verify.sh's PYTHONPATH or prepend root globally.
source = checked_module("natural_verified_root_source", "source.py")
previous_source = sys.modules.get("source")
try:
    sys.modules["source"] = source
    tools = checked_module("natural_verified_root_tools", "common.py")
finally:
    if previous_source is None:
        sys.modules.pop("source", None)
    else:
        sys.modules["source"] = previous_source
FOUNDATION = tools.module("natural_foundation_manifest", ROOT / "support/prepare.py").FOUNDATION


def prepare_reference(root):
    tools.remote_only()
    return tools.prepare_foundation(root)


def prepare_kit(root):
    return tools.source_kit(root)


if __name__ == "__main__":
    if sys.argv[1:] != ["--source-smoke"] or "torch" in sys.modules:
        raise ValueError("only source smoke is allowed from this standalone CLI")
    print(json.dumps({"status": "SOURCE_ONLY_READY", "foundation_source_hashes": FOUNDATION,
                      "root_dependency_hashes_checked": True, "neural_imports": False}))
