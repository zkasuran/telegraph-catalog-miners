# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Reuse the verified public foundation preparation without local neural work."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("natural_verified_root_tools", ROOT / "common.py")
tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tools)
FOUNDATION = tools.module("natural_foundation_manifest", ROOT / "support/prepare.py").FOUNDATION


def prepare_reference(root):
    tools.remote_only()
    return tools.prepare_foundation(root)


def prepare_kit(root):
    return tools.source_kit(root)
