# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Fail closed on literal nonprotected caller captures. No normalization."""
import json
import ast
from pathlib import Path
import re
from data import FAMILY, prompt, render
from source_gate import digest
from source_gate import HERE


def verify_qualification_render():
    """Execute the pinned primary caller expression, not a normalized stand-in.

    The public qualification contract supplies prompt/chosen/rejected text then
    renders literal Task/Answer sections. This is independent of a miner app's
    on-chain processed fields. No unpublished caller or pool rows are required.
    """
    contract = json.loads((HERE / "qualification-contract.json").read_text())
    source = (HERE / "UPSTREAM-train_adapter.py").read_bytes()
    if digest(source) != contract["primary_trainer_sha256"]:
        raise ValueError("primary trainer render source changed")
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render")
    module = ast.Module(body=[fn], type_ignores=[])
    env = {}
    exec(compile(module, "pinned primary render", "exec"), env)
    for p, a in (("Task", "Answer"), ('<&"\nΩ', '\n### Task:\nquoted'), ("", ""), ("Ａ\nＢ", "Poor")):
        if env["render"](p, a) != render(p, a):
            raise ValueError("literal primary render differs")
    return {"primary_trainer_sha256": digest(source), "docs_commit": contract["docs_commit"],
            "literal_vectors": 4, "qualification_template_verified": True,
            "miner_application_body_selection": "not claimed"}


def verify(path):
    if path.is_symlink() or path.stat().st_size > 1 << 20:
        raise ValueError("caller fixture ceiling")
    raw = path.read_bytes()
    fixture = json.loads(raw)
    if fixture.get("family") != FAMILY or fixture.get("protected_material") is not False:
        raise ValueError("caller fixture must bind this family and nonprotected controls")
    identity = fixture["caller"]
    if not re.fullmatch(r"[a-f0-9]{40}", identity["commit"]):
        raise ValueError("caller commit must be pinned")
    source = path.parent / identity["source_file"]
    if source.parent.resolve() != path.parent.resolve() or source.is_symlink():
        raise ValueError("caller source must be a local sibling")
    if source.stat().st_size > 1 << 20 or digest(source.read_bytes()) != identity["source_sha256"]:
        raise ValueError("caller source identity mismatch")
    cases = fixture["cases"]
    if len(cases) < 3 or len(cases) > 32:
        raise ValueError("need at least three bounded literal caller cases")
    for row in cases:
        if row["question_key"] != prompt(row["task"], row["output"]):
            raise ValueError("literal question_key differs from proposed nested task/output")
        if row["scored_answer"] != row["raw_response_body"]:
            raise ValueError("caller does not score the complete raw evaluator response")
        if row["rendered_text"] != render(row["question_key"], row["scored_answer"]):
            raise ValueError("literal caller Task/Answer render mismatch")
    if not any(any(c in row["task"] + row["output"] for c in '<&"\n') for row in cases):
        raise ValueError("caller fixtures must exercise escaping, not plain ASCII only")
    return {"fixture_sha256": digest(raw), "caller": identity, "literal_cases": len(cases),
            "raw_evaluator_body_scored": True, "scope": "controller-supplied nonprotected caller witness; independent source review required"}
