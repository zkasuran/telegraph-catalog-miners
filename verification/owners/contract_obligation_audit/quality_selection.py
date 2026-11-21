# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Validation-only minimum gate slack. Not accuracy or a model quality audit."""
import ast
import hashlib
import math
import re

from contracts import FAMILY, FORMATS, LABELS, quality

SELECTORS = ("worst-format-accuracy", "quality-slack")
TRAINER_SHA256 = "5da2919c66c2d0840104628ab88e98a9c8e5e578686f71ca9be4dcd91bd18d71"


def check_experiments(workflow):
    inputs = workflow.get("on", {}).get("workflow_dispatch", {}).get("inputs")
    if not isinstance(inputs, dict) or not 1 <= len(inputs) <= 10:
        raise ValueError("workflow must fit the ten-input release contract")
    objective = inputs.get("objective")
    if (not isinstance(objective, dict) or objective.get("type") != "choice" or objective.get("default") != "independent"
            or objective.get("options") != ["independent", "paired-worst-format", "quality-slack", "paired-quality-slack", "syntactic-quality-slack"]):
        raise ValueError("exact safe workflow experiment choices required")


def gate_slack(formats, original_pairs):
    if type(original_pairs) is not int or original_pairs < 16:
        raise ValueError("original validation population required")
    if not isinstance(formats, dict) or set(formats) != set(FORMATS):
        raise ValueError("complete validation formats required for selection")
    for metrics in formats.values():
        if (not isinstance(metrics, dict) or metrics.get("fit_split") != "validation"
                or metrics.get("original_pairs") != original_pairs
                or not {"accuracy", "per_label", "untrained_accuracy", "shortcut_accuracy"} <= set(metrics)
                or not isinstance(metrics["per_label"], dict) or set(metrics["per_label"]) != set(LABELS)):
            raise ValueError("validation-only complete label/baseline identities required")
    quality(formats)  # The exact existing finite metric and unchanged gate checks.
    constraints = {}
    for style, metrics in formats.items():
        accuracy = metrics["accuracy"]
        constraints[style + ":overall"] = accuracy - .80
        constraints[style + ":untrained"] = accuracy + 1e-9 - metrics["untrained_accuracy"] - .05
        constraints[style + ":shortcut"] = accuracy + 1e-9 - metrics["shortcut_accuracy"] - .10
        for label, value in metrics["per_label"].items():
            constraints[style + ":" + label] = value - .70
    key = min(constraints, key=constraints.get)
    return {"metric_name": "minimum_quality_gate_slack", "score": constraints[key],
            "weakest_constraint": key, "constraints": constraints,
            "role": "validation_selection_only", "independent_audit": False}


def attach_selection(measured, formats, original_pairs):
    if (not isinstance(measured, dict) or type(measured.get("pairwise_accuracy")) not in (int, float)
            or not math.isfinite(measured["pairwise_accuracy"]) or not 0 <= measured["pairwise_accuracy"] <= 1
            or type(measured.get("val_loss")) not in (int, float) or not math.isfinite(measured["val_loss"])):
        raise ValueError("actual finite trainer accuracy and loss required")
    selection = gate_slack(formats, original_pairs)
    return {**measured, "selection_gate_slack": selection["score"]}, selection


def require_selection_trace(freeze, trace, trace_hash):
    selector = freeze.get("validation_selector", "worst-format-accuracy")
    if selector not in SELECTORS:
        raise ValueError("unknown frozen validation selector")
    if selector != "quality-slack":
        return
    if (not isinstance(trace, dict) or trace.get("schema_version") != 1 or trace.get("selector") != selector
            or freeze.get("family") != FAMILY or trace.get("family") != FAMILY
            or type(trace.get("run_id")) is not int or trace["run_id"] < 1 or trace["run_id"] != freeze.get("run_id")
            or trace.get("commit") != freeze.get("commit") or trace.get("audit_used_for_selection") is not False
            or trace.get("source_manifest_sha256") != freeze.get("source_manifest_sha256")
            or not isinstance(trace_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", trace_hash)
            or freeze.get("validation_selection_trace_sha256") != trace_hash):
        raise ValueError("actual validation selection trace is not bound to the freeze")
    records = trace.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= 6:
        raise ValueError("bounded actual validation selection records required")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or type(record.get("epoch_index")) is not int or record["epoch_index"] != index:
            raise ValueError("actual selection record order differs")
        selected = gate_slack(record.get("formats"), 30)
        observed = record.get("selection")
        if (not isinstance(observed, dict) or observed.get("score") != selected["score"]
                or observed.get("constraints") != selected["constraints"]):
            raise ValueError("selection score differs from actual validation constraints")


def compile_selector(source):
    """Compile one exact upstream expression change. Does not execute training."""
    if not isinstance(source, bytes) or len(source) > 128 << 10 or hashlib.sha256(source).hexdigest() != TRAINER_SHA256:
        raise ValueError("exact primary trainer source required for selector adapter")
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train"]
    if len(functions) != 1:
        raise ValueError("exact upstream train function inventory differs")
    expected = ast.dump(ast.parse('val["val_loss"] if lower_better else val["pairwise_accuracy"]', mode="eval").body)
    matches = []
    for node in ast.walk(functions[0]):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "metric" and ast.dump(node.value) == expected):
            matches.append(node)
    if len(matches) != 1:
        raise ValueError("exact upstream selection expression differs")
    matches[0].value = ast.parse('val["selection_gate_slack"]', mode="eval").body
    return compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), "pinned-upstream-quality-selector", "exec")
