# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure contracts plus injectable RNG isolation. No framework import."""
import ast
import hashlib
import json
import math
import os
import random
import re
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent / "training"
CONTRACT = json.loads((HERE / "contract.json").read_text())
FORMATS = ("json", "prose", "bare_label")
LABELS = ("Flagged", "Safe")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def bindings():
    for name, expected in CONTRACT["source_bindings"].items():
        if sha((TRAINING / name).read_bytes()) != expected:
            raise ValueError("shared source binding changed: " + name)


def load_shared():
    bindings()
    # These modules contain no top-level neural imports. Never call run.main.
    sys.path.insert(0, str(TRAINING))
    import data
    import preserve
    import run
    return data, run, preserve


def remote_only():
    import os
    expected = {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": CONTRACT["repository"],
                "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_OS": "Linux",
                "GITHUB_WORKFLOW": "Bounded moderation schedule diagnostic",
                "DIAGNOSTIC_PUBLIC_REPOSITORY": "true"}
    if any(os.environ.get(k) != v for k, v in expected.items()):
        raise RuntimeError("diagnostic requires its guarded free public CPU workflow")


def schedule(epochs, n=768):
    hp = CONTRACT["hyperparameters"]
    micro = math.ceil(n / hp["batch_size"])
    if micro % hp["grad_accum"]:
        raise ValueError("incomplete accumulation window")
    steps = micro // hp["grad_accum"] * epochs
    warm = int(steps * 0.06)
    def lr(s):
        factor = min(1.0, s / max(1, warm)) if s < warm else max(0.0, (steps - s) / max(1, steps - warm))
        return hp["lr"] * factor
    return {"epochs": epochs, "microbatches_per_epoch": micro, "steps_per_epoch": micro // hp["grad_accum"],
            "optimizer_calls": steps, "warmup_calls": warm,
            "optimizer_lr_before_call": [lr(s) for s in range(steps)], "terminal_lr": lr(steps),
            "optimizer": "AdamW", "gradient_clip": 1.0,
            "limit": "Six epochs changes the warmup/decay prefix. Not exact continuation."}


@contextmanager
def isolated_rng(torch, np):
    py_before, np_before = random.getstate(), np.random.get_state()
    before = torch.get_rng_state().clone()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(py_before)
        np.random.set_state(np_before)
        if not torch.equal(before, torch.get_rng_state()):
            raise RuntimeError("extra tracing changed Torch RNG")
        if random.getstate() != py_before or sha(repr(np.random.get_state()).encode()) != sha(repr(np_before).encode()):
            raise RuntimeError("extra tracing changed Python or NumPy RNG")


def summary(rows, margins):
    if len(rows) != len(margins) or not rows or any(not math.isfinite(x) for x in margins):
        raise ValueError("missing or nonfinite margins")
    def cell(values):
        n = len(values)
        if not n:
            raise ValueError("missing supported label")
        correct, errors, ties = sum(x > 0 for x in values), sum(x < 0 for x in values), sum(x == 0 for x in values)
        losses = [max(0, -x) + math.log1p(math.exp(-abs(x))) for x in values]
        return {"n": n, "correct": correct, "strict_errors": errors, "ties": ties,
                "accuracy": correct / n, "bt_loss": sum(losses) / n,
                "signed_margin_mean": sum(values) / n, "nonpositive_share": (errors + ties) / n}
    if {r["label"] for r in rows} != set(LABELS):
        raise ValueError("unsupported or missing source label")
    result = cell(margins)
    result["by_label"] = {label: cell([m for r, m in zip(rows, margins) if r["label"] == label]) for label in LABELS}
    wins = sum(len(r["chosen"]) > len(r["rejected"]) for r in rows)
    ties = sum(len(r["chosen"]) == len(r["rejected"]) for r in rows)
    baseline = (wins + 0.5 * ties) / len(rows)
    result["length_baseline"] = max(baseline, 1 - baseline)
    result["margin_sha256"] = sha(json.dumps(margins, allow_nan=False).encode())
    return result


def quality(formats, untrained, dataset_ok):
    failures = []
    if not dataset_ok:
        failures.append("upstream dataset gate failed")
    if set(formats) != set(FORMATS) or set(untrained) != set(FORMATS):
        return failures + ["missing response format"]
    for name in FORMATS:
        row = formats[name]
        if set(row["by_label"]) != set(LABELS):
            failures.append(name + " missing label")
            continue
        values = [row["accuracy"], row["length_baseline"], untrained[name]["accuracy"]]
        values += [r["accuracy"] for r in row["by_label"].values()]
        values += [row["bt_loss"]] + [r["bt_loss"] for r in row["by_label"].values()]
        if any(not math.isfinite(v) for v in values):
            failures.append(name + " nonfinite metric")
            continue
        if row["n"] <= 0 or any(r["n"] <= 0 for r in row["by_label"].values()):
            failures.append(name + " empty denominator")
        for cell in [row, *row["by_label"].values()]:
            if cell["correct"] + cell["strict_errors"] + cell["ties"] != cell["n"] or cell["accuracy"] != cell["correct"] / max(1, cell["n"]):
                failures.append(name + " inconsistent counts")
        if sum(c["n"] for c in row["by_label"].values()) != row["n"] or len({c["n"] for c in row["by_label"].values()}) != 1:
            failures.append(name + " label denominators changed")
        if row["accuracy"] < 0.8 or row["accuracy"] + 1e-12 < row["length_baseline"] + 0.1:
            failures.append(name + " accuracy or length baseline gate")
        if row["accuracy"] + 1e-12 < untrained[name]["accuracy"] + 0.05:
            failures.append(name + " untrained improvement gate")
        if any(r["accuracy"] < 0.7 for r in row["by_label"].values()):
            failures.append(name + " per-label gate")
    return failures


def rank_key(record):
    cells = [record["formats"][f]["by_label"][label] for f in FORMATS for label in LABELS]
    return (-min(c["accuracy"] for c in cells), max(c["bt_loss"] for c in cells),
            -min(record["formats"][f]["accuracy"] for f in FORMATS), record["epoch"], record["arm"])


def choose(records, untrained, dataset_ok):
    eligible = [r for r in records if not quality(r["formats"], untrained, dataset_ok)]
    return min(eligible, key=rank_key) if eligible else None


def deny_measurement_reads(root):
    forbidden = (root / "audit-private").resolve()
    old_test = (root / "originals" / "test.jsonl").resolve()
    def guard(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, Path)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path == old_test or path == forbidden or forbidden in path.parents:
            raise PermissionError("trainer/selector cannot open old test or fresh audit")
    sys.addaudithook(guard)


def overlay(source, namespace):
    """Reuse the original train body with two observation-only callbacks."""
    if sha(source.encode()) != CONTRACT["trainer_sha256"]:
        raise ValueError("pinned trainer source changed")
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    class Insert(ast.NodeTransformer):
        steps = 0
        epochs = 0
        def visit_Expr(self, node):
            text = ast.unparse(node)
            if text == "optim.step()":
                self.steps += 1
                return [ast.parse("diagnostic_step(optim, params, gstep)").body[0], node]
            return node
        def visit_Assign(self, node):
            if ast.unparse(node).startswith("avg_loss = "):
                self.epochs += 1
                return [node, ast.parse("diagnostic_epoch(epoch, avg_loss, gstep)").body[0]]
            return node
    insert = Insert()
    fn = insert.visit(fn)
    if (insert.steps, insert.epochs) != (1, 1):
        raise ValueError("trainer overlay anchors changed")
    unit = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    private = dict(namespace)
    # Only the hash-verified pinned train function and two fixed callbacks enter this AST.
    exec(compile(unit, "<pinned-diagnostic-overlay>", "exec"), private)  # noqa: S102
    return private["train"], ast.unparse(unit)


def validate_exclusions(value):
    if not isinstance(value, list) or len(value) > 10000 or any(not isinstance(g, str) or not re.fullmatch(r"[0-9a-f]{64}", g) for g in value):
        raise ValueError("campaign exclusions must be at most 10000 normalized SHA-256 groups")
    if len(set(value)) != len(value):
        raise ValueError("duplicate campaign exclusion")
    return set(value)


def audit_select(rows, excluded, per_label=128):
    # Full official test-file pass records conflicts before bounded hash selection.
    groups = {}
    contradictions = set()
    for row in rows:
        g = row["group_id"]
        if g in excluded:
            continue
        if g in groups and groups[g]["label"] != row["label"]:
            contradictions.add(g)
        groups.setdefault(g, row)
    candidates = [r for g, r in groups.items() if g not in contradictions]
    selected = []
    for label in LABELS:
        bucket = sorted((r for r in candidates if r["label"] == label),
                        key=lambda r: sha((CONTRACT["audit"]["salt"] + ":" + r["group_id"]).encode()))
        if len(bucket) < per_label:
            raise ValueError("insufficient unused official audit originals")
        selected.extend(bucket[:per_label])
    return sorted(selected, key=lambda r: r["group_id"]), len(contradictions)
