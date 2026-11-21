# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Check the actual primary update condition without a model or optimizer."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

PRIMARY_SHA256 = "5da2919c66c2d0840104628ab88e98a9c8e5e578686f71ca9be4dcd91bd18d71"


def primary_update_schedule(micro_batches, grad_accum, epochs=1, source=None):
    if any(type(value) is not int or value < 1 for value in (micro_batches, grad_accum, epochs)) or micro_batches > 10000:
        raise ValueError("invalid bounded source schedule")
    raw = (Path(__file__).parent / "UPSTREAM-train_adapter.py").read_bytes() if source is None else source
    if hashlib.sha256(raw).hexdigest() != PRIMARY_SHA256:
        raise ValueError("primary optimizer source changed")
    train = next(node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef) and node.name == "train")
    guards = [node.test for node in ast.walk(train) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "(i + 1) % hp.grad_accum == 0"]
    if len(guards) != 1:
        raise ValueError("primary optimizer update guard differs")
    guard = compile(ast.Expression(guards[0]), "actual primary optimizer update guard", "eval")
    positions = [i + 1 for i in range(micro_batches)
                 if eval(guard, {"__builtins__": {}}, {"i": i, "hp": SimpleNamespace(grad_accum=grad_accum)})]
    covered = positions[-1] if positions else 0
    return {"micro_batches_per_epoch": micro_batches, "gradient_accumulation": grad_accum, "epochs": epochs,
            "updates_per_epoch": len(positions), "total_updates": len(positions) * epochs,
            "covered_batches_per_epoch": covered, "unapplied_batches_per_epoch": micro_batches - covered,
            "primary_source_sha256": PRIMARY_SHA256, "model_or_optimizer_executed": False}


def require_complete_accumulation(pair_count, batch_size, grad_accum, epochs):
    if type(batch_size) is not int or batch_size != 1:
        raise ValueError("bounded owner path requires microbatch one")
    report = primary_update_schedule(pair_count, grad_accum, epochs)
    if report["unapplied_batches_per_epoch"]:
        raise ValueError("training views leave an unapplied terminal accumulation; use a complete versioned recipe")
    return report
