# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Exact within-job boundaries. Pure checks never load neural dependencies."""
import base64
import importlib.metadata
import json
import os
import platform
import random
from pathlib import Path

from core import CONTRACT, remote_only, sha, write


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def require_match(expected, actual):
    fields = CONTRACT["matched_trial"]["matched_fields"]
    mismatches = [field for field in fields if field not in expected or field not in actual
                  or canonical(expected[field]) != canonical(actual[field])]
    if mismatches:
        raise ValueError("within-job boundary differs: " + ", ".join(mismatches))
    return {"matched_fields": fields, "exact_match": True, "historical_comparison_valid": False}


def validate_orders(plan, rows):
    n = len(rows)
    if plan.get("source_groups") != [r["group_id"] for r in rows] or len(plan.get("epochs", [])) != 6:
        raise ValueError("frozen order source or epoch count changed")
    for order in plan["epochs"]:
        if len(order) != n or any(type(i) is not int for i in order) or sorted(order) != list(range(n)):
            raise ValueError("frozen order is not an exact source permutation")


def rng_record(torch, np):
    remote_only()
    name, keys, index, gaussian, cached = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [name, keys.tolist(), int(index), int(gaussian), float(cached)],
            "torch_cpu": base64.b64encode(torch.get_rng_state().numpy().tobytes()).decode()}


def restore_rng(torch, np, record):
    remote_only()
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, (tuple, list)) else value
    random.setstate(tuples(record["python"]))
    name, keys, index, gaussian, cached = record["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), index, gaussian, cached))
    raw = base64.b64decode(record["torch_cpu"], validate=True)
    torch.set_rng_state(torch.tensor(list(raw), dtype=torch.uint8))
    if canonical(rng_record(torch, np)) != canonical(record):
        raise ValueError("initial RNG restoration differs")


def trainable_record(model):
    remote_only()
    record = [{"name": name, "dtype": str(p.dtype), "shape": list(p.shape),
             "sha256": sha(p.detach().cpu().contiguous().numpy().tobytes())}
            for name, p in model.named_parameters() if p.requires_grad]
    if not record:
        raise ValueError("no initial trainable parameters")
    return record


def environment(torch):
    remote_only()
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise ValueError("hash seed must be set before process startup")
    cpu = []
    for line in Path("/proc/cpuinfo").read_text().split("\n"):
        if line.startswith(("model name", "flags", "vendor_id")) and line not in cpu:
            cpu.append(line)
    return {"run_id": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"],
            "python": platform.python_version(), "machine": platform.machine(), "cpu": cpu,
            "hash_seed": os.environ["PYTHONHASHSEED"], "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), "default_dtype": str(torch.get_default_dtype()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_build": torch.__config__.show(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "numpy", "transformers", "peft", "safetensors", "tokenizers")}}


def boundary(root, model, torch, np, plan, loader_rng):
    return {"trainable": trainable_record(model), "rng": rng_record(torch, np),
            "environment": environment(torch),
            "train_sha256": sha((root / "originals" / "train.jsonl").read_bytes()),
            "validation_sha256": sha((root / "originals" / "validation.jsonl").read_bytes()),
            "plan_sha256": sha(canonical(plan)),
            "loader_rng": base64.b64encode(loader_rng.get_state().numpy().tobytes()).decode()}


def freeze_initial(root, model, torch, np, rows):
    remote_only()
    order_rng = torch.Generator().manual_seed(42)
    plan = {"source_groups": [r["group_id"] for r in rows],
            "epochs": [torch.randperm(len(rows), generator=order_rng).tolist() for _ in range(6)],
            "scope": "Frozen training-order prefix shared by new C3 and new H6 only."}
    validate_orders(plan, rows)
    loader_rng = torch.Generator().manual_seed(42)
    record = boundary(root, model, torch, np, plan, loader_rng)
    record["initial_state_sha256"] = sha(canonical(record))
    write(root / "evidence" / "training-orders.json", plan)
    write(root / "evidence" / "initial-state.json", record)
    return record


def initial_spec(root):
    record = json.loads((root / "evidence" / "initial-state.json").read_text())
    unsigned = {k: v for k, v in record.items() if k != "initial_state_sha256"}
    if sha(canonical(unsigned)) != record.get("initial_state_sha256"):
        raise ValueError("initial state record changed")
    plan = json.loads((root / "evidence" / "training-orders.json").read_text())
    if sha(canonical(plan)) != record["plan_sha256"]:
        raise ValueError("initial order plan changed")
    return record, plan


def matched_report(root):
    """Selector checks only captured development boundaries, never audit data."""
    initial, plan = initial_spec(root)
    reports = {}
    for arm in ("C3", "H6"):
        actual = json.loads((root / "evidence" / (arm + "-initial.json")).read_text())
        reports[arm] = require_match(initial, actual)
        if actual.get("fresh_optimizer") is not True or actual.get("initial_lr") != 0 or actual.get("scheduler_epoch") != 0:
            raise ValueError("missing fresh zero-step arm initialization")
        orders = json.loads((root / "evidence" / (arm + "-orders.json")).read_text())
        if not orders or len(orders) > CONTRACT["arms"][arm]:
            raise ValueError("missing exact observed arm order")
        for epoch, row in enumerate(orders):
            expected = [plan["source_groups"][i] for i in plan["epochs"][epoch]]
            if row["epoch"] != epoch or row["groups"] != expected or row["order_matches_plan"] is not True:
                raise ValueError("observed arm order differs from exact plan")
        history = json.loads((root / "evidence" / (arm + "-validation-history.json")).read_text())
        if len(history) != len(orders):
            raise ValueError("observed orders do not cover every trained epoch")
    return {"experiment": CONTRACT["name"], "arms": reports,
            "historical_comparison_valid": False, "comparison": "NEW C3 BASELINE versus NEW H6"}
