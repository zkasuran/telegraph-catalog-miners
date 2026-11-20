# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Strict development-only inputs. The exposed audit is never read or scored."""
import json
import math
import tarfile

from data import HERE, TOKENIZER_SHA, check_splits, digest

RECIPE = {
    "epochs": 24, "lr": 0.001, "weight_decay": 0.01, "grad_accum": 4,
    "lora_rank": 8, "lora_alpha": 32, "lora_dropout": 0.0,
    "max_length": 256, "batch_size": 1, "eval_batch": 1, "seed": 42,
}
SPLITS = ("train", "validation")
EXPOSED_AUDIT_SHA256 = "4772203facbf9fa3f6883f71375b5348e9ad348745a1be4fda85abfc98857a01"


def tokenizer_bytes():
    source = HERE / "sources/tokenizer.json"
    if source.is_symlink():
        raise ValueError("released tokenizer must not be a symlink")
    if source.exists():
        if source.stat().st_size > 8 << 20:
            raise ValueError("released tokenizer byte ceiling")
        raw = source.read_bytes()
    else:
        # A standalone sparse checkout has the exact source archive, not the
        # owner's expanded cache. Extract only the public tokenizer, no rows.
        archive = HERE / "sources.tar.gz"
        pin = json.loads((HERE / "sources.json").read_text())["source_archive_sha256"]
        if archive.is_symlink() or archive.stat().st_size > 32 << 20 or digest(archive.read_bytes()) != pin:
            raise ValueError("primary source archive identity or byte ceiling")
        with tarfile.open(archive, "r:gz") as packed:
            member = packed.getmember("tokenizer.json")
            if not member.isfile() or member.size > 8 << 20:
                raise ValueError("primary tokenizer archive entry ceiling")
            raw = packed.extractfile(member).read()
    if digest(raw) != TOKENIZER_SHA:
        raise ValueError("released tokenizer pin mismatch")
    return raw


def prepare_development(root):
    contract = json.loads((HERE / "contract.json").read_text())
    root.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name in SPLITS:
        path = HERE / "bundle" / (name + ".jsonl")
        raw = path.read_bytes()
        if path.is_symlink() or len(raw) > 8 << 20 or digest(raw) != contract["splits"][name]["sha256"]:
            raise ValueError("development input identity or ceiling")
        values = [json.loads(line) for line in raw.decode().splitlines()]
        if len(values) != contract["splits"][name]["rows"]:
            raise ValueError("development count differs from the original freeze")
        rows[name] = values
        destination = root / (name + ".jsonl")
        if destination.exists() and destination.read_bytes() != raw:
            raise ValueError("refuse replacement of existing development data")
        destination.write_bytes(raw)
    check_splits(rows)
    raw = tokenizer_bytes()
    (root / "sources").mkdir(exist_ok=True)
    (root / "sources/tokenizer.json").write_bytes(raw)
    if (root / "audit.jsonl").exists():
        raise ValueError("development sandbox contains forbidden audit material")
    return rows, {name: contract["splits"][name] for name in SPLITS}


def select(best, epoch, metrics):
    if type(epoch) is not int or not 1 <= epoch <= RECIPE["epochs"]:
        raise ValueError("development epoch outside the predeclared recipe")
    loss = metrics.get("val_loss")
    if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
        raise ValueError("development loss must be finite and nonnegative")
    if best is not None and epoch <= best["epoch"]:
        raise ValueError("development selection must advance")
    improved = best is None or loss < best["val_loss"]
    return ({"epoch": epoch, "val_loss": loss} if improved else best), improved


def quality(trained, untrained, controls):
    failures = []
    labels = trained.get("accuracy_by_label", {})
    if set(labels) != {"True", "False"} or set(controls) != {"length", "token_length", "overlap", "answer_only", "surface"}:
        return {"passed": False, "failures": ["missing supported labels or shortcuts"]}
    accuracy = trained.get("pairwise_accuracy")
    baseline = untrained.get("pairwise_accuracy")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
           for value in [accuracy, baseline, *labels.values(), *controls.values()]):
        return {"passed": False, "failures": ["nonfinite or invalid development metrics"]}
    if accuracy < .8:
        failures.append("development overall accuracy below 80 percent")
    if min(labels.values()) < .7:
        failures.append("development label accuracy below 70 percent")
    if accuracy < baseline + .05:
        failures.append("less than five points over untrained development")
    if accuracy < max(controls.values()) + .1:
        failures.append("less than ten points over strongest frozen shortcut")
    if max(controls.values()) > .65:
        failures.append("development shortcut above 65 percent")
    return {"passed": not failures, "failures": failures, "independent_audit": False}
