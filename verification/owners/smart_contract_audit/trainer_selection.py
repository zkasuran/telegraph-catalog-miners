# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Isolated source adaptation of the MIT pinned trainer, no shared kit changes.

Upstream Copyright (c) 2026 telegraphprotocol. Its full MIT notice is retained
in licences.json and all package/checkpoint exports. Our additions use SAND.
"""
import hashlib
import argparse
import json
import sys
import types
from pathlib import Path
from urllib.request import Request, urlopen

from policy import remote_only
from selection import selection_metrics
from paired_sampling_remote import sampler as sibling_sampler

UPSTREAM_SHA256 = "5da2919c66c2d0840104628ab88e98a9c8e5e578686f71ca9be4dcd91bd18d71"
UPSTREAM_URL = "https://raw.githubusercontent.com/telegraphprotocol/telegraph-wasm-baseline/1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd/float_runtime/train_adapter.py"


def adapt_source(raw):
    if len(raw) > 1 << 20 or hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("exact pinned primary trainer source required")
    text = raw.decode()
    replacements = (
        ("    n = len(diffs)\n", "    n = len(diffs)\n    owner_selection = owner_selection_metrics(pairs, diffs)\n"),
        ('        "n_eval": n,\n', '        "n_eval": n,\n        "owner_selection": owner_selection,\n        "selection_score": owner_selection["selection_score"],\n'),
        ('        metric = val["val_loss"] if lower_better else val["pairwise_accuracy"]\n',
         '        metric = val["selection_score"] if hp.es_metric == "worst_gate_margin" else (val["val_loss"] if lower_better else val["pairwise_accuracy"])\n'),
        ('    loader = DataLoader(train_pairs, batch_size=hp.batch_size, shuffle=True,\n                        collate_fn=PairCollator(tok, hp.max_length))\n',
         '    loader = DataLoader(train_pairs,\n                        batch_sampler=owner_training_batches(train_pairs, hp),\n                        collate_fn=PairCollator(tok, hp.max_length))\n'),
    )
    for original, changed in replacements:
        if text.count(original) != 1:
            raise ValueError("primary trainer selection anchor changed")
        text = text.replace(original, changed)
    compile(text, "isolated-owner-trainer.py", "exec")  # Source syntax only.
    return text.encode()


def install(source_path, output_root):
    remote_only()  # Before exec or any upstream/model import.
    source_path, output_root = Path(source_path), Path(output_root)
    original = source_path.read_bytes()
    adapted = adapt_source(original)
    history = []

    def record(rows, differences):
        result = selection_metrics(rows, differences)
        history.append(result)
        if len(history) > 100:
            raise ValueError("validation-selection report ceiling")
        (output_root / "validation-selection.json").write_text(json.dumps({"evaluations": history, "audit_used_for_selection": False}, indent=2) + "\n")
        return result

    # Preserve the original __file__ so tokenizer conformance resources resolve.
    # Only this private module is adapted; the pinned kit source stays untouched.
    name = "float_runtime.owner_train_adapter"
    module = types.ModuleType(name)
    module.__file__ = str(source_path)
    module.__dict__["owner_selection_metrics"] = record
    frozen_path = Path(__file__).with_name("frozen-sampling-v1.json")
    frozen = json.loads(frozen_path.read_text())
    module.__dict__["owner_training_batches"] = lambda rows, hp: sibling_sampler(rows, frozen, hp)
    sys.modules[name] = module
    exec(compile(adapted.decode(), str(source_path), "exec"), module.__dict__)
    report = {"upstream_sha256": hashlib.sha256(original).hexdigest(), "adapted_source_sha256": hashlib.sha256(adapted).hexdigest(),
              "method": "worst_gate_margin", "upstream_grant": "MIT, Copyright (c) 2026 telegraphprotocol",
              "additions_grant": "LicenseRef-zkasuran-SAND-1.0", "shared_kit_modified": False,
              "optimizer_and_training_data_changed": False, "aggregate_accuracy_still_reported": True,
              "training_sampling": "frozen inverse-group weights with replacement on literal reversed sibling minibatches",
              "frozen_sampling_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
              "audit_or_quality_gates_modified": False}
    (output_root / "trainer-selection-source.json").write_text(json.dumps(report, indent=2) + "\n")
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-check-report", required=True, type=Path)
    args = parser.parse_args()
    with urlopen(Request(UPSTREAM_URL, headers={"User-Agent": "smart-audit-source-selection-contract/1"}), timeout=30) as response:
        original = response.read((1 << 20) + 1)
    adapted = adapt_source(original)
    # Actual syntax and exact retained training/optimizer/gate source, no exec.
    assert b'"pairwise_accuracy": round(acc, 4)' in adapted
    assert b'loss = bt_loss(rc, rr, b["margin"].to(device)) / hp.grad_accum' in adapted
    assert b'if hp.early_stopping and avg_loss < hp.loss_floor:' in adapted
    report = {"primary_source_url": UPSTREAM_URL, "upstream_sha256": hashlib.sha256(original).hexdigest(),
              "adapted_source_sha256": hashlib.sha256(adapted).hexdigest(), "primary_source_compiled": True,
               "neural_import_or_execution": False, "source_adaptation_anchors": 4,
              "aggregate_accuracy_optimizer_loss_floor_and_quality_gates_retained": True}
    args.source_check_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
