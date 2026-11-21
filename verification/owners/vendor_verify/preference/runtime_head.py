# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only frozen-feature pairwise ridge head, not independent qualification.

Use the exact classifier input of the pinned primary ModernBERT model. Preserve
all frozen pooling/dense/norm weights and zero LoRA branches. Only the supported
linear reward head changes. No neural imports occur before the remote guard.
"""

import data
from development_controls import RIDGE_GRID, aggregate, select_candidate, variants


def train_frozen_head(
    trainer, base, tok, train, validation, adapter, hp, root, strongest
):
    import remote

    remote.remote_only()
    import numpy as np
    import torch

    if torch.cuda.is_available():
        raise ValueError("standard CPU only")
    torch.manual_seed(hp.seed)
    model, _ = trainer.build_model(str(base), hp)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    classifier = model.get_base_model().classifier
    head = classifier.modules_to_save["default"]
    if not isinstance(head, torch.nn.Linear) or head.out_features != 1:
        raise ValueError("literal supported classifier head required")
    # A frozen feature model must have no active neural delta.
    branches = [
        parameter for name, parameter in model.named_parameters() if ".lora_B." in name
    ]
    if not branches or any(torch.count_nonzero(p).item() for p in branches):
        raise ValueError("nonzero LoRA in frozen-feature model")
    captured = []

    def capture(_module, args):
        if len(args) != 1 or args[0].ndim != 2 or args[0].shape[1] != head.in_features:
            raise ValueError("unsupported primary classifier input")
        captured.append(args[0].detach().float().cpu().numpy().copy())

    hook = head.register_forward_pre_hook(capture)
    deadline = __import__("time").monotonic() + 120 * 60

    def features(rows, permit_control_refusal=False):
        good, bad, diffs = [], [], []
        refused = 0
        for row in rows:
            pair_features, scores = [], []
            tokens = []
            for side in ("chosen", "rejected"):
                text = data.render(row["prompt"], row[side])
                if text != trainer.render(row["prompt"], row[side]):
                    raise ValueError("literal input renderer mismatch")
                encoded = tok(text, truncation=False, return_tensors="pt")
                if encoded["input_ids"].shape[1] > 512:
                    if not permit_control_refusal:
                        raise ValueError("literal development exceeds exact token cap")
                    tokens = []
                    break
                tokens.append(encoded)
            if not tokens:
                refused += 1
                continue
            for encoded in tokens:
                captured.clear()
                with torch.no_grad():
                    score = trainer._reward(
                        model, encoded["input_ids"], encoded["attention_mask"]
                    )
                if len(captured) != 1 or not torch.isfinite(score).all():
                    raise ValueError("actual classifier capture or finite score failed")
                vector = captured.pop()[0]
                if not np.isfinite(vector).all():
                    raise ValueError("nonfinite frozen feature")
                pair_features.append(vector)
                scores.append(float(score.item()))
                if __import__("time").monotonic() > deadline:
                    raise ValueError("bounded frozen-feature runtime exceeded")
            good.append(pair_features[0])
            bad.append(pair_features[1])
            diffs.append(scores[0] - scores[1])
        if refused:
            return (
                None,
                None,
                {
                    "pairs_requested": len(rows),
                    "pairs_scored": len(diffs),
                    "over_cap_pairs_refused": refused,
                    "complete": False,
                    "accuracy": None,
                    "reason": "attribution control exceeds literal token cap, no truncation",
                },
            )
        return np.asarray(good), np.asarray(bad), diffs

    try:
        tc, tr, train_initial = features(train)
        vc, vr, val_initial = features(validation)
        deltas = (tc - tr).astype(np.float64)
        # All normalization and solving use training only. Fold scale into the
        # final head so serving needs no extra operation or feature side channel.
        scale = np.maximum(np.sqrt(np.mean(deltas * deltas, axis=0)), 1e-6)
        standardized = deltas / scale
        gram = standardized @ standardized.T
        records, weights = [], []
        for ridge in RIDGE_GRID:
            dual = np.linalg.solve(
                gram + ridge * np.eye(len(train)), np.ones(len(train))
            )
            weight = ((standardized.T @ dual) / scale).astype(np.float32)
            if not np.isfinite(weight).all():
                raise ValueError("nonfinite fitted reward head")
            train_diff = (tc @ weight - tr @ weight).tolist()
            val_diff = (vc @ weight - vr @ weight).tolist()
            records.append(
                {
                    "ridge": ridge,
                    "train": aggregate(train, train_diff),
                    "validation": aggregate(validation, val_diff),
                }
            )
            weights.append(weight)
        selected = select_candidate(records)
        with torch.no_grad():
            head.weight.copy_(torch.from_numpy(weights[selected]).reshape(1, -1))
            if head.bias is not None:
                head.bias.zero_()
        # Primary evaluator executes the actual selected model, not fitted
        # feature scores. This remains native CPU, never a reference-engine pass.
        untrained = {
            "pairwise_accuracy": aggregate(validation, val_initial)["accuracy"]
        }
        trained = trainer.evaluate(model, tok, validation, hp)
        actual = {}
        for name, rows in variants(validation).items():
            _, _, differences = features(rows, permit_control_refusal=name != "literal")
            actual[name] = (
                differences
                if isinstance(differences, dict)
                else aggregate(validation, differences)
            )
        fitted_acc = records[selected]["validation"]["accuracy"]
        if abs(actual["literal"]["accuracy"] - fitted_acc) > 1e-9:
            raise ValueError(
                "frozen-feature and literal selected-model ranking mismatch"
            )
        if actual["prompt_only"]["ties"] != len(validation):
            raise ValueError("prompt-only identical answers must tie")
        model.save_pretrained(adapter)
        # Exporter checks the tokenizer bytes literally. Save the verified
        # original rather than HF's differently serialized equivalent tokenizer.
        (adapter / "tokenizer.json").write_bytes((base / "tokenizer.json").read_bytes())
        trainer.save_reward_head(model, adapter)
        mechanism = {
            "schema_version": 1,
            "family": "VENDOR_VERIFY",
            "fit_scope": "training only",
            "mechanism": "frozen primary classifier-input features with pairwise dual ridge head",
            "train_pairs": len(train),
            "validation_pairs": len(validation),
            "training_initial": aggregate(train, train_initial),
            "untrained_validation": aggregate(validation, val_initial),
            "candidate_grid": records,
            "selected_index": selected,
            "selection": "validation accuracy, strongest ridge on ties",
            "selected_validation": actual["literal"],
            "development_controls": actual,
            "strongest_classical_shortcut_accuracy": strongest,
            "weights_sha256": data.sha(weights[selected].tobytes()),
            "feature_width": head.in_features,
            "zero_lora_branches": len(branches),
            "audit_evaluated": False,
            "raw_text_exported": False,
            "reference_engine_passed": False,
            "registration_ready": False,
            "control_limits": "Displaced-prompt agreement is attribution only, not a newly labelled truth set. No exact Vendor format audit.",
        }
        from remote import write_report

        write_report(root / "frozen-head-development.json", mechanism)
        return model, untrained, trained, mechanism
    finally:
        hook.remove()
