# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""One guarded arm. Old test and audit reads are denied in this process."""
import argparse
import base64
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from core import (
    CONTRACT,
    deny_measurement_reads,
    isolated_rng,
    overlay,
    remote_only,
    schedule,
    sha,
    write,
)
from matched import boundary, initial_spec, require_match, restore_rng, validate_orders
from model_ops import (
    read_development,
    reload_snapshot,
    runtime,
    save_snapshot,
    score_views,
    tokenize,
    tokenizer,
    views,
)


def fit(root, arm):
    deny_measurement_reads(root)
    with (root / "evidence" / (arm + "-started.json")).open("x") as out:
        out.write(json.dumps({"arm": arm, "fresh_optimizer": True}) + "\n")
    torch, np, trainer, hp = runtime(root, CONTRACT["arms"][arm])
    splits = read_development(root)
    formats = views(splits["validation"])
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from balanced import style_positions, training_batch, training_views as anchored_views
    positions = style_positions(splits["train"])
    data = __import__("core").load_shared()[0]
    training_views = anchored_views(splits["train"], data.pair)
    style_epoch = {"epoch": 0}
    style_gates = {name: trainer.gate_dataset(rows, hp) for name, rows in training_views.items()}
    if any(not value["ok"] for value in style_gates.values()):
        raise ValueError("actual training-format dataset gate failed")
    write(root / "evidence/training-format-gates.json", {"gates": style_gates, "groups": len(positions), "per_label_format": 256, "policy": "source-anchored JSON/prose/label-led views, stable per-label position plus epoch modulo 3; no exact bare-label training"})
    tok, node = tokenizer(root)
    gate = trainer.gate_dataset(splits["train"], hp)
    write(root / "evidence" / (arm + "-dataset-gate.json"), gate)
    if not gate["ok"]:
        raise RuntimeError("upstream dataset gate failed before model loading")
    tokenize(trainer, tok, node, [*training_views.values(), splits["validation"], *formats.values()], root / "evidence" / (arm + "-tokenizer-check.json"))
    planned = schedule(hp.epochs, len(splits["train"]))
    initial, plan = initial_spec(root)
    validate_orders(plan, splits["train"])
    loader_rng = torch.Generator()
    loader_rng.set_state(torch.tensor(list(base64.b64decode(initial["loader_rng"], validate=True)), dtype=torch.uint8))
    class FrozenSampler:
        epoch = 0
        def __len__(self):
            return len(splits["train"])
        def __iter__(self):
            if self.epoch >= hp.epochs:
                raise ValueError("extra training loader iteration")
            style_epoch["epoch"] = self.epoch
            order = plan["epochs"][self.epoch]
            self.epoch += 1
            return iter(order)
    write(root / "evidence" / (arm + "-schedule.json"), {**planned, "hyperparameters": asdict(hp)})
    history, telemetry, epoch_state, window = [], [], {}, Counter()
    observed_order, observed_epochs = [], []
    class ObservedCollator(trainer.PairCollator):
        def __call__(self, batch):
            batch = training_batch(batch, positions, style_epoch["epoch"], data.pair)
            for row in batch:
                window[(row["label"], row["style"])] += 1
                observed_order.append(row["group_id"])
            return super().__call__(batch)
    def step(optim, params, gstep):
        if gstep >= planned["optimizer_calls"]:
            raise ValueError("optimizer call ceiling")
        lr = optim.param_groups[0]["lr"]
        if not math.isclose(lr, planned["optimizer_lr_before_call"][gstep], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("optimizer schedule differs from inspected source")
        if sum(window.values()) != hp.batch_size * hp.grad_accum:
            raise ValueError("incomplete observed accumulation window")
        record = {"call": gstep + 1, "actual_lr": lr,
                  "label_style_pairs": {label + ":" + str(style): n for (label, style), n in sorted(window.items())}}
        window.clear()
        if gstep % 10 == 0:
            grads = [p.grad for p in params if p.grad is not None]
            if not grads or any(not torch.isfinite(g).all() for g in grads):
                raise ValueError("absent or nonfinite gradients")
            record["post_clip_gradient_l2"] = math.sqrt(sum(float(g.float().square().sum()) for g in grads))
        telemetry.append(record)
        write(root / "evidence" / (arm + "-optimizer.json"), telemetry)
    def epoch(index, train_loss, steps):
        if window or steps != (index + 1) * planned["steps_per_epoch"]:
            raise ValueError("discarded accumulation tail or step-count drift")
        epoch_state.update(epoch=index, train_loss=train_loss, optimizer_calls=steps)
        expected = [plan["source_groups"][i] for i in plan["epochs"][index]]
        actual = {"epoch": index, "groups": list(observed_order), "order_matches_plan": observed_order == expected}
        observed_epochs.append(actual)
        write(root / "evidence" / (arm + "-orders.json"), observed_epochs)
        if not actual["order_matches_plan"]:
            raise ValueError("actual training order differs from frozen plan")
        observed_order.clear()
    original_evaluate = trainer.evaluate
    def traced(model, token, rows, hparams, train_negatives=None):
        # This original evaluation must consume exactly its usual RNG draws.
        original = original_evaluate(model, token, rows, hparams, train_negatives)
        if rows is not splits["validation"]:
            raise ValueError("unexpected trainer validation input")
        if len(history) >= hp.epochs:
            raise ValueError("snapshot ceiling")
        with isolated_rng(torch, np):
            snapshot = save_snapshot(root, arm + "-epoch-" + str(epoch_state["epoch"]), model, token, trainer)
            write(root / "evidence" / (snapshot["snapshot"] + "-saved.json"), snapshot)
            metrics = score_views(trainer, model, token, formats, hp)
        record = {"arm": arm, **epoch_state, "original_mixed_validation": original,
                  "formats": metrics, **snapshot, "extra_rng_restored": True}
        history.append(record)
        write(root / "evidence" / (arm + "-validation-history.json"), history)
        return original
    private = dict(trainer.__dict__)
    original_build = trainer.build_model
    def observed_build(base, hparams):
        model, token = original_build(base, hparams)
        frozen = json.loads((root / "evidence" / "training-start.json").read_text())
        directory = root / "adapters" / "training-start"
        if sha((directory / "adapter_model.safetensors").read_bytes()) != frozen["adapter_sha256"] or sha((directory / "adapter_config.json").read_bytes()) != frozen["config_sha256"]:
            raise ValueError("frozen initial adapter identity changed")
        reload_snapshot(root, "training-start", model)
        restore_rng(torch, np, initial["rng"])
        names = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
        write(root / "evidence" / (arm + "-trainable.json"), {"parameters": names, "count": sum(names.values())})
        return model, token
    def start(model, optim, sched):
        actual = boundary(root, model, torch, np, plan, loader_rng)
        actual["match"] = require_match(initial, actual)
        actual["new_baseline"] = True
        actual["fresh_optimizer"] = not optim.state
        actual["initial_lr"] = optim.param_groups[0]["lr"]
        actual["scheduler_epoch"] = sched.last_epoch
        write(root / "evidence" / (arm + "-initial.json"), actual)
        if not actual["fresh_optimizer"] or actual["initial_lr"] != 0 or sched.last_epoch != 0:
            raise ValueError("new arm must start with a fresh zero-step optimizer/scheduler")
    private.update(evaluate=traced, diagnostic_step=step, diagnostic_epoch=epoch,
                   PairCollator=ObservedCollator, build_model=observed_build,
                   diagnostic_sampler=FrozenSampler(), diagnostic_loader_rng=loader_rng, diagnostic_start=start)
    source = Path(trainer.__file__).read_text()
    train, overlay_text = overlay(source, private, matched=True)
    (root / "evidence" / (arm + "-trainer-overlay.py")).write_text(overlay_text + "\n")
    model, tok, _ = train(str(root / "base"), splits["train"], splits["validation"], root / "best" / arm, hp)
    # Verify the trainer's best-loss restoration against its disk snapshot.
    best_loss, best_index = math.inf, 0
    for i, r in enumerate(history):
        loss = r["original_mixed_validation"]["val_loss"]
        if i == 0 or loss < best_loss - hp.es_min_delta:
            best_loss, best_index = loss, i
    best = history[best_index]
    from peft import get_peft_model_state_dict
    before = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    reload_result = reload_snapshot(root, best["snapshot"], model)
    after = get_peft_model_state_dict(model)
    if set(before) != set(after) or any(not torch.equal(before[k], after[k]) for k in before):
        raise ValueError("trainer best-epoch restoration did not match saved snapshot")
    with isolated_rng(torch, np):
        replay = score_views(trainer, model, tok, formats, hp)
    if replay != best["formats"]:
        raise ValueError("saved snapshot validation margin replay changed")
    write(root / "evidence" / (arm + "-new-baseline.json"), {
          "scope": "New within-job arm, not historical reproduction", "historical_comparison_valid": False,
          "initial_state_sha256": initial["initial_state_sha256"], "plan_sha256": initial["plan_sha256"]})
    write(root / "evidence" / (arm + "-restoration.json"), {**reload_result, "best_epoch": best["epoch"],
          "validation_replay_equal": True, "optimizer_calls_executed": len(telemetry),
          "original_trainer_global_unchanged": trainer.evaluate is original_evaluate})
    return history


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(CONTRACT["arms"]), required=True)
    args = parser.parse_args()
    fit(args.root, args.arm)
