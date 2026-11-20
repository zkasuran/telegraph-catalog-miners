# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote actual-model repeat control with no optimizer or audit data access."""
import os
from common import remote_only, dump


def verify(model, tok, rows, hp, trainer, root):
    remote_only()
    import torch
    if hp.lora_dropout != 0.0 or not rows or rows[0]["split"] != "train":
        raise ValueError("zero-dropout control requires actual training input")
    active = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Dropout) and module.p != 0]
    if active:
        raise ValueError("nonzero model dropout remains: " + ",".join(active[:4]))
    batch = trainer.PairCollator(tok, hp.max_length)(rows[:1])
    model.train()
    with torch.no_grad():
        first = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
        second = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
        model.eval()
        evaluation = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
    if (not torch.isfinite(first).all() or not torch.equal(first, second) or not torch.equal(first, evaluation)):
        raise ValueError("zero-dropout native training/evaluation repeat control failed")
    dump(root / "comparison-dropout-control.json", {"passed": True,
        "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
        "lora_dropout": hp.lora_dropout, "nonzero_dropout_modules": len(active),
        "native_training_repeat_identical": True, "native_train_eval_identical": True,
        "actual_training_text_id": rows[0]["text_id"], "optimizer_updates": 0, "audit_used": False,
        "scope": "Actual native repeat only, not an independent quality or node-engine result."})
