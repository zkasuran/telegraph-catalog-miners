# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only source-origin supervision through a separate unexported readout.

[MODERNBERT-5.22.1] and [PEFT-BASE-0.21.2] define the adapted encoder API.
Source text prefixes have equal byte lengths within each paired source view.
"""
import json
import os
from common import remote_only, dump, HERE, sha
from origin_contract import check_origin_training, origin_policy, fit_length, length_accuracy


class OriginAuxiliary:
    def __init__(self, model, tok, rows, validation, recipe, root):
        remote_only()
        import torch
        self.torch, self.model, self.tok, self.root = torch, model, tok, root
        self.policy = origin_policy(recipe)
        train, self.validation = check_origin_training(rows, validation)
        self.binding = {"run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
             "source_manifest_sha256": sha((HERE / "sources.json").read_bytes()),
             "development_sha256": sha((root / "development-generation.jsonl").read_bytes()),
             "augmentation_sha256": sha((root / "augmentation-generation.jsonl").read_bytes()),
             "additional_augmentation_sha256": sha((root / "additional-augmentation-generation.jsonl").read_bytes()),
             "training_view_sha256": sha(json.dumps(train, sort_keys=True, ensure_ascii=False).encode())}
        self.by_original_id = {r["original_text_id"]: r for r in train}
        baseline = fit_length(train)
        if length_accuracy(train, baseline) != 0.5 or length_accuracy(self.validation, baseline) != 0.5:
            raise ValueError("equal-byte origin auxiliary retained a length shortcut")
        base = model.get_base_model()
        if base.config.model_type != "modernbert":
            raise ValueError("origin readout requires pinned ModernBERT encoder")
        self.encoder = base.model
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1042)
            self.head = torch.nn.Linear(base.config.hidden_size, 1).to(dtype=torch.float32, device="cpu")
        self.assert_excluded()
        from tokenizers import Tokenizer
        node = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
        node.no_padding()
        node.no_truncation()
        for view in train + self.validation:
            ids = tok(view["input"], truncation=False)["input_ids"]
            if not 1 <= len(ids) <= 256 or ids != node.encode(view["input"]).ids:
                raise ValueError("origin-only tokenizer identity or token ceiling mismatch")
        dump(root / "origin-source-contract.json", {**self.binding, "policy": self.policy, "training_passages": len(train),
             "training_groups": 128, "validation_groups": 32, "normalized_overlap": 0,
             "auxiliary_length_baseline": baseline, "controlled_length_accuracy": 0.5,
             "original_training_pairs_unchanged": True, "audit_used": False,
             "head_in_node_package": False,
             "provenance": "Source annotations on journal controls and recorded GPT-2 outputs, not human preference review."})
        self.diagnostics = []
        # Real primary-path gradient smoke. No optimizer update, no held-out row.
        view = train[0]
        model.zero_grad(set_to_none=True)
        loss, _ = self.loss(view)
        loss.backward()
        touched = [name for name, p in model.named_parameters() if "lora_" in name and p.grad is not None and torch.count_nonzero(p.grad).item()]
        if (not touched or self.head.weight.grad is None or not torch.count_nonzero(self.head.weight.grad).item() or
                any(not torch.isfinite(p.grad).all() for p in list(model.parameters()) + list(self.head.parameters()) if p.grad is not None)):
            raise ValueError("origin-only loss did not produce finite adapted-encoder/head gradients")
        dump(root / "origin-gradient-control.json", {**self.binding, "passed": True, "nonzero_lora_gradient_tensors": len(touched),
             "training_only_head_gradient_nonzero": True, "optimizer_updates": 0,
             "head_excluded_from_model": True, "actual_training_text_id": view["original_text_id"],
             "source_only_input": True, "audit_used": False})
        model.zero_grad(set_to_none=True)
        self.head.zero_grad(set_to_none=True)

    def assert_excluded(self):
        remote_only()
        if {id(p) for p in self.head.parameters()}.intersection(id(p) for p in self.model.parameters()):
            raise ValueError("training-only origin head attached to production model")
        from peft import get_peft_model_state_dict
        if len(get_peft_model_state_dict(self.model, save_embedding_layers=False)) != 178:
            raise ValueError("origin head changed the expected learned adapter tensor inventory")

    def loss(self, view):
        remote_only()
        torch = self.torch
        encoded = self.tok(view["input"], truncation=False, return_tensors="pt")
        hidden = self.encoder(**encoded).last_hidden_state
        if hidden.ndim != 3 or hidden.shape[0] != 1 or not torch.isfinite(hidden).all():
            raise ValueError("invalid actual adapted encoder output")
        logit = self.head(hidden[:, 0]).squeeze(-1)
        target = self.policy["human_target"] if view["human"] else self.policy["synthetic_target"]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.full_like(logit, target))
        if not torch.isfinite(loss).all():
            raise ValueError("nonfinite source-origin loss")
        return loss, int((logit.detach().item() > 0) == bool(view["human"]))

    def save(self):
        remote_only()
        self.assert_excluded()
        from safetensors.torch import save_file
        sources = json.loads((HERE / "sources.json").read_text())
        meta = {"x-terms-" + k: item["text"] for k, item in sources["terms"].items()}
        meta["x-control"] = "Training-only source-origin readout. Excluded from the production adapter and node package."
        save_file({k: v.detach().cpu().contiguous() for k, v in self.head.state_dict().items()},
                  str(self.root / "origin-head.safetensors"), metadata=meta)

    def measure(self, epoch, training_loss, training_correct, n):
        remote_only()
        self.head.eval()
        results = {0: [], 1: []}
        with self.torch.no_grad():
            for row in self.validation:
                _, correct = self.loss(row)
                results[row["human"]].append(correct)
        report = {**self.binding, "epoch": epoch, "training_loss": training_loss / n, "training_accuracy": training_correct / n,
             "validation_accuracy": sum(sum(v) for v in results.values()) / 64,
             "validation_by_origin": {str(k): sum(v) / len(v) for k, v in results.items()},
             "scope": "Native training-only readout diagnostic, not node runtime or candidate selection.",
             "used_for_selection": False, "audit_used": False}
        self.diagnostics.append(report)
        dump(self.root / "origin-training-report.json", {"epochs": self.diagnostics, "used_for_selection": False})
        return report
