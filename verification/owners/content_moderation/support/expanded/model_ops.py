# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only operations. Frameworks are imported inside guarded entry points."""
import json
import random
import struct
from pathlib import Path

from core import CONTRACT, FORMATS, load_shared, remote_only, sha, summary, write


def runtime(root, epochs):
    remote_only()
    import numpy as np
    import torch
    if torch.cuda.is_available() or torch.__version__ != "2.15.0+cpu":
        raise RuntimeError("only the pinned CPU Torch build is allowed")
    torch.set_num_threads(4)
    kit = root / ("telegraph-wasm-baseline-" + CONTRACT["kit_revision"])
    import sys
    sys.path.insert(0, str(kit))
    from float_runtime import train_adapter as trainer
    if sha((kit / "float_runtime" / "train_adapter.py").read_bytes()) != CONTRACT["trainer_sha256"]:
        raise ValueError("trainer hash mismatch")
    hp = trainer.Hyperparams(**CONTRACT["hyperparameters"], epochs=epochs)
    return torch, np, trainer, hp


def read_development(root):
    data, _, _ = load_shared()
    splits = {}
    for split in ("train", "validation"):
        raw = (root / "originals" / (split + ".jsonl")).read_bytes()
        if sha(raw) != json.loads((root / "evidence/source-manifest.json").read_text())["sources"]["source_files"][split]["sha256"]:
            raise ValueError("development source hash mismatch")
        splits[split] = [{**r, "margin": 0.0} for r in data.decode_jsonl(raw)]
    if len(splits["train"]) != 1536 or len(splits["validation"]) != 256:
        raise ValueError("expanded source split counts changed")
    data.check_splits(splits)
    return splits


def views(rows):
    data, _, _ = load_shared()
    return {name: [{**data.pair("CONTENT_MODERATION", {**r, "text": json.loads(r["prompt"])["text"]}, style), "margin": 0.0}
                   for r in rows] for style, name in enumerate(FORMATS)}


def tokenize(trainer, tok, node, sets, report_path):
    if not trainer.TOKENIZER_VECTORS.is_file():
        raise ValueError("missing tokenizer conformance vectors")
    trainer.check_tokenizer(tok)
    count, lengths = 0, []
    for rows in sets:
        for row in rows:
            for side in ("chosen", "rejected"):
                rendered = trainer.render(row["prompt"], row[side])
                ids = tok(rendered, truncation=False)["input_ids"]
                if len(ids) > 256 or ids != node.encode(rendered).ids:
                    raise ValueError("token identity or complete-claim ceiling failed")
                lengths.append(len(ids))
                count += 1
    write(report_path, {"matches_node": True, "rendered_texts": count, "max_tokens": max(lengths),
                        "min_tokens": min(lengths), "mean_tokens": sum(lengths) / count,
                        "tokenizer_sha256": load_shared()[1].FOUNDATION["tokenizer.json"]})


def tokenizer(root):
    remote_only()
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(root / "base", local_files_only=True)
    node = Tokenizer.from_file(str(root / "foundation" / "tokenizer.json"))
    node.no_padding()
    node.enable_truncation(max_length=256)
    return tok, node


def score(trainer, model, tok, rows, hp):
    remote_only()
    import torch
    from torch.utils.data import DataLoader
    device = next(model.parameters()).device
    if device.type != "cpu":
        raise RuntimeError("non-CPU model")
    model.eval()
    margins = []
    loader = DataLoader(rows, batch_size=hp.eval_batch, shuffle=False,
                        collate_fn=trainer.PairCollator(tok, hp.max_length))
    with torch.no_grad():
        for b in loader:
            rc = trainer._reward(model, b["chosen_input_ids"], b["chosen_attention_mask"])
            rr = trainer._reward(model, b["rejected_input_ids"], b["rejected_attention_mask"])
            # The inspected loss is still exercised. Pure summaries use its exact formula.
            if not torch.isfinite(trainer.bt_loss(rc.float(), rr.float(), b["margin"])):
                raise ValueError("nonfinite BT loss")
            margins.extend((rc - rr).float().tolist())
    result = summary(rows, margins)
    result["margin_bits_sha256"] = sha(b"".join(struct.pack("<f", m) for m in margins))
    return result


def score_views(trainer, model, tok, formats, hp):
    return {name: score(trainer, model, tok, rows, hp) for name, rows in formats.items()}


def save_snapshot(root, name, model, tok, trainer):
    _data, _, preserve = load_shared()
    directory = root / "adapters" / name
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tok.save_pretrained(directory)
    trainer.save_reward_head(model, directory)
    licences = json.loads((Path(__file__).parent.parent / "training" / "licences.json").read_text())
    tensor = directory / "adapter_model.safetensors"
    output = root / "evidence" / (name + ".safetensors")
    output.write_bytes(preserve.stamp(tensor.read_bytes(), licences))
    config = directory / "adapter_config.json"
    (root / "evidence" / (name + "-config.json")).write_bytes(config.read_bytes())
    return {"snapshot": name, "checkpoint_sha256": sha(output.read_bytes()),
            "adapter_sha256": sha(tensor.read_bytes()), "config_sha256": sha(config.read_bytes())}


def seed(torch, np, hp):
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)


def reload_snapshot(root, name, model):
    remote_only()
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    expected = load_file(str(root / "adapters" / name / "adapter_model.safetensors"), device="cpu")
    result = set_peft_model_state_dict(model, expected, adapter_name="default")
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if unexpected or any("lora_" in key or "modules_to_save.default" in key for key in missing):
        raise ValueError("adapter reload missing or unexpected trainable keys")
    actual = get_peft_model_state_dict(model)
    if set(actual) != set(expected) or any(not torch.equal(actual[k].cpu(), expected[k]) for k in expected):
        raise ValueError("snapshot reload changed parameter bytes")
    return {"parameter_bytes_equal": True, "missing_keys": missing, "unexpected_keys": unexpected,
            "missing_scope": "Frozen backbone keys may be absent from an adapter-only checkpoint."}
