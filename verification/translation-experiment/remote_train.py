# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Fixed one-epoch, four-thread remote CPU experiment using unchanged kit helpers."""
import argparse
import ast
import bz2
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path

from checkpoint_terms import family_bundle, remote_guard, stamp
from pilot import COUNTS, checks, load_helper, render_pairs
from remote_runtime import (
    FAMILY,
    FOUNDATION,
    metrics,
    parser_check,
    quality,
    runtime,
    tools,
)
from source_probe import sha, write

HERE = Path(__file__).resolve().parent
SHARED_PREPARE_SHA256 = "5b2ba64a11ff141ac96a20c491fc282f917e722183b2ab67c17e6a12c8ff519b"


def trusted_prepare_ast(raw):
    """Select one function from exactly the pinned controller source bytes."""
    if hashlib.sha256(raw).hexdigest() != SHARED_PREPARE_SHA256:
        raise ValueError("trusted shared preparation source hash mismatch")
    tree = ast.parse(raw.decode())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
    return ast.Module(body=[function], type_ignores=[])


def source_check(root, manifest):
    from source_probe import LINKS_SHA, records

    shared = load_helper("translation_source_fetch", HERE.parent / "training/data.py")
    spec = json.loads((HERE / "general-sources.json").read_text())
    selected = manifest["originals"]
    source = root / "source"
    source.mkdir(exist_ok=True)
    for filename, pin in spec["files"].items():
        path = source / filename
        shared.fetch(pin["url"], path, pin["sha256"], 160 << 20)
        detailed = "detailed" in filename
        language = filename[:3]
        found = set()
        with bz2.open(path, "rb") as stream:
            for line, fields in records(stream, 6 if detailed else 3):
                sid = fields[0]
                if sid not in selected:
                    continue
                row = selected[sid]
                if fields[1] != language or fields[2] != row["text"] or line != row["detailed_source_row" if detailed else "source_row"]:
                    raise ValueError("primary original text/line/identity changed")
                if detailed and fields[3:] != [row["credited_contributor"], row["created"], row["modified"]]:
                    raise ValueError("primary attribution changed")
                found.add(sid)
        if found != {sid for sid, row in selected.items() if row["language"] == language}:
            raise ValueError("primary selected source identities missing")
    links = source / "links.tar.bz2"
    shared.fetch("https://downloads.tatoeba.org/exports/links.tar.bz2", links, LINKS_SHA, 160 << 20)
    expected = {(e, f) for blocks in manifest["splits"].values() for block in blocks for e, f in block["links"]}
    expected |= {(b, a) for a, b in expected}
    found = set()
    with tarfile.open(links, "r|bz2") as archive:
        member = next(iter(archive))
        if member.name != "links.csv":
            raise ValueError("primary link archive member changed")
        with archive.extractfile(member) as stream:
            for _, fields in records(stream, 2):
                edge = tuple(map(int, fields))
                if edge in expected:
                    found.add(edge)
    if found != expected:
        raise ValueError("primary reciprocal translation links missing")
    write(root / "source-check.json", {"originals": len(selected), "directed_original_links": len(found),
          "export_hashes_verified": True, "original_text_and_attribution_verified": True, "links_sha256": LINKS_SHA})


def preserve(root):
    remote_guard()
    bundle = family_bundle(json.loads((HERE / "rights.json").read_text()), json.loads((HERE / "pilot-manifest.json").read_text()))
    public = root / "public"
    public.mkdir(exist_ok=True)
    for directory in ("initial", "checkpoint", "adapter"):
        source = root / directory
        for path in source.glob("*.safetensors"):
            output = public / (directory + "-" + path.name)
            partial = output.with_suffix(".partial")
            partial.write_bytes(stamp(path.read_bytes(), bundle))
            partial.replace(output)
        config = source / "adapter_config.json"
        if config.exists():
            shutil.copyfile(config, public / (directory + "-adapter_config.json"))
        head = source / "reward_head.pt"
        if head.exists():
            if head.stat().st_size > 2 << 20:
                raise ValueError("reward head checkpoint ceiling")
            output = public / (directory + "-reward_head.pt")
            partial = output.with_suffix(".partial")
            shutil.copyfile(head, partial)
            with zipfile.ZipFile(partial, "a") as archive:
                prefix = archive.namelist()[0].split("/")[0]
                archive.writestr(prefix + "/NOTICE-TATOEBA.json", json.dumps(bundle, ensure_ascii=False))
            partial.replace(output)
    shutil.copyfile(HERE / "rights.json", public / "rights.json")
    shutil.copyfile(HERE / "pilot-manifest.json", public / "pilot-manifest.json")


def prepare(root, source_only=False):
    shared = HERE.parent / "training"
    data = load_helper("translation_shared_data", shared / "data.py")
    pin = data.SOURCES["kit"]
    archive = root / "kit.tar.gz"
    data.fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + pin["revision"], archive, pin["sha256"], 32 << 20)
    with tarfile.open(archive) as source:
        if sum(m.size for m in source.getmembers()) > 128 << 20:
            raise ValueError("reference kit expansion ceiling")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + pin["revision"])
    if source_only:
        return kit
    sys.path.insert(0, str(HERE.parent))
    from prepare import RELEASE
    raw = (shared / "run.py").read_bytes()
    trusted = trusted_prepare_ast(raw)
    env = {"FOUNDATION": FOUNDATION, "RELEASE": RELEASE, "SOURCES": data.SOURCES, "fetch": data.fetch,
           "tarfile": tarfile, "sys": sys, "os": os}
    # Execute only the prepare function from these verified controller bytes.
    # No module-level code or neural import is executed by this extraction.
    exec(compile(trusted, "unchanged-shared-reference-prepare", "exec"), env)  # noqa: S102
    return env["prepare"](root)


def main():
    remote_guard()  # Before even importing or accessing any neural tooling.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--preserve-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("remote experiment root is outside RUNNER_TEMP")
    root.mkdir(exist_ok=True)
    if args.preserve_only:
        preserve(root)
        return
    manifest = json.loads((HERE / "pilot-manifest.json").read_text())
    report = json.loads((HERE / "pilot-checks.json").read_text())
    if report["pilot_manifest_sha256"] != sha(HERE / "pilot-manifest.json") or not manifest["source_review_before_split"]:
        raise ValueError("reviewed final source boundary changed")
    source_check(root, manifest)
    kit = prepare(root, True)
    sys.path.insert(0, str(kit))
    from float_runtime import train_adapter as trainer
    splits = {s: render_pairs(manifest, s) for s in COUNTS}
    gates = {s: trainer.gate_dataset(rows, trainer.Hyperparams(max_length=256)) for s, rows in splits.items()}
    write(root / "dataset-gate.json", gates)
    if not all(g["ok"] for g in gates.values()):
        raise ValueError("unchanged upstream data gate failed before any model")
    tools(root, kit)
    parser_check(root, True)
    foundation, base, kit = prepare(root)
    import numpy as np
    import torch
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import (
        metadata,
        target_dims,
        tensor_names,
        write_package,
    )
    from tokenizers import Tokenizer
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoTokenizer

    if torch.cuda.is_available():
        raise RuntimeError("fixed public CPU experiment refuses GPU execution")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    hp = trainer.Hyperparams(max_length=256, batch_size=1, grad_accum=16, epochs=1, eval_batch=1,
                            lora_rank=8, lora_alpha=32, lora_dropout=0.1, lr=1e-4,
                            gate_min_acc=0.80, grad_checkpointing=False)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    node = Tokenizer.from_file(str(foundation / "tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    checks(manifest, splits, node, trainer.render)
    trainer.check_tokenizer(tok)
    for rows in splits.values():
        for row in rows:
            for side in ("chosen", "rejected"):
                text = trainer.render(row["prompt"], row[side])
                if tok(text, truncation=False)["input_ids"] != node.encode(text).ids:
                    raise ValueError("native/released tokenizer mismatch")
    views = {name: render_pairs(manifest, "test", name) for name in ("raw", "json", "prose")}
    def seed():
        random.seed(hp.seed)
        np.random.seed(hp.seed)
        torch.manual_seed(hp.seed)
    def scores(model, rows):
        model.eval()
        result = []
        with torch.no_grad():
            for row in rows:
                for side in ("chosen", "rejected"):
                    encoded = tok(trainer.render(row["prompt"], row[side]), truncation=False, return_tensors="pt")
                    result.append(float(trainer._reward(model, encoded["input_ids"], encoded["attention_mask"]).item()))
        return result
    def save(model, directory):
        destination = root / directory
        model.save_pretrained(destination)
        trainer.save_reward_head(model, destination)
        preserve(root)
    seed()
    initial, _ = trainer.build_model(str(base), hp)
    save(initial, "initial")
    untrained = {name: metrics(rows, scores(initial, rows)) for name, rows in views.items()}
    del initial
    gc.collect()
    # Fixed one epoch. Final baseline cannot alter recipe or epoch selection.
    seed()
    model, tok = trainer.build_model(str(base), hp)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=hp.lr, weight_decay=hp.weight_decay)
    loader = DataLoader(splits["train"], batch_size=1, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(hp.seed), collate_fn=trainer.PairCollator(tok, 256))
    if len(loader) != 512 or len(loader) % hp.grad_accum:
        raise ValueError("fixed 32-step accumulation budget changed")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    timings, started = [], time.monotonic()
    try:
        for i, batch in enumerate(loader):
            rc = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rr = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = trainer.bt_loss(rc, rr, batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            (loss / hp.grad_accum).backward()
            if (i + 1) % hp.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                save(model, "checkpoint")
                timings.append((time.monotonic() - started) / len(timings + [0]))
                write(root / "progress.json", {"optimizer_steps": len(timings), "required_steps": 32,
                      "epoch_selection": "fixed one epoch, no final selection", "model_quality": "pending", "runtime": "pending"})
                if time.monotonic() - started + (32 - len(timings)) * max(timings) > 6900:
                    raise TimeoutError("measured CPU throughput exceeds fixed training reserve")
        save(model, "adapter")
        native = {name: scores(model, rows) for name, rows in views.items()}
        trained = {name: metrics(views[name], values) for name, values in native.items()}
        strongest = max(report["shortcut_baselines"]["by_split"]["test"].values())
        verdict = quality(trained, untrained, strongest)
        write(root / "result.json", {"family": FAMILY, **verdict, "formats": trained, "untrained": untrained,
              "source_manifest_sha256": report["pilot_manifest_sha256"], "run_id": os.environ["GITHUB_RUN_ID"],
              "commit": os.environ["GITHUB_SHA"], "parser": "pending", "runtime": "pending", "node_qualification": "not attempted"})
        if not verdict["passed"]:
            raise ValueError("source-held-out native quality gates failed")
        cfg = AutoConfig.from_pretrained(base, local_files_only=True)
        tensors, config = read_peft(root / "adapter", cfg)
        meta = metadata(cls="judgment", intent=FAMILY, foundation_hash=FOUNDATION["foundation_lora_fp32.onnx"],
            foundation_version="modernbert-base-lora16-v1", rank=8, alpha=32,
            target_modules=config["target_modules"], max_tokens=256,
            training={"family": FAMILY, "children": [{"intent": FAMILY + "_GENERAL", "dataset_version": report["pilot_manifest_sha256"],
                "pairs": 256, "accuracy": trained["raw"]["pairwise_accuracy"], "held_out": True}],
                "preference_authorship": "AI reviewed original CC BY source translations and donor contrasts before new split. Constructed preferences, not human votes."})
        bundle = family_bundle(json.loads((HERE / "rights.json").read_text()), manifest)
        meta.update({"x-terms-" + key: value for key, value in bundle.items()})
        package = root / (FAMILY + ".safetensors")
        write_package(package, meta, {name: tensors[name] for name in tensor_names(cfg.num_hidden_layers)})
        # Leave a source-bound candidate with explicit pending parser/runtime
        # state if the full integration later times out or refuses it.
        shutil.copyfile(package, root / "public" / package.name)
        write(root / "foundation.json", {"foundation_hash": meta["foundation_hash"], "tokenizer_hash": meta["tokenizer_hash"],
              "foundation_version": meta["foundation_version"], "base_model": str(base), "r_max": 16,
              "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
              "targets": {key: list(value) for key, value in target_dims(cfg).items()}})
        del model
        gc.collect()
        parsed = parser_check(root)
        runtime(root, parsed, native, untrained, strongest)
        completed = json.loads((root / "result.json").read_text())
        completed.update({"parser": "PASS_FULL_STAGE1", "runtime": "PASS_REFERENCE_ALL_FORMATS", "node_qualification": "not attempted"})
        write(root / "result.json", completed)
        shutil.copyfile(package, root / "public" / package.name)
    finally:
        preserve(root)


if __name__ == "__main__":
    main()
