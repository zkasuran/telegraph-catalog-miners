# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded remote-only generation for new windows of existing training articles."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

from common import HERE, audit_excerpts, dump, fetch, identity, lines, remote_only, sha
from source import parse_article
from window_contract import validate_plan


def main():
    remote_only()
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--additional", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    source_raw = (HERE / "sources.json").read_bytes()
    sources = json.loads(source_raw)
    plan_path = HERE / ("additional-train-window-plan.json" if args.additional else "train-window-plan.json")
    plan = json.loads(plan_path.read_text())
    articles = validate_plan(plan, sources, sha(source_raw))
    if args.additional and plan["prior_plan_sha256"] != sha((HERE / "train-window-plan.json").read_bytes()):
        raise ValueError("additional source windows do not bind original disjoint plan")
    prefix_name = "additional-augmentation" if args.additional else "augmentation"
    output = root / (prefix_name + "-generation.jsonl")
    if output.exists():
        raise ValueError("augmentation already exists; never replace or reroll it")
    locked_path = HERE / "recovery/locked-augmentation.jsonl"
    if locked_path.exists() and not args.additional:
        from window_contract import validate_augmented
        producer_raw = (HERE / "recovery/locked-augmentation-producer.json").read_bytes()
        producer = json.loads(producer_raw)
        lineage = json.loads((HERE / "recovery/locked-augmentation-lineage.json").read_text())
        raw = locked_path.read_bytes()
        if (sha(raw) != lineage["data_sha256"] or sha(producer_raw) != lineage["producer_report_sha256"] or
                lineage["source_manifest_sha256"] != sha(source_raw) or producer["run_id"] != lineage["producer_run_id"] or
                producer["commit"] != lineage["producer_commit"]):
            raise ValueError("locked actual augmentation byte/producer/source identity mismatch")
        restored = validate_augmented(raw, plan, sources, producer)
        output.write_bytes(raw)
        dump(root / "augmentation-generation-report.json", {**producer, "producer_run_id": producer["run_id"],
             "producer_commit": producer["commit"], "generation_performed_here": False,
             "preparation": "Byte-identical reuse of actual prior train-only generation; no new model generation claim.",
             "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"]})
        print(json.dumps({"restored_actual_train_records": len(restored), "data_sha256": sha(raw),
                          "new_generation_performed": False}), flush=True)
        return
    for article in articles.values():
        raw = fetch(article["url"], root / article["path"], article["sha256"], 2 << 20)
        parsed = parse_article(raw, {"sha": article["git_blob"], "size": article["bytes"], "path": article["path"]})
        if parsed != article:
            raise ValueError("actual original article changed before train-only generation")
    generator = sources["generators"]["gpt2"]
    model_path = root / "augmentation-gpt2"
    model_path.mkdir(exist_ok=True)
    for filename, expected in generator["files"].items():
        fetch(f"https://huggingface.co/{generator['id']}/resolve/{generator['revision']}/{filename}", model_path / filename, expected, 600 << 20)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True, dtype=torch.float32).eval()
    original = lines((root / "development-generation.jsonl").read_bytes())
    if args.additional:
        original += lines((root / "augmentation-generation.jsonl").read_bytes())
    seen = {identity(r["text"]) for r in original}
    accepted, refusals = 0, []
    started = time.monotonic()
    with output.open("w") as log:
        for attempt, view in enumerate(plan["windows"]):
            if attempt >= 128 or time.monotonic() - started > 1000:
                raise ValueError("train-only generation count/time ceiling")
            torch.manual_seed(view["seed"])
            prefix_words = view["prefix"].split()
            prefix = " ".join(prefix_words)
            inputs = tok(prefix, return_tensors="pt")
            while inputs["input_ids"].shape[1] > 64 and len(prefix_words) > 1:
                prefix_words = prefix_words[1:]
                prefix = " ".join(prefix_words)
                inputs = tok(prefix, return_tensors="pt")
            if not 1 <= inputs["input_ids"].shape[1] <= 64:
                raise ValueError("source prefix cannot fit unchanged token ceiling")
            with torch.no_grad():
                generated = model.generate(**inputs, do_sample=True, top_p=0.9, top_k=50, temperature=0.9,
                                           min_new_tokens=64, max_new_tokens=64, pad_token_id=tok.eos_token_id)
            continuation = generated[0, inputs["input_ids"].shape[1]:].tolist()
            decoded = tok.decode(continuation, skip_special_tokens=True)
            try:
                human, synthetic, crop = audit_excerpts(view["human_source_window"], decoded)
                ids = {identity(human), identity(synthetic)}
                if seen.intersection(ids):
                    raise ValueError("structural normalized-text duplicate")
            except ValueError as error:
                refusals.append({"view_id": view["view_id"], "source_id": view["source_id"], "seed": view["seed"], "reason": str(error)})
                continue
            base = {"source_id": view["source_id"], "group_id": view["group_id"], "view_id": view["view_id"],
                    "split": "train", "year": view["year"], "source_sha256": view["source_sha256"], "excerpt_view": crop,
                    "source_window": {"prefix_span": view["prefix_span"], "human_span": view["human_span"]},
                    "source_attribution": view["url"], "licence": view["licence"]}
            records = [{**base, "origin": "human_control", "text": human},
                       {**base, "origin": "synthetic", "text": synthetic, "generator": "gpt2",
                        "generator_revision": generator["revision"], "generator_weight_sha256": generator["weight_sha256"],
                        "seed": view["seed"], "prefix_used": prefix, "prefix_token_ids": inputs["input_ids"][0].tolist(),
                        "continuation_token_ids": continuation, "decoded_continuation": decoded,
                        "disclosure": "Actual bounded GPT-2 output for a training-only source window. Not human-reviewed."}]
            for record in records:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            seen.update(ids)
            accepted += 1
            print(json.dumps({"train_views_generated": accepted, "refused_views": len(refusals)}), flush=True)
    del model, tok
    gc.collect()
    report = {"schema_version": 1, "source_manifest_sha256": sha(source_raw), "plan_sha256": sha(plan_path.read_bytes()),
              "output_sha256": sha(output.read_bytes()), "generator": generator, "valid_views": accepted, "refusals": refusals,
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "audit_used": False,
              "elapsed_generation_seconds": time.monotonic() - started}
    dump(root / (prefix_name + "-generation-report.json"), report)
    if accepted < plan["minimum_valid_windows"]:
        raise ValueError("insufficient structurally valid new train windows; reports preserved")


if __name__ == "__main__":
    main()
