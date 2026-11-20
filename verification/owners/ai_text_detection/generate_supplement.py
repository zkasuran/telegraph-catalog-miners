# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded actual GPT-2 generation for independent training-origin sources only."""
import argparse
import gc
import json
import os
import time
from pathlib import Path
from common import HERE, remote_only, fetch, dump, sha, audit_excerpts, identity, lines
from supplement_contract import source_bundle, validate_sources, validate_generation
from source import parse_article


def main():
    remote_only()
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root.resolve()
    bundle_sha, extra, base = source_bundle()
    articles = validate_sources(extra, base)
    output = root / "supplement-generation.jsonl"
    if output.exists():
        raise ValueError("supplement already generated; never reroll or replace")
    locked = HERE / "recovery/locked-supplement.jsonl"
    if locked.exists():
        raw = locked.read_bytes()
        producer_raw = (HERE / "recovery/locked-supplement-producer.json").read_bytes()
        producer = json.loads(producer_raw)
        lineage = json.loads((HERE / "recovery/locked-supplement-lineage.json").read_text())
        if (sha(raw) != lineage["data_sha256"] or sha(producer_raw) != lineage["producer_sha256"] or
                producer["run_id"] != lineage["run_id"] or producer["commit"] != lineage["commit"] or
                lineage["source_bundle_sha256"] != bundle_sha):
            raise ValueError("retained supplement producer/byte/source identity mismatch")
        originals = []
        for name in ("development-generation.jsonl", "augmentation-generation.jsonl", "additional-augmentation-generation.jsonl"):
            originals += [{"text_id": identity(r["text"])} for r in lines((root / name).read_bytes())]
        views = validate_generation(raw, producer, extra, base, originals)
        output.write_bytes(raw)
        dump(root / "supplement-generation-report.json", {**producer, "producer_run_id": producer["run_id"],
            "producer_commit": producer["commit"], "generation_performed_here": False,
            "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"]})
        print(json.dumps({"restored_source_rows": len(views), "output_sha256": sha(raw), "new_generation_performed": False}), flush=True)
        return
    for row in articles:
        raw = fetch(row["url"], root / row["path"], row["sha256"], 2 << 20)
        if parse_article(raw, {"sha": row["git_blob"], "size": row["bytes"], "path": row["path"]}) != row:
            raise ValueError("supplementary original source/grant changed")
    generator = base["generators"]["gpt2"]
    directory = root / "supplement-gpt2"
    directory.mkdir(exist_ok=True)
    for name, expected in generator["files"].items():
        fetch(f"https://huggingface.co/{generator['id']}/resolve/{generator['revision']}/{name}", directory / name, expected, 600 << 20)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True, dtype=torch.float32).eval()
    started = time.monotonic()
    seen, count = set(), 0
    with output.open("w") as log:
        for row in articles:
            if time.monotonic() - started > 1100 or count >= 64:
                raise ValueError("supplement generation time/count ceiling")
            seed = int(row["group_id"][:8], 16)
            torch.manual_seed(seed)
            encoded = tok(row["prefix"], return_tensors="pt")
            n = encoded["input_ids"].shape[1]
            if not 1 <= n <= 64:
                raise ValueError("supplement source prefix token ceiling")
            with torch.no_grad():
                ids = model.generate(**encoded, do_sample=True, top_p=0.9, top_k=50, temperature=0.9,
                                     min_new_tokens=64, max_new_tokens=64, pad_token_id=tok.eos_token_id)
            continuation = ids[0, n:].tolist()
            decoded = tok.decode(continuation, skip_special_tokens=True)
            human, synthetic, crop = audit_excerpts(row["human_text"], decoded)
            marks = {identity(human), identity(synthetic)}
            if seen.intersection(marks):
                raise ValueError("supplement normalized text duplicate; no score-based retries")
            common = {"source_id": row["source_id"], "group_id": row["group_id"], "split": "train", "year": row["year"],
                      "source_sha256": row["sha256"], "licence": row["licence"], "source_attribution": row["url"],
                      "excerpt_view": crop, "label_kind": "source_provenance_not_human_preference_review"}
            records = [{**common, "origin": "human_control", "text": human},
                       {**common, "origin": "synthetic", "text": synthetic, "generator": "gpt2", "seed": seed,
                        "generator_weight_sha256": generator["weight_sha256"], "generator_revision": generator["revision"],
                        "prefix_token_ids": encoded["input_ids"][0].tolist(), "continuation_token_ids": continuation,
                        "decoded_continuation": decoded, "disclosure": "Actual bounded GPT-2 training output, not human-reviewed."}]
            for record in records:
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            seen.update(marks)
            count += 1
    del model, tok
    gc.collect()
    dump(root / "supplement-generation-report.json", {"schema_version": 1, "groups": count,
        "output_sha256": sha(output.read_bytes()), "source_bundle_sha256": bundle_sha,
        "extra_source_manifest_sha256": sha((HERE / "extra-training-sources.json").read_bytes()),
        "generator": generator, "audit_used": False, "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "commit": os.environ["GITHUB_SHA"], "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
