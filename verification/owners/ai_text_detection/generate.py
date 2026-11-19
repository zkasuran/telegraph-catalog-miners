# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Remote-only source-bound generation. Audit generation is a separate step."""
import argparse
import gc
import json
import os
import re
import time
from pathlib import Path

from common import HERE, audit_excerpts, dump, fetch, identity, remote_only, sha
from source import parse_article


def main():
    remote_only()  # before model dependencies or downloads
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--partition", required=True, choices=("development", "audit"))
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = json.loads((HERE / "sources.json").read_text())
    wanted = ("train", "validation") if args.partition == "development" else ("test",)
    if args.partition == "audit":
        if not (root / "frozen-candidate.json").exists():
            raise ValueError("audit generation requires a frozen candidate")
        frozen = json.loads((root / "frozen-candidate.json").read_text())
        if frozen.get("validation_gate", {}).get("passed") is not True:
            raise ValueError("audit remains sealed until every validation format passes")
    name = "gpt2" if args.partition == "development" else "distilgpt2"
    generator = spec["generators"][name]
    model_dir = root / ("generator-" + name)
    model_dir.mkdir(exist_ok=True)
    for filename, expected in generator["files"].items():
        fetch(f"https://huggingface.co/{generator['id']}/resolve/{generator['revision']}/{filename}",
              model_dir / filename, expected, 600 << 20)
    if (model_dir / "model.safetensors").stat().st_size != generator["weight_bytes"]:
        raise ValueError("generator weight length mismatch")
    # Safe source verification precedes model import. Each raw source must still
    # produce the exact pinned source record, including rights and attribution.
    for split in wanted:
        for row in spec["source_report"]["articles"][split]:
            path = root / row["path"]
            raw = fetch(row["url"], path, row["sha256"], 2 << 20)
            parsed = parse_article(raw, {"sha": row["git_blob"], "size": row["bytes"], "path": row["path"]})
            if parsed != row:
                raise ValueError("source manifest differs from original article")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                               trust_remote_code=False, use_safetensors=True,
                                               dtype=torch.float32).eval()
    started, count = time.monotonic(), 0
    log = root / (args.partition + "-generation.jsonl")
    if log.exists():
        raise ValueError("generation output already exists; resume without replacing it")
    # Fixed source order and seed. Never retry a candidate for a favourable score.
    with log.open("w") as output:
        for split in wanted:
            for row in spec["source_report"]["articles"][split]:
                seed = int(row["group_id"][:8], 16)
                torch.manual_seed(seed)
                inputs = tok(row["prefix"], return_tensors="pt")
                prefix_tokens = inputs["input_ids"].shape[1]
                if not 1 <= prefix_tokens <= 64:
                    raise ValueError("prefix token ceiling")
                if count >= 224 or time.monotonic() - started > 1100:
                    raise ValueError("generation sample/time ceiling")
                with torch.no_grad():
                    tokens = model.generate(**inputs, do_sample=True, top_p=0.9, top_k=50,
                                            temperature=0.9, min_new_tokens=64, max_new_tokens=64,
                                            pad_token_id=tok.eos_token_id)
                continuation = tokens[0, prefix_tokens:].tolist()
                decoded = tok.decode(continuation, skip_special_tokens=True)
                # Diagnostic counts are retained even if the unchanged byte or
                # identity gate refuses this source. No generated text becomes
                # a local fixture or is selected by reward-model outcomes.
                dump(root / (args.partition + "-generation-progress.json"),
                     {"source_id": row["source_id"], "generated_groups": count,
                      "decoded_bytes": len(decoded.encode()), "generated_words": len(decoded.split()),
                      "seed": seed, "generator": name})
                spans = list(re.finditer(r"\S+", decoded))
                if len(spans) < 24:
                    raise ValueError("fixed generation produced fewer than 24 words")
                if args.partition == "audit":
                    human_text, text, view = audit_excerpts(row["human_text"], decoded)
                else:
                    # Original development bytes are deliberately unchanged for
                    # safe continuation of the source-fitted checkpoint.
                    human_text, text = row["human_text"], decoded[spans[0].start():spans[23].end()]
                    view = {"word_count": 24, "changes": "Original development view retained."}
                    if not 32 <= len(text.encode()) <= 256:
                        raise ValueError("development excerpt exceeds unchanged byte gate")
                    if identity(text) == identity(human_text):
                        raise ValueError("development origins have identical visible text")
                base = {"source_id": row["source_id"], "group_id": row["group_id"],
                        "split": split, "year": row["year"], "source_sha256": row["sha256"]}
                human = {**base, "origin": "human_control", "text": human_text}
                if args.partition == "audit":
                    human["excerpt_view"] = view
                synthetic = {**base, "origin": "synthetic", "text": text, "generator": name,
                             "generator_revision": generator["revision"],
                             "generator_weight_sha256": generator["weight_sha256"], "seed": seed,
                             "prefix_token_ids": inputs["input_ids"][0].tolist(),
                             "continuation_token_ids": continuation, "decoded_continuation": decoded,
                             "generation": {"top_p": 0.9, "top_k": 50, "temperature": 0.9, "new_tokens": 64},
                             "disclosure": "Synthetic text created using " + name + ". Not human-reviewed.",
                             "licence": row["licence"], "source_attribution": row["url"]}
                if args.partition == "audit":
                    synthetic["excerpt_view"] = view
                for item in (human, synthetic):
                    output.write(json.dumps(item, ensure_ascii=False) + "\n")
                output.flush()
                count += 1
                print(json.dumps({"partition": args.partition, "groups_generated": count}), flush=True)
    del model, tok
    gc.collect()
    dump(root / (args.partition + "-generation-report.json"),
         {"partition": args.partition, "groups": count, "generator": generator,
          "source_manifest_sha256": sha((HERE / "sources.json").read_bytes()),
          "output_sha256": sha(log.read_bytes()), "elapsed_seconds": time.monotonic() - started,
          "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"]})


if __name__ == "__main__":
    main()
