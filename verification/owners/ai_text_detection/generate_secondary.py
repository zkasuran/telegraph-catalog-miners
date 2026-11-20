# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Fixed-seed public CPU generation on a new independent licensed training block."""
import argparse
import gc
import json
import os
import time
from pathlib import Path

from common import remote_only, fetch, dump, sha, audit_excerpts, identity, lines, HERE
from secondary_contract import expanded_source_bundle, validate_secondary
from generation_resume import retained_prefix, generation_policy, check_token_prefix
from source import parse_article


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root.resolve()
    bundle, secondary, _, base = expanded_source_bundle()
    output = root / "secondary-generation.jsonl"
    if output.exists():
        raise ValueError("secondary generation already exists; do not reroll")
    locked = HERE / "recovery/locked-secondary.jsonl"
    if locked.exists():
        raw = locked.read_bytes()
        producer = json.loads((HERE / "recovery/locked-secondary-producer.json").read_text())
        lineage = json.loads((HERE / "recovery/locked-secondary-lineage.json").read_text())
        if sha(raw) != lineage["data_sha256"] or sha((HERE / "recovery/locked-secondary-producer.json").read_bytes()) != lineage["producer_sha256"] or producer["run_id"] != 44789054038:
            raise ValueError("completed secondary generation byte/producer binding changed")
        originals = []
        for name in ("development-generation.jsonl", "augmentation-generation.jsonl", "additional-augmentation-generation.jsonl", "supplement-generation.jsonl"):
            originals += [{"text_id": identity(r["text"])} for r in lines((root / name).read_bytes())]
        validate_secondary(raw, producer, originals)
        output.write_bytes(raw)
        journal_raw = (HERE / "recovery/locked-secondary-journal.jsonl").read_bytes()
        if sha(journal_raw) != producer["raw_candidate_journal_sha256"]:
            raise ValueError("completed secondary raw journal changed")
        (root / "secondary-generation-journal.jsonl").write_bytes(journal_raw)
        dump(root / "secondary-generation-report.json", {**producer, "producer_run_id": producer["run_id"],
            "producer_commit": producer["commit"], "generation_performed_here": False,
            "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"]})
        print(json.dumps({"restored_records": len(lines(raw)), "data_sha256": sha(raw), "new_generation": False}), flush=True)
        return
    policy = generation_policy(json.loads((HERE / "recipe.json").read_text()))
    retained_raw = (HERE / "recovery/secondary-prefix.jsonl").read_bytes()
    retained_producer = json.loads((HERE / "recovery/secondary-prefix-producer.json").read_text())
    retained = retained_prefix(retained_raw, retained_producer)
    retained_count = len(retained) // 2
    if retained_count != policy["retained_groups"]:
        raise ValueError("retained source-group count changed before generation")
    for row in secondary["articles"]:
        raw = fetch(row["url"], root / row["path"], row["sha256"], 2 << 20)
        if parse_article(raw, {"path": row["path"], "sha": row["git_blob"], "size": row["bytes"]}) != row:
            raise ValueError("secondary original grant/source changed")
    generator = base["generators"]["gpt2"]
    directory = root / "secondary-gpt2"
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
    seen = {identity(r["text"]) for r in retained}
    count, prefix_controls = retained_count, 0
    output.write_bytes(retained_raw)
    journal = root / "secondary-generation-journal.jsonl"
    if journal.exists():
        raise ValueError("secondary raw-candidate journal already exists")
    with output.open("ab") as log, journal.open("w") as history:
        for row in secondary["articles"][retained_count:]:
            if time.monotonic() - started > 1100 or count >= 64:
                raise ValueError("secondary generation time/count ceiling")
            seed = int(row["group_id"][:8], 16)
            torch.manual_seed(seed)
            encoded = tok(row["prefix"], return_tensors="pt")
            n = encoded["input_ids"].shape[1]
            if not 1 <= n <= 64:
                raise ValueError("secondary prefix token ceiling")
            with torch.no_grad():
                short_tokens = model.generate(**encoded, do_sample=True, top_p=0.9, top_k=50, temperature=0.9,
                    min_new_tokens=64, max_new_tokens=64, pad_token_id=tok.eos_token_id)
                torch.manual_seed(seed)
                tokens = model.generate(**encoded, do_sample=True, top_p=0.9, top_k=50, temperature=0.9,
                    min_new_tokens=128, max_new_tokens=128, pad_token_id=tok.eos_token_id)
            continuation = tokens[0, n:].tolist()
            decoded = tok.decode(continuation, skip_special_tokens=True)
            short = short_tokens[0, n:].tolist()
            raw_candidate = {"source_id": row["source_id"], "source_sha256": row["sha256"], "source_attribution": row["url"],
                "source_authors": row["authors"], "source_grant": row["grant_clause"], "source_licence": row["licence"],
                "generator": "gpt2", "generator_weight_sha256": generator["weight_sha256"], "seed": seed,
                "short_token_ids": short, "extended_token_ids": continuation, "decoded_continuation": decoded,
                "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
                "kind": "Raw GPT-2 candidate before excerpt acceptance, not human-reviewed preference data."}
            history.write(json.dumps(raw_candidate, ensure_ascii=False) + "\n")
            history.flush()
            check_token_prefix(short, continuation)
            prefix_controls += 1
            human, machine, crop = audit_excerpts(row["human_text"], decoded)
            if seen.intersection({identity(human), identity(machine)}):
                raise ValueError("secondary normalized duplicate; no score-based retry")
            common = {"source_id": row["source_id"], "group_id": row["group_id"], "split": "train", "year": row["year"],
                "source_sha256": row["sha256"], "licence": row["licence"], "source_attribution": row["url"],
                "excerpt_view": crop, "label_kind": "source_provenance_not_human_preference_review",
                "generation_producer_run_id": int(os.environ["GITHUB_RUN_ID"]), "generation_token_budget": 128}
            records = [{**common, "origin": "human_control", "text": human},
                {**common, "origin": "synthetic", "text": machine, "generator": "gpt2", "seed": seed,
                 "generator_weight_sha256": generator["weight_sha256"], "generator_revision": generator["revision"],
                 "prefix_token_ids": encoded["input_ids"][0].tolist(), "continuation_token_ids": continuation,
                 "decoded_continuation": decoded, "disclosure": "Actual bounded GPT-2 training output, not human-reviewed."}]
            for record in records:
                log.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
            log.flush()
            seen.update({identity(human), identity(machine)})
            count += 1
    del model, tok
    gc.collect()
    report = {"schema_version": 1, "groups": count, "output_sha256": sha(output.read_bytes()),
        "source_bundle_sha256": bundle, "secondary_source_manifest_sha256": sha((HERE / "secondary-training-sources.json").read_bytes()),
        "generator": generator, "audit_used": False, "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "commit": os.environ["GITHUB_SHA"], "elapsed_seconds": time.monotonic() - started,
        "retained_records": len(retained), "retained_groups": retained_count, "retained_sha256": sha(retained_raw),
        "retained_producer_run_id": retained_producer["run_id"], "retained_producer_commit": retained_producer["commit"],
        "remaining_generation_token_budget": 128, "same_job_prefix_controls": prefix_controls,
        "historical_rejected_draw_bytes_available": False,
        "raw_candidate_journal_sha256": sha(journal.read_bytes())}
    if prefix_controls != 64 - retained_count or not output.read_bytes().startswith(retained_raw):
        raise ValueError("generation did not preserve retained bytes and every new prefix control")
    originals = []
    for name in ("development-generation.jsonl", "augmentation-generation.jsonl", "additional-augmentation-generation.jsonl", "supplement-generation.jsonl"):
        originals += [{"text_id": identity(r["text"])} for r in lines((root / name).read_bytes())]
    validate_secondary(output.read_bytes(), report, originals)
    dump(root / "secondary-generation-report.json", report)
    print(json.dumps({"actual_groups": count, "output_sha256": report["output_sha256"], "audit_used": False}), flush=True)


if __name__ == "__main__":
    main()
