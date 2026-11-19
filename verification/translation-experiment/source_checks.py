# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Tokenizer and dataset-gate controls only. No neural training or inference."""
import argparse
import hashlib
import importlib
import json
import random
import string
import subprocess
import sys
from pathlib import Path

from source_probe import LINKS_SHA, SENTENCES, sentences, sha, write

KIT_REVISION = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
TOKENIZER_SHA = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"
LINK_IDENTITIES_SHA = "bbcf61daff786fc4acf512e098848b74ed49989cd3c1da34a3747d66dc021c97"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--source-cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    trainer_path = args.kit / "float_runtime/train_adapter.py"
    pinned = subprocess.run(["git", "show", KIT_REVISION + ":float_runtime/train_adapter.py"],
                            cwd=args.kit, capture_output=True, check=True, timeout=15).stdout
    if trainer_path.read_bytes() != pinned or sha(args.tokenizer) != TOKENIZER_SHA:
        raise ValueError("pinned kit source or tokenizer identity mismatch")
    sys.path.insert(0, str(args.kit.resolve()))
    trainer = importlib.import_module("float_runtime.train_adapter")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(args.tokenizer))
    tok.no_padding()
    tok.no_truncation()
    vectors = json.loads((args.kit / "float_runtime/testdata/tokenizer_vectors.json").read_text())["vectors"]
    if not vectors:
        raise ValueError("empty conformance vectors")
    for vector in vectors:
        if tok.encode(vector["text"], add_special_tokens=vector["add_special_tokens"]).ids != vector["ids"]:
            raise ValueError("tokenizer conformance mismatch")
    originals = {}
    for lang in SENTENCES:
        originals.update(sentences(args.source_cache / (lang + "-cc0.tsv.bz2"), lang))
    identities_path = args.out / "link-identities.json"
    if sha(identities_path) != LINK_IDENTITIES_SHA:
        raise ValueError("verified primary link identity manifest hash mismatch")
    identities = json.loads(identities_path.read_text())
    archive = args.out / "links.tar.bz2"
    if sha(archive) != LINKS_SHA or identities["source_sha256"] != LINKS_SHA:
        raise ValueError("original link archive hash mismatch")
    links = identities["eng_fra_original_ids"]
    if not links:
        raise ValueError("no original aligned identity probes")
    lengths, rendered = [], hashlib.sha256()
    request_oversize = 0
    for left, right in links:
        for source, target in ((left, right), (right, left)):
            text, answer = originals[source]["text"], originals[target]["text"]
            target_language = "French" if originals[target]["language"] == "fra" else "English"
            prompt = json.dumps({"text": text, "target_language": target_language}, ensure_ascii=False)
            request_oversize += len(prompt.encode()) > 256
            for response in (answer, json.dumps({"translation": answer}, ensure_ascii=False),
                             "Translation: " + answer):
                rendered_text = trainer.render(prompt, response)
                lengths.append(len(tok.encode(rendered_text).ids))
                rendered.update(rendered_text.encode())
    # A deliberately oversized common task loses the differing answer at 256.
    full = [tok.encode(trainer.render("context " * 1024, s)).ids for s in ("alpha", "omega")]
    retained = [v[:255] + v[-1:] for v in full]
    if full[0] == full[1] or retained[0] != retained[1]:
        raise ValueError("truncation collision control did not fire")
    hp = trainer.Hyperparams(max_length=256)
    # Random equal-length strings test the real gate. They are not translations,
    # labels or corpus rows and must never enter a reward-model experiment.
    rng = random.Random(891)
    def control_text():
        return " ".join("".join(rng.choice(string.ascii_lowercase) for _ in range(5)) for _ in range(8))
    controls = [{"prompt": "synthetic gate control", "chosen": control_text(), "rejected": control_text()}
                for _ in range(200)]
    reports = {"synthetic_balanced": trainer.gate_dataset(controls, hp),
               "empty_actual_preferences": trainer.gate_dataset([], hp),
               "reused_negative": trainer.gate_dataset([{**r, "rejected": "fixed"} for r in controls], hp),
               "identical": trainer.gate_dataset([{**r, "rejected": r["chosen"]} for r in controls], hp),
               "length": trainer.gate_dataset([{**r, "chosen": r["chosen"] * 4} for r in controls], hp)}
    if not reports["synthetic_balanced"]["ok"] or any(reports[k]["ok"] for k in reports if k != "synthetic_balanced"):
        raise ValueError("dataset gate positive or negative control failed")
    if any(name in sys.modules for name in ("torch", "transformers", "peft", "onnxruntime")):
        raise RuntimeError("source check imported neural tooling")
    result = {"state": "SOURCE_CONTROLS_PASS_DATASET_BLOCKED", "kit_revision": KIT_REVISION,
              "trainer_sha256": sha(trainer_path), "tokenizer_sha256": TOKENIZER_SHA,
              "tokenizer_vectors": len(vectors), "vector_mismatches": 0,
              "original_linked_renders": len(lengths), "render_sha256": rendered.hexdigest(),
              "primary_link_archive_sha256": LINKS_SHA,
              "primary_identity_manifest_sha256": LINK_IDENTITIES_SHA,
              "maximum_original_render_tokens": max(lengths), "renders_above_256_tokens": sum(n > 256 for n in lengths),
              "bidirectional_requests_above_256_bytes": request_oversize,
              "long_prompt_collision_rejected": True, "dataset_gate_controls": reports,
              "dataset_ready": False, "model_execution": False,
              "scope": "Original linked source render probes only. No preferences, scores or bilingual quality result."}
    write(args.out / "source-checks.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
