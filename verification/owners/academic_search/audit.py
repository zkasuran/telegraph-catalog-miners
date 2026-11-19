# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Independent frozen audit, licensed source export and actual engine proof."""
import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

from baselines import replay as baseline_replay
from bootstrap import dump, file_sha
from checkpoint import licence_metadata, preserve_artifact
from data import FORMATS, render
from engine import package_identity, score_all, server
from live_class import capture
from overlap import replay as overlap_replay
from report import metrics, quality, ready
from scorer import FAMILY, load_freeze, remote_only
from source import HERE, sha
from wire import FOUNDATION, ENGINE, TOKENIZER, score_value, strict_json


def native_model(root, kit, selected=False):
    remote_only()
    sys.path.insert(0, str(kit))
    import numpy as np
    import torch
    from float_runtime import train_adapter as trainer
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    hp = trainer.Hyperparams(max_length=256, batch_size=1, grad_accum=16, eval_batch=1, lora_rank=8, lora_alpha=32, epochs=3, seed=42, grad_checkpointing=False)
    model, tokenizer = trainer.build_model(str(root / "base"), hp)
    report = json.loads((root / "selected-model-report.json").read_bytes())
    if selected:
        model.load_adapter(str(root / "selected"), adapter_name="default", is_trainable=True)
        model.set_adapter("default")
        # Explicit external head restore, not an assumption about PEFT names.
        head = torch.load(root / "selected/reward_head.pt", map_location="cpu", weights_only=True)
        targets = [m for name, m in model.named_modules() if isinstance(m, torch.nn.Linear) and m.out_features == 1 and
                   (name.endswith("classifier.modules_to_save.default") or name.endswith("score.modules_to_save.default") or name.endswith("classifier") or name.endswith("score"))]
        if not targets:
            raise ValueError("selected one-logit reward head missing")
        with torch.no_grad():
            for module in targets:
                module.weight.copy_(head["weight"])
                if module.bias is not None:
                    module.bias.copy_(head["bias"])
    digest = __import__("hashlib").sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    expected = report["selected_trainable_sha256"] if selected else report["initial_sha256"]
    if digest.hexdigest() != expected:
        raise ValueError("restored native parameters do not match frozen checkpoint")
    return model, tokenizer, trainer


def native_scores(model, tokenizer, trainer, rows):
    import torch
    model.eval()
    scores = {}
    with torch.no_grad():
        for f in FORMATS:
            values = []
            for row in rows:
                for side in ("chosen", "rejected"):
                    tokens = tokenizer(render(row["prompt"], row["formats"][f][side]), truncation=False, return_tensors="pt")
                    value = trainer._reward(model, tokens["input_ids"], tokens["attention_mask"]).item()
                    if not math.isfinite(value):
                        raise ValueError("native audit returned a nonfinite logit")
                    values.append(value)
            scores[f] = values
    return scores


def export(root, kit, directory, output, training=None):
    remote_only()
    sys.path.insert(0, str(kit))
    from transformers import AutoConfig
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata, target_dims, tensor_names, write_package
    config = AutoConfig.from_pretrained(str(root / "base"), local_files_only=True)
    tensors, parameters = read_peft(directory, config)
    spec = {"foundation_hash": FOUNDATION, "tokenizer_hash": TOKENIZER, "foundation_version": "modernbert-base-lora16-v1",
            "base_model": str(root / "base"), "r_max": 16, "num_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size, "targets": {k: list(v) for k, v in target_dims(config).items()}}
    dump(root / "foundation.json", spec)
    header = metadata(cls="judgment", intent=FAMILY, foundation_hash=FOUNDATION,
                      foundation_version="modernbert-base-lora16-v1", rank=parameters["rank"], alpha=parameters["alpha"],
                      target_modules=parameters["target_modules"], max_tokens=256, training=training)
    header.update(licence_metadata())
    write_package(output, header, {name: tensors[name] for name in tensor_names(config.num_hidden_layers)})
    preserve_artifact(output)


def parser(root, package):
    captured = capture(root / "class")
    command = [str(root / "packagecheck"), "--intent", FAMILY, "--child", FAMILY + "_GENERAL",
               "--class-snapshot", str(root / "class/class-snapshot.json"), "--snapshot-sha256", captured["snapshot_sha256"],
               "--package", str(package), "--sha256", file_sha(package), "--foundation", str(root / "foundation.json"),
               "--foundation-sha256", file_sha(root / "foundation.json"), "--out", str(root / "packaged")]
    done = subprocess.run(command, capture_output=True, timeout=60)
    (root / "full-parser.log").write_bytes(done.stdout + done.stderr)
    if done.returncode:
        raise ValueError("real current full package parser refused candidate")
    result = json.loads(done.stdout)
    if result["manifest"]["verification"] != {"stage1": True, "loaded": False} or result["sha256"] != file_sha(package):
        raise ValueError("real full-parser identity/coverage mismatch")
    dump(root / "full-parser.json", result)
    return captured


def audit(root, kit):
    remote_only()
    import torch
    from tokenizers import Tokenizer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise RuntimeError("independent audit requires the standard public CPU runner")
    selected_report = json.loads((root / "selected-model-report.json").read_bytes())
    if len(selected_report["history"]) != 3 or selected_report["audit_used_for_selection"] is not False:
        raise ValueError("validation-selected checkpoint incomplete")
    for name, expected in selected_report["selected_files"].items():
        if file_sha(root / "selected" / name) != expected:
            raise ValueError("selected checkpoint bytes changed before independent audit")
    for name, expected in selected_report["initial_files"].items():
        if file_sha(root / "initial" / name) != expected:
            raise ValueError("initial checkpoint bytes changed before independent audit")
    if (root / "audit-started.json").exists():
        raise ValueError("independent audit already started; preserve and reconcile the same attempt")
    dump(root / "audit-started.json", {"selected_report_sha256": file_sha(root / "selected-model-report.json"),
         "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"]})
    manifest, splits = load_freeze(HERE / "frozen")
    rows = splits["audit"]
    initial_model, initial_tokenizer, initial_trainer = native_model(root, kit, selected=False)
    del initial_model, initial_tokenizer, initial_trainer
    gc.collect()
    model, tokenizer, trainer = native_model(root, kit, selected=True)
    native = native_scores(model, tokenizer, trainer, rows)
    dump(root / "native-audit-scores.json", {"formats": native, "audit_sha256": manifest["splits"]["audit"]["sha256"],
         "selection_completed_before_measurement": True, "scope": "native conformance only, not reference-engine bits"})
    del model, tokenizer, trainer
    gc.collect()
    native_acc = sum(native["title"][i] > native["title"][i + 1] for i in range(0, 128, 2)) / 64
    training = {"family": FAMILY, "children": [{"intent": FAMILY + "_GENERAL", "dataset_version": manifest["splits"]["audit"]["sha256"],
                "pairs": 64, "accuracy": native_acc, "held_out": True}],
                "trainer": "Frozen three-epoch RELISH title scorer; code-created preferences and AI-assisted visibility review",
                "backbone": "ModernBERT-base", "backbone_version": "8949b909ec900327062f0ebf497f51aef5e6f0c8"}
    candidate = root / (FAMILY + ".safetensors")
    baseline = root / "untrained.safetensors"
    export(root, kit, root / "selected", candidate, training)
    # The initial delta is a seed-matched untrained control, not a trained registration.
    export(root, kit, root / "initial", baseline)
    node = parser(root, candidate)
    tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    vectors = {f: [tokenizer.encode(render(row["prompt"], row["formats"][f][side])).ids for row in rows for side in ("chosen", "rejected")] for f in FORMATS}
    if any(len(v) != 128 or any(not t or len(t) > 256 for t in v) for v in vectors.values()):
        raise ValueError("audit vectors incomplete or clipped")
    untrained_bits, candidate_bits = {}, {}
    with server(root, baseline, "untrained") as (score, post, initial_identity):
        for f in FORMATS:
            untrained_bits[f] = score_all(score, vectors[f])
            dump(root / ("untrained-engine-" + f + ".json"), {"score_bits": untrained_bits[f], "package": initial_identity})
    negative_controls = 0
    with server(root, candidate, "candidate") as (score, post, candidate_identity):
        for f in FORMATS:
            candidate_bits[f] = score_all(score, vectors[f])
            dump(root / ("candidate-engine-" + f + ".json"), {"score_bits": candidate_bits[f], "package": candidate_identity})
        fixed = vectors["title"][:16]
        repeat = score_all(score, fixed)
        if repeat != candidate_bits["title"][:16]:
            raise ValueError("same reference engine changed repeated score bits")
        for key in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
            try:
                score(fixed[:1], {key: "0" * 64})
            except ValueError:
                negative_controls += 1
            else:
                raise ValueError("negative identity control was accepted")
        invalid = {"batch": [{"tokens": [-1]}], "foundation_hash": FOUNDATION, "adapter_hash": candidate_identity["keccak"], "tokenizer_hash": TOKENIZER}
        refusal = strict_json(post(invalid))
        if not isinstance(refusal.get("results"), list) or len(refusal["results"]) != 1 or set(refusal["results"][0]) != {"error"}:
            raise ValueError("actual engine accepted an invalid token")
    with server(root, candidate, "fresh") as (score, post, fresh_identity):
        fresh = score_all(score, fixed)
    if fresh != repeat or fresh_identity != candidate_identity:
        raise ValueError("fresh reference process changed package identity or score bits")
    fixed_controls = json.loads((HERE / "frozen/frozen-baselines.json").read_bytes())
    extra_controls = json.loads((HERE / "frozen/unseen-term-baselines.json").read_bytes())
    if sha((HERE / "frozen/frozen-baselines.json").read_bytes()) != selected_report["frozen_baseline_sha256"] or sha((HERE / "frozen/unseen-term-baselines.json").read_bytes()) != selected_report["unseen_term_baseline_sha256"]:
        raise ValueError("frozen control coefficients changed")
    formats, agreement = {}, {}
    for f in FORMATS:
        formatted = [{**row, "chosen_title": row["formats"][f]["chosen"], "rejected_title": row["formats"][f]["rejected"]} for row in rows]
        controls = baseline_replay(fixed_controls, formatted, manifest["splits"]["train"]["sha256"])
        controls.update(overlap_replay(extra_controls, rows, manifest["splits"]["train"]["sha256"], f))
        measured = metrics(rows, candidate_bits[f], f)
        measured.update({"untrained_accuracy": metrics(rows, untrained_bits[f], f)["accuracy"], "shortcut_accuracy": max(controls.values()), "shortcut_controls": controls})
        formats[f] = measured
        actual = [score_value(b) for b in candidate_bits[f]]
        errors = [abs(a - b) for a, b in zip(native[f], actual)]
        agreement[f] = {"numerical": all(e <= .001 + .001 * abs(a) for e, a in zip(errors, native[f])),
                        "ordering": all((native[f][i] > native[f][i + 1]) == (actual[i] > actual[i + 1]) for i in range(0, 128, 2)),
                        "maximum_absolute_error": max(errors), "exact_native_bits_claimed": False}
    source = json.loads((root / "source-proof.json").read_bytes())
    runtime = {"full_parser_passed": True, "reference_engine_passed": all(a["numerical"] and a["ordering"] for a in agreement.values()),
               "repeat_bits_identical": True, "negative_identity_controls": negative_controls, "invalid_token_refused": True,
               "vectors": [{"tokens": tokens, "score_bits": bits} for tokens, bits in zip(fixed, repeat)], "native_engine_agreement": agreement,
               "node_class_snapshot_sha256": node["snapshot_sha256"], "engine_hashes_checked": True}
    result = {"schema_version": 1, "status": "MEASURED_CANDIDATE_NOT_READY", "family": FAMILY,
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "serving_commit": os.environ["GITHUB_SHA"],
              "package_sha256": candidate_identity["sha256"], "package_keccak": candidate_identity["keccak"], "package_bytes": candidate_identity["bytes"],
              "foundation_sha256": FOUNDATION, "engine_sha256": ENGINE, "tokenizer_sha256": TOKENIZER,
              "source": source, "quality": quality(formats), "runtime": runtime, "selected_report_sha256": file_sha(root / "selected-model-report.json"),
              "runtime_environment": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
                                      "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "tokenizers", "numpy")}}}
    dump(root / "audit-runtime-report.json", result)
    if result["quality"]["passed"] and runtime["reference_engine_passed"]:
        ready_report = ready(result)
        dump(root / "registration-ready.json", ready_report)
        return 0
    return 1


if __name__ == "__main__":
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("--root", type=Path, required=True)
    parser_arg.add_argument("--kit", type=Path, required=True)
    args = parser_arg.parse_args()
    raise SystemExit(audit(args.root, args.kit))
