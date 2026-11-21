# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""A1 human safety adaptation. All neural and reference work is public-CPU only."""
import argparse
import gc
import importlib
import json
import os
import random
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from aegis import FORMATS, grouped, select, sha, source_spec, training_pairs, validation_views
from baselines import fit as fit_baselines, scores as baseline_scores
from contracts import FAMILY, FOUNDATION, HERE, eligibility, remote_only, write
from remote import capture_class, compile_tools, engine_audit, finish_report, read


def shared():
    sys.path.insert(0, str(HERE / "support/expanded"))
    return {name: importlib.import_module(name) for name in ("core", "model_ops")}


def plan():
    value = read(HERE / "aegis-plan.json")
    if value["family"] != FAMILY or value["epochs"] != 2 or value["max_tokens"] != 512 or value["training_sources"] != 1584 or value["validation_sources"] != 256 or value["audit_sources"] != 256:
        raise ValueError("frozen A1 experiment/source/audit contract differs")
    return value


def prepare(root):
    remote_only()
    recipe, m = plan(), shared()
    data, original, _ = m["core"].load_shared()
    original.prepare(root)
    compile_tools(root)
    capture_class(root)
    (root / "originals").mkdir()
    corpus = root / "originals/corpus"
    corpus.mkdir()
    spec = source_spec()
    def rows():
        for item in spec["files"]:
            path = corpus / item["path"]
            url = "https://huggingface.co/datasets/" + spec["dataset_id"] + "/resolve/" + spec["revision"] + "/" + item["path"]
            data.fetch(url, path, item["sha256"], 32 << 20)
            if path.stat().st_size != item["bytes"]:
                raise ValueError("exact full primary file size differs")
            values = json.loads(path.read_bytes())
            if not isinstance(values, list):
                raise ValueError("primary JSON must be an original row array")
            for row in values:
                yield item["path"], item["split"], row
    excluded_raw = (HERE / "aegis-exclusions.json").read_bytes()
    if sha(excluded_raw) != recipe["exclusions_sha256"]:
        raise ValueError("all exposed original audits must remain excluded")
    excluded = set(json.loads(excluded_raw))
    if len(excluded) != 768:
        raise ValueError("three exposed original audits required")
    candidates, inventory = grouped(rows(), excluded)
    from tokenizers import Tokenizer
    if sha((root / "foundation/tokenizer.json").read_bytes()) != FOUNDATION["tokenizer.json"]:
        raise ValueError("actual authoritative tokenizer identity differs")
    node = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    def eligible(row):
        return len(node.encode(row["text"]).ids) <= 96
    selected, rejected = select(candidates, {"train": 1584, "validation": 256, "audit": 256}, recipe["selection_salts"], eligible)
    del candidates
    for split, values in selected.items():
        directory = root / "audit-private" if split == "audit" else root / "originals"
        directory.mkdir(exist_ok=True)
        write(directory / (split + "-sources.json"), values)
    trains = [training_pairs(selected["train"], epoch) for epoch in range(2)]
    development = {**{name: rows for name, rows in validation_views(selected["validation"]).items()},
                   **{"context_" + name: rows for name, rows in validation_views(selected["validation"], True).items()}}
    lengths = []
    for values in [*trains, *development.values()]:
        for row in values:
            for side in ("chosen", "rejected"):
                tokens = node.encode("### Task:\n" + row["prompt"] + "\n\n### Answer:\n" + row[side]).ids
                if len(tokens) > 512:
                    raise ValueError("complete A1 input exceeds real node context ceiling")
                lengths.append(len(tokens))
    warm = recipe["warm_start"]
    for name, tensor, expected in (("training-start", warm["checkpoint"], warm["checkpoint_sha256"]), ("untrained", warm["untrained"], warm["untrained_sha256"])):
        source = HERE / "assets" / tensor
        if sha(source.read_bytes()) != expected:
            raise ValueError("licensed fitted/untrained source bytes changed")
        directory = root / "adapters" / name
        directory.mkdir(parents=True)
        shutil.copyfile(source, directory / "adapter_model.safetensors")
        config = HERE / "assets" / (warm["config"] if name == "training-start" else warm["untrained_config"])
        shutil.copyfile(config, directory / "adapter_config.json")
    coefficients = fit_baselines(trains[0] + trains[1])
    write(root / "evidence/baseline-coefficients.json", coefficients)
    shortcuts = [baseline_scores(rows, coefficients) for rows in trains]
    write(root / "evidence/training-shortcut-gates.json", {"scores": shortcuts, "ceiling": 0.65})
    write(root / "evidence/training-label-counts.json", [dict(Counter(row["label"] for row in values)) for values in trains])
    if any(max(values.values()) > 0.65 for values in shortcuts):
        raise ValueError("A1 strongest training shortcut gate failed")
    files = {split: {"sha256": sha((root / ("audit-private" if split == "audit" else "originals") / (split + "-sources.json")).read_bytes()),
                    "groups": [row["group_id"] for row in values]} for split, values in selected.items()}
    write(root / "evidence/audit-freeze.json", {"source_files": files, "normalized_overlap": 0, "excluded_audit_overlap": 0,
          "full_source_inventory": inventory, "files": spec["files"], "annotation_scope": "Only original human prompt labels. Generated text and response labels are distinct; no synthetic response label is fitted or audited.",
          "frozen_before_model_loading": True, "plan_sha256": sha((HERE / "aegis-plan.json").read_bytes()), "max_tokens": max(lengths),
          "old_256_token_truncations": sum(n > 256 for n in lengths), "source_admission_rejected": rejected})
    write(root / "evidence/source-manifest.json", {"sources": read(root / "evidence/audit-freeze.json"), "dataset": spec,
          "coefficient_sha256": sha((root / "evidence/baseline-coefficients.json").read_bytes()),
          "preference_authorship": "AI-assisted programmatic conditional and single-message preferences from original human source labels, not human-reviewed preferences"})


def read_sources(root, split):
    source = read(root / "evidence/audit-freeze.json")["source_files"][split]
    path = root / ("audit-private" if split == "audit" else "originals") / (split + "-sources.json")
    raw = path.read_bytes()
    if sha(raw) != source["sha256"]:
        raise ValueError("original grouped source bytes differ")
    return json.loads(raw)


def runtime(root):
    remote_only()
    m = shared()
    torch, np, trainer, old_hp = m["model_ops"].runtime(root, 2)
    hp = replace(old_hp, max_length=512)
    tok, node = m["model_ops"].tokenizer(root)
    node.no_padding()
    node.no_truncation()
    return m, torch, np, trainer, hp, tok, node


def save(root, name, model, token, trainer):
    m = shared()
    _, _, preserve = m["core"].load_shared()
    directory = root / "adapters" / name
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    trainer.save_reward_head(model, directory)
    token.save_pretrained(directory)
    licences = read(HERE / "support/training/licences.json")
    licences["notice"] += "\nNVIDIA Nemotron Content Safety Dataset V2 (Aegis 2.0), Ghosh et al. 2025, CC BY 4.0. Original human prompt labels retained. New preferences and conditional prompts are AI-assisted constructed work, not human-reviewed preferences.\nhttps://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0\n"
    licences["nvidia_cc_by_4_0"] = (HERE / "CC-BY-4.0.txt").read_text()
    tensor = directory / "adapter_model.safetensors"
    tensor.write_bytes(preserve.stamp(tensor.read_bytes(), licences))
    output = root / "evidence" / (name + ".safetensors")
    shutil.copyfile(tensor, output)
    config = directory / "adapter_config.json"
    shutil.copyfile(config, root / "evidence" / (name + "-config.json"))
    return {"snapshot": name, "checkpoint_sha256": sha(output.read_bytes()), "adapter_sha256": sha(tensor.read_bytes()), "config_sha256": sha(config.read_bytes())}


def fit(root):
    remote_only()
    if root.exists():
        raise ValueError("fresh A1 fit directory required; do not redraw an old audit")
    (root / "evidence").mkdir(parents=True)
    write(root / "evidence/run-binding.json", {"family": FAMILY, "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "plan_sha256": sha((HERE / "aegis-plan.json").read_bytes())})
    prepare(root)
    m, torch, np, trainer, hp, tok, node = runtime(root)
    train_source, validation_source = read_sources(root, "train"), read_sources(root, "validation")
    views, context = validation_views(validation_source), validation_views(validation_source, True)
    all_views = {**views, **{"context_" + key: value for key, value in context.items()}}
    for epoch in range(2):
        epoch_pairs = training_pairs(train_source, epoch)
        gate = trainer.gate_dataset(epoch_pairs, hp)
        write(root / "evidence" / ("dataset-gate-" + str(epoch) + ".json"), gate)
        if not gate["ok"]:
            raise ValueError("original dataset/repetition/leakage gate refused A1 before model loading")
        style_gates = {str(style): trainer.gate_dataset([row for row in epoch_pairs if row["style"] == style], hp) for style in range(3)}
        write(root / "evidence" / ("training-format-gates-" + str(epoch) + ".json"), style_gates)
        if any(value["ok"] is not True for value in style_gates.values()):
            raise ValueError("actual per-format training data gate refused A1")
    trainer.check_tokenizer(tok)
    for values in [training_pairs(train_source, 0), training_pairs(train_source, 1), *all_views.values()]:
        for row in values:
            for side in ("chosen", "rejected"):
                text = trainer.render(row["prompt"], row[side])
                actual = tok(text, truncation=False)["input_ids"]
                if len(actual) > 512 or actual != node.encode(text).ids:
                    raise ValueError("literal RenderForJudgment/tokenizer/complete-claim mismatch")
    m["core"].deny_measurement_reads(root)
    m["model_ops"].seed(torch, np, hp)
    model, tok = trainer.build_model(str(root / "base"), hp)
    m["model_ops"].reload_snapshot(root, "untrained", model)
    with m["core"].isolated_rng(torch, np):
        untrained = m["model_ops"].score_views(trainer, model, tok, all_views, hp)
        untrained_saved = save(root, "untrained", model, tok, trainer)
    write(root / "evidence/untrained-validation.json", {"formats": untrained, **untrained_saved})
    m["model_ops"].reload_snapshot(root, "training-start", model)
    diagnosis = {str(length): m["model_ops"].score_views(trainer, model, tok, all_views, replace(hp, max_length=length)) for length in (256, 512)}
    diagnosis["answer_only_512"] = m["model_ops"].score_views(trainer, model, tok,
        {name: [{**row, "prompt": ""} for row in values] for name, values in all_views.items()}, hp)
    gc.collect()
    write(root / "evidence/M2-development-diagnosis.json", {"formats": diagnosis, "scope": "Native scores on legitimate development only; not original protected node sample or canonical runtime proof", "source_token_lengths": read(root / "evidence/audit-freeze.json")["old_256_token_truncations"]})
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=hp.lr, weight_decay=hp.weight_decay)
    history, steps, losses = [], 0, []
    total_steps = 198
    from torch.utils.data import DataLoader
    for epoch in range(2):
        rows = training_pairs(train_source, epoch)
        order = list(range(len(rows)))
        random.Random(hp.seed + epoch).shuffle(order)
        loader = DataLoader([rows[index] for index in order], batch_size=2, shuffle=False, collate_fn=trainer.PairCollator(tok, 512))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            chosen = trainer._reward(model, batch["chosen_input_ids"], batch["chosen_attention_mask"])
            rejected = trainer._reward(model, batch["rejected_input_ids"], batch["rejected_attention_mask"])
            loss = trainer.bt_loss(chosen.float(), rejected.float(), batch["margin"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite A1 loss")
            (loss / 8).backward()
            losses.append(float(loss.detach()))
            if (index + 1) % 8 == 0:
                if steps >= total_steps:
                    raise ValueError("A1 optimizer ceiling")
                torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                warm = max(1, int(total_steps * 0.06))
                factor = min(1.0, (steps + 1) / warm) if steps < warm else max(0.0, (total_steps - steps) / (total_steps - warm))
                for group in optimizer.param_groups:
                    group["lr"] = hp.lr * factor
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
        if len(rows) != 1584 or (index + 1) % 8:
            raise ValueError("discarded accumulation tail")
        with m["core"].isolated_rng(torch, np):
            record = {"epoch": epoch, "arm": "A1", "formats": m["model_ops"].score_views(trainer, model, tok, views, hp),
                      "context_formats": m["model_ops"].score_views(trainer, model, tok, context, hp), **save(root, "A1-epoch-" + str(epoch), model, tok, trainer)}
        history.append(record)
        write(root / "evidence/A1-validation-history.json", history)
    coefficients = read(root / "evidence/baseline-coefficients.json")
    shortcuts = {name: max(baseline_scores(rows, coefficients).values()) for name, rows in views.items()}
    context_shortcuts = {name: max(baseline_scores(rows, coefficients).values()) for name, rows in context.items()}
    eligible = [r for r in history if not eligibility(r["formats"], {name: untrained[name] for name in FORMATS}, shortcuts)
                and not eligibility(r["context_formats"], {name: untrained["context_" + name] for name in FORMATS}, context_shortcuts)]
    write(root / "evidence/selection.json", {"selection_uses_audit": False, "eligible_epochs": [r["epoch"] for r in eligible], "shortcuts": shortcuts, "context_shortcuts": context_shortcuts})
    if not eligible:
        raise ValueError("no A1 validation-eligible candidate; new audit stays unscored")
    candidate = min(eligible, key=m["core"].rank_key)
    write(root / "evidence/candidate-freeze.json", {"candidate": candidate, "untrained": {**untrained_saved, "formats": {name: untrained[name] for name in FORMATS}},
          "source_manifest_sha256": sha((root / "evidence/source-manifest.json").read_bytes()), "selection_sha256": sha((root / "evidence/selection.json").read_bytes()),
          "audit_sha256": read(root / "evidence/audit-freeze.json")["source_files"]["audit"]["sha256"], "optimizer_calls": steps, "audit_used_for_selection": False})


def bundle(root, destination):
    remote_only()
    frozen = read(root / "evidence/candidate-freeze.json")
    destination.mkdir()
    for directory in ("evidence", "originals", "audit-private"):
        shutil.copytree(root / directory, destination / directory)
    (destination / "adapters").mkdir()
    for record in (frozen["candidate"], frozen["untrained"]):
        shutil.copytree(root / "adapters" / record["snapshot"], destination / "adapters" / record["snapshot"])


def audit(root):
    remote_only()
    m = shared()
    _, original, _ = m["core"].load_shared()
    original.prepare(root)
    compile_tools(root)
    capture_class(root)
    frozen = read(root / "evidence/candidate-freeze.json")
    if frozen["source_manifest_sha256"] != sha((root / "evidence/source-manifest.json").read_bytes()) or frozen["selection_sha256"] != sha((root / "evidence/selection.json").read_bytes()) or frozen["audit_used_for_selection"] is not False:
        raise ValueError("candidate/source/selection freeze differs")
    for item in source_spec()["files"]:
        if sha((root / "originals/corpus" / item["path"]).read_bytes()) != item["sha256"]:
            raise ValueError("complete original human source bytes changed across jobs")
    seen = set()
    for split in ("train", "validation", "audit"):
        source_rows = read_sources(root, split)
        ids = {row["group_id"] for row in source_rows}
        if len(ids) != len(source_rows) or seen & ids:
            raise ValueError("source/entity overlap across fit/validation/audit")
        from aegis import group_id
        if any(row["annotation_source"] != "human" or row["group_id"] != group_id(row["text"]) for row in source_rows):
            raise ValueError("original human annotation or normalized identity differs")
        seen |= ids
    from aegis_export import export
    trained, trained_manifest = export(root, FAMILY + ".safetensors", frozen["candidate"])
    untrained, untrained_manifest = export(root, "UNTRAINED.safetensors", frozen["untrained"])
    with (root / "evidence/audit-access.json").open("x") as out:
        out.write(json.dumps({"frozen_candidate_sha256": frozen["candidate"]["checkpoint_sha256"], "audit_sha256": frozen["audit_sha256"]}))
    sources = read_sources(root, "audit")
    if len(sources) != 256 or len({row["group_id"] for row in sources}) != 256:
        raise ValueError("full new independent audit required")
    views = validation_views(sources)
    from tokenizers import Tokenizer
    node = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    node.no_padding()
    node.no_truncation()
    vectors = {name: [node.encode("### Task:\n" + row["prompt"] + "\n\n### Answer:\n" + row[side]).ids for row in rows for side in ("chosen", "rejected")] for name, rows in views.items()}
    boundary_tokens = [50281] + [42] * 510 + [50282]
    vectors["context_boundary"] = [boundary_tokens]
    if any(len(ids) > 512 for values in vectors.values() for ids in values):
        raise ValueError("audit complete-claim token ceiling")
    trained_bits, fixed, negative = engine_audit(root, trained, trained_manifest, vectors)
    untrained_bits, _, _ = engine_audit(root, untrained, untrained_manifest, vectors)
    def summaries(streams):
        from contracts import bits
        return {name: m["core"].summary(rows, [bits(streams[name][2*i]) - bits(streams[name][2*i+1]) for i in range(len(rows))]) for name, rows in views.items()}
    trained_scores, untrained_scores = summaries(trained_bits), summaries(untrained_bits)
    coefficients = read(root / "evidence/baseline-coefficients.json")
    if sha((root / "evidence/baseline-coefficients.json").read_bytes()) != read(root / "evidence/source-manifest.json")["coefficient_sha256"]:
        raise ValueError("train-fitted coefficient bytes changed")
    shortcuts = {name: max(baseline_scores(rows, coefficients).values()) for name, rows in views.items()}
    if eligibility(trained_scores, untrained_scores, shortcuts):
        write(root / "evidence/result.json", {"status": "AUDIT_FAILED", "trained": trained_scores, "untrained": untrained_scores, "shortcuts": shortcuts})
        raise ValueError("A1 actual-reference independent audit failed")
    # Existing finish_report expects these original field names; only the source
    # reader changes, not quality/identity/runtime gates.
    finish_report(root, frozen, trained, trained_manifest, sources, views, trained_bits, untrained_bits, shortcuts, fixed, negative)
    report = read(root / "evidence/registration-ready.json")
    report["source"]["scope"] = plan()["scope"]
    report["source"]["annotations"] = "Original human NVIDIA prompt labels; AI-assisted constructed preferences are distinct and are not human-reviewed preferences."
    report["evidence"]["development_diagnosis_sha256"] = sha((root / "evidence/M2-development-diagnosis.json").read_bytes())
    report["runtime"]["extra_context_boundary"] = {"tokens": boundary_tokens, "score_bits": trained_bits["context_boundary"][0], "repeated_reference_engine": True}
    write(root / "evidence/registration-ready.json", report)


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "bundle", "audit"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.phase == "bundle":
        bundle(args.root, args.destination)
    else:
        globals()[args.phase](args.root)
