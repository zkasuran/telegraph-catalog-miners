# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Train only training, freeze on validation then audit once in the real engine."""
import argparse
import gc
import importlib.metadata
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

from contracts import BASE, FAMILY, FOUNDATION, HERE, data_gate, label, metrics, quality, read_split, ready_contract, remote_only, score_bits, sha, source_manifest, write
from remote_io import Reference, compile_runtime, full_parser, prepare
from scorer import render


def native_metrics(model,tok,rows,hp,trainer):
    # Actual scores for each label slice, not reconstructed rounded fractions.
    import torch
    from torch.utils.data import DataLoader
    device = next(model.parameters()).device
    model.eval()
    wins = []
    with torch.no_grad():
        loader = DataLoader(rows,batch_size=hp.eval_batch,shuffle=False,collate_fn=trainer.PairCollator(tok,hp.max_length))
        for batch in loader:
            chosen = trainer._reward(model,batch["chosen_input_ids"].to(device),batch["chosen_attention_mask"].to(device))
            rejected = trainer._reward(model,batch["rejected_input_ids"].to(device),batch["rejected_attention_mask"].to(device))
            if not torch.isfinite(chosen).all() or not torch.isfinite(rejected).all():
                raise ValueError("nonfinite native audit score")
            wins.extend(bool(x) for x in (chosen > rejected).cpu().tolist())
    return metrics(rows,wins)


def preserve(root):
    # Always called, including failed audits. Do not publish raw unlicensed PEFT.
    path = root/"adapter/adapter_model.safetensors"
    if path.exists():
        from preserve import stamp
        licences = json.loads((HERE/"licences.json").read_text())
        (root/"checkpoint.safetensors").write_bytes(stamp(path.read_bytes(),licences))
        (root/"checkpoint-config.json").write_bytes((root/"adapter/adapter_config.json").read_bytes())


def run(root,preflight):
    remote_only()
    source = source_manifest()
    train,validation = read_split("train"),read_split("validation")
    invariant = data_gate({"train":train,"validation":validation})
    kit,base = prepare(root,model=not preflight)
    sys.path.insert(0,str(kit))
    os.environ["HF_HUB_OFFLINE"] = "1"
    from float_runtime import train_adapter as trainer
    hp = trainer.Hyperparams(max_length=512,batch_size=2,grad_accum=4,epochs=8,
                            seed=23,eval_batch=2,grad_checkpointing=False,
                            lr=3e-4,es_metric="pairwise_accuracy",gate_min_acc=.80,
                            gate_min_margin_over_baseline=.10)
    gate = trainer.gate_dataset(train,hp)
    write(root/"dataset-gate.json",gate)
    if not gate["ok"]:
        raise ValueError("unmodified upstream dataset gate failed before model loading")
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base,local_files_only=True)
    node_tok = Tokenizer.from_file(str(base/"tokenizer.json"))
    node_tok.no_padding()
    node_tok.no_truncation()
    count,maximum = 0,0
    for rows in (train,validation):
        for row in rows:
            for side in ("chosen","rejected"):
                text = render(row["prompt"],row[side])
                ids = tok(text,truncation=False)["input_ids"]
                if ids != node_tok.encode(text).ids or len(ids) > hp.max_length:
                    raise ValueError("literal tokenizer or decisive truncation mismatch")
                count,maximum = count+1,max(maximum,len(ids))
    trainer.check_tokenizer(tok)
    report = {"schema_version":1,"family":FAMILY,"owner_attempt":source["run_id"],
              "run_id":int(os.environ["GITHUB_RUN_ID"]),"run_attempt":int(os.environ["GITHUB_RUN_ATTEMPT"]),
              "commit":os.environ["GITHUB_SHA"],"source_manifest_sha256":sha((HERE/"source-manifest.json").read_bytes()),
              "source_manifest":source,"dataset_gate":gate,"hyperparameters":asdict(hp),
              "training_validation_data":invariant,"rendered_texts":count,"max_tokens_observed":maximum,
              "runtime_versions":{"python":sys.version,**{n:importlib.metadata.version(n) for n in ("torch","transformers","peft","numpy","safetensors","tokenizers")}},
              "node":"https://devnode.telegraphprotocol.com","scope":source["source_scope"],"phase":"preflight"}
    write(root/"training-report.json",report)
    if preflight:
        print(json.dumps({"phase":"preflight","rendered_texts":count,"max_tokens_observed":maximum}))
        return
    import numpy as np
    import torch
    torch.set_num_threads(2)
    # The trainer initializes independently with the same seed and selects only
    # validation. The audit is not read until this selected candidate is frozen.
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    model,tok,_ = trainer.train(str(base),train,validation,root/"adapter",hp)
    preserve(root)
    frozen = sha((root/"checkpoint.safetensors").read_bytes())
    report.update(phase="candidate_frozen",checkpoint_sha256=frozen)
    write(root/"training-report.json",report)
    marker = root/"audit-started.json"
    if marker.exists():
        raise ValueError("audit already exposed; do not select another candidate on it")
    write(marker,{"checkpoint_sha256":frozen,"commit":report["commit"],"run_id":report["run_id"]})
    audit = read_split("audit")
    for row in audit:
        for side in ("chosen","rejected"):
            text = render(row["prompt"],row[side])
            if tok(text,truncation=False)["input_ids"] != node_tok.encode(text).ids or len(node_tok.encode(text).ids) > hp.max_length:
                raise ValueError("audit evidence truncated or token path differs")
    native = native_metrics(model,tok,audit,hp,trainer)
    # Export the frozen selected candidate before freeing its native model.
    from float_runtime.export_adapter import read_peft
    from float_runtime.lora_format import metadata,tensor_names,write_package
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(base,local_files_only=True)
    tensors,params = read_peft(root/"adapter",config)
    training = {"family":FAMILY,"children":[{"intent":FAMILY+"_GENERAL","dataset_version":report["source_manifest_sha256"],
                "pairs":len(audit),"accuracy":native["accuracy"],"held_out":True}],
                "scope":source["source_scope"],"preference_authorship":"AI-assisted construction, not human reviewed", "supported_formats":["json"]}
    meta = metadata(cls="judgment",intent=FAMILY,foundation_hash=FOUNDATION["foundation_lora_fp32.onnx"],
                    foundation_version="modernbert-base-lora16-v1",rank=params["rank"],alpha=params["alpha"],
                    target_modules=params["target_modules"],max_tokens=512,training=training)
    licences = json.loads((HERE/"licences.json").read_text())
    for key,value in licences.items():
        meta["x-terms-"+key] = value
    package = root/(FAMILY+".safetensors")
    write_package(package,meta,{name:tensors[name] for name in tensor_names(config.num_hidden_layers)})
    config_dict = config.to_dict()
    del model,tensors,config
    gc.collect()
    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    initial,_ = trainer.build_model(str(base),hp)
    untrained = native_metrics(initial,tok,audit,hp,trainer)
    del initial
    gc.collect()
    source_check = data_gate({"train":train,"validation":validation,"audit":audit})
    baselines = source_check["splits"]["audit"]["shortcut_baselines"]
    shortcut = max(v["correct_twice"]/(2*v["pairs"]) for v in baselines.values())
    report.update(phase="audit",native_audit=native,untrained_audit=untrained,shortcut_baselines=baselines,
                  audit_used_for_selection=False)
    write(root/"training-report.json",report)
    if not quality(native,untrained,shortcut):
        raise ValueError("frozen candidate failed genuine independent audit quality gates")
    compile_runtime(root,kit)
    parsed = full_parser(root,package,config_dict)
    keccak = parsed["keccak256"]
    if not keccak.startswith("0x") or len(keccak) != 66:
        raise ValueError("actual parser keccak wire contract differs")
    keccak = keccak[2:]
    vectors = [node_tok.encode(render(row["prompt"],row[side])).ids for row in audit for side in ("chosen","rejected")]
    with Reference(root,package,keccak) as runtime:
        negative_count = runtime.negative_controls()
        scores = []
        for start in range(0,len(vectors),8):
            scores.extend(runtime.score(vectors[start:start+8]))
        repeated = runtime.score(vectors[:8])
        if repeated != scores[:8]:
            raise ValueError("actual reference score bits changed on repeat")
    ref_metrics = metrics(audit,[score_bits(scores[2*i]) > score_bits(scores[2*i+1]) for i in range(len(audit))])
    if not quality(ref_metrics,untrained,shortcut):
        report.update(reference_audit=ref_metrics,phase="reference_quality_failed")
        write(root/"training-report.json",report)
        raise ValueError("real reference engine failed independent quality gate")
    ready = {"schema_version":1,"status":"REGISTRATION_READY","family":FAMILY,"run_id":report["run_id"],"commit":report["commit"],
             "owner_attempt":source["run_id"],"package_sha256":sha(package.read_bytes()),"package_keccak":keccak,
             "package_bytes":package.stat().st_size,"foundation_sha256":FOUNDATION["foundation_lora_fp32.onnx"],
             "engine_sha256":FOUNDATION["foundation_engine.wasm"],"tokenizer_sha256":FOUNDATION["tokenizer.json"],
             "source":{"normalized_overlap":source_check["normalized_cross_split_overlap"],"audit_used_for_selection":False,
                       "grants_verified":True,"source_manifest_sha256":report["source_manifest_sha256"],
                       "independent_groups":source_check["original_groups"],"scope":source["source_scope"],"limits":source["limits"]},
             "quality":{"passed":True,"audit_pairs":len(audit),"all_supported_formats_tested":True,
                        "formats":{"json":{**ref_metrics,"untrained_accuracy":untrained["accuracy"],"shortcut_accuracy":shortcut,
                                             "native_accuracy":native["accuracy"]}}},
             "runtime":{"full_parser_passed":True,"reference_engine_passed":True,"repeat_bits_identical":True,
                        "negative_identity_controls":negative_count,"vectors":[{"tokens":v,"score_bits":s} for v,s in zip(vectors[:8],scores[:8])],
                        "parser_manifest_sha256":sha((root/"package-manifest.json").read_bytes()),"literal_bare_hash_echoes":True},
             "runtime_versions":report["runtime_versions"],"serving_commit":report["commit"],
             "scope":source["source_scope"],"production_trusted_report_delivery":"unverified; not a universal authenticity proof"}
    ready_contract(ready)
    write(root/"registration-ready.json",ready)
    report.update(phase="registration_ready",reference_audit=ref_metrics,ready_report_sha256=sha((root/"registration-ready.json").read_bytes()))
    write(root/"training-report.json",report)
    print(json.dumps({"phase":"registration_ready","quality":ref_metrics,"package_sha256":ready["package_sha256"]}))


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--preflight",action="store_true")
    args = parser.parse_args()
    args.root.mkdir(exist_ok=True)
    try:
        run(args.root,args.preflight)
    except Exception as exc:
        write(args.root/"failure.json",{"type":type(exc).__name__,"failure":str(exc),
                                      "run_id":int(os.environ["GITHUB_RUN_ID"]),"commit":os.environ["GITHUB_SHA"]})
        raise
    finally:
        preserve(args.root)


if __name__ == "__main__":
    main()
