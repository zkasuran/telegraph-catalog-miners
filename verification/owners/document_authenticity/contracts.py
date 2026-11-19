# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Closed source, audit and literal runnable contracts. No neural imports."""
import hashlib
import json
import math
import os
import re
import struct
from pathlib import Path

from check_data import checks
from scorer import strict_json

HERE = Path(__file__).resolve().parent
FAMILY = "DOCUMENT_AUTHENTICITY"
CHILD = FAMILY+"_GENERAL"
KIT = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
KIT_SHA = "18d7dbe7425e977b576a70146f901a006b8e4abddbfd351276205faa7f287243"
FOUNDATION = {"foundation_lora_fp32.onnx":"0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
              "foundation_engine.wasm":"77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de",
              "tokenizer.json":"67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"}
RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"
BASE = {"id":"answerdotai/ModernBERT-base", "revision":"8949b909ec900327062f0ebf497f51aef5e6f0c8",
        "files":{"model.safetensors":"340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9",
                 "config.json":"1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f",
                 "tokenizer_config.json":"3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2",
                 "special_tokens_map.json":"ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049"}}
HASH = re.compile(r"[a-f0-9]{64}")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners" or
            os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch" or os.environ.get("OWNER_SOURCE_ONLY") == "1"):
        raise RuntimeError("model operations require the controller's bounded public CPU workflow")


def source_manifest():
    value = json.loads((HERE/"source-manifest.json").read_text())
    if value["family"] != FAMILY or value["original_groups"] != 32 or value["audit_used_for_selection"] is not False:
        raise ValueError("source identity or frozen audit boundary changed")
    for filename, expected in value["files"].items():
        if sha((HERE/"data"/filename).read_bytes()) != expected:
            raise ValueError("frozen source file changed")
    for filename, expected in value["primary"].items():
        if sha((HERE/filename).read_bytes()) != expected:
            raise ValueError("pinned primary source changed")
    if sha((HERE/"DATA-SOURCES.md").read_bytes()) != value["grants_file_sha256"]:
        raise ValueError("retained source grants changed")
    return value


def read_split(split):
    if split not in ("train", "validation", "audit"):
        raise ValueError("invalid source split")
    manifest = source_manifest()
    raw = (HERE/"data"/(split+".jsonl")).read_bytes()
    if sha(raw) != manifest["files"][split+".jsonl"]:
        raise ValueError("source byte binding changed")
    return [{**strict_json(line), "margin":0.0} for line in raw.decode().split("\n") if line]


def data_gate(splits):
    baseline = json.loads((HERE/"data/baseline.json").read_text())
    return checks(splits, baseline)


def label(row):
    truth = strict_json(row["prompt"])["report"][row["field"]]
    return row["field"]+":"+str(truth).lower()


def metrics(rows, wins):
    if len(rows) != len(wins) or not rows or any(type(w) is not bool for w in wins):
        raise ValueError("actual audit results must cover every pair")
    labels = sorted({label(row) for row in rows})
    return {"accuracy":sum(wins)/len(wins), "per_label":{
        name:sum(w for row,w in zip(rows,wins) if label(row)==name)/sum(label(row)==name for row in rows)
        for name in labels}, "pairs":len(rows)}


def quality(trained, untrained, shortcut):
    if type(shortcut) not in (float,int) or not math.isfinite(shortcut) or not 0 <= shortcut <= 1:
        raise ValueError("invalid frozen shortcut metric")
    return (trained["accuracy"] >= .80 and min(trained["per_label"].values()) >= .70 and
            trained["accuracy"]+1e-9 >= untrained["accuracy"]+.05 and trained["accuracy"]+1e-9 >= shortcut+.10)


def score_bits(value):
    if type(value) is not int or not 0 <= value < 1 << 32:
        raise ValueError("invalid score bits")
    number = struct.unpack("<f",struct.pack("<I",value))[0]
    if not math.isfinite(number):
        raise ValueError("nonfinite reference score")
    return number


def literal_response(value, request):
    # No case, prefix or hash normalization. This is the actual node wire.
    for name in ("foundation_hash","adapter_hash","tokenizer_hash"):
        if value.get(name) != request[name] or not HASH.fullmatch(value.get(name,"")):
            raise ValueError("literal bare-hash echo differs")
    if request.get("input_root") and value.get("input_root") != request["input_root"]:
        raise ValueError("literal input-root echo differs")
    results = value.get("results")
    if not isinstance(results,list) or len(results) != len(request["batch"]):
        raise ValueError("wrong reference result count")
    for result in results:
        if set(result) != {"score"}:
            raise ValueError("reference refused a valid vector")
        score_bits(result["score"])
    return [result["score"] for result in results]


def ready_contract(report):
    if report.get("status") != "REGISTRATION_READY" or report.get("schema_version") != 1 or report.get("family") != FAMILY:
        raise ValueError("candidate is not registration ready")
    if type(report.get("run_id")) is not int or not re.fullmatch(r"[a-f0-9]{40}",report.get("commit","")):
        raise ValueError("actual remote run/commit required")
    for key in ("package_sha256","package_keccak","foundation_sha256","engine_sha256","tokenizer_sha256"):
        if not HASH.fullmatch(report.get(key,"")):
            raise ValueError("invalid ready identity")
    if (report["foundation_sha256"] != FOUNDATION["foundation_lora_fp32.onnx"] or
            report["engine_sha256"] != FOUNDATION["foundation_engine.wasm"] or report["tokenizer_sha256"] != FOUNDATION["tokenizer.json"]):
        raise ValueError("pinned foundation identity differs")
    source = report["source"]
    if source["audit_used_for_selection"] is not False or source["normalized_overlap"] != 0 or source["independent_groups"] < 32 or source["grants_verified"] is not True:
        raise ValueError("source gate not passed")
    q = report["quality"]
    if q["passed"] is not True or q["audit_pairs"] < 64 or set(q["formats"]) != {"json"} or q["all_supported_formats_tested"] is not True:
        raise ValueError("complete supported-format audit required")
    for m in q["formats"].values():
        if set(m["per_label"]) != {"integrity:true","integrity:false","whole_document:true","whole_document:false"} or not quality(m, {"accuracy":m["untrained_accuracy"]}, m["shortcut_accuracy"]):
            raise ValueError("unchanged quality gates failed")
    rt = report["runtime"]
    if not all(rt[k] is True for k in ("full_parser_passed","reference_engine_passed","repeat_bits_identical")) or rt["negative_identity_controls"] < 3 or not 8 <= len(rt["vectors"]) <= 100:
        raise ValueError("actual complete runtime proof required")
    for vector in rt["vectors"]:
        if not 1 <= len(vector["tokens"]) <= 512 or any(type(t) is not int or not 0 <= t < 50368 for t in vector["tokens"]):
            raise ValueError("invalid fixed vector")
        score_bits(vector["score_bits"])
    return report
