# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Endpoint identity recovery of the exact passing v3 tensor function, never fitting."""
import argparse
import copy
import http.client
import json
import os
import struct
from pathlib import Path

from contracts import FAMILY, FORMATS, FOUNDATION, HERE, OWNER, file_sha, finite_bits, package_parts, remote_only, sha, strict_json, validate_ready
from common import json_bytes
from remote import assets, executor, identity_controls, score, stage_check
from v3_data import LABELS, baselines, dump, pairs, prepare

PLAN = HERE / "endpoint-recovery-plan.json"
ORIGIN = HERE / "model/SENTIMENT_ANALYSIS-v3.safetensors"
RECOVERED = HERE / "model/SENTIMENT_ANALYSIS-endpoint-recovery.safetensors"
ORIGIN_REPORT = HERE / "model/v3-registration-ready.json"


def verify_identity(original, recovered):
    before, first = package_parts(original)
    after, second = package_parts(recovered)
    if first != second or {key: value for key, value in before.items() if key != "__metadata__"} != {
            key: value for key, value in after.items() if key != "__metadata__"}:
        raise ValueError("endpoint recovery changed tensor bytes or descriptors")
    old, new = before["__metadata__"], after["__metadata__"]
    if any(new.get(key) != value for key, value in old.items()) or set(new) - set(old) != {"x-endpoint-recovery"}:
        raise ValueError("endpoint recovery changed original model fields or grants")
    plan = strict_json(PLAN.read_bytes())
    record = strict_json(new["x-endpoint-recovery"])
    if record["origin_package_sha256"] != plan["origin_package_sha256"] or record["recovery_id"] != plan["recovery_id"]:
        raise ValueError("endpoint recovery lineage mismatch")
    return sha(first)


def build():
    plan = strict_json(PLAN.read_bytes())
    original = ORIGIN.read_bytes()
    if sha(original) != plan["origin_package_sha256"] or file_sha(ORIGIN_REPORT) != plan["origin_report_sha256"]:
        raise ValueError("passing origin model/report changed")
    header, payload = package_parts(original)
    updated = copy.deepcopy(header)
    if "x-endpoint-recovery" in updated["__metadata__"]:
        raise ValueError("origin already has endpoint recovery identity")
    updated["__metadata__"]["x-endpoint-recovery"] = json.dumps({
        "SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0", "origin_package_sha256": sha(original),
        "origin_report_sha256": plan["origin_report_sha256"], "recovery_id": plan["recovery_id"],
        "rejected_registration_id": 161,
        "change": "ignored endpoint registration identity only; all original tensors, model fields and grants unchanged",
        "reason": "bounded serving lease ended before delayed node qualification; no quality loss observed"
    }, sort_keys=True, separators=(",", ":"))
    encoded = json.dumps(updated, ensure_ascii=False, separators=(",", ":")).encode()
    recovered = struct.pack("<Q", len(encoded)) + encoded + payload
    payload_sha = verify_identity(original, recovered)
    if RECOVERED.exists() and RECOVERED.read_bytes() != recovered:
        raise ValueError("do not replace a different recovered package")
    RECOVERED.write_bytes(recovered)
    proof = {"SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0", "schema_version": 1, "family": FAMILY, "origin_package_sha256": sha(original),
             "package_sha256": sha(recovered), "package_bytes": len(recovered), "tensor_payload_sha256": payload_sha,
             "original_fields_grants_and_descriptors_identical": True, "tensor_payload_identical": True,
             "scope": "Ignored endpoint recovery identity only, no model loading, fitting or inference."}
    dump(HERE / "model/endpoint-byte-preservation.json", proof)
    return proof


def verify_remote(root):
    remote_only()
    preservation = build()
    origin = validate_ready(strict_json(ORIGIN_REPORT.read_bytes()))
    assets(root)
    parsed = stage_check(root, RECOVERED, preservation["package_sha256"], "endpoint-recovery-parser")
    groups, manifest = prepare(root / "data")
    if file_sha(root / "data/source-manifest.json") != origin["source"]["source_manifest_sha256"]:
        raise ValueError("frozen origin audit source changed")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    views = {name: pairs(groups["audit"], style) for style, name in enumerate(FORMATS)}
    token_views = {name: [tokenizer.encode(f"### Task:\n{row['prompt']}\n\n### Answer:\n{row[side]}").ids
                         for row in rows for side in ("chosen", "rejected")] for name, rows in views.items()}
    if any(not 1 <= len(ids) <= 256 for rows in token_views.values() for ids in rows):
        raise ValueError("frozen integration claim exceeds original token limit")
    expected_vectors = [vector["tokens"] for vector in origin["runtime"]["vectors"]]
    expected_bits = [vector["score_bits"] for vector in origin["runtime"]["vectors"]]
    keccak = parsed["keccak256"].removeprefix("0x")
    quality = {}
    actual_bits = {}
    coefficient = strict_json((HERE / "v3-baseline.json").read_bytes())
    with executor(root, RECOVERED, keccak) as (backend, row, server):
        negatives = identity_controls(server, row)
        first, second = score(backend, row, expected_vectors), score(backend, row, expected_vectors)
        if first != expected_bits or second != expected_bits:
            raise ValueError("ignored metadata recovery changed actual origin reference bits")
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=180)
        try:
            request = {"batch": [{"tokens": ids} for ids in expected_vectors],
                       "foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"], "adapter_hash": keccak,
                       "tokenizer_hash": FOUNDATION["tokenizer.json"]}
            connection.request("POST", "/runnable", json_bytes(request), {"Content-Type": "application/json"})
            response = connection.getresponse()
            body = strict_json(response.read((1 << 20) + 1))
            if response.status != 200 or any(body.get(key) != request[key] for key in
                    ("foundation_hash", "adapter_hash", "tokenizer_hash")) or body.get("results") != [{"score": bits} for bits in first]:
                raise ValueError("actual endpoint-recovery literal wire contract failed")
        finally:
            connection.close()
        for name, rows in views.items():
            bits = score(backend, row, token_views[name])
            actual_bits[name] = bits
            wins = [finite_bits(bits[2*i]) > finite_bits(bits[2*i+1]) for i in range(len(rows))]
            metrics = {"accuracy": sum(wins)/len(wins),
                       "per_label": {label: sum(win for original_row, win in zip(rows, wins) if original_row["label"] == label) /
                                     sum(original_row["label"] == label for original_row in rows) for label in LABELS}}
            old = origin["quality"]["formats"][name]
            if metrics["accuracy"] != old["accuracy"] or metrics["per_label"] != old["per_label"]:
                raise ValueError("fixed tensor integration replay did not preserve actual quality aggregates")
            shortcuts = baselines(rows, coefficient)
            if shortcuts != old["shortcuts"]:
                raise ValueError("immutable train-fitted origin shortcut measurement changed")
            metrics.update(shortcuts=shortcuts, shortcut_accuracy=max(shortcuts.values()),
                           untrained_accuracy=old["untrained_accuracy"],
                           untrained_scope="Retained actual origin audit baseline, not a new untrained-model measurement",
                           utf8_length_range=old["utf8_length_range"])
            quality[name] = metrics
    report = {"schema_version": 1, "status": "REGISTRATION_READY", "family": FAMILY, "owner": OWNER,
              "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"], "serving_commit": os.environ["GITHUB_SHA"],
              "package_sha256": preservation["package_sha256"], "package_keccak": keccak, "package_bytes": preservation["package_bytes"],
              "foundation_sha256": FOUNDATION["foundation_lora_fp32.onnx"], "engine_sha256": FOUNDATION["foundation_engine.wasm"],
              "tokenizer_sha256": FOUNDATION["tokenizer.json"], "supported_labels": list(LABELS),
              "source": {"normalized_overlap": 0, "audit_used_for_selection": False, "grants_verified": True,
                         "source_manifest_sha256": file_sha(root / "data/source-manifest.json"), "independent_groups": len(groups["audit"]),
                         "construction": manifest["construction"],
                         "scope": "Frozen passing origin audit replay for byte-preserving integration only, not a new independent audit or selection"},
              "quality": {"passed": True, "audit_pairs": len(views["json"]), "all_supported_formats_tested": True, "formats": quality},
              "runtime": {"full_parser_passed": True, "reference_engine_passed": True, "repeat_bits_identical": True,
                          "negative_identity_controls": negatives, "vectors": [{"tokens": ids, "score_bits": bits} for ids, bits in zip(expected_vectors, first)]},
              "origin_audit": {"run_id": origin["run_id"], "commit": origin["commit"], "report_sha256": file_sha(ORIGIN_REPORT),
                               "package_sha256": origin["package_sha256"], "no_new_fitting": True, "no_audit_used_for_selection": True},
              "byte_preservation": preservation,
              "limits": ["Byte-preserving endpoint integration replay, not new human review, a new independent audit or canonical qualification.",
                         "Prior source proxy, composition, reason/confidence and pretraining limitations remain unchanged."]}
    validate_ready(report)
    dump(root / "actual-quality.json", {"passed": True, "formats": quality})
    dump(root / "audit-score-bits.json", actual_bits)
    dump(root / "endpoint-byte-preservation.json", preservation)
    (root / "SENTIMENT_ANALYSIS-endpoint-recovery.safetensors").write_bytes(RECOVERED.read_bytes())
    dump(root / "registration-ready.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("build", "verify"))
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    if args.phase == "build":
        print(json.dumps(build(), indent=2))
    else:
        remote_only()
        args.root.mkdir(parents=True, exist_ok=True)
        verify_remote(args.root)
