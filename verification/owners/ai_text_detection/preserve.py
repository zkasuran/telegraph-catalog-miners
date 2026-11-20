# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Preserve only bounded safe results and licensed checkpoints, including failures."""
import argparse
import json
import struct
from pathlib import Path

from common import HERE, dump, sha


def stamp(raw, terms):
    if not 9 <= len(raw) <= 64 << 20:
        raise ValueError("checkpoint byte ceiling")
    size = struct.unpack("<Q", raw[:8])[0]
    if not 0 < size <= 1 << 20 or size + 8 > len(raw):
        raise ValueError("checkpoint header ceiling")
    head = json.loads(raw[8:8+size])
    before = {k: v for k, v in head.items() if k != "__metadata__"}
    meta = head.setdefault("__metadata__", {})
    for name, item in terms.items():
        meta["x-terms-" + name] = item["text"]
    header = json.dumps(head, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    if len(header) > 1 << 20 or before != {k: v for k, v in json.loads(header).items() if k != "__metadata__"}:
        raise ValueError("stamp changed tensor descriptors")
    return struct.pack("<Q", len(header)) + header + raw[8+size:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    root = p.parse_args().root
    out = root / "public-result"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("dataset-gate.json", "frozen-candidate.json", "package-manifest.json", "class-snapshot.json",
                 "frozen-baselines.json", "audit-runtime-result.json", "registration-ready.json", "dependency-audit.json",
                 "validation-result.json", "checkpoint-continuation.json", "checkpoint-loaded.json", "format-training-report.json", "development-generation-report.json",
                 "audit-generation-report.json", "development-generation-progress.json", "audit-generation-progress.json"):
        source = root / name
        if source.is_file() and source.stat().st_size <= 256 << 10:
            (out / name).write_bytes(source.read_bytes())
    if (root / "literal-repetition-gate.json").is_file():
        (out / "literal-repetition-gate.json").write_bytes((root / "literal-repetition-gate.json").read_bytes())
    for name in ("augmentation-generation-report.json", "additional-augmentation-generation-report.json", "training-expansion-report.json", "training-weight-report.json",
                 "training-objective-report.json", "objective-tensor-controls.json"):
        path = root / name
        if path.is_file() and path.stat().st_size <= 256 << 10:
            (out / name).write_bytes(path.read_bytes())
    source_manifest = (HERE / "sources.json").read_bytes()
    for name in ("origin-source-contract.json", "origin-gradient-control.json", "origin-training-report.json", "origin-checkpoint-loaded.json", "coupled-tensor-controls.json", "margin-tensor-controls.json"):
        path = root / name
        if path.is_file() and path.stat().st_size <= 256 << 10:
            (out / name).write_bytes(path.read_bytes())
    for name in ("supplement-generation-report.json", "supplement-source-contract.json"):
        path = root / name
        if path.is_file() and path.stat().st_size <= 256 << 10:
            (out / name).write_bytes(path.read_bytes())
    (out / "source-manifest.json").write_bytes(source_manifest)
    for name in ("development-generation.jsonl", "augmentation-generation.jsonl", "additional-augmentation-generation.jsonl", "audit-generation.jsonl", "supplement-generation.jsonl"):
        path = root / name
        if path.is_file() and path.stat().st_size <= 16 << 20:
            (out / name).write_bytes(path.read_bytes())
    from supplement_contract import source_bundle
    bundle_sha, extra, _ = source_bundle()
    (out / "base-source-manifest.json").write_bytes(source_manifest)
    (out / "source-manifest.json").write_bytes((HERE / "training-source-bundle.json").read_bytes())
    (out / "extra-training-sources.json").write_bytes((HERE / "extra-training-sources.json").read_bytes())
    dump(out / "composite-source-binding.json", {"source_manifest_sha256": bundle_sha, "original_grants_retained": True})
    dump(out / "source-binding.json", {"source_manifest_sha256": bundle_sha,
        "base_source_manifest_sha256": sha(source_manifest), "source_grants_attached": True})
    dump(out / "source-binding.json", {"source_manifest_sha256": sha(source_manifest), "source_grants_attached": True})
    for path in (root / "AI_TEXT_DETECTION.safetensors", root / "UNTRAINED.safetensors", root / "adapter/adapter_model.safetensors", root / "origin-head.safetensors"):
        if path.is_file():
            source = json.loads(source_manifest)
            terms = {**source["terms"], "source-attribution": {"text": json.dumps([{k: r[k] for k in
                     ("source_id", "authors", "url", "licence", "licence_url", "changes", "sha256")}
                     for rows in source["source_report"]["articles"].values() for r in rows], ensure_ascii=False)},
                      "source-manifest-sha256": {"text": sha(source_manifest)}, "notice": {"text": (HERE / "NOTICE").read_text()},
                      "training-source-bundle-sha256": {"text": bundle_sha},
                      "extra-source-attribution": {"text": json.dumps(extra["articles"], ensure_ascii=False)}}
            (out / path.name).write_bytes(stamp(path.read_bytes(), terms))
            # A released package is already stamped. Re-stamping must not change
            # its bytes or its audited identity. Only raw checkpoints need stamps.
            if path.name in ("AI_TEXT_DETECTION.safetensors", "UNTRAINED.safetensors"):
                (out / path.name).write_bytes(path.read_bytes())
    dump(out / "result.json", {"status": "REGISTRATION_READY" if (out / "registration-ready.json").exists() else "INCOMPLETE_OR_FAILED",
                               "limits": "Only present observed reports are retained. Missing runtime or audit is not a pass."})
    dump(out / "artifact-manifest.json", {"files": [{"path": p.name, "sha256": sha(p.read_bytes()), "bytes": p.stat().st_size}
         for p in sorted(out.iterdir()) if p.is_file()]})


if __name__ == "__main__":
    main()
