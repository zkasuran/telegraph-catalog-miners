# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pinned source/model inputs. Neural assets are downloaded on GitHub only."""
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

from scorer import BASE, KIT, remote_only
from source import HERE, sha
from wire import FOUNDATION, ENGINE, TOKENIZER

KIT_SHA = "18d7dbe7425e977b576a70146f901a006b8e4abddbfd351276205faa7f287243"
RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"
MODEL_FILES = {"foundation_lora_fp32.onnx": FOUNDATION, "foundation_engine.wasm": ENGINE, "tokenizer.json": TOKENIZER}
BASE_FILES = {"model.safetensors": "340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9",
              "config.json": "1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f",
              "tokenizer_config.json": "3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2",
              "special_tokens_map.json": "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049"}
MAX_DOWNLOAD = 2 << 30
MAX_SOURCE = 32 << 20


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = urllib.parse.urlsplit(newurl).hostname or ""
        allowed = (".githubusercontent.com", ".github.com", ".huggingface.co", ".hf.co", ".amazonaws.com", ".figshare.com")
        if not newurl.startswith("https://") or not (host.endswith(allowed) or host == "44c1cec316ec56bca76dfd65d2882b20.r2.cloudflarestorage.com"):
            raise ValueError("unexpected pinned-source redirect host")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch(url, path, expected, ceiling=MAX_DOWNLOAD):
    if path.exists():
        if path.is_symlink() or path.stat().st_size > ceiling or file_sha(path) != expected:
            raise ValueError("cached immutable input differs: " + path.name)
        return
    if not url.startswith("https://") or urllib.parse.urlsplit(url).hostname not in {"github.com", "codeload.github.com", "huggingface.co", "ndownloader.figshare.com", "raw.githubusercontent.com"}:
        raise ValueError("pinned public source URL required")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    start, count, digest = time.monotonic(), 0, hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "academic-owner-pinned-input/1"})
        with urllib.request.build_opener(SafeRedirect()).open(request, timeout=45) as response, partial.open("wb") as output:
            if response.status != 200:
                raise ValueError("pinned source HTTP status")
            while chunk := response.read(1 << 20):
                count += len(chunk)
                if count > ceiling or time.monotonic() - start > 1200:
                    raise ValueError("pinned input byte/time ceiling")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("pinned input SHA-256 mismatch: " + path.name)
        partial.replace(path)
        dump(path.with_suffix(path.suffix + ".receipt.json"), {"requested_url": url, "sha256": expected, "bytes": count,
             "elapsed_seconds": time.monotonic() - start, "redirect_urls_retained": False})
    finally:
        partial.unlink(missing_ok=True)


def prepare_kit(root):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + KIT, archive, KIT_SHA, MAX_SOURCE)
    kit = root / ("telegraph-wasm-baseline-" + KIT)
    if not kit.exists():
        with tarfile.open(archive) as source:
            members = source.getmembers()
            if sum(m.size for m in members) > 128 << 20 or any(m.name.startswith("/") or ".." in Path(m.name).parts or m.isdev() for m in members):
                raise ValueError("reference source archive expansion/path ceiling")
            source.extractall(root, filter="data")
    if file_sha(kit / "float_runtime/train_adapter.py") != "5da2919c66c2d0840104628ab88e98a9c8e5e578686f71ca9be4dcd91bd18d71":
        raise ValueError("primary trainer source identity changed")
    dump(kit / "source-identity.json", {"commit": KIT, "verified_archive_sha256": KIT_SHA})
    return kit


def compile_tools(root, kit):
    commands = [["go", "test", "-mod=readonly", "-v", str(HERE / "packagecheck.go"), str(HERE / "packagecheck_test.go"), str(HERE / "inputroot_test.go")],
                ["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / "packagecheck"), str(HERE / "packagecheck.go")],
                ["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / "digest"), str(HERE / "digest.go")],
                ["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / "adapterexec"), "./cmd/adapterexec"]]
    reports = []
    for i, command in enumerate(commands):
        done = subprocess.run(command, cwd=kit, capture_output=True, timeout=600)
        log = root / ("tool-build-" + str(i) + ".log")
        log.write_bytes(done.stdout + done.stderr)
        reports.append({"command": command, "returncode": done.returncode, "log_sha256": file_sha(log)})
        if done.returncode:
            raise ValueError("reference source tool compilation failed")
    dump(root / "tool-build.json", reports)


def prepare_models(root):
    remote_only()
    foundation, base = root / "foundation", root / "base"
    for name, expected in MODEL_FILES.items():
        fetch(RELEASE + name, foundation / name, expected)
    for name, expected in BASE_FILES.items():
        fetch("https://huggingface.co/answerdotai/ModernBERT-base/resolve/" + BASE + "/" + name, base / name, expected)
    (base / "tokenizer.json").write_bytes((foundation / "tokenizer.json").read_bytes())
    dump(base / "source-identity.json", {"revision": BASE, "files": BASE_FILES, "tokenizer_sha256": TOKENIZER})
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return foundation, base


def source_gate(root):
    from checkpoint import licence_metadata
    from scorer import load_freeze
    from source import annotations, consensus
    licence_metadata()
    manifest, splits = load_freeze(HERE / "frozen")
    # Recheck the complete original label inventory, not only a selected page.
    original = root / "RELISH_v1.json.gz"
    fetch("https://ndownloader.figshare.com/files/14376173", original, "1c592f487af7e0b5ca3113a1ea8f49662135efd90cbd3b28da5712f4b351986a", 2 << 20)
    labels, counts = consensus(annotations(original.read_bytes()))
    metadata = json.loads((HERE / "frozen/source-metadata.json").read_bytes())
    for rows in splits.values():
        for row in rows:
            ids = [row[k] for k in ("seed_pmid", "chosen_pmid", "rejected_pmid")]
            for ident, field in zip(ids, ("query_title", "chosen_title", "rejected_title")):
                if metadata[ident]["title"] != row[field]:
                    raise ValueError("frozen original metadata changed")
            for label, ident in (("relevant", ids[1]), ("irrelevant", ids[2])):
                if ident not in {r["pmid"] for r in labels[ids[0]][label]}:
                    raise ValueError("frozen relation disagrees with primary original annotations")
    report = {"normalized_overlap": manifest["normalized_overlap"], "audit_used_for_selection": False,
              "grants_verified": True, "source_manifest_sha256": sha((HERE / "frozen/freeze.json").read_bytes()),
              "independent_groups": len(splits["audit"]), "all_original_annotation_counts": counts,
              "metadata_sha256": sha((HERE / "frozen/source-metadata.json").read_bytes()), "audit_metrics_measured": False}
    dump(root / "source-proof.json", report)
    return report
