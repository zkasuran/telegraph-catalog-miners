# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Recover one immutable trained artifact. No training or registration calls."""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from data import SOURCES, check_splits, decode_jsonl, digest, fetch, write_json

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from prepare import FOUNDATION, RELEASE

REPO = "zkasuran/telegraph-catalog-miners"
RUN = 44266324976
COMMIT = "a3a6f86c1bd831b41de8d99d12b0620c3a287f6a"
ARTIFACT = 14413342566
ZIP_SHA = "b647e50c448fe42431a84da0509a9f0ed119f618f0786435c9a93b14048ed32b"
ZIP_BYTES = 12603244
PACKAGE_SHA = "f2eefcc7b7fd807bcbd260d97c8b3b0a6b50276199b475131df968a5b60d43d9"
REPORT_SHA = "4ffc26072a347b9af879afbd3c17fbecc1c1115bf6fe141943088577dff23422"
DATA_SHA = "417f9901e5bfc7ddbff1b3cf016f973649f9c50ebf0410f2391583e80d214451"
FAMILY = "SENTIMENT_ANALYSIS"
CHILD = FAMILY + "_GENERAL"
FORMATS = ("json", "prose", "bare_label")
LABELS = {"Negative", "Positive"}
NODE = "https://devnode.telegraphprotocol.com"
URLS = {"family": NODE + "/api/wasm?resolve=" + FAMILY,
        "child": NODE + "/engine/v1/intents/" + CHILD}
FILES = {
    "result.json": REPORT_SHA,
    FAMILY + ".safetensors": PACKAGE_SHA,
    "checkpoint.safetensors": "8886ef641979885d300dad7087ea92d55038c252275d0ed7c813996efaad4a38",
    "checkpoint-config.json": "be58b8853f4ac64cc6f761f5c13497b060c061e334e1a6da74238e589922dbe5",
    "dataset-gate.json": "cc17a2726c653211a304a45e0f31fd39f296ffab0a81186c8e8cd3940dc60647",
    "tokenizer-check.json": "d889a8faabe6c9fe866228659a6efcf62d84411ab16f21123fa1c57a846fb015",
    "data/data-report.json": DATA_SHA,
}
SOURCE_HASHES = {
    "data.py": "82709147c3d628c258e871c414195396097af8c228370c53e10c110018ddf9ce",
    "sources.json": "73acea9578e0bdda4b7f47724ab4a34ec055abb1bd4b10c50a65ad2927874430",
    "requirements.lock": "4c2b80fd53899579cccb949ef49760b5f53d32cb00e2e3c85bdc857345c832ac",
    "licences.json": "99ced81fe4590ace72b41e69f55464694d1306fbecc69388108c08a2670b2b1f",
}


def check_source_inputs():
    for name, sha in SOURCE_HASHES.items():
        if digest((HERE / name).read_bytes()) != sha:
            raise ValueError("original recovery source hash mismatch: " + name)


def remote_only():
    if (os.environ.get("GITHUB_ACTIONS") != "true" or
            os.environ.get("GITHUB_REPOSITORY") != REPO or
            os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"):
        raise RuntimeError("recovery runtime runs only in the manually dispatched public workflow")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def https_read(url, ceiling=1 << 20, token=None):
    """Fixed GET only. Never forward authorization through a redirect."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password:
        raise ValueError("HTTPS URL without userinfo required")
    if token and parsed.netloc != "api.github.com":
        raise ValueError("artifact token may only reach the GitHub API")
    headers = {"User-Agent": "Telegraph-sentiment-recovery/1.0", "Cache-Control": "no-cache"}
    if token:
        headers["Authorization"] = "Bearer " + token
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    request = urllib.request.Request(url, headers=headers)
    started = time.monotonic()
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=25) as response:
        if response.status != 200 or response.geturl() != url:
            raise ValueError("unexpected HTTPS status or redirect")
        chunks, total = [], 0
        while chunk := response.read(min(1 << 20, ceiling + 1 - total)):
            total += len(chunk)
            if total > ceiling or time.monotonic() - started > 120:
                raise ValueError("HTTPS byte or time ceiling exceeded")
            chunks.append(chunk)
        return b"".join(chunks)


def check_metadata(run, artifact):
    if (run.get("id") != RUN or run.get("head_sha") != COMMIT or
            run.get("repository", {}).get("full_name") != REPO or
            run.get("head_repository", {}).get("full_name") != REPO or
            run.get("status") != "completed" or run.get("conclusion") != "failure"):
        raise ValueError("original artifact run or commit mismatch")
    if (artifact.get("id") != ARTIFACT or artifact.get("name") != "training-" + FAMILY or
            artifact.get("expired") is not False or artifact.get("size_in_bytes") != ZIP_BYTES or
            artifact.get("digest") != "sha256:" + ZIP_SHA or
            artifact.get("workflow_run", {}).get("id") != RUN or
            artifact.get("workflow_run", {}).get("head_sha") != COMMIT or
            artifact.get("archive_download_url") !=
            f"https://api.github.com/repos/{REPO}/actions/artifacts/{ARTIFACT}/zip"):
        raise ValueError("original artifact identity or hash mismatch")


def unpack_original(raw, root):
    if len(raw) != ZIP_BYTES or digest(raw) != ZIP_SHA:
        raise ValueError("original artifact ZIP hash or size mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if len(entries) != 9 or sum(e.file_size for e in entries) > 32 << 20:
            raise ValueError("original artifact member ceiling or count mismatch")
        names = [e.filename for e in entries]
        if len(set(names)) != len(names):
            raise ValueError("duplicate artifact member")
        verified = {}
        for path, expected in FILES.items():
            matches = [e for e in entries if e.filename.endswith("/telegraph-training/" + path)]
            if len(matches) != 1 or matches[0].file_size > 8 << 20:
                raise ValueError("missing or oversized original artifact file: " + path)
            contents = archive.read(matches[0])
            if digest(contents) != expected:
                raise ValueError("original artifact file hash mismatch: " + path)
            verified[path] = contents
        # Write fixed paths only after every required member has passed.
        for path, contents in verified.items():
            target = root / ("original-data-report.json" if path == "data/data-report.json" else path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)


def check_quality(report):
    if (report.get("run_id") != str(RUN) or report.get("commit") != COMMIT or
            report.get("family") != FAMILY or report.get("verdict") != "PASS" or
            report.get("dataset_gate", {}).get("ok") is not True or
            set(report.get("formats", {})) != set(FORMATS) or
            set(report.get("untrained", {})) != set(FORMATS)):
        raise ValueError("original quality identity or missing format")
    for name in FORMATS:
        metrics = report["formats"][name]
        initial = report["untrained"][name]
        values = [metrics["pairwise_accuracy"], initial["pairwise_accuracy"],
                  metrics["length_baseline"], *metrics["accuracy_by_label"].values()]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in values):
            raise ValueError("nonfinite or invalid original quality metric")
        if (metrics["n_eval"] != 256 or initial["n_eval"] != 256 or
                set(metrics["accuracy_by_label"]) != LABELS or
                min(metrics["accuracy_by_label"].values()) < 0.70 or
                metrics["pairwise_accuracy"] < 0.80 or
                metrics["pairwise_accuracy"] < initial["pairwise_accuracy"] + 0.05 or
                metrics["pairwise_accuracy"] < metrics["length_baseline"] + 0.10 or
                metrics["beats_length_baseline"] is not True):
            raise ValueError("original source-held-out quality gate failed: " + name)


def original_report(root):
    raw = (root / "result.json").read_bytes()
    if digest(raw) != REPORT_SHA or digest((root / (FAMILY + ".safetensors")).read_bytes()) != PACKAGE_SHA:
        raise ValueError("original report or package file hash mismatch")
    report = json.loads(raw)
    check_quality(report)
    return report


def retrieve(root):
    remote_only()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("ephemeral actions:read token is required for artifact download")
    api = "https://api.github.com/repos/" + REPO + "/actions/"
    run = json.loads(https_read(api + "runs/" + str(RUN), token=token))
    artifact = json.loads(https_read(api + "artifacts/" + str(ARTIFACT), token=token))
    check_metadata(run, artifact)
    try:
        raw = https_read(artifact["archive_download_url"], 32 << 20, token)
    except urllib.error.HTTPError as error:
        if error.code != 302:
            raise ValueError("artifact download API refused the request") from None
        location = error.headers.get("Location", "")
        host = urllib.parse.urlsplit(location).hostname or ""
        if not host.endswith((".blob.core.windows.net", ".githubusercontent.com")):
            raise ValueError("unexpected artifact archive host") from None
        # The signed archive URL is used once without authorization and never logged.
        raw = https_read(location, 32 << 20)
    unpack_original(raw, root)
    original_report(root)
    write_json(root / "artifact-binding.json", {"repository": REPO, "run_id": RUN,
               "commit": COMMIT, "artifact_id": ARTIFACT, "zip_sha256": ZIP_SHA,
               "zip_bytes": len(raw), "file_sha256": FILES})


def verify_data(root):
    check_source_inputs()
    report = original_report(root)
    data = root / "data"
    if digest((data / "data-report.json").read_bytes()) != DATA_SHA:
        raise ValueError("regenerated source-only data report hash mismatch")
    splits = {}
    for name, count in (("train", 768), ("validation", 128), ("test", 256)):
        raw = (data / (name + ".jsonl")).read_bytes()
        expected = report["sources"]["files"][name]
        rows = decode_jsonl(raw)
        if (digest(raw) != expected["sha256"] or len(rows) != count or
                dict(Counter(r["label"] for r in rows)) != expected["labels"]):
            raise ValueError("regenerated source split hash or count mismatch: " + name)
        splits[name] = rows
    check_splits(splits)
    write_json(root / "data-binding.json", {"data_report_sha256": DATA_SHA,
               "files": report["sources"]["files"], "normalized_source_overlap": 0})


def capture_snapshot(root):
    snapshot = {"schema": 1}
    for name, url in URLS.items():
        snapshot[name] = {"url": url, "raw": https_read(url).decode("utf-8")}
    snapshot["fetched_at"] = datetime.now(timezone.utc).isoformat()
    write_json(root / "class-snapshot.json", snapshot)
    return digest((root / "class-snapshot.json").read_bytes())


def prepare_runtime(root):
    remote_only()
    check_source_inputs()
    original_report(root)
    foundation = root / "foundation"
    foundation.mkdir(exist_ok=True)
    for name, sha in FOUNDATION.items():
        fetch(RELEASE + name, foundation / name, sha)
    base = SOURCES["base"]
    config = root / "base-config.json"
    fetch(f"https://huggingface.co/{base['id']}/resolve/{base['revision']}/config.json",
          config, base["files"]["config.json"], 1 << 20)
    cfg = json.loads(config.read_bytes())
    h, i = cfg["hidden_size"], cfg["intermediate_size"]
    spec = {"foundation_hash": FOUNDATION["foundation_lora_fp32.onnx"],
            "tokenizer_hash": FOUNDATION["tokenizer.json"],
            "foundation_version": "modernbert-base-lora16-v1", "r_max": 16,
            "num_layers": cfg["num_hidden_layers"], "hidden_size": h,
            "targets": {"attn.Wqkv": [h, 3*h], "attn.Wo": [h, h],
                        "mlp.Wi": [h, 2*i], "mlp.Wo": [i, h]}}
    write_json(root / "foundation.json", spec)
    kit = SOURCES["kit"]
    archive = root / "kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/" + kit["revision"],
          archive, kit["sha256"], 32 << 20)
    with tarfile.open(archive) as source:
        if len(source.getmembers()) > 5000 or sum(m.size for m in source.getmembers()) > 128 << 20:
            raise ValueError("reference kit ceiling exceeded")
        source.extractall(root, filter="data")
    kit_root = root / ("telegraph-wasm-baseline-" + kit["revision"])
    subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / "packagecheck"),
                    str(HERE / "packagecheck.go")], cwd=kit_root, check=True, timeout=600)
    subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / "adapterexec"),
                    "./cmd/adapterexec"], cwd=kit_root, check=True, timeout=600)


def validate(root):
    original_report(root)
    snapshot_sha = capture_snapshot(root)
    arguments = [str(root / "packagecheck"), "--intent", FAMILY, "--child", CHILD,
                 "--class-snapshot", str(root / "class-snapshot.json"),
                 "--snapshot-sha256", snapshot_sha, "--package", str(root / (FAMILY + ".safetensors")),
                 "--sha256", PACKAGE_SHA, "--foundation", str(root / "foundation.json"),
                 "--foundation-sha256", digest((root / "foundation.json").read_bytes()), "--out", str(root)]
    with (root / "package-check.stdout.txt").open("wb") as out, (root / "package-check.stderr.txt").open("wb") as err:
        try:
            result = subprocess.run(arguments, stdout=out, stderr=err, timeout=60, check=False)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
    oversized = False
    for name in ("stdout", "stderr"):
        path = root / ("package-check." + name + ".txt")
        if path.stat().st_size > 1 << 20:
            oversized = True
            with path.open("rb") as stream:
                bounded = stream.read(1 << 20)
            path.write_bytes(bounded)
    status = {"phase": "package_validation", "returncode": returncode,
              "quality_verdict": "PASS", "diagnostics_truncated": oversized,
              "snapshot_sha256": snapshot_sha}
    write_json(root / "package-check-status.json", status)
    if returncode or oversized:
        raise ValueError("package validator refused; see persisted stderr and status")
    manifest = json.loads((root / "package-check.stdout.txt").read_bytes())
    if manifest["sha256"] != PACKAGE_SHA or manifest["manifest"]["verification"] != {"stage1": True, "loaded": False}:
        raise ValueError("package validator identity or Stage 1 mismatch")
    write_json(root / "package-manifest.json", manifest)


def candidate(root):
    remote_only()
    original_report(root)
    verify_data(root)
    manifest = json.loads((root / "package-manifest.json").read_bytes())
    runtime = json.loads((root / "runtime-result.json").read_bytes())
    if (runtime.get("verdict") != "PASS" or set(runtime.get("formats", {})) != set(FORMATS) or
            runtime.get("adapter_hash") != manifest["keccak256"] or manifest["sha256"] != PACKAGE_SHA):
        raise ValueError("candidate requires the complete exported-runtime gate")
    from runtime_all_formats import format_rows, heldout_verdict, metrics
    views = format_rows(decode_jsonl((root / "data/test.jsonl").read_bytes()))
    score_file = json.loads((root / "runtime-score-bits.json").read_bytes())
    if set(score_file) != set(FORMATS):
        raise ValueError("candidate is missing a scored format")
    recomputed = {name: metrics(views[name], score_file[name]["score_bits"],
                              score_file[name]["repeated_score_bits"]) for name in FORMATS}
    if heldout_verdict(recomputed, original_report(root))["verdict"] != "PASS" or recomputed != runtime["formats"]:
        raise ValueError("candidate runtime score evidence mismatch")
    identity = {"run_id": os.environ["GITHUB_RUN_ID"], "commit": os.environ["GITHUB_SHA"]}
    if not re.fullmatch(r"[0-9a-f]{40}", identity["commit"]) or not identity["run_id"].isdigit():
        raise ValueError("invalid recovery identity")
    evidence_files = ["result.json", "runtime-result.json", "package-manifest.json", "artifact-binding.json",
                      "data-binding.json", "class-snapshot.json", "foundation.json", "package-check-status.json",
                      "package-check.stderr.txt", "runtime-score-bits.json"]
    value = {"schema_version": 1, "SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0",
             "status": "CANDIDATE_PENDING_CONTROLLER_REGISTRATION_TEST", "family": FAMILY, "child": CHILD,
             "package": {"file": FAMILY + ".safetensors", "sha256": PACKAGE_SHA,
                         "keccak256": manifest["keccak256"], "bytes": manifest["bytes"]},
             "training_identity": {"repository": REPO, "run_id": RUN, "commit": COMMIT,
                                   "artifact_id": ARTIFACT, "quality_report_sha256": REPORT_SHA},
             "recovery_identity": identity, "kit_revision": SOURCES["kit"]["revision"],
             "runtime_inputs": FOUNDATION, "formats": runtime["formats"], "originals": 256,
             "source_scope": SOURCES[FAMILY]["scope"], "retrained": False,
             "registration": "not attempted", "node_qualification": "not attempted",
             "endpoint_lease": "not attempted", "signing": "not attempted",
             "evidence_sha256": {name: digest((root / name).read_bytes()) for name in evidence_files}}
    write_json(root / "candidate-manifest.json", value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["fetch", "verify-data", "prepare", "validate", "candidate", "snapshot"])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    actions = {"fetch": retrieve, "verify-data": verify_data, "prepare": prepare_runtime,
               "validate": validate, "candidate": candidate, "snapshot": capture_snapshot}
    try:
        actions[args.phase](root)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError) as error:
        # Do not serialize signed archive URLs, tokens or environment values.
        write_json(root / "recovery-error.json", {"phase": args.phase, "verdict": "FAIL",
                   "error_type": type(error).__name__, "training_report_sha256": REPORT_SHA})
        raise SystemExit("Recovery failed in " + args.phase + "; inspect persisted diagnostics") from None


if __name__ == "__main__":
    main()
