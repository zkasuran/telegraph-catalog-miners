# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded public-CPU general reward-model development, never qualification.

No neural imports at module scope. main refuses local execution before fetching
anything. Validation may select an epoch. The sealed audit is not measured here.
"""

import gc
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import tarfile
import time
import traceback
import urllib.request
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import data
import token_contract
from baselines import load_frozen

REPO = "zkasuran/telegraph-catalog-miners"
KIT = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
KIT_SHA = "18d7dbe7425e977b576a70146f901a006b8e4abddbfd351276205faa7f287243"
TRAINER_SHA = "5da2919c66c2d0840104628ab88e98a9c8e5e578686f71ca9be4dcd91bd18d71"
BASE = "https://huggingface.co/answerdotai/ModernBERT-base/resolve/8949b909ec900327062f0ebf497f51aef5e6f0c8/"
BASE_FILES = {
    "model.safetensors": "340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9",
    "config.json": "1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f",
    "tokenizer_config.json": "3cd2017ff46d0a527e5d39cae39272eccfa1f19bb9f89b05d166aab2e38354e2",
    "special_tokens_map.json": "ea97ecdbcc73713039d8d64dbb05e3689495c96657fbd9a18f5bed381be81049",
}
FOUNDATION = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"
TOKENIZER_SHA = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"
SOURCE_BASE = (
    "https://huggingface.co/datasets/nvidia/HelpSteer2/resolve/"
    + data.SOURCE_COMMIT
    + "/"
)
HERE = Path(__file__).resolve().parent


def remote_only(env=None):
    env = os.environ if env is None else env
    if (
        env.get("GITHUB_ACTIONS") != "true"
        or env.get("GITHUB_REPOSITORY") != REPO
        or not re.fullmatch(r"[a-f0-9]{40}", env.get("GITHUB_SHA", ""))
        or not re.fullmatch(r"[1-9][0-9]*", env.get("GITHUB_RUN_ID", ""))
        or not re.fullmatch(r"[a-f0-9]{32}", env.get("OWNER_ATTEMPT", ""))
    ):
        raise RuntimeError("neural development only in the owner public CPU workflow")
    root = Path(env.get("RUNNER_TEMP", ""))
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise RuntimeError("actual runner temporary directory required")
    return root / "vendor-general-preference"


def download(url, target, expected, ceiling):
    allowed = (
        BASE,
        SOURCE_BASE,
        FOUNDATION,
        "https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/"
        + KIT,
    )
    if not url.startswith(allowed):
        raise ValueError("unsupported public source URL")
    partial = target.with_name(target.name + ".partial")
    if target.exists():
        if data.sha(data.bounded_read(target, ceiling)) != expected:
            raise ValueError("retained download identity mismatch")
        return
    deadline = time.monotonic() + 600
    h, size = hashlib.sha256(), 0
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "vendor-preference-development/1.0",
            "Accept-Encoding": "identity",
        },
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=45) as response,
            partial.open("xb") as out,
        ):
            final = urlparse(response.geturl())
            host = final.hostname or ""
            if (
                response.status != 200
                or final.scheme != "https"
                or not any(
                    host == suffix or host.endswith("." + suffix)
                    for suffix in (
                        "huggingface.co",
                        "hf.co",
                        "github.com",
                        "githubusercontent.com",
                        "githubassets.com",
                        "xethub.hf.co",
                        "amazonaws.com",
                    )
                )
            ):
                raise ValueError("public download status/redirect refused")
            while chunk := response.read(1 << 20):
                size += len(chunk)
                if size > ceiling or time.monotonic() > deadline:
                    raise ValueError("download byte/time ceiling")
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != expected:
            raise ValueError("download hash mismatch")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def development_gate(metrics, untrained, strongest_shortcut):
    accuracy = metrics["pairwise_accuracy"]
    return (
        accuracy >= 0.80
        and accuracy + 1e-9 >= untrained["pairwise_accuracy"] + 0.05
        and accuracy + 1e-9 >= strongest_shortcut + 0.10
        and metrics["beats_length_baseline"]
    )


def write_report(path, value):
    with path.open("x") as target:
        target.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    root = remote_only()
    if root.exists():
        raise ValueError("retained remote experiment cannot be overwritten")
    root.mkdir(mode=0o700)
    source_dir, base = root / "source", root / "base"
    source_dir.mkdir()
    base.mkdir()
    # Grants are in the public source bundle. No raw rows are uploaded.
    original = source_dir / "preferences.source.gz"
    download(
        SOURCE_BASE + "preference/preference.jsonl.gz",
        original,
        data.SOURCE_SHA256,
        data.MAX_COMPRESSED,
    )
    report = data.prepare(
        original,
        HERE / "UPSTREAM-CARD.md",
        HERE / "CC-BY-4.0.txt",
        root / "data",
        HERE / "BASELINES.json",
    )
    expected = json.loads((HERE / "EXPERIMENT-IDENTITY.json").read_bytes())
    if (
        report["files"] != expected["files"]
        or report["baseline_sha256"] != expected["baseline_sha256"]
    ):
        raise ValueError(
            "remote source split/baseline differs from frozen development identity"
        )
    archive = root / "kit.tar.gz"
    download(
        "https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/"
        + KIT,
        archive,
        KIT_SHA,
        32 << 20,
    )
    with tarfile.open(archive) as source:
        members = source.getmembers()
        if len(members) > 10000 or sum(m.size for m in members) > 128 << 20:
            raise ValueError("reference kit extraction ceiling")
        if any(
            m.issym()
            or m.islnk()
            or not (m.isfile() or m.isdir())
            or Path(m.name).is_absolute()
            or ".." in Path(m.name).parts
            for m in members
        ):
            raise ValueError("unsafe reference kit member")
        source.extractall(root, filter="data")
    kit = root / ("telegraph-wasm-baseline-" + KIT)
    if data.sha((kit / "float_runtime/train_adapter.py").read_bytes()) != TRAINER_SHA:
        raise ValueError("primary trainer identity drift")
    sys.path.insert(0, str(kit))
    from float_runtime import train_adapter as trainer

    train_raw = (root / "data/train.jsonl").read_bytes()
    train = [json.loads(line) for line in train_raw.decode().split("\n") if line]
    validation = [
        json.loads(line)
        for line in (root / "data/validation.jsonl").read_text().split("\n")
        if line
    ]
    hp = trainer.Hyperparams(
        max_length=512,
        batch_size=1,
        grad_accum=8,
        epochs=3,
        eval_batch=2,
        gate_min_acc=0.80,
        grad_checkpointing=True,
    )
    gate = trainer.gate_dataset(train, hp)
    write_report(
        root / "dataset-gate.json",
        {
            "ok": gate["ok"],
            "n": gate["n"],
            "metrics": gate["metrics"],
            "failure_count": len(gate["failures"]),
            "trainer_sha256": TRAINER_SHA,
        },
    )
    if not gate["ok"]:
        raise ValueError("unchanged primary dataset gate failed before model loading")
    download(
        FOUNDATION + "tokenizer.json", base / "tokenizer.json", TOKENIZER_SHA, 8 << 20
    )
    for name, expected_sha in BASE_FILES.items():
        download(BASE + name, base / name, expected_sha, 700 << 20)
    os.environ["HF_HUB_OFFLINE"] = "1"
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    if torch.cuda.is_available():
        raise RuntimeError("this pipeline is standard CPU only")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    node_tok = Tokenizer.from_file(str(base / "tokenizer.json"))
    node_tok.no_padding()
    node_tok.enable_truncation(max_length=512)
    if not trainer.TOKENIZER_VECTORS.is_file():
        raise ValueError("primary tokenizer conformance vectors missing")
    trainer.check_tokenizer(tok)
    for row in train + validation:
        for side in ("chosen", "rejected"):
            text = trainer.render(row["prompt"], row[side])
            if text != data.render(row["prompt"], row[side]):
                raise ValueError("literal caller render mismatch")
    token_report = token_contract.inspect_development(
        {"train": train, "validation": validation},
        lambda text: tok(text, truncation=False)["input_ids"],
        lambda text: tok(text, truncation=True, max_length=512)["input_ids"],
        lambda text: node_tok.encode(text).ids,
    )
    token_report.update(
        {
            "family": "VENDOR_VERIFY",
            "run_id": int(os.environ["GITHUB_RUN_ID"]),
            "owner_attempt": os.environ["OWNER_ATTEMPT"],
            "commit": os.environ["GITHUB_SHA"],
            "tokenizer_sha256": TOKENIZER_SHA,
            "train_sha256": data.sha(train_raw),
            "validation_sha256": data.sha(
                (root / "data/validation.jsonl").read_bytes()
            ),
            "conformance_vectors_passed": True,
        }
    )
    write_report(root / "tokenizer-check.json", token_report)
    if not token_report["passed"]:
        raise ValueError(
            "development token contract failed; aggregate evidence retained"
        )
    baseline_raw = (root / "data/BASELINES.json").read_bytes()
    load_frozen(baseline_raw, data.sha(train_raw))
    strongest = max(
        v["accuracy"] for v in report["development_baselines"]["validation"].values()
    )
    licences = json.loads((HERE / "licences.json").read_bytes())
    adapter = root / "adapter"
    adapter.mkdir()
    # Attach retained grants before training can create a partial checkpoint.
    (adapter / "licences.json").write_bytes((HERE / "licences.json").read_bytes())
    (adapter / "NOTICE").write_text(licences["notice"])
    (adapter / "LICENSE").write_text(licences["sand"])
    (adapter / "CC-BY-4.0.txt").write_bytes((HERE / "CC-BY-4.0.txt").read_bytes())
    initial, _ = trainer.build_model(str(base), hp)
    untrained = trainer.evaluate(initial, tok, validation, hp)
    del initial
    gc.collect()
    start = time.monotonic()
    model, tok, _ = trainer.train(str(base), train, validation, root / "adapter", hp)
    trained = trainer.evaluate(model, tok, validation, hp)
    del model
    gc.collect()
    files = {
        p.name: data.sha(data.bounded_read(p, 64 << 20))
        for p in sorted(adapter.iterdir())
        if p.is_file()
    }
    write_report(
        root / "SELECTED-CHECKPOINT.json",
        {
            "files": files,
            "selection": "Primary trainer validation-only best epoch",
            "audit_used_for_selection": False,
        },
    )
    with tarfile.open(root / "selected-adapter.tar.gz", "x:gz") as output:
        for path in sorted(adapter.iterdir()):
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size > 64 << 20
            ):
                raise ValueError("unsafe selected checkpoint file")
            output.add(path, arcname="adapter/" + path.name, recursive=False)
    result = {
        "schema_version": 1,
        "family": "VENDOR_VERIFY",
        "status": "GENERAL_DEVELOPMENT_COMPLETE",
        "owner_attempt": os.environ["OWNER_ATTEMPT"],
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "commit": os.environ["GITHUB_SHA"],
        "kit_commit": KIT,
        "dataset_manifest_sha256": data.sha(
            (root / "data/DATA-REPORT.json").read_bytes()
        ),
        "baseline_sha256": data.sha(baseline_raw),
        "backbone_sha256": BASE_FILES["model.safetensors"],
        "tokenizer_sha256": TOKENIZER_SHA,
        "checkpoint_sha256": data.sha((root / "SELECTED-CHECKPOINT.json").read_bytes()),
        "hyperparameters": asdict(hp),
        "untrained_validation": untrained,
        "trained_validation": trained,
        "strongest_shortcut_accuracy": strongest,
        "development_gate_passed": development_gate(trained, untrained, strongest),
        "elapsed_training_seconds": time.monotonic() - start,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "peft", "numpy", "tokenizers")
        },
        "audit_evaluated": False,
        "exact_vendor_audit_admitted": False,
        "registration_ready": False,
        "full_package_parser_passed": False,
        "reference_engine_passed": False,
        "scope": "General preference validation selects an epoch, not an independent Vendor audit or canonical qualification",
    }
    write_report(root / "development-result.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        # Errors are metadata-only. Do not print third-party exception payloads,
        # which may quote historical corpus text or large tensor representations.
        frames = [
            {
                "file": Path(frame.filename).name,
                "line": frame.lineno,
                "function": frame.name,
            }
            for frame in traceback.extract_tb(error.__traceback__)[-8:]
        ]
        failure = {
            "status": "GENERAL_DEVELOPMENT_FAILED",
            "error_type": type(error).__name__,
            "frames": frames,
            "registration_ready": False,
        }
        print(json.dumps(failure))
        if (
            os.environ.get("GITHUB_ACTIONS") == "true"
            and os.environ.get("GITHUB_REPOSITORY") == REPO
        ):
            directory = Path(os.environ["RUNNER_TEMP"]) / "vendor-general-preference"
            if directory.is_dir():
                write_report(directory / "development-failure.json", failure)
        sys.exit(1)
