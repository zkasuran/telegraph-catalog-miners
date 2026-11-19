# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Full unchanged Stage 1 parser and controller reference-runtime contracts."""
import json
import math
import os
import socket
import struct
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from checkpoint_terms import remote_guard
from pilot import render_pairs
from reuse_helpers import package_source, runtime_functions
from source_probe import sha, write

HERE = Path(__file__).resolve().parent
FAMILY = "LANGUAGE_TRANSLATION"
FOUNDATION = {"foundation_lora_fp32.onnx": "0a49b325304bb4cf292e683929689d0457966be264f60bfc94d0072b614bf8e6",
              "foundation_engine.wasm": "77643d6e1176ab8744801cef93ccb844d0136ed32b8daa865d541f37d13292de",
              "tokenizer.json": "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"}


def tools(root, kit):
    path = root / "translation-packagecheck.go"
    path.write_text(package_source())
    for name, source in (("packagecheck", str(path)), ("adapterexec", "./cmd/adapterexec")):
        subprocess.run(["go", "build", "-mod=readonly", "-trimpath", "-o", str(root / name), source], cwd=kit, check=True, timeout=600)


def parser_check(root, preflight=False):
    remote_guard()
    snapshot = {"schema": 1, "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")}
    for label, url in (("family", "https://devnode.telegraphprotocol.com/api/wasm?resolve=" + FAMILY),
                       ("child", "https://devnode.telegraphprotocol.com/engine/v1/intents/" + FAMILY + "_GENERAL")):
        req = urllib.request.Request(url, headers={"User-Agent": "Translation-source-check/2.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read((1 << 20) + 1)
            if response.status != 200 or len(raw) > 1 << 20:
                raise ValueError("primary class source status/size")
        json.loads(raw)
        snapshot[label] = {"url": url, "raw": raw.decode()}
    path = root / "class-snapshot.json"
    write(path, snapshot)
    command = [str(root / "packagecheck"), "--intent", FAMILY, "--child", FAMILY + "_GENERAL",
               "--class-snapshot", str(path), "--snapshot-sha256", sha(path)]
    if preflight:
        command.append("--preflight")
    else:
        command.extend(["--package", str(root / (FAMILY + ".safetensors")), "--sha256", sha(root / (FAMILY + ".safetensors")),
                        "--foundation", str(root / "foundation.json"), "--foundation-sha256", sha(root / "foundation.json"), "--out", str(root)])
    value = subprocess.run(command, capture_output=True, check=True, timeout=60)
    if len(value.stdout) > 1 << 20:
        raise ValueError("parser output ceiling")
    result = json.loads(value.stdout)
    if result.get("class") != "judgment":
        raise ValueError("the live translation class does not match the reviewed judgment recipe")
    if not preflight and result["manifest"]["verification"] != {"stage1": True, "loaded": False}:
        raise ValueError("complete Stage 1 parser did not pass")
    write(root / ("class-preflight.json" if preflight else "package-manifest.json"), result)
    return result


def metrics(rows, values):
    if len(values) != 2 * len(rows) or len(rows) != 256 or len({r["group_id"] for r in rows}) != 64:
        raise ValueError("complete 64-group final score coverage required")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("nonfinite final score")
    wins = [values[2 * i] > values[2 * i + 1] for i in range(len(rows))]
    masks = {d: [i for i, r in enumerate(rows) if r["direction"] == d] for d in ("eng-fra", "fra-eng")}
    masks.update({"chosen_longer": [i for i, r in enumerate(rows) if len(r["chosen"]) > len(r["rejected"])],
                  "chosen_shorter": [i for i, r in enumerate(rows) if len(r["chosen"]) < len(r["rejected"])]})
    subsets = {key: sum(wins[i] for i in indices) / len(indices) for key, indices in masks.items() if indices}
    return {"pairwise_accuracy": sum(wins) / len(wins), "subsets": subsets,
            "groups": 64, "pairs": 256, "ties_are_failures": True,
            "wins_by_group": {g: sum(w for r, w in zip(rows, wins) if r["group_id"] == g) / 4 for g in sorted({r["group_id"] for r in rows})}}


def quality(trained, untrained, strongest):
    failures = []
    if (set(trained) != {"raw", "json", "prose"} or set(untrained) != set(trained)
            or type(strongest) not in (int, float) or not math.isfinite(strongest) or not 0 <= strongest <= 0.65):
        return {"passed": False, "failures": ["missing final format or seed-matched baseline"]}
    for name, metric in trained.items():
        accuracy = metric["pairwise_accuracy"]
        baseline = untrained[name]["pairwise_accuracy"]
        if (not math.isfinite(accuracy) or not math.isfinite(baseline) or not 0 <= baseline <= 1
                or accuracy < 0.80 or accuracy < baseline + 0.05
                or accuracy < strongest + 0.10 or strongest > 0.65 or not metric["subsets"]
                or set(metric["subsets"]) != {"eng-fra", "fra-eng", "chosen_longer", "chosen_shorter"}
                or any(not math.isfinite(v) or v < 0.70 for v in metric["subsets"].values())):
            failures.append(name + " failed 80/70 accuracy or five/ten-point improvement gates")
    return {"passed": not failures, "failures": failures}


def runtime(root, package, native, untrained, strongest):
    remote_guard()
    from tokenizers import Tokenizer
    for name, expected in FOUNDATION.items():
        if sha(root / "foundation" / name) != expected:
            raise ValueError("reference runtime foundation identity changed")
    manifest = json.loads((HERE / "pilot-manifest.json").read_text())
    tok = Tokenizer.from_file(str(root / "foundation/tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    views = {name: render_pairs(manifest, "test", name) for name in ("raw", "json", "prose")}
    vectors = {name: [tok.encode(f"### Task:\n{r['prompt']}\n\n### Answer:\n{r[side]}").ids for r in rows for side in ("chosen", "rejected")] for name, rows in views.items()}
    if any(len(ids) > 256 for batches in vectors.values() for ids in batches):
        raise ValueError("reference runtime would clip a final distinction")
    env = {"struct": struct, "math": math, "socket": socket, "os": os, "subprocess": subprocess, "time": time,
           "urllib": __import__("urllib"), "contextmanager": contextmanager, "model_remote_only": remote_guard,
           "RELEASE_HASHES": FOUNDATION, "MAX_RUNTIME_SECONDS": 6900, "MAX_LOG_BYTES": 16 << 20, "BATCH": 4, "FAMILY": FAMILY}
    helper = runtime_functions(env)
    streams = {}
    with helper["server"](root, package, "first") as score:
        for name, batches in vectors.items():
            streams[name] = [[bit for start in range(0, len(batches), 4) for bit in score(batches[start:start + 4])] for _ in range(2)]
            if streams[name][0] != streams[name][1]:
                raise ValueError("repeated reference scores changed bits")
        for identity in ("foundation_hash", "tokenizer_hash", "adapter_hash"):
            try:
                score(vectors["raw"][:1], {identity: "0" * 64})
            except ValueError:
                pass
            else:
                raise ValueError("wrong requested identity passed verification")
    with helper["server"](root, package, "fresh") as score:
        for name, batches in vectors.items():
            fresh = [bit for start in range(0, len(batches), 4) for bit in score(batches[start:start + 4])]
            if fresh != streams[name][0]:
                raise ValueError("fresh process reference score bits changed")
    evaluated = {}
    for name, rows in views.items():
        values = [helper["decode_bits"](b) for b in streams[name][0]]
        if len(native[name]) != len(values) or any(not math.isfinite(v) for v in native[name]):
            raise ValueError("native/reference score count or finiteness changed")
        if any(abs(a - b) > 1e-3 + 1e-3 * abs(a) for a, b in zip(native[name], values)):
            raise ValueError("native/reference numerical tolerance failed")
        if any((native[name][i] > native[name][i + 1]) != (values[i] > values[i + 1]) for i in range(0, len(values), 2)):
            raise ValueError("native/reference pair ordering differs")
        evaluated[name] = metrics(rows, values)
    verdict = quality(evaluated, untrained, strongest)
    write(root / "runtime-score-bits.json", streams)
    write(root / "runtime-result.json", {"formats": evaluated, **verdict, "repeat_and_fresh_exact_bits": True,
          "native_comparison": "numerical and pair-order agreement, not an exact native-bit claim",
          "wrong_identity_controls": 3, "parser_stage1": True, "node_qualification": "not attempted"})
    if not verdict["passed"]:
        raise ValueError("reference-runtime final quality gates failed")
