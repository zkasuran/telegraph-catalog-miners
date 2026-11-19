# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Standalone source-only release gate. It never imports or runs neural code."""
import argparse
import datetime as dt
import io
import json
import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path


class SourceOnly:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in {"torch", "tensorflow", "onnxruntime", "transformers", "peft"}:
            raise RuntimeError("neural imports forbidden in source-only gate")


sys.meta_path.insert(0, SourceOnly())

import reference as ref
import tests

HERE = Path(__file__).resolve().parent


def suite():
    return unittest.defaultTestLoader.loadTestsFromTestCase(tests.PublicTests)


def run_suite():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite())
    return result, stream.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    for path in HERE.glob("*.py"):
        compile(path.read_text(), str(path), "exec")
    subprocess.run(["bash", "-n", str(HERE / "verify.sh")], check=True, timeout=10)
    records = ref.selected_rules(HERE)
    groups = {str(network.supernet(new_prefix=24)) if network.prefixlen > 24 else str(network)
              for record in records for network in ref.direct_rule(record)}
    bsd = (HERE / "UPSTREAM-ET-BSD.txt").read_bytes()
    mit = (HERE / "UPSTREAM-MIT.txt").read_bytes()
    if ref.sha(bsd) != "2fba5e256c061fd24855c2a46d2ecf9f4eed7869cdf3b5a22d53fc26578ae906" or ref.sha(mit) != "e20635bc37e1ec73cc962c20cf23b0043bfcf20d01dbcefb959238ebc2f8fde1":
        raise ValueError("original licence bytes changed")
    cards = json.loads((HERE / "cards.json").read_text())
    if cards["upstream_notice"].encode() != bsd or cards["training_authorized"] is not False:
        raise ValueError("source rights or mechanical-data boundary changed")
    result, output = run_suite()
    if not result.wasSuccessful():
        print(output)
        raise RuntimeError("standalone contract gate failed")
    mutations = {}
    old_score, old_ttl, old_complete = ref.score, ref.SNAPSHOT_TTL_SECONDS, ref.require_complete_selection
    try:
        for name in ("accept-all", "never-stale", "allow-source-omission"):
            ref.score = (lambda *args: 1.0) if name == "accept-all" else old_score
            ref.SNAPSHOT_TTL_SECONDS = 10**20 if name == "never-stale" else old_ttl
            ref.require_complete_selection = (lambda records: None) if name == "allow-source-omission" else old_complete
            mutant, _ = run_suite()
            if mutant.wasSuccessful() or mutant.errors:
                raise RuntimeError("a broken reference escaped the assertion gate")
            mutations[name] = {"failures": len(mutant.failures), "errors": len(mutant.errors)}
    finally:
        ref.score, ref.SNAPSHOT_TTL_SECONDS, ref.require_complete_selection = old_score, old_ttl, old_complete
    protected = ("torch", "tensorflow", "onnxruntime", "transformers", "peft")
    if any(name in sys.modules for name in protected):
        raise RuntimeError("source-only gate imported neural code")
    files = {path.name: ref.sha(path.read_bytes()) for path in HERE.iterdir() if path.is_file()}
    identity = {"python": platform.python_version(), "node": platform.node(), "platform": platform.platform(),
                "runner": os.environ.get("RUNNER_NAME"), "github_run_id": int(os.environ["GITHUB_RUN_ID"]) if os.environ.get("GITHUB_RUN_ID") else None,
                "commit": os.environ.get("GITHUB_SHA")}
    report = {"schema_version": 1, "family": ref.FAMILY, "status": "SOURCE_REFERENCE_VERIFIED",
              "at": dt.datetime.now(dt.timezone.utc).isoformat(), "source_only": True,
              "runtime_identity": identity, "files": files, "tests": result.testsRun,
              "parser_controls": 500, "mutations": mutations, "source_cards": len(records),
              "source_cards_sha256": ref.PINNED_CARDS_SHA256, "source_sha256": ref.PINNED_MALWARE_SHA256,
              "source_captured_at": ref.PINNED_CAPTURED_AT, "independent_groups": len(groups),
              "model_operations": 0, "qualification_ready": False, "training_authorized": False,
              "limits": ["Source code and mechanical contracts only, not a neural model quality audit.",
                         "Only six source/entity groups, below the unchanged qualification minimum of 32.",
                         "No approved current-positive observation source. Rule edit dates are not last-seen timestamps.",
                         "Literal render is tested. Tokenizer and FP32 reference engine are not executed."]}
    if args.report:
        if not args.report.parent.is_dir():
            raise ValueError("report parent directory must exist")
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    print("SOURCE REFERENCE GREEN; NOT REGISTRATION READY")


if __name__ == "__main__":
    main()
