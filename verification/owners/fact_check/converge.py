# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Bounded public CPU convergence comparison. Never an independent audit."""
import argparse
import gc
import json
import os
import shutil
import signal
from pathlib import Path

from convergence_policy import RECIPE, STRATEGIES, select
from data import write_json
from develop import fit
from preserve import preserve
from run import remote_only


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    if not root.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("convergence outputs must stay in bounded runner storage")
    def stop(signum, frame):
        raise TimeoutError("bounded convergence comparison stopped")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(260 * 60)
    preserve(root)
    reports = []
    try:
        for index, strategy in enumerate(STRATEGIES):
            trial = root / "convergence" / strategy
            preserve(trial)
            if index:
                for name in ("foundation", "base"):
                    shutil.copytree(root / "convergence" / STRATEGIES[0] / name, trial / name, copy_function=os.link)
            fit(trial, RECIPE, strategy, threads_configured=bool(index))
            reports.append(json.loads((trial / "result.json").read_text()))
            write_json(root / "convergence-progress.json", {"family": "FACT_CHECK", "run_id": int(os.environ["GITHUB_RUN_ID"]),
                       "commit": os.environ["GITHUB_SHA"], "completed_strategies": [item["strategy"] for item in reports],
                       "audit_read": False, "registration_ready": False})
            gc.collect()
        chosen = select(reports)
        shutil.copytree(root / "convergence" / STRATEGIES[chosen] / "adapter", root / "selected-adapter", dirs_exist_ok=True)
        report = {"family": "FACT_CHECK", "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
                  "status": "CONVERGENCE_DEVELOPMENT_PASS" if reports[chosen]["quality"]["passed"] else "CONVERGENCE_DEVELOPMENT_FAIL",
                  "selected_strategy": STRATEGIES[chosen], "selection": "minimum validation loss across complete declared candidates",
                  "candidates": reports, "audit_read": False, "audit_scored": False,
                  "registration_ready": False, "package_exported": False}
        write_json(root / "result.json", report)
        print(json.dumps({key: report[key] for key in ("status", "selected_strategy", "selection")}), flush=True)
        return 0 if reports[chosen]["quality"]["passed"] else 1
    finally:
        preserve(root)


if __name__ == "__main__":
    raise SystemExit(main())
