# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Sequential bounded public CPU ablations. No audit or registration claim."""
import argparse
import gc
import json
import os
import shutil
import signal
from pathlib import Path

from data import write_json
from develop import fit
from preserve import preserve
from recovery_policy import RECOVERY_RECIPE, STRATEGIES, recovery_selection
from run import remote_only


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    root = parser.parse_args().root
    if not root.resolve().is_relative_to(Path(os.environ["RUNNER_TEMP"]).resolve()):
        raise ValueError("recovery outputs must remain in bounded runner storage")
    def stop(signum, frame):
        raise TimeoutError("bounded sequential recovery stopped")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(260 * 60)
    preserve(root)
    reports = []
    try:
        for index, strategy in enumerate(STRATEGIES):
            trial = root / "recovery" / strategy
            trial.mkdir(parents=True, exist_ok=True)
            preserve(trial)
            # Identical immutable foundation downloads and source kit are shared,
            # not checkpoint state. Each candidate starts from the same seed.
            if index:
                for name in ("foundation", "base"):
                    source = root / "recovery" / STRATEGIES[0] / name
                    shutil.copytree(source, trial / name, copy_function=os.link)
            fit(trial, RECOVERY_RECIPE, strategy, threads_configured=bool(index))
            reports.append(json.loads((trial / "result.json").read_text()))
            write_json(root / "recovery-progress.json", {"family": "FACT_CHECK", "run_id": int(os.environ["GITHUB_RUN_ID"]),
                       "commit": os.environ["GITHUB_SHA"], "completed_strategies": [item["strategy"] for item in reports],
                       "audit_read": False, "registration_ready": False})
            gc.collect()
        chosen = recovery_selection(reports)
        selected = root / "recovery" / STRATEGIES[chosen] / "adapter"
        shutil.copytree(selected, root / "selected-adapter", dirs_exist_ok=True)
        report = {"family": "FACT_CHECK", "run_id": int(os.environ["GITHUB_RUN_ID"]), "commit": os.environ["GITHUB_SHA"],
                  "status": "RECOVERY_DEVELOPMENT_PASS" if reports[chosen]["quality"]["passed"] else "RECOVERY_DEVELOPMENT_FAIL",
                  "selected_strategy": STRATEGIES[chosen], "selection": "minimum validation loss across complete predeclared ablations",
                  "candidates": reports, "audit_read": False, "audit_scored": False,
                  "registration_ready": False, "package_exported": False}
        write_json(root / "result.json", report)
        print(json.dumps({key: report[key] for key in ("status", "selected_strategy", "selection")}), flush=True)
        return 0 if reports[chosen]["quality"]["passed"] else 1
    finally:
        preserve(root)


if __name__ == "__main__":
    raise SystemExit(main())
