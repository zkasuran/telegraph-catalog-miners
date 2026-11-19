# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bind an actual public CPU source run. This report is never registration-ready."""
import hashlib
import json
import os
import platform
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    if (os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_REPOSITORY") != "zkasuran/telegraph-catalog-miners"
            or os.environ.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or os.environ.get("OWNER_MODE") != "source_verify"):
        raise ValueError("only the bounded public source workflow may bind a run")
    commit = os.environ.get("GITHUB_SHA", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    owner_run = os.environ.get("OWNER_RUN_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"[1-9][0-9]*", run_id) or not re.fullmatch(r"[0-9a-f]{32}", owner_run):
        raise ValueError("actual source commit/run/attempt identity required")
    path = HERE / ".gate-work/source-report.json"
    raw = path.read_bytes()
    source = json.loads(raw)
    if source["status"] != "SOURCE_GATE_PASSED" or source["registration_ready"] is not False:
        raise ValueError("completed honest source gate required")
    url = "https://devnode.telegraphprotocol.com/engine/v1/intents/OPTIMAL_EXECUTION_ROUTE_GENERAL"
    request = urllib.request.Request(url, headers={"User-Agent": "Telegraph-source-run/1", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        live_raw = response.read((1 << 20) + 1)
        if response.geturl() != url or len(live_raw) > 1 << 20 or response.status != 200:
            raise ValueError("live source response bound")
    live = json.loads(live_raw)
    result = {"schema_version": 1, "status": "SOURCE_VERIFIED_NOT_REGISTRATION_READY",
              "family": "OPTIMAL_EXECUTION_ROUTE", "owner_run_id": owner_run, "run_id": int(run_id),
              "commit": commit, "at": datetime.now(timezone.utc).isoformat(),
              "source_report_sha256": hashlib.sha256(raw).hexdigest(),
              "source_manifest_sha256": source["source_manifest_sha256"],
              "implementation_sha256": source["implementation_sha256"],
              "runtime": {"python": platform.python_version(), "go": source["primary_wire"]["go_version"],
                          "runner_os": os.environ.get("RUNNER_OS"), "machine": platform.machine(),
                          "reference_engine_executed": False, "neural_operations": 0},
              "node": {"url": url, "definition_sha256": hashlib.sha256(live_raw).hexdigest(),
                       "live_class": live["class"], "input_fields": [p["name"] for p in live["request"]["params"]]},
              "tests": source["tests"], "qualification": source["qualification"],
              "registration_ready": False, "package": None,
              "limit": "Source compilation and primary handler mock only, not model training/audit/parser/reference-engine qualification"}
    (HERE / ".gate-work/remote-source-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "run_id": int(run_id), "commit": commit}, sort_keys=True))


if __name__ == "__main__":
    main()
