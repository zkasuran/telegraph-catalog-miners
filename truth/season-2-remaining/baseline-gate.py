# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Run the baseline suite in an owned isolated copy so its reports stay private."""
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
target = ROOT / '.baseline-gate'
for row in json.loads((ROOT / 'BASELINE-HASHES.json').read_text())['files']:
    name = row['file'].removeprefix('baseline/')
    destination = target / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / row['file'], destination)
# Replay the captured source fixtures at their recorded publication interval.
# The offset clock advances normally, so request deadlines retain real elapsed time.
environment = dict(os.environ)
environment['NODE_OPTIONS'] = '--import=' + (ROOT / 'baseline-fixture-clock.mjs').as_uri()
subprocess.run(['bash', str(target / 'verify.sh')], check=True, timeout=120, env=environment)
