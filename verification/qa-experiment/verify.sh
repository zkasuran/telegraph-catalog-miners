#!/usr/bin/env bash
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
python3 -W error -m unittest discover -s verification/qa-experiment -p 'test_qa.py' -v
python3 -c 'import ast,pathlib,json; p=pathlib.Path("verification/qa-experiment"); [ast.parse(f.read_text(),filename=str(f)) for f in p.glob("*.py")]; json.loads((p/"sources.json").read_text()); assert (p/"LICENSE").stat().st_size > 6000; assert (p/"UPSTREAM-APACHE-2.0.txt").stat().st_size > 10000; assert (p/"KIT-LICENSE").stat().st_size > 1000; print("QA source syntax, JSON and separate full own/upstream licences checked")'
