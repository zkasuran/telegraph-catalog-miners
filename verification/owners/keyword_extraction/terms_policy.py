# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""One licence policy for successful exports and failed checkpoints."""
import json
from pathlib import Path


def licences_for(root):
    licences = json.loads((Path(__file__).parent / "licences.json").read_text())
    report = root / "data-report.json"
    if report.exists():
        sources = json.loads(report.read_text())["sources"]
        fields = ("doi", "title", "authors", "url", "licence", "licence_uri", "copyright", "grant", "modifications")
        credits = [{field: source[field] for field in fields} for source in sources]
        licences["source_attribution"] = json.dumps(credits, ensure_ascii=False, separators=(",", ":"))
    if len(json.dumps(licences, ensure_ascii=False).encode()) > 768 << 10:
        raise ValueError("licence/attribution header budget exceeded")
    return licences
