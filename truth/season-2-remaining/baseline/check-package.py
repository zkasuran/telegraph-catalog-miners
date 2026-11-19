# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Portable source and descriptor checks. Private source inputs are never required."""
import hashlib
import json
import re
from pathlib import Path

import jsonschema
import yaml

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST.json").read_text())
for record in manifest["files"]:
    data = (root / record["file"]).read_bytes()
    assert len(data) == record["bytes"]
    assert hashlib.sha256(data).hexdigest() == record["sha256"]
for pin in json.loads((root / "MODULE-HASHES.json").read_text())["modules"]:
    data = (root / pin["file"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == pin["sha256"]
schema = json.loads((root / "descriptor-schema.json").read_text())["json_schema"]
for source in json.loads((root / "CONTRACTS.json").read_text())["sources"]:
    text = (root / source["yaml"]).read_text()
    doc = yaml.safe_load(text)
    jsonschema.validate(doc, schema)
    row = json.loads((root / source["jsonl"]).read_text())
    assert row["source"] == text and row["label_kind"] == "truth_source"
    assert row["intent_id"] == source["intent_id"]
    assert doc["base_url"] + doc["endpoints"][0]["external_path"] == "https://telegraph-s2-hashprice.margyn.workers.dev" + source["public_path"]
    assert not {"prompt", "chosen", "rejected", "signature", "public_key"} & set(row)
print("PACKAGE HASH AND DOCUMENTED SCHEMA PASS. Actual backend loading is unverified.")
body_file = root / ".test-results/dispatcher-contracts.json"
if body_file.exists():
    from decimal import Decimal
    for item in json.loads(body_file.read_text())["observations"]:
        body = json.loads(item["body"], parse_float=Decimal)
        if item["name"].startswith("balance-") and item["status"] == 200:
            assert re.search(r'"balance_gwei":\d+\.\d{6}[,}]', item["body"])
            assert body["balance_gwei"] == Decimal((int(body["balance_wei"]) + 500) // 1000) / Decimal(1000000)
    print("EXACT DECIMAL WIRE CONTRACT PASS")
