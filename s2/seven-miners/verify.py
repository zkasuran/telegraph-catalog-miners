"""SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
Validate the scoped YAML contract and bind every retained runtime source byte.
The descriptor is emitted in YAML's JSON subset, so only the stdlib is needed.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def validate(value):
    allowed = {"version", "kind", "id", "slug", "protocol", "name", "description", "base_url", "cache_ttl_sec",
               "rate_limit_per_sec", "circuit_threshold", "circuit_cooldown_seconds", "endpoints", "input_schema",
               "output_schema", "semantics", "limitations", "errors", "docs"}
    assert isinstance(value, dict) and not value.keys() - allowed, "unknown YAML root key"
    assert value["version"] == "1" and value["kind"] == "miner"
    assert value["id"] == 202611201 and value["slug"] == "zkasuran-s2-digital-stock"
    assert value["base_url"] == "https://telegraph-s2-hashprice.margyn.workers.dev"
    assert value["semantics"]["supported_intents"] == ["SKU_IN_STOCK_GENERAL"]
    assert len(value["endpoints"]) == 1
    endpoint = value["endpoints"][0]
    assert not endpoint.keys() - {"path", "external_path", "method", "description", "intents", "params"}
    assert endpoint["path"] == "/stock" and endpoint["external_path"] == "/truth/v2/sku"
    assert endpoint["method"] == "GET" and endpoint["intents"] == ["SKU_IN_STOCK_GENERAL"]
    assert endpoint["description"]
    params = endpoint["params"]["query"]["required"]
    assert {item["name"] for item in params} == {"merchant", "sku", "location"}
    assert all(item["type"] == "string" and item["intents"] == ["*"] and item["description"] for item in params)
    assert not value["semantics"]["signal_mapping"].keys() - {"label_field", "reason_field", "confidence_field"}
    assert value["errors"] == {"message_path":"error"}


if __name__ == "__main__":
    raw = (ROOT / "sku-in-stock.yaml").read_bytes()
    assert raw.startswith(b"# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0\n")
    value = json.loads(raw.split(b"\n", 1)[1])
    validate(value)
    # Prove a forbidden endpoint-local schema cannot pass this gate.
    bad = json.loads(json.dumps(value))
    bad["endpoints"][0]["input_schema"] = {}
    try:
        validate(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("descriptor negative control did not fail")
    provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
    assert provenance["binary_wrap"] == 0
    for item in provenance["files"]:
        assert item["lineage"] == "own"
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    print("PASS: descriptor allowlist, negative control and deployed source hashes")
