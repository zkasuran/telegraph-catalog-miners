# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Grade conditional source evidence, not universal IP reputation.

No network, clock read, payload execution, fitting or neural work occurs here.
"""
import argparse
import datetime as dt
import hashlib
import ipaddress
import json
import re
from pathlib import Path

MAX_REQUEST = 16384
MAX_ANSWER = 4096
MAX_RULES = 64
MAX_RULE_BYTES = 8192
SNAPSHOT_TTL_SECONDS = 86400  # Our policy, not an ET observation lifetime.
FAMILY = "THREAT_IP_REPUTATION"
ANSWER_KEYS = {"ip", "rule_ids", "evidence_state", "current_abuse", "basis"}
BASIS = "conditional_detection_rule_not_live_observation"
PINNED_MALWARE_SHA256 = "e1d7a3bc20a8344d845d1d4cdad8b88ad4d189efb68bb0e9055ee2a9bb48d0f7"
PINNED_CARDS_SHA256 = "fc57c0dba06af58ba6a4274fe26b2a187ff378afb7037f6a9102aac56bcfb891"
PINNED_CAPTURED_AT = "2026-11-19T11:52:04.418529+00:00"
MAX_SOURCE_BYTES = 32 << 20
MAX_RECEIPT_BYTES = 512 << 10


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_limited(path, ceiling):
    with path.open("rb") as stream:
        raw = stream.read(ceiling + 1)
    if len(raw) > ceiling:
        raise ValueError("file byte ceiling exceeded")
    return raw


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def decode(raw, ceiling):
    if not isinstance(raw, str) or len(raw.encode()) > ceiling:
        raise ValueError("text byte ceiling exceeded")
    try:
        obj = json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except RecursionError as exc:
        raise ValueError("JSON nesting ceiling exceeded") from exc
    if not isinstance(obj, dict):
        raise ValueError("JSON object required")
    return obj


def instant(value):
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError("invalid timestamp")
    value = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.utcoffset() != dt.timedelta(0):
        raise ValueError("explicit UTC timestamp required")
    return value


def ip(value):
    if not isinstance(value, str) or len(value) > 64 or "%" in value:
        raise ValueError("invalid IP literal")
    address = ipaddress.ip_address(value)
    return address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped else address


def direct_rule(record):
    """Extract a finite IPv4 header predicate. Do not execute payload conditions."""
    line = record["rule"]
    if not isinstance(line, str) or len(line.encode()) > MAX_RULE_BYTES:
        raise ValueError("rule byte ceiling exceeded")
    if sha(line.encode()) != record["rule_sha256"]:
        raise ValueError("rule hash mismatch")
    if record["file"] != "emerging-malware.rules" or record["references"]:
        raise ValueError("not a direct ET malware source rule")
    if any(term in line.casefold() for term in ("reference:", "abuse.ch", "spamhaus")):
        raise ValueError("third-party feed or report not cleared")
    head, separator, _ = line.partition(" (")
    fields = head.split()
    if not separator or len(fields) != 7 or fields[0:3] != ["alert", "http", "$HOME_NET"] or fields[4] != "->":
        raise ValueError("unsupported rule direction or protocol")
    literals = fields[5].strip("[]").split(",")
    if any("$" in item or "!" in item for item in literals):
        raise ValueError("non-finite or negative address predicate")
    networks = [ipaddress.ip_network(item, strict=True) for item in literals]
    if any(network.version != 4 for network in networks):
        raise ValueError("only source IPv4 predicates inspected")
    sid = re.search(r"\bsid:(\d+);", line)
    updated = re.search(r"updated_at (\d{4}_\d{2}_\d{2})", line)
    if not sid or int(sid[1]) != record["sid"] or not 2000000 <= int(sid[1]) <= 2799999:
        raise ValueError("SID or grant range mismatch")
    if not updated or updated[1].replace("_", "-") != record["updated_at"]:
        raise ValueError("rule date mismatch")
    return networks


def selected_rules(root):
    if not (root / "sources" / "emerging-malware.rules").is_file():
        cards = decode(read_limited(root / "cards.json", MAX_RECEIPT_BYTES).decode(), MAX_RECEIPT_BYTES)
        if cards.get("source_sha256") != PINNED_MALWARE_SHA256 or cards.get("captured_at") != PINNED_CAPTURED_AT:
            raise ValueError("source or capture identity mismatch")
        require_complete_selection(cards.get("records"))
        return cards["records"]
    manifest = decode(read_limited(root / "capture-manifest.json", MAX_RECEIPT_BYTES).decode(), MAX_RECEIPT_BYTES)
    if not manifest["complete"] or manifest["family"] != FAMILY:
        raise ValueError("incomplete or foreign source capture")
    if manifest["captured_at"] != PINNED_CAPTURED_AT:
        raise ValueError("capture identity mismatch")
    inspection = decode(read_limited(root / "inspection.json", MAX_RECEIPT_BYTES).decode(), MAX_RECEIPT_BYTES)
    source = read_limited(root / "sources" / "emerging-malware.rules", MAX_SOURCE_BYTES)
    if sha(source) != PINNED_MALWARE_SHA256 or manifest["files"]["emerging-malware.rules"]["sha256"] != PINNED_MALWARE_SHA256:
        raise ValueError("complete source hash mismatch")
    lines = source.decode().splitlines()
    selected = []
    for record in inspection["rules"]:
        if record["file"] != "emerging-malware.rules" or record["references"]:
            continue
        if not 1 <= record["line"] <= len(lines) or lines[record["line"] - 1] != record["rule"]:
            raise ValueError("extracted rule does not match primary source bytes")
        direct_rule(record)
        selected.append(record)
    if not 1 <= len(selected) <= MAX_RULES:
        raise ValueError("no bounded useful direct source predicates")
    require_complete_selection(selected)
    return selected


def cards_sha(records):
    return sha(json.dumps(records, sort_keys=True, separators=(",", ":")).encode())


def require_complete_selection(records):
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_RULES:
        raise ValueError("complete source selection required")
    seen = set()
    for record in records:
        if not isinstance(record, dict) or type(record.get("sid")) is not int or type(record.get("line")) is not int or not isinstance(record.get("file"), str):
            raise ValueError("invalid source selection metadata")
        if record["sid"] in seen:
            raise ValueError("duplicate source SID")
        seen.add(record["sid"])
    canonical = sorted(records, key=lambda record: (record["file"], record["line"]))
    if cards_sha(canonical) != PINNED_CARDS_SHA256:
        raise ValueError("complete source selection hash mismatch")


def expected(prompt, records, captured_at):
    require_complete_selection(records)
    if captured_at != PINNED_CAPTURED_AT:
        raise ValueError("capture identity mismatch")
    request = decode(prompt, MAX_REQUEST)
    if set(request) != {"ip", "as_of", "source_sha256", "captured_at"}:
        raise ValueError("strict source-evidence request required")
    if request["source_sha256"] != cards_sha(records) or request["captured_at"] != captured_at:
        raise ValueError("prompt does not bind source cards and capture")
    address = ip(request["ip"])
    as_of, captured = instant(request["as_of"]), instant(captured_at)
    age = (as_of - captured).total_seconds()
    if age < 0:
        raise ValueError("future snapshot cannot support earlier as-of")
    if not 1 <= len(records) <= MAX_RULES:
        raise ValueError("rule count out of bounds")
    ids, seen = [], set()
    for record in records:
        networks = direct_rule(record)
        if record["sid"] in seen:
            raise ValueError("duplicate source SID")
        seen.add(record["sid"])
        if dt.date.fromisoformat(record["updated_at"]) > captured.date():
            raise ValueError("rule update after capture")
        if any(address.version == network.version and address in network for network in networks):
            ids.append(record["sid"])
    state = "stale_snapshot" if age >= SNAPSHOT_TTL_SECONDS else "rule_address_match" if ids else "no_rule_address_match"
    return {"ip": str(address), "rule_ids": sorted(ids), "evidence_state": state,
            "current_abuse": "unknown", "basis": BASIS}


def score(prompt, answer, records, captured_at):
    truth = expected(prompt, records, captured_at)
    candidate = decode(answer, MAX_ANSWER)
    if set(candidate) != ANSWER_KEYS:
        return 0.0
    ids = candidate["rule_ids"]
    if not isinstance(ids, list) or any(type(sid) is not int for sid in ids) or len(ids) > MAX_RULES:
        return 0.0
    try:
        candidate["ip"] = str(ip(candidate["ip"]))
    except ValueError:
        return 0.0
    return float(candidate == truth)


def render(prompt, answer):
    # Retain literal strings. Do not compact JSON or normalize whitespace.
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def prompt_for(address, records, captured_at, as_of=None):
    return json.dumps({"ip": address, "as_of": as_of or captured_at, "captured_at": captured_at,
                       "source_sha256": cards_sha(records)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = decode(read_limited(args.request, MAX_REQUEST + MAX_ANSWER).decode(), MAX_REQUEST + MAX_ANSWER)
        if set(request) != {"prompt", "answer"} or not all(isinstance(request[key], str) for key in request):
            raise ValueError("strict string envelope required")
        records = selected_rules(args.sources)
        result = score(request["prompt"], request["answer"], records, PINNED_CAPTURED_AT)
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({"error": "invalid_request", "kind": "auxiliary_source_evidence_reference", "live_qualified": False}))
        raise SystemExit(2)
    print(json.dumps({"score": result, "kind": "auxiliary_source_evidence_reference", "live_qualified": False}))
    raise SystemExit(0 if result == 1.0 else 1)


if __name__ == "__main__":
    main()
