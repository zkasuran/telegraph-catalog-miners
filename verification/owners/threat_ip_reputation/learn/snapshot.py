# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Strict observation-feed validation and conservative entity grouping.

Source facts are publisher observations. Membership is not universal proof of
maliciousness and absence is not a benign label. No model operations occur here.
"""
import argparse
import copy
import csv
import datetime as dt
import hashlib
import io
import ipaddress
import json
import re
from pathlib import Path

COLUMNS = ("ip", "tier", "bans", "attempts", "first_seen", "last_seen", "first_banned", "asn")
TIERS = {"credential", "scanner", "loader"}
MAX_BYTES = 2 << 20
MAX_ROWS = 20000
MAX_COUNTER = (1 << 63) - 1
ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
TRAIN_END = "2026-11-07T23:59:59Z"
VALIDATION_END = "2026-11-18T23:59:59Z"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def unique(items):
    obj = {}
    for key, value in items:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def read(path):
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("source byte ceiling exceeded")
    return raw


def decode(raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_BYTES:
        raise ValueError("source byte ceiling exceeded")
    try:
        obj = json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant)
    except RecursionError as exc:
        raise ValueError("source JSON nesting ceiling") from exc
    if not isinstance(obj, dict):
        raise ValueError("source JSON root must be an object")
    return obj


def instant(value):
    if not isinstance(value, str) or not ISO.fullmatch(value):
        raise ValueError("literal UTC observation timestamp required")
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def address(value, query=False):
    if not isinstance(value, str) or len(value) > 64 or "%" in value:
        raise ValueError("invalid IP literal")
    ip = ipaddress.ip_address(value)
    if query and isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if not query and (ip.version != 4 or str(ip) != value or not ip.is_global or ip.is_reserved or ip.is_multicast):
        raise ValueError("feed address must be canonical global IPv4")
    return ip


def counter(value):
    if type(value) is not int or not 0 <= value <= MAX_COUNTER:
        raise ValueError("source counter must be a bounded nonnegative integer")
    return value


def validate_json(obj):
    if not isinstance(obj, dict) or set(obj) != {"meta", "ips"} or not isinstance(obj["meta"], dict):
        raise ValueError("strict meta/ips source structure required")
    meta, rows = obj["meta"], obj["ips"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
        raise ValueError("source row count out of bounds")
    if meta.get("license") != "CC0-1.0" or type(meta.get("window_days")) is not int or not 1 <= meta["window_days"] <= 366:
        raise ValueError("source grant or recency window changed")
    if meta.get("schema_version") not in (None, "1.0"):
        raise ValueError("unrecognized source schema version")
    if meta.get("csv_columns") != list(COLUMNS):
        raise ValueError("literal CSV column contract changed")
    updated = instant(meta.get("updated"))
    floor = counter(meta.get("scanner_min_events"))
    if floor < 1000:
        raise ValueError("published scanner floor cannot be weakened")
    seen, counts = set(), {tier: 0 for tier in TIERS}
    cutoff = updated - dt.timedelta(days=meta["window_days"])
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(COLUMNS):
            raise ValueError("source observation columns changed")
        ip = str(address(row["ip"]))
        if ip in seen:
            raise ValueError("duplicate source IP")
        seen.add(ip)
        if row["tier"] not in TIERS:
            raise ValueError("unknown source tier")
        bans, attempts = counter(row["bans"]), counter(row["attempts"])
        if row["tier"] == "credential" and attempts < 50:
            raise ValueError("credential count below source floor")
        if row["tier"] == "scanner" and attempts < floor:
            raise ValueError("scanner count below source floor")
        if row["tier"] != "credential" and bans != 0:
            raise ValueError("scanner/loader ban counter must remain zero")
        first, last = instant(row["first_seen"]), instant(row["last_seen"])
        if not cutoff <= first <= last <= updated:
            raise ValueError("observation outside exact source time window")
        if row["first_banned"] is not None:
            banned = instant(row["first_banned"])
            if bans == 0 or banned > updated:
                raise ValueError("inconsistent ban bookkeeping")
        elif bans > 0:
            raise ValueError("positive bans require bookkeeping timestamp")
        if not isinstance(row["asn"], str) or not re.fullmatch(r"AS\d{1,10}", row["asn"]) or int(row["asn"][2:]) > 0xffffffff:
            raise ValueError("source ASN must be a bounded AS number: " + repr(row["asn"]))
        counts[row["tier"]] += 1
    if type(meta.get("count")) is not int or meta["count"] != len(rows) or meta.get("count_by_tier") != counts:
        raise ValueError("published row or tier counts disagree")
    return meta, rows


def read_csv(raw, headed):
    if len(raw) > MAX_BYTES:
        raise ValueError("CSV byte ceiling")
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
    if headed:
        if not rows or rows.pop(0) != list(COLUMNS):
            raise ValueError("CSV header order mismatch")
    elif rows and rows[0] and rows[0][0] == "ip":
        raise ValueError("header-less MISP contract received a header")
    if not 1 <= len(rows) <= MAX_ROWS or any(len(row) != len(COLUMNS) for row in rows):
        raise ValueError("CSV shape mismatch")
    for row in rows:
        if any(cell.lstrip().startswith(("=", "+", "-", "@")) or "\n" in cell or "\r" in cell for cell in row):
            raise ValueError("CSV formula/control character refused")
    return rows


def validate_formats(root):
    meta, rows = validate_json(decode(read(root / "blocklist.json")))
    expected = [[str(row[key]) if row[key] is not None else "" for key in COLUMNS] for row in rows]
    if read_csv(read(root / "blocklist.csv"), True) != expected or read_csv(read(root / "blocklist.misp.csv"), False) != expected:
        raise ValueError("source format values or order disagree")
    values = []
    for line in read(root / "blocklist.txt").decode().splitlines():
        if line and not line.startswith("#"):
            values.append(str(address(line)))
    if values != [row["ip"] for row in rows]:
        raise ValueError("literal TXT publication contract differs")
    return meta, rows


def curate_formats(root):
    """Keep the raw failure visible and quarantine rows, never repair source facts.

    This is not a pass of the publisher's whole feed. Every retained row must
    independently pass the same strict row gate, with no lowered thresholds.
    """
    obj = decode(read(root / "blocklist.json"))
    if not isinstance(obj, dict) or set(obj) != {"meta", "ips"} or not isinstance(obj["meta"], dict) or not isinstance(obj["ips"], list):
        raise ValueError("source structure invalid before curation")
    meta, rows = obj["meta"], obj["ips"]
    if not 1 <= len(rows) <= MAX_ROWS or any(not isinstance(row, dict) or set(row) != set(COLUMNS) for row in rows):
        raise ValueError("source row shape invalid before curation")
    if type(meta.get("count")) is not int or meta["count"] != len(rows) or len({row["ip"] for row in rows}) != len(rows):
        raise ValueError("source count or duplicate identity invalid before curation")
    raw_counts = {tier: sum(row["tier"] == tier for row in rows) for tier in TIERS}
    if meta.get("count_by_tier") != raw_counts:
        raise ValueError("source tier counts invalid before curation")
    expected = [[str(row[key]) if row[key] is not None else "" for key in COLUMNS] for row in rows]
    if read_csv(read(root / "blocklist.csv"), True) != expected or read_csv(read(root / "blocklist.misp.csv"), False) != expected:
        raise ValueError("uncurated source formats disagree")
    txt = [line for line in read(root / "blocklist.txt").decode().splitlines() if line and not line.startswith("#")]
    if txt != [row["ip"] for row in rows]:
        raise ValueError("uncurated TXT source identity order differs")
    kept, rejected = [], []
    for row in rows:
        single = copy.deepcopy(meta)
        single["count"] = 1
        single["count_by_tier"] = {tier: int(row["tier"] == tier) for tier in TIERS}
        try:
            validate_json({"meta": single, "ips": [row]})
        except ValueError as exc:
            rejected.append({"ip": row["ip"], "reason": str(exc)})
        else:
            kept.append(row)
    derived = copy.deepcopy(meta)
    derived["count"] = len(kept)
    derived["count_by_tier"] = {tier: sum(row["tier"] == tier for row in kept) for tier in TIERS}
    validate_json({"meta": derived, "ips": kept})
    return derived, kept, {"raw_publication_rows": len(rows), "retained_rows": len(kept), "quarantined_rows": rejected,
                           "raw_publication_passed_strict_gate": not rejected,
                           "all_four_raw_formats_match": True, "row_facts_modified": False}


def groups(rows):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    keys = {}
    for i, row in enumerate(rows):
        net = str(ipaddress.ip_network(row["ip"] + "/24", strict=False))
        for key in (("network", net), ("asn", row["asn"])):
            if key in keys:
                parent[find(i)] = find(keys[key])
            keys[key] = i
    result = {}
    for i, row in enumerate(rows):
        result.setdefault(find(i), []).append(row)
    output = []
    for values in result.values():
        values = sorted(values, key=lambda row: int(address(row["ip"])))
        end = max(row["last_seen"] for row in values)
        output.append({"group_id": sha(json.dumps([row["ip"] for row in values]).encode()), "rows": values,
                       "last_seen_max": end, "unknown_asn": any(row["asn"] == "AS0" for row in values),
                       "phase": "train" if end <= TRAIN_END else "validation" if end <= VALIDATION_END else "reserved_later"})
    return sorted(output, key=lambda group: group["group_id"])


def evidence(meta, rows, query, as_of):
    ip = str(address(query, query=True))
    at = instant(as_of)
    updated = instant(meta["updated"])
    if at < updated:
        raise ValueError("future snapshot cannot answer earlier as-of")
    row = next((row for row in rows if row["ip"] == ip), None)
    if row is None:
        return {"ip": ip, "source_state": "not_observed_in_snapshot", "current_universal_reputation": "unknown"}
    try:
        end = instant(row["last_seen"]) + dt.timedelta(days=meta["window_days"])
    except OverflowError as exc:
        raise ValueError("source expiry exceeds timestamp bounds") from exc
    state = "publisher_observed_in_window" if at <= end else "historical_publisher_observation"
    return {"ip": ip, "source_state": state, "tier": row["tier"], "last_seen": row["last_seen"],
            "window_expires": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "current_universal_reputation": "unknown"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--curate", action="store_true")
    args = parser.parse_args()
    curation = None
    if args.curate:
        meta, rows, curation = curate_formats(args.source)
    else:
        meta, rows = validate_formats(args.source)
    components = groups(rows)
    stats = {name: {"groups": sum(group["phase"] == name and not group["unknown_asn"] for group in components),
                    "rows": sum(len(group["rows"]) for group in components if group["phase"] == name and not group["unknown_asn"])}
             for name in ("train", "validation", "reserved_later")}
    print(json.dumps({"rows": len(rows), "tiers": meta["count_by_tier"], "updated": meta["updated"], "window_days": meta["window_days"],
                      "network_asn_components": len(components), "unknown_asn_groups": sum(group["unknown_asn"] for group in components),
                      "development_partition_stats": stats, "curation": curation,
                      "independent_sensors_claimed": False, "model_operations": 0}))


if __name__ == "__main__":
    main()
