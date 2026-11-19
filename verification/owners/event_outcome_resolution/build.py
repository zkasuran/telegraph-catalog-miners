# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Build counterbalanced preferences from complete official award records."""
import argparse
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from acquire import sha, write
from scorer import LABELS, MAX_EVENT_BYTES, normalized, response

HERE = Path(__file__).resolve().parent


def names(record):
    return [p.get("knownName", p.get("orgName", p.get("fullName", {}))).get("en", "")
            for p in record.get("laureates", [])]


def period(year):
    return "train" if year <= 1950 else "validation" if year <= 1970 else "audit"


def event_text(r, candidate, mode):
    announcement = date.fromisoformat(r["dateAwarded"])
    cutoff = announcement + timedelta(days=1) if mode != "pre_cutoff" else announcement - timedelta(days=1)
    winners = "unknown" if mode == "missing_evidence" else " | ".join(names(r))
    return (f"{r['awardYear']} {r['category']['en']}: Did {candidate} win? Winners: {winners}; "
            f"announced {announcement}; cutoff {cutoff}; evidence-only.")


def check_splits(splits):
    seen_people, seen_events, seen_prompts = set(), set(), set()
    for split in ("train", "validation", "audit"):
        rows = splits[split]
        people = {p for r in rows for p in r["people"]}
        events = {r["event_id"] for r in rows}
        prompts = {normalized(r["prompt"]) for r in rows}
        if people & seen_people or events & seen_events or prompts & seen_prompts:
            raise ValueError("cross-split normalized entity/event/prompt overlap")
        seen_people |= people
        seen_events |= events
        seen_prompts |= prompts
        if len(events) < 20 or not rows:
            raise ValueError("insufficient grouped source events")
        # Every directed label pair occurs equally, within each source event.
        for eid in events:
            counts = Counter((r["label"], r["wrong_label"]) for r in rows if r["event_id"] == eid)
            if len(counts) != 6 or len(set(counts.values())) != 1:
                raise ValueError("label-pair counterbalance failed")


def shortcuts(rows):
    def metric(feature):
        wins = sum((feature(r["prompt"], r["chosen"]) > feature(r["prompt"], r["rejected"])) +
                   0.5 * (feature(r["prompt"], r["chosen"]) == feature(r["prompt"], r["rejected"]))
                   for r in rows)
        return wins / len(rows)
    return {"prefer_longer": metric(lambda p, a: len(a)),
            "utf8_length": metric(lambda p, a: len(a.encode("utf-8"))),
            "token_overlap": metric(lambda p, a: len(set(normalized(p).split()) & set(normalized(a).split()))),
            "answer_only_label": "label token counts counterbalanced; neural surface probe still required"}


def build(source, out):
    receipts = json.loads((source / "receipts.json").read_text())
    for item in receipts["sources"]:
        # Original terms-page HTML and raw live miner inventory are private
        # evidence, not redistributable training inputs. Data and MIT primary
        # code must be present and byte-identical in the remote publication.
        if (item["path"].startswith(("prizes-", "kit-")) and
                sha((source / item["path"]).read_bytes()) != item["sha256"]):
            raise ValueError("source receipt hash mismatch")
    records = json.loads((source / "prizes.json").read_text())
    if len(records) != receipts["reported_count"] or len({(r["awardYear"], r["category"]["en"]) for r in records}) != len(records):
        raise ValueError("source inventory incomplete")
    # Rebuild from the original pages, not a separately editable merged file.
    pages = [i for i in receipts["sources"] if i["path"].startswith("prizes-")]
    merged = [r for i in pages for r in json.loads((source / i["path"]).read_text())["nobelPrizes"]]
    if records != merged:
        raise ValueError("merged source differs from original pages")
    ownership = defaultdict(set)
    for r in records:
        for p in r.get("laureates", []):
            ownership[("id", p["id"])].add(period(int(r["awardYear"])))
        for n in names(r):
            ownership[("name", normalized(n))].add(period(int(r["awardYear"])))
    split_records = defaultdict(list)
    excluded = Counter()
    for r in records:
        if not names(r) or any(not n for n in names(r)):
            excluded["no_published_recipients"] += 1
            continue
        if not r.get("dateAwarded"):
            excluded["missing_announcement_date"] += 1
            continue
        if date.fromisoformat(r["dateAwarded"]).year < int(r["awardYear"]):
            excluded["source_date_precedes_edition_quarantine"] += 1
            print("Quarantined inconsistent source date:", r["awardYear"], r["category"]["en"], r["dateAwarded"])
            continue
        if any(len(ownership[("id", p["id"])]) > 1 for p in r["laureates"]) or any(
                len(ownership[("name", normalized(n))]) > 1 for n in names(r)):
            excluded["entity_crosses_time_boundary"] += 1
            continue
        split_records[period(int(r["awardYear"]))].append(r)
    splits = {s: [] for s in ("train", "validation", "audit")}
    retained = {}
    for split, rs in split_records.items():
        for r in rs:
            # Negatives use real recipients of another category in the same year.
            # Their identity stays in the same time split. No fabricated person.
            peers = [x for x in rs if x["awardYear"] == r["awardYear"] and x["category"] != r["category"]]
            peer_names = sorted({n for x in peers for n in names(x)} - set(names(r)))
            if not peer_names:
                excluded["no_same_year_negative"] += 1
                continue
            candidates = [(names(r)[0], "after", "yes"), (peer_names[0], "after", "no"),
                          (names(r)[0], "pre_cutoff", "unresolved")]
            prompts = [(event_text(r, person, mode), label, mode, person) for person, mode, label in candidates]
            if any(len(p.encode("utf-8")) > MAX_EVENT_BYTES for p, *_ in prompts):
                excluded["onchain_event_byte_ceiling"] += 1
                continue
            eid = r["awardYear"] + "/" + r["category"]["en"]
            retained[eid] = {"split": split, "source_sha256": sha(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()),
                             "recipient_ids": [p["id"] for p in r["laureates"]]}
            people = sorted({normalized(n) for n in names(r)} | {normalized(peer_names[0])})
            for prompt, label, mode, person in prompts:
                for wrong in LABELS:
                    if wrong == label:
                        continue
                    splits[split].append({"prompt": prompt, "chosen": response(prompt, label),
                        "rejected": response(prompt, wrong), "label": label, "wrong_label": wrong,
                        "event_id": eid, "people": people, "mode": mode, "claimant": person,
                        "source_label_kind": "published_official_award_membership_and_date",
                        "outcome_label_kind": "constructed_evidence_cutoff_policy" if mode == "pre_cutoff" else "derived_complete_list_membership",
                        "preference_authorship": "constructed by AI-assisted code, not human-reviewed",
                        "source": "NP_API", "source_record_sha256": retained[eid]["source_sha256"]})
    check_splits(splits)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in splits.items():
        raw = b"".join((json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n").encode() for r in rows)
        (out / (split + ".jsonl")).write_bytes(raw)
        files[split] = {"sha256": sha(raw), "pairs": len(rows), "events": len({r["event_id"] for r in rows}),
                        "period": "<=1950" if split == "train" else "1951-1970" if split == "validation" else ">=1971",
                        "labels": dict(Counter(r["label"] for r in rows)), "shortcuts": shortcuts(rows),
                        "max_event_bytes": max(len(r["prompt"].encode()) for r in rows)}
        if any(max(v, 1-v) > 0.65 for k, v in files[split]["shortcuts"].items() if isinstance(v, float)):
            raise ValueError("shortcut gate failed")
    write(out / "group-manifest.json", retained)
    report = {"schema_version": 1, "family": "EVENT_OUTCOME_RESOLUTION", "source_inventory": len(records),
              "source_receipts_sha256": sha((source / "receipts.json").read_bytes()),
              "files": files, "excluded": dict(excluded), "normalized_cross_split_overlap": 0,
              "untouched_neural_audit": True, "audit_rule_measurement": "deterministic reference only",
              "source_scope": "official historical award-list entailment with supplied evidence, not future prediction"}
    write(out / "data-report.json", report)
    print(json.dumps(report, indent=2))
    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=HERE / "sources")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.out)
