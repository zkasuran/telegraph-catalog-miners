# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Pure training-format schedule and clean source-held-out audit selection."""
import json
import re
from collections import Counter

from contracts import FAMILY, HERE, sha


def plan():
    value = json.loads((HERE / "balanced.json").read_bytes())
    validate_plan(value)
    return value


def validate_plan(value):
    expected = {"schema_version": 1, "family": FAMILY, "arm": "B3", "epochs": 3, "source_groups": 768,
                "validation_groups": 128, "audit_groups": 256, "audit_per_label": 128, "job_minutes": 330,
                "audit_salt": "moderation-format-balanced-independent-v1"}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("frozen format-balanced recipe/source/audit counts changed")
    if not re.fullmatch(r"[0-9a-f]{64}", value.get("excluded_audit_groups_sha256", "")):
        raise ValueError("exposed-audit exclusions unbound")
    for key in ("checkpoint_sha256", "config_sha256", "untrained_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", value.get("warm_start", {}).get(key, "")):
            raise ValueError("fitted and genuine-untrained inputs unbound")


def style_positions(rows):
    by_label = {label: sorted(r["group_id"] for r in rows if r["label"] == label) for label in ("Flagged", "Safe")}
    if len(rows) != len({r["group_id"] for r in rows}) or set(r["label"] for r in rows) != set(by_label) or len(by_label["Safe"]) != len(by_label["Flagged"]) or len(by_label["Safe"]) % 3:
        raise ValueError("unique, balanced source groups divisible into three styles required")
    return {group: i for groups in by_label.values() for i, group in enumerate(groups)}


def training_batch(batch, positions, epoch, pair):
    if type(epoch) is not int or not 0 <= epoch < 3:
        raise ValueError("extra format epoch refused")
    result = []
    for row in batch:
        if row["group_id"] not in positions:
            raise ValueError("non-training source cannot enter a format batch")
        original = {**row, "text": json.loads(row["prompt"])["text"]}
        result.append(training_pair(original, (positions[row["group_id"]] + epoch) % 3, pair))
    return result


def training_pair(original, style, pair):
    """Source-anchored training only. Bare labels remain an honest unseen format."""
    row = {**pair(FAMILY, original, style), "margin": 0.0}
    if style == 2:
        opposite = {"Flagged": "Safe", "Safe": "Flagged"}[original["label"]]
        row["chosen"] = original["label"] + ". Text assessed: " + original["text"]
        row["rejected"] = opposite + ". Text assessed: " + original["text"]
    return row


def training_views(rows, pair):
    return {name: [training_pair({**r, "text": json.loads(r["prompt"])["text"]}, style, pair) for r in rows]
            for style, name in enumerate(("json", "prose", "label_led_source"))}


def audit_select(rows, excluded, salt, per_label=128):
    groups, conflicts = {}, set()
    scanned = 0
    for row in rows:
        scanned += 1
        if scanned > 300000:
            raise ValueError("official source scan ceiling")
        group = row["group_id"]
        if group in excluded:
            continue
        if group in groups and groups[group]["label"] != row["label"]:
            conflicts.add(group)
        groups.setdefault(group, row)
    selected = []
    for label in ("Flagged", "Safe"):
        eligible = [r for g, r in groups.items() if g not in conflicts and r["label"] == label]
        eligible.sort(key=lambda r: sha((salt + ":" + r["group_id"]).encode()))
        if len(eligible) < per_label:
            raise ValueError("insufficient clean independent audit groups")
        selected.extend(eligible[:per_label])
    if Counter(r["label"] for r in selected) != {"Flagged": per_label, "Safe": per_label}:
        raise ValueError("independent audit label coverage changed")
    return sorted(selected, key=lambda r: r["group_id"]), {"scanned_rows": scanned, "unique_groups": len(groups), "conflicting_groups": len(conflicts)}
