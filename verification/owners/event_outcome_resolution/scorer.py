# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Strict source-scoped settlement diagnostic, not a registered neural adapter."""
import json
import re
import unicodedata
from datetime import date

MAX_EVENT_BYTES = 256
MAX_ANSWER_BYTES = 4096
LABELS = ("yes", "no", "unresolved")
PATTERN = re.compile(r"^(\d{4}) ([A-Za-z ]+): Did (.+) win\? Winners: (.+); announced ([0-9-]+|unknown); cutoff ([0-9-]+); evidence-only\.$")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def render(prompt, answer):
    # Literal reference trainer render, including whitespace. Do not normalize.
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def parse_event(event):
    if not isinstance(event, str) or len(event.encode("utf-8")) > MAX_EVENT_BYTES:
        raise ValueError("event must be a string of at most 256 UTF-8 bytes")
    match = PATTERN.fullmatch(event)
    if match is None:
        raise ValueError("unsupported evidence-scoped event grammar")
    year, category, candidate, winners, announced, cutoff = match.groups()
    cutoff = date.fromisoformat(cutoff)
    announced = None if announced == "unknown" else date.fromisoformat(announced)
    # Some historical editions were announced in a later year. Preserve that
    # authoritative date rather than rewriting it to the award's nominal year.
    if announced is not None and announced.year < int(year):
        raise ValueError("announcement precedes award edition")
    people = [] if winners == "unknown" else winners.split(" | ")
    if not 0 <= len(people) <= 3 or len({normalized(p) for p in people}) != len(people):
        raise ValueError("invalid recipient list")
    if any(not p.strip() or any(c in p for c in "\n\r;") for p in people):
        raise ValueError("invalid recipient name")
    return {"year": int(year), "category": category, "candidate": candidate,
            "winners": people, "announced": announced, "cutoff": cutoff}


def resolve(event):
    """Entail only supplied evidence. Unknown and pre-announcement are not no."""
    p = parse_event(event)
    # Date-only API does not supply time of day. Same-day cutoff is conservative.
    if not p["winners"] or p["announced"] is None or p["cutoff"] <= p["announced"]:
        return "unresolved"
    return "yes" if normalized(p["candidate"]) in {normalized(x) for x in p["winners"]} else "no"


def event_key(event):
    p = parse_event(event)
    return f"{p['year']} {p['category']}: Did {p['candidate']} win? Cutoff: {p['cutoff']}."


def response(event, label, style="json"):
    if label not in LABELS:
        raise ValueError("unsupported outcome")
    if style == "json":
        return json.dumps({"event": event_key(event), "outcome": label, "resolved_yes": label == "yes"},
                          ensure_ascii=False, separators=(",", ":"))
    if style == "prose":
        return f"Event: {event_key(event)}\nOutcome: {label}\nResolved YES: {'true' if label == 'yes' else 'false'}."
    raise ValueError("unsupported response style")


def unique_object(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate JSON key")
        out[k] = v
    return out


def score(event, answer, style="json"):
    """Integer reference score with no fitted weights and no hidden archive lookup."""
    expected = resolve(event)
    if not isinstance(answer, str) or len(answer.encode("utf-8")) > MAX_ANSWER_BYTES:
        raise ValueError("answer ceiling")
    if style == "prose":
        return 10000 if answer == response(event, expected, style) else 0
    if style != "json":
        raise ValueError("unsupported response style")
    try:
        obj = json.loads(answer, object_pairs_hook=unique_object)
    except (ValueError, RecursionError):
        return 0
    if not isinstance(obj, dict) or set(obj) != {"event", "outcome", "resolved_yes"}:
        return 0
    if (obj["event"] != event_key(event) or obj["outcome"] not in LABELS or
            type(obj["resolved_yes"]) is not bool or obj["resolved_yes"] != (obj["outcome"] == "yes")):
        return 0
    return 10000 if obj["outcome"] == expected else 0
