# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Freeze real source observations and constructed preferences separately."""
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from core import ArchiveScorer, Observation, identity, inputs_question, parse_dly
from inventory import STATIONS

HERE = Path(__file__).resolve().parent
SPLIT_POLICY = {
    "train": {"stations": (STATIONS[1], STATIONS[4], STATIONS[5]), "start": "2025-09-01", "end": "2025-10-31", "count": 128},
    "validation": {"stations": (STATIONS[3],), "start": "2025-11-01", "end": "2025-12-31", "count": 32},
    "audit": {"stations": (STATIONS[0], STATIONS[2]), "start": "2026-02-01", "end": "2026-03-31", "count": 64},
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def jsonl(rows):
    return "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows).encode()


def load_archive():
    sources = HERE / "sources"
    if not all((sources / ("all-" + station + ".dly")).exists() for station in STATIONS):
        # The published projection contains only granted NWS flag-1 material.
        # Unreviewed foreign-source raw records never enter the public bundle.
        raw = (HERE / "data/observations.jsonl").read_bytes()
        report = json.loads((HERE / "data/data-report.json").read_text())
        if sha(raw) != report["observation_sha256"]:
            raise ValueError("granted source projection digest mismatch")
        rows = [Observation(**json.loads(line)) for line in raw.splitlines()]
        observations = {(r.station, r.day, r.element): r for r in rows}
        if len(observations) != len(rows) or any(r.source != "1" or r.line < 1 for r in rows):
            raise ValueError("invalid source projection identity or grant")
        names = json.loads((HERE / "data/station-names.json").read_text())
        return observations, names
    manifest = json.loads((sources / "capture-manifest.json").read_text())
    observations = {}
    for station in STATIONS:
        filename = "all-" + station + ".dly"
        raw = (sources / filename).read_bytes()
        if sha(raw) != manifest[filename]["sha256"]:
            raise ValueError("raw source digest mismatch")
        observations.update(parse_dly(raw, station))
    raw = (sources / "ghcnd-stations.txt").read_bytes()
    if sha(raw) != manifest["ghcnd-stations.txt"]["sha256"]:
        raise ValueError("station inventory digest mismatch")
    names = {line[:11]: line[41:71].strip() for line in raw.decode("ascii").splitlines() if line[:11] in STATIONS}
    return observations, names


def candidate(value, station_name, day_text, style):
    if style == "json":
        return '{"location":' + json.dumps(station_name) + ',"date":' + json.dumps(day_text) + ',"temp_max_c":' + str(value) + '}'
    if style == "celsius":
        return str(value) + " degrees Celsius"
    if style == "bare":
        return str(value)
    raise ValueError("unknown style")


def construct(records, names):
    rows = []
    for index, observation in enumerate(records):
        # Exactly balance directions. No score or audit result selects candidates.
        offset = 20 if index % 2 == 0 else -20
        wrong = Decimal(observation.tenths + offset) / 10
        if not -90 <= wrong <= 60:
            raise ValueError("constructed negative outside live range")
        name = names[observation.station]
        rows.append({"SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0", "prompt": inputs_question(name, observation.day),
                     "chosen": candidate(observation.value, name, observation.day, "json"),
                     "rejected": candidate(wrong, name, observation.day, "json"),
                     "margin": 0.0, "station": observation.station, "day": observation.day,
                     "group": observation.station + ":" + observation.day,
                     "event_window": observation.day[:7], "offset_tenths": offset,
                     "authorship": "AI-assisted deterministic preference construction; not human-reviewed"})
    return rows


def shortcut_features(row):
    # Integer-only frozen coefficients. Directions fit exclusively on train.
    chosen, rejected = row["chosen"], row["rejected"]
    def numeric(text):
        if text.startswith("{"):
            value = json.loads(text, parse_float=Decimal, parse_int=Decimal)["temp_max_c"]
        else:
            value = Decimal(text.partition(" ")[0])
        return int(value * 10)
    def tokens(text):
        return set(re.findall(r"\w+", identity(text)))
    return {
        "length": len(chosen.encode()) - len(rejected.encode()),
        "answer_only_numeric": numeric(chosen) - numeric(rejected),
        "answer_only_digits": sum(c.isdigit() for c in chosen) - sum(c.isdigit() for c in rejected),
        "overlap": len(set(re.findall(r"\w+", identity(row["prompt"]))) & tokens(chosen))
                   - len(set(re.findall(r"\w+", identity(row["prompt"]))) & tokens(rejected)),
    }


def surface_tokens(text):
    return Counter(re.findall(r"[a-z0-9']+", text.lower()))


def surface_accuracy(rows, weights):
    wins = 0
    for row in rows:
        difference = surface_tokens(row["chosen"])
        difference.subtract(surface_tokens(row["rejected"]))
        score = sum(weights.get(token, 0) * value for token, value in difference.items())
        wins += 2 if score > 0 else 1 if score == 0 else 0
    return {"wins_half_units": wins, "pairs": len(rows), "accuracy": wins / (2 * len(rows))}


def control_accuracy(rows, coefficients):
    wins = Counter()
    for row in rows:
        for name, value in shortcut_features(row).items():
            signed = coefficients[name] * value
            wins[name] += 2 if signed > 0 else 1 if signed == 0 else 0
    return {name: {"wins_half_units": wins[name], "pairs": len(rows), "accuracy": wins[name] / (2 * len(rows))}
            for name in coefficients}


def verify_splits(splits):
    previous_stations, previous_groups, previous_events, previous_prompts = set(), set(), set(), set()
    for split, rows in splits.items():
        stations = {r["station"] for r in rows}
        groups = {r["group"] for r in rows}
        events = {r["event_window"] for r in rows}
        prompts = {identity(r["prompt"]) for r in rows}
        if len(groups) != len(rows) or len(prompts) != len(rows):
            raise ValueError("duplicate normalized station-day or request")
        if stations & previous_stations or groups & previous_groups or events & previous_events or prompts & previous_prompts:
            raise ValueError("station/entity/time/normalized overlap")
        policy = SPLIT_POLICY[split]
        if any(not policy["start"] <= r["day"] <= policy["end"] or r["station"] not in policy["stations"] for r in rows):
            raise ValueError("row outside frozen split boundary")
        if Counter(r["offset_tenths"] for r in rows) != {20: len(rows) // 2, -20: len(rows) // 2}:
            raise ValueError("unbalanced constructed error direction")
        previous_stations |= stations
        previous_groups |= groups
        previous_events |= events
        previous_prompts |= prompts


def main():
    root = HERE / "data"
    root.mkdir(exist_ok=True)
    observations, names = load_archive()
    # Archive is a source table, never a learned model coefficient or split label.
    selected = [o for o in observations.values() if o.day <= "2026-09-30"]
    original = jsonl([asdict(o) for o in sorted(selected, key=lambda o: (o.station, o.day, o.element))])
    (root / "observations.jsonl").write_bytes(original)
    write_json(root / "station-names.json", names)
    splits, split_reports = {}, {}
    for split, policy in SPLIT_POLICY.items():
        eligible = [o for o in selected if o.element == "TMAX" and o.station in policy["stations"]
                    and policy["start"] <= o.day <= policy["end"]]
        # Deterministic hash spread over the complete eligible population.
        eligible.sort(key=lambda o: sha(("weather-source-v1:" + o.station + o.day).encode()))
        if len(eligible) < policy["count"]:
            raise ValueError("insufficient independently grouped source records")
        rows = construct(eligible[:policy["count"]], names)
        splits[split] = rows
        raw = jsonl(rows)
        (root / (split + ".jsonl")).write_bytes(raw)
        split_reports[split] = {"eligible_originals": len(eligible), "originals": len(rows), "sha256": sha(raw),
                               "stations": sorted({r["station"] for r in rows}), "time_policy": policy,
                               "tmax_negative_c": sum(o.tenths < 0 for o in eligible[:policy["count"]])}
    verify_splits(splits)
    names_features = shortcut_features(splits["train"][0])
    positive = control_accuracy(splits["train"], {k: 1 for k in names_features})
    coefficients = {k: 1 if v["wins_half_units"] >= v["pairs"] else -1 for k, v in positive.items()}
    controls = {"SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0", "fit_split": "train", "integer_coefficients": coefficients,
                "selection": "greater train accuracy; ties choose +1", "validation": control_accuracy(splits["validation"], coefficients)}
    weights = Counter()
    for row in splits["train"]:
        weights.update(surface_tokens(row["chosen"]))
        weights.subtract(surface_tokens(row["rejected"]))
    controls["answer_only_token_weights"] = {key: value for key, value in sorted(weights.items()) if value}
    controls["validation"]["answer_only_bag_of_words"] = surface_accuracy(splits["validation"], controls["answer_only_token_weights"])
    write_json(root / "frozen-controls.json", controls)
    report = {"SPDX-License-Identifier": "LicenseRef-zkasuran-SAND-1.0", "family": "WEATHER_FORECAST_VERIFY", "scope": "NWS CF6 station-specific archived TMAX; lookup is not universal factual proof",
              "source_fact_terms": "NWS public-domain facts, not subject to our SAND copyright",
              "source_observations": len(selected), "source_station_days": len({(o.station, o.day) for o in selected}),
              "observation_sha256": sha(original), "splits": split_reports, "normalized_cross_split_overlap": 0,
              "precipitation": {"trace": sum(o.element == "PRCP" and o.measurement == "T" for o in selected),
                                "measurable_positive": sum(o.element == "PRCP" and o.tenths > 0 for o in selected),
                                "observed_zero": sum(o.element == "PRCP" and o.tenths == 0 and o.measurement == " " for o in selected)},
              "audit": "prepared, no learned model or shortcut audit metric evaluated",
              "forecast_issue_time": "absent and not needed for recorded TMAX; forecast skill is not claimed",
              "station_binding": "official station name or exact ID only; ambiguous bare city is unsupported",
              "preference_authorship": "constructed by AI-assisted code from unchanged source facts, not human-reviewed",
              "views": ["json", "celsius", "bare"], "views_are_not_independent_originals": True}
    write_json(root / "data-report.json", report)
    scorer = ArchiveScorer(observations, names)
    smoke = splits["train"][0]
    good = scorer.score(names[smoke["station"]], smoke["day"], smoke["chosen"])
    bad = scorer.score(names[smoke["station"]], smoke["day"], smoke["rejected"])
    if Decimal(good["score"]) != 0 or Decimal(bad["absolute_error_c"]) != 2:
        raise ValueError("primary scorer smoke failed")
    write_json(root / "primary-smoke.json", {"good": good, "bad": bad, "station": smoke["station"], "day": smoke["day"]})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
