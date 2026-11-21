# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source-only input contracts for a separate weather development executor."""
from collections import Counter
from datetime import date
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
FAMILY_ROOT = HERE.parent
sys.path.insert(0, str(FAMILY_ROOT))
from core import identity, inputs_question  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def view_row(row, view):
    result = dict(row)
    for side in ("chosen", "rejected"):
        if view == "json":
            continue
        if view not in {"bare", "celsius"}:
            raise ValueError("unsupported development answer view")
        value = json.loads(row[side], parse_float=Decimal, parse_int=Decimal)["temp_max_c"]
        result[side] = format(value, ".6f") + (" degrees Celsius" if view == "celsius" else "")
    return result


def training_representations(rows):
    # One view per source question, not three copies counted as independent data.
    views = ("json", "celsius", "bare")
    return [{**view_row(row, views[index % 3]), "answer_view": views[index % 3]} for index, row in enumerate(rows)]


def validate_rows(splits, spec):
    if set(splits) != {"train", "validation"}:
        raise ValueError("development may load only its train and validation splits")
    previous_entities, previous_months, previous_prompts = set(), set(), set()
    for split, rows in splits.items():
        if not isinstance(rows, list) or len(rows) != spec["pairs"][split] or len(rows) > 512:
            raise ValueError("development pair population differs")
        entities, months, prompts, groups = set(), set(), set(), set()
        for row in rows:
            station, day = row["station"], row["day"]
            if not isinstance(station, str) or not re.fullmatch(r"USW[0-9]{8}", station):
                raise ValueError("literal WBAN source identity required")
            if station not in spec["policy"][split] or station in spec["excluded_original_nontraining_entities"]:
                raise ValueError("source entity outside development role")
            bounds = spec["policy"][split][station]
            date.fromisoformat(day)
            if not bounds["start"] <= day <= bounds["end"]:
                raise ValueError("source date outside frozen development window")
            name = spec["station_names"][station]
            if row["prompt"] != inputs_question(name, day) or row["group"] != station + ":" + day or row["event_window"] != day[:7]:
                raise ValueError("literal input or source grouping differs")
            if type(row.get("margin")) not in (int, float) or not math.isfinite(row["margin"]) or row["margin"] != 0:
                raise ValueError("development margin must be the frozen finite zero")
            if row["offset_tenths"] not in (-20, 20) or type(row["offset_tenths"]) is not int:
                raise ValueError("constructed offset is not the declared integer")
            source = row["source_evidence"]
            if (source.get("flag") != "1" or source.get("quality") != " " or source.get("measurement") != " "
                    or type(source.get("line")) is not int or source["line"] < 1
                    or not re.fullmatch(r"[a-f0-9]{64}", source.get("sha256", ""))
                    or source.get("url") != "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/" + station + ".dly"):
                raise ValueError("source grant, annotation or capture identity differs")
            cell, header = source.get("cell"), source.get("record_header")
            if (not isinstance(cell, str) or len(cell) != 8 or cell[5:] != "  1" or not isinstance(header, str)
                    or len(header) != 21 or header != station + day[:4] + day[5:7] + "TMAX"):
                raise ValueError("unaltered admitted source cell/header required")
            observed = Decimal(int(cell[:5])) / 10
            for side in ("chosen", "rejected"):
                answer = json.loads(row[side], parse_float=Decimal, parse_int=Decimal)
                value = answer.get("temp_max_c")
                if (set(answer) != {"location", "date", "temp_max_c"} or answer["location"] != name or answer["date"] != day
                        or not isinstance(value, Decimal) or not value.is_finite() or not -90 <= value <= 60
                        or value.as_tuple().exponent != -6):
                    raise ValueError("literal six-decimal source answer required")
                expected = observed if side == "chosen" else observed + Decimal(row["offset_tenths"]) / 10
                if value != expected:
                    raise ValueError("constructed answer no longer matches admitted source")
            entities.add(station)
            months.add(day[:7])
            prompts.add(identity(row["prompt"]))
            groups.add(row["group"])
        if len(prompts) != len(rows) or len(groups) != len(rows):
            raise ValueError("duplicate normalized source question or station-day")
        if entities & previous_entities or months & previous_months or prompts & previous_prompts:
            raise ValueError("development entity/time/normalized overlap")
        if Counter(row["offset_tenths"] for row in rows) != {20: len(rows) // 2, -20: len(rows) // 2}:
            raise ValueError("development offset directions are not balanced")
        previous_entities |= entities
        previous_months |= months
        previous_prompts |= prompts
    return True


def load_inputs(root=HERE):
    spec = json.loads((root / "input-contract.json").read_text())
    if spec.get("schema_version") != 1 or spec.get("family") != "WEATHER_FORECAST_VERIFY" or spec.get("development_only") is not True:
        raise ValueError("exact development-only family contract required")
    for name, expected in spec["files"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not (root / path).is_file() or (root / path).stat().st_size > 2 << 20:
            raise ValueError("bounded portable source path required")
        if digest(root / path) != expected:
            raise ValueError("development source bytes changed")
    rows = {split: [json.loads(line) for line in (root / "data" / (split + ".jsonl")).read_text().splitlines()]
            for split in ("train", "validation")}
    validate_rows(rows, spec)
    report = json.loads((root / "data/data-report.json").read_text())
    if report.get("independent_audit") is not False or report.get("source_evidence_in_prompt") is not False:
        raise ValueError("development material cannot masquerade as an audit or injected source passage")
    return spec, rows, report


def validate_development_workflow(spec):
    inputs = spec["on"]["workflow_dispatch"]["inputs"]
    flag = inputs.get("development_only")
    if flag != {"type": "boolean", "default": False, "required": False}:
        raise ValueError("opt-in development flag must default false")
    train_steps = spec["jobs"]["train"]["steps"]
    original = [step for step in train_steps if step.get("name") == "Train only on frozen training then export selected validation candidate"]
    development = [step for step in train_steps if step.get("name") == "Separate development-only cold-source experiment"]
    if len(original) != 1 or original[0].get("if") != "${{ !inputs.development_only }}":
        raise ValueError("original recipe must not run on development inputs")
    if len(development) != 1 or development[0].get("if") != "${{ inputs.development_only }}" or development[0].get("timeout-minutes") != 130:
        raise ValueError("development executor must remain opt-in and bounded")
    if spec["jobs"]["audit"]["if"] != "${{ inputs.mode == 'experiment' && !inputs.development_only }}":
        raise ValueError("development must never consume the original reserved audit")
    return True
