# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Strict source-bound historical temperature scorer, not a neural adapter."""
import calendar
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import json
import re
import unicodedata

MAX_SOURCE_BYTES = 16 << 20
MAX_RECORDS = 100000
MAX_ANSWER_BYTES = 4096
MAX_LOCATION_BYTES = 256
ELEMENTS = {"TMAX": "temp_max_c", "TMIN": "temp_min_c", "PRCP": "precipitation_mm"}


def identity(text):
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_LOCATION_BYTES:
        raise ValueError("invalid location length or type")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def iso_day(text):
    if not isinstance(text, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError("date must be a literal ISO day, not an instant")
    return date.fromisoformat(text)


@dataclass(frozen=True)
class Observation:
    station: str
    day: str
    element: str
    tenths: int
    measurement: str
    line: int
    source: str = "1"

    @property
    def value(self):
        return Decimal(self.tenths) / 10


def parse_dly(raw, station):
    """Inspect every record. Only QC-clean NWS CF6 facts are admitted."""
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("station source byte ceiling")
    rows = raw.decode("ascii").splitlines()
    if len(rows) > MAX_RECORDS:
        raise ValueError("station record ceiling")
    out, seen = {}, set()
    for number, line in enumerate(rows, 1):
        if len(line) != 269 or line[:11] != station:
            raise ValueError("record size or station identity mismatch")
        year, month = int(line[11:15]), int(line[15:17])
        last = calendar.monthrange(year, month)[1]
        element = line[17:21]
        record_key = (year, month, element)
        if record_key in seen:
            raise ValueError("duplicate station month element")
        seen.add(record_key)
        for day in range(1, 32):
            start = 21 + (day - 1) * 8
            value = int(line[start:start + 5])
            measurement, qc, source = line[start + 5:start + 8]
            if day > last:
                if value != -9999 or (measurement, qc, source) != (" ", " ", " "):
                    raise ValueError("nonmissing nonexistent calendar day")
                continue
            if element not in ELEMENTS or source != "1" or qc != " " or value == -9999:
                continue
            # H and P are not exact observed zero. Unknown measurement flags refuse.
            if measurement not in ((" ", "T") if element == "PRCP" else (" ",)):
                continue
            if measurement == "T" and value != 0:
                raise ValueError("trace must not be assigned a measurable quantity")
            if element == "PRCP" and not 0 <= value <= 20000:
                raise ValueError("precipitation outside live range")
            if element != "PRCP" and not -900 <= value <= 600:
                raise ValueError("temperature outside live range")
            day_text = date(year, month, day).isoformat()
            key = (station, day_text, element)
            out[key] = Observation(station, day_text, element, value, measurement, number)
    for station_id, day_text, element in out:
        if element == "TMAX" and (station_id, day_text, "TMIN") in out:
            if out[(station_id, day_text, "TMIN")].tenths > out[(station_id, day_text, element)].tenths:
                raise ValueError("high lower than low")
    return out


def reject_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate answer field")
        out[key] = value
    return out


def answer_value(answer, expected_location, expected_day):
    if not isinstance(answer, str) or len(answer.encode("utf-8")) > MAX_ANSWER_BYTES:
        raise ValueError("answer byte ceiling or type")
    text = answer.strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text, parse_float=Decimal, parse_int=Decimal,
                             parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
                             object_pairs_hook=reject_duplicates)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("invalid JSON answer") from exc
        if set(obj) - {"location", "date", "temp_max_c", "temp_min_c", "precipitation_mm", "condition"}:
            raise ValueError("unknown answer field")
        if "location" in obj and identity(obj["location"]) != identity(expected_location):
            raise ValueError("answer location differs from request")
        if "date" in obj and obj["date"] != expected_day:
            raise ValueError("answer date differs from request")
        value = obj.get("temp_max_c")
        if not isinstance(value, Decimal):
            raise ValueError("temp_max_c must be a JSON number")
    else:
        # Honest numeric-only contract. Never mine an arbitrary first number.
        match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*(?:°?C|degrees Celsius)?", text)
        if match is None:
            raise ValueError("unsupported prose or units")
        try:
            value = Decimal(match[1])
        except InvalidOperation as exc:
            raise ValueError("invalid decimal") from exc
    if not value.is_finite() or not -90 <= value <= 60:
        raise ValueError("nonfinite or out-of-range Celsius")
    if value.as_tuple().exponent < -6:
        raise ValueError("more precision than live six-decimal contract")
    return value


class ArchiveScorer:
    def __init__(self, observations, station_names):
        self.observations = observations
        self.names = {}
        for station, name in station_names.items():
            for alias in (station, name):
                key = identity(alias)
                if key in self.names and self.names[key] != station:
                    raise ValueError("ambiguous station alias")
                self.names[key] = station

    def score(self, location, day_text, answer):
        iso_day(day_text)
        station = self.names.get(identity(location))
        observed = self.observations.get((station, day_text, "TMAX"))
        if observed is None:
            return {"status": "unsupported", "score": None, "reason": "no trusted station-day TMAX"}
        try:
            value = answer_value(answer, location, day_text)
        except ValueError as exc:
            return {"status": "invalid_answer", "score": None, "reason": str(exc)}
        error = abs(value - observed.value)
        return {"status": "supported", "score": str(-error), "absolute_error_c": str(error),
                "reference_c": str(observed.value), "source": "NOAA/NWS CF6 via NCEI GHCN-Daily",
                "source_flag": observed.source, "source_line": observed.line,
                "scope": "archive TMAX only; not forecast accuracy or arbitrary city weather"}


def inputs_question(location, day_text):
    # Literal expression from pinned responseparser.InputsQuestion. No JSON stand-in.
    return ("WEATHER_FORECAST_VERIFY_GENERAL request. location (City or place name): "
            + location + ". date (Past date to verify): " + day_text + ".")


def render(prompt, answer):
    return "### Task:\n" + prompt + "\n\n### Answer:\n" + answer
