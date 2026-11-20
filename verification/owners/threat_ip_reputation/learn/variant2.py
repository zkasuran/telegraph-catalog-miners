# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Compact elapsed-time development formulation. No new source/audit rows."""
import datetime as dt
import json
from decimal import Decimal
from pathlib import Path

import contracts

HERE = Path(__file__).resolve().parent


def instant(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def formulate(row):
    original = json.loads(row["prompt"])
    observation = original["observation"]
    seen = instant(observation["last_seen"])
    updated = instant(original["source_updated"])
    expiry = seen + dt.timedelta(days=original["window_days"])
    jitter = int(contracts.sha(observation["ip"].encode())[:8], 16) % 45
    times = (("publisher_observed_in_window", updated),
             ("publisher_observed_in_window", expiry),
             ("historical_publisher_observation", expiry + dt.timedelta(days=1 + jitter)),
             ("historical_publisher_observation", expiry + dt.timedelta(days=50 + jitter)))
    for index, (label, at) in enumerate(times):
        if updated == expiry and index in (1, 3):
            continue  # Do not salt duplicate temporal contexts to satisfy a data gate.
        elapsed = int((at - seen).total_seconds())
        expected = "publisher_observed_in_window" if elapsed <= original["window_days"] * 86400 else "historical_publisher_observation"
        if label != expected or at < updated:
            raise ValueError("derived source-time contrast is not supported")
        age = str((Decimal(elapsed) / Decimal(86400)).quantize(Decimal("0.00000001")))
        prompt = ("Assess source-specific IP reputation for " + observation["ip"] + ". "
                  "The jacobrakai-honeypot source reports " + observation["tier"] + " activity. "
                  "Last observation: " + observation["last_seen"] + ". Assessment time: " + at.strftime("%Y-%m-%dT%H:%M:%SZ") + ". "
                  "Elapsed observation age: " + age + " days. Source activity window: " + str(original["window_days"]) + " days. "
                  "publisher_observed_in_window means no older than the window; historical_publisher_observation means older. "
                  "Shared addresses, reassignment and universal present reputation remain unknown.")
        other = "historical_publisher_observation" if label == "publisher_observed_in_window" else "publisher_observed_in_window"
        def answer(state, fmt):
            if fmt == "json":
                return json.dumps({"ip": observation["ip"], "source": "jacobrakai-honeypot", "source_state": state,
                                   "tier": observation["tier"], "as_of": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                   "current_universal_reputation": "unknown"}, separators=(",", ":"))
            return "IP " + observation["ip"] + " has source state " + state + " in jacobrakai-honeypot. Its reported tier is " + observation["tier"] + ". Assessment time is " + at.strftime("%Y-%m-%dT%H:%M:%SZ") + ". Current universal reputation is unknown."
        for fmt in ("json", "prose"):
            yield {**row, "prompt": prompt, "chosen": answer(label, fmt), "rejected": answer(other, fmt),
                   "format": fmt, "label": label, "variant": "compact-elapsed-v2", "contrast_index": index,
                   "source_observation": observation, "observation_modified": False, "elapsed_seconds": elapsed,
                   "elapsed_days_derived": age, "as_of_is_constructed": index != 0,
                   "preference_provenance": "AI-assisted elapsed-time source contrast; source values unchanged; not human reviewed"}
