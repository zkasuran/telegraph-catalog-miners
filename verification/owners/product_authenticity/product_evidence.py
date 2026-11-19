# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""A source-bound archived-report diagnostic using the live response fields."""
import json
import re
import sys

import scorer
from source_inventory import all_records, SANADA_DOI


def product_for(record_id):
    if record_id.startswith("newton2008-"):
        return scorer.product_for(record_id)
    return scorer.PREFIX + record_id + "; doi:" + SANADA_DOI


def expectation(product, rows):
    # Use the old primary path unchanged for Newton records and unknown products.
    default = scorer.expectation(product, rows)
    match = re.fullmatch(re.escape(scorer.PREFIX) + r"(sanada2020-(?:GC|FC|GL|FL|GV|FV)[1-9][0-9]*); doi:" + re.escape(SANADA_DOI), product)
    if not match or match.group(1) not in rows:
        return default
    record = rows[match.group(1)]
    label = record["source_label"]
    return {"product": product, "verdict": label, "authentic": label == "Authentic", "evidence": [
        "Source: doi:" + SANADA_DOI,
        "Published manufacturer-assessed label for " + record["sample"] + ": " + label,
        "Raw spectra and batch identity are not publicly released.", scorer.SCOPE]}


def score(product, answer, rows):
    expected = expectation(product, rows)
    try:
        actual = scorer.decode_answer(answer)
    except (ValueError, UnicodeError, TypeError):
        return {"score": 0, "reason": "invalid_response", "mode": "diagnostic_only"}
    if actual != expected:
        return {"score": 0, "reason": "unsupported_or_unbound_report", "mode": "diagnostic_only"}
    return {"score": 10000, "reason": "archived_source_report", "mode": "diagnostic_only"}


def main():
    limit = scorer.MAX_PRODUCT_BYTES + scorer.MAX_ANSWER_BYTES + 2048
    raw = sys.stdin.buffer.read(limit)
    if len(raw) >= limit:
        raise SystemExit("input envelope exceeds byte limit")
    try:
        obj = json.loads(raw, object_pairs_hook=scorer.strict_object, parse_constant=scorer.reject_constant)
        if not isinstance(obj, dict) or set(obj) != {"product", "answer"}:
            raise ValueError("expected product and answer strings")
        print(json.dumps(score(obj["product"], obj["answer"], all_records()), sort_keys=True))
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
