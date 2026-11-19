# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Exact bounded premise-agreement comparator, not a universal truth oracle.

This source-only comparator is a meaningful baseline for the prepared subset.
It is not a deployable FP32 judgment package and is not a canonical candidate.
"""
import json
import sys

from source import DISALLOWED, EFFECT, OPPOSITES, make_answer, make_prompt

PREFIX = "Does the claim agree with the excerpt?\nExcerpt: "


def resolve(prompt):
    if not isinstance(prompt, str) or len(prompt.encode()) > 256 or not prompt.startswith(PREFIX):
        raise ValueError("complete literal research question required")
    rest = prompt[len(PREFIX):]
    if rest.count("\nClaim: ") != 1:
        raise ValueError("exact excerpt and claim fields required")
    excerpt, claim = rest.split("\nClaim: ")
    if "\n" in excerpt or "\n" in claim or not excerpt or not claim:
        raise ValueError("empty or multiline premise")
    matches = list(EFFECT.finditer(excerpt))
    if len(matches) != 1 or DISALLOWED.search(excerpt) or not excerpt.endswith("."):
        raise ValueError("unsupported or ambiguous premise")
    if claim == excerpt:
        verdict = "Yes"
    else:
        match = matches[0]
        opposite = excerpt[:match.start()] + OPPOSITES[match.group()] + excerpt[match.end():]
        if claim != opposite:
            raise ValueError("unsupported claim relation, no inferred truth label")
        verdict = "No"
    if make_prompt(excerpt, claim) != prompt:
        raise ValueError("wire changed")
    return claim, verdict


def score(prompt, answer):
    claim, verdict = resolve(prompt)
    if not isinstance(answer, str) or len(answer.encode()) > 1024:
        raise ValueError("bounded literal answer required")
    if answer == make_answer(claim, verdict):
        return 1.0
    if answer == make_answer(claim, "No" if verdict == "Yes" else "Yes"):
        return 0.0
    # Unknown wording is out of scope, not scored as a false factual claim.
    raise ValueError("unrecognized raw answer wording")


def main():
    raw = sys.stdin.buffer.read(8193)
    if len(raw) > 8192:
        raise ValueError("request byte ceiling")
    request = json.loads(raw)
    if not isinstance(request, dict) or set(request) != {"question", "answer"}:
        raise ValueError("question and literal raw answer required")
    result = score(request["question"], request["answer"])
    print(json.dumps({"score": result, "scope": "literal supplied-excerpt agreement only", "neural": False}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError) as exc:
        print("research-comparator refused: " + str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
