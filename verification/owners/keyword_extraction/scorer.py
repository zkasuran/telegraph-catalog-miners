# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Transparent lexical keyword diagnostic, not a neural adapter or gold oracle."""
import argparse
import json
import re
from collections import Counter

from source import contains, norm

MAX_TEXT_BYTES = 16384
MAX_ANSWER_BYTES = 4096
MAX_TERMS = 32
WORD = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)


def tokens(value):
    return WORD.findall(norm(value))


def render(prompt, answer):
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def terms(answer):
    if not isinstance(answer, str) or len(answer.encode()) > MAX_ANSWER_BYTES:
        raise ValueError("invalid or oversized answer")
    if answer.lstrip().startswith(("[", "{")):
        value = json.loads(answer)
        if isinstance(value, dict):
            if set(value) != {"keywords"}:
                raise ValueError("unsupported keyword object")
            value = value["keywords"]
        if not isinstance(value, list) or any(not isinstance(t, str) for t in value):
            raise ValueError("keywords must be a string list")
    else:
        # Commas inside PI(4,5)P2 and other bracketed terms are not delimiters.
        stack, begin, value = [], 0, []
        for index, char in enumerate(answer):
            if char in "([{":
                stack.append(char)
            elif char in ")]}":
                if not stack or {")": "(", "]": "[", "}": "{"}[char] != stack.pop():
                    raise ValueError("unbalanced keyword brackets")
            elif char in ",;\n" and not stack:
                if char == "," and index > 0 and index + 1 < len(answer) and answer[index - 1].isdigit() and answer[index + 1].isdigit():
                    raise ValueError("ambiguous numeric comma in keyword; send a JSON string list")
                value.append(answer[begin:index])
                begin = index + 1
        if stack:
            raise ValueError("unbalanced keyword brackets")
        value.append(answer[begin:])
    value = [v.strip() for v in value]
    if not value or len(value) > MAX_TERMS or any(not v or len(v.encode()) > 256 for v in value):
        raise ValueError("empty, oversized or excessive terms")
    return value


def score(text, answer):
    """Fixed integer arithmetic. No training, corpus lookup or external request."""
    if not isinstance(text, str) or not text.strip() or len(text.encode()) > MAX_TEXT_BYTES:
        raise ValueError("empty or oversized passage")
    values = terms(answer)
    title = text.split("\n", 1)[0]
    full, heading = Counter(tokens(text)), set(tokens(title))
    details, unique = [], set()
    for value in values:
        words = tokens(value)
        if not words:
            raise ValueError("term has no lexical content")
        key = norm(value)
        duplicate = key in unique
        unique.add(key)
        coverage = sum(w in full for w in words) * 1000 // len(words)
        title_coverage = sum(w in heading for w in words) * 1000 // len(words)
        repetition = sum(min(full[w], 3) for w in words) * 1000 // (3 * len(words))
        literal = contains(text, value)
        reward = 4 * coverage + title_coverage + repetition + (1000 if literal else 0)
        if duplicate:
            reward = 0
        details.append({"term": value, "literal": literal, "token_coverage_milli": coverage,
                        "title_coverage_milli": title_coverage, "duplicate": duplicate, "reward_milli": reward})
    return {"score_milli": sum(r["reward_milli"] for r in details) // len(details), "terms": details,
            "scope": "Fixed lexical diagnostic only. Absent phrases may still be relevant; literal substrings may not be central. Not factual proof."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--answer", required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.text, args.answer), ensure_ascii=False))


if __name__ == "__main__":
    main()
