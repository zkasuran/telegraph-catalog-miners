# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""A narrow supplied-evidence comparator, not browsing or universal factual proof."""
import json
import re

MAX_BYTES = 4096


def object_text(text):
    if not isinstance(text, str) or len(text.encode()) > MAX_BYTES:
        raise ValueError("bounded literal JSON string required")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def score(prompt, response):
    task = object_text(prompt)
    body = object_text(response)
    if set(body) != {"answer", "sources", "brief"}:
        raise ValueError("literal WEB_SEARCH body fields required")
    if (not isinstance(body["sources"], list) or len(body["sources"]) > 8 or
            any(not isinstance(url, str) or not url.startswith("https://") for url in body["sources"]) or
            not isinstance(body["answer"], str) or not isinstance(body["brief"], str) or len(body["brief"]) > 80):
        raise ValueError("invalid search body")
    query = task.get("query", "")
    evidence = task.get("evidence", "")
    if not isinstance(query, str) or not isinstance(evidence, str):
        raise ValueError("string query and evidence required")
    q = re.fullmatch(r"Which (Python|Node\.js) version added ([A-Za-z_][\w.]*)\?", query)
    if not q or not evidence:
        return 0.0  # unobserved support, never a fabricated truth label
    records = re.findall(r"([A-Za-z_][\w.]*): added (v?\d+(?:\.\d+){1,2})(?:;|\.)", evidence)
    bindings = {name: version for name, version in records}
    if len(bindings) != len(records) or q[2] not in bindings:
        return 0.0
    expected = f"{q[2]} was added in {q[1]} {bindings[q[2]]}."
    return float(body["answer"] == expected and body["brief"] == expected)
