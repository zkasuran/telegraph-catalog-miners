# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Offline DNS answer relevance and internal consistency, never live DNS truth."""
import argparse
import json
import math
import re
import sys

MAX_TEXT_BYTES = 16384
MAX_DEPTH = 12
MAX_NODES = 2048
TYPES = {"A": 1, "NS": 2, "CNAME": 5, "SOA": 6, "PTR": 12, "MX": 15,
         "TXT": 16, "AAAA": 28, "SRV": 33, "DS": 43, "RRSIG": 46,
         "DNSKEY": 48, "CAA": 257}
RCODES = {"NOERROR": 0, "FORMERR": 1, "SERVFAIL": 2, "NXDOMAIN": 3,
          "NOTIMP": 4, "REFUSED": 5}
DOMAIN_DESC = "Email domain"
TYPE_DESC = "DNS record type to look up, e.g. MX or TXT"


def limited_text(text):
    if not isinstance(text, str):
        raise ValueError("text type or byte ceiling")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError("invalid text encoding") from error
    if size > MAX_TEXT_BYTES:
        raise ValueError("text type or byte ceiling")
    return text


def domain(value):
    limited_text(value)
    # No URL stripping, whitespace stripping or underscore removal. Those can
    # turn an unrelated query into an apparent identity match.
    name = value[:-1] if value.endswith(".") else value
    if not name or len(name.encode("ascii", errors="replace")) > 253:
        raise ValueError("DNS name length")
    labels = name.split(".")
    if any(not 1 <= len(label) <= 63 or not re.fullmatch(r"[A-Za-z0-9_-]+", label)
           for label in labels):
        raise ValueError("only literal ASCII DNS labels supported")
    return name.lower()


def record_type(value):
    if type(value) is int and 1 <= value <= 65535:
        return value
    if not isinstance(value, str):
        raise ValueError("record type")
    if value.upper() in TYPES:
        return TYPES[value.upper()]
    if value.isascii() and value.isdigit() and 1 <= int(value) <= 65535:
        return int(value)
    raise ValueError("unsupported record type")


def request_prompt(name, kind):
    # Literal InputsQuestion expression, no normalized substitute. Wire test
    # invokes the original Go implementation on these exact descriptions.
    domain(name)
    record_type(kind)
    return (f"EMAIL_SECURITY_GENERAL request. domain ({DOMAIN_DESC}): {name}. "
            f"record_type ({TYPE_DESC}): {kind}.")


def parse_prompt(prompt):
    limited_text(prompt)
    start = f"EMAIL_SECURITY_GENERAL request. domain ({DOMAIN_DESC}): "
    middle = f". record_type ({TYPE_DESC}): "
    if not prompt.startswith(start) or not prompt.endswith("."):
        raise ValueError("unsupported request rendering")
    values = prompt[len(start):-1].split(middle)
    if len(values) != 2:
        raise ValueError("ambiguous request")
    return domain(values[0]), record_type(values[1])


def render(prompt, answer):
    return "### Task:\n" + limited_text(prompt) + "\n\n### Answer:\n" + limited_text(answer)


def pairs_object(items):
    out = {}
    for key, value in items:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def parse_json(text):
    limited_text(text)
    try:
        value = json.loads(text, object_pairs_hook=pairs_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid JSON") from error
    stack, nodes = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > MAX_DEPTH or nodes > MAX_NODES:
            raise ValueError("JSON structure ceiling")
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON")
    if not isinstance(value, dict):
        raise ValueError("answer must be a JSON object")
    return value


def status(value):
    if type(value) is int and 0 <= value <= 4095:
        return value
    if isinstance(value, str) and value.upper() in RCODES:
        return RCODES[value.upper()]
    raise ValueError("missing or invalid observed DNS Status")


def inspect(prompt, answer):
    wanted_name, wanted_type = parse_prompt(prompt)
    body = parse_json(answer)
    if "Status" not in body:
        return {"score": None, "reason": "no_dns_status", "scope": "internal_consistency_only"}
    code = status(body["Status"])
    questions = body.get("Question")
    if not isinstance(questions, list) or len(questions) != 1 or not isinstance(questions[0], dict):
        return {"score": None, "reason": "no_unique_question", "scope": "internal_consistency_only"}
    query = questions[0]
    matches = domain(query.get("name")) == wanted_name and record_type(query.get("type")) == wanted_type
    if not matches:
        return {"score": -1, "reason": "wrong_question", "scope": "internal_consistency_only"}
    ok = code == 0
    if "lookup_ok" in body and type(body["lookup_ok"]) is not bool:
        raise ValueError("lookup_ok must be boolean")
    consistent = "lookup_ok" not in body or body["lookup_ok"] == ok
    return {"score": 1 if consistent else -1, "reason": "bound_status" if consistent else "contradiction",
            "lookup_ok": ok, "scope": "internal_consistency_only", "externally_authenticated": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--record-type", required=True)
    args = parser.parse_args()
    try:
        raw = sys.stdin.buffer.read(MAX_TEXT_BYTES + 1)
        if len(raw) > MAX_TEXT_BYTES:
            raise ValueError("answer byte ceiling")
        result = inspect(request_prompt(args.domain, args.record_type), raw.decode("utf-8"))
        print(json.dumps(result, allow_nan=False))
        return 0 if result["score"] == 1 else 1
    except (ValueError, UnicodeError) as error:
        print(json.dumps({"score": None, "error": str(error)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
