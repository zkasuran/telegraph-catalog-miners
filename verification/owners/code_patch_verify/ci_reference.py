# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""CI observation consistency, not semantic patch repair or a network oracle."""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

MAX_TEXT = 20000
MAX_JOBS = 1000
MAX_JSON_DEPTH = 64
CONCLUSIONS = frozenset(("success", "failure", "cancelled", "skipped", "neutral", "timed_out"))
PENDING = frozenset(("queued", "in_progress", "waiting", "requested", "pending"))
SHA = re.compile(r"[0-9a-fA-F]{40}\Z")
NAME = r"[A-Za-z0-9_.-]+"


class EvidenceError(ValueError):
    pass


def strict_json(text):
    if not isinstance(text, str) or len(text.encode("utf-8", errors="surrogatepass")) > MAX_TEXT:
        raise EvidenceError("JSON byte ceiling")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise EvidenceError("duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise EvidenceError("nonfinite JSON")

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
        stack = [(value, 0)]
        while stack:
            item, depth = stack.pop()
            if depth > MAX_JSON_DEPTH:
                raise EvidenceError("JSON depth ceiling")
            if isinstance(item, dict):
                stack.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                stack.extend((child, depth + 1) for child in item)
        return value
    except (ValueError, RecursionError) as exc:
        raise EvidenceError("invalid JSON: " + str(exc)) from exc


def run_identity(url):
    if not isinstance(url, str) or len(url) > MAX_TEXT or any(c.isspace() for c in url):
        raise EvidenceError("invalid workflow URL")
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise EvidenceError("invalid workflow URL") from exc
    if parts.scheme != "https" or parts.query or parts.fragment:
        raise EvidenceError("exact HTTPS workflow URL required")
    # netloc, not hostname, denies credentials, explicit ports and spoofed hosts.
    if parts.netloc == "github.com":
        pattern = rf"/({NAME})/({NAME})/actions/runs/([1-9][0-9]*)(?:/attempts/([1-9][0-9]*))?"
    elif parts.netloc == "api.github.com":
        pattern = rf"/repos/({NAME})/({NAME})/actions/runs/([1-9][0-9]*)(?:/attempts/([1-9][0-9]*))?"
    else:
        raise EvidenceError("unsupported workflow host")
    match = re.fullmatch(pattern, parts.path)
    if not match or any(x in (".", "..") for x in match.groups()[:2]):
        raise EvidenceError("unsupported workflow path")
    owner, repo, run, attempt = match.groups()
    if len(run) > 19 or (attempt and len(attempt) > 6):
        raise EvidenceError("workflow integer ceiling")
    return (owner.lower() + "/" + repo.lower(), int(run), int(attempt) if attempt else None)


def positive_integer(value):
    return type(value) is int and 0 < value < 2**63


def expected(request, evidence):
    """Only caller-authenticated evidence is authoritative. Never use answer evidence."""
    if not isinstance(request, dict) or set(request) - {"patch", "commit"} or "patch" not in request:
        raise EvidenceError("request requires patch with optional pinned commit")
    repository, run_id, requested_attempt = run_identity(request["patch"])
    if not isinstance(evidence, dict) or evidence.get("transport_status") != 200:
        raise EvidenceError("source unavailable")
    run = evidence.get("run")
    if not isinstance(run, dict) or not positive_integer(run.get("id")) or run["id"] != run_id:
        raise EvidenceError("wrong run identity")
    attempt = run.get("run_attempt")
    if not positive_integer(attempt) or (requested_attempt is not None and attempt != requested_attempt):
        raise EvidenceError("wrong attempt identity")
    repo = run.get("repository", {}).get("full_name") if isinstance(run.get("repository"), dict) else None
    if not isinstance(repo, str) or repo.lower() != repository:
        raise EvidenceError("wrong repository identity")
    if run_identity(run.get("html_url")) != (repository, run_id, None):
        raise EvidenceError("wrong canonical pipeline identity")
    commit = run.get("head_sha")
    if not isinstance(commit, str) or not SHA.fullmatch(commit):
        raise EvidenceError("invalid source commit")
    if "commit" in request and (not isinstance(request["commit"], str)
                               or not SHA.fullmatch(request["commit"])
                               or request["commit"].lower() != commit.lower()):
        raise EvidenceError("wrong commit binding")
    observed = evidence.get("observed_at")
    try:
        when = datetime.fromisoformat(observed.replace("Z", "+00:00"))
    except (AttributeError, ValueError, TypeError) as exc:
        raise EvidenceError("missing observation time") from exc
    if when.tzinfo is None or when.utcoffset() != timezone.utc.utcoffset(when):
        raise EvidenceError("observation must be UTC")
    status, conclusion = run.get("status"), run.get("conclusion")
    if status == "completed" and isinstance(conclusion, str) and conclusion in CONCLUSIONS:
        pass
    elif isinstance(status, str) and status in PENDING and conclusion is None:
        conclusion = "pending"
    else:
        raise EvidenceError("unsupported or contradictory run state")
    if conclusion not in ("success", "failure"):
        raise EvidenceError("no independently verified live passed mapping for " + conclusion)
    return {"passed": conclusion == "success", "conclusion": conclusion,
            "commit": commit, "pipeline": run["html_url"]}


def score(request, answer, evidence):
    """Three separate results: unavailable source, wrong candidate or consistent claim."""
    try:
        truth = expected(request, evidence)
    except EvidenceError as exc:
        return {"available": False, "score": None, "reason": str(exc)}
    try:
        candidate = strict_json(answer)
        if not isinstance(candidate, dict) or type(candidate.get("passed")) is not bool:
            raise EvidenceError("passed must be a native boolean")
        if not isinstance(candidate.get("conclusion"), str) or candidate["conclusion"] not in CONCLUSIONS | {"pending"}:
            raise EvidenceError("unsupported conclusion")
        mismatches = [key for key, value in truth.items() if candidate.get(key) != value]
        return {"available": True, "score": int(not mismatches), "mismatches": mismatches,
                "scored_field": "passed", "scope": "CI evidence consistency only"}
    except EvidenceError as exc:
        return {"available": True, "score": 0, "reason": str(exc)}


def render(prompt, answer):
    # Literal published train_adapter.render, without whitespace normalization.
    return "### Task:\n" + prompt + "\n\n### Answer:\n" + answer


def collect_jobs(fetch_page):
    """Consume the whole attempt-pinned inventory. Never call page one a population."""
    jobs, ids, total = [], set(), None
    for page in range(1, MAX_JOBS // 100 + 1):
        data = fetch_page(page)
        count, batch = data.get("total_count"), data.get("jobs")
        if type(count) is not int or not 0 <= count <= MAX_JOBS or not isinstance(batch, list) or len(batch) > 100:
            raise EvidenceError("invalid job inventory")
        if total is None:
            total = count
        if count != total:
            raise EvidenceError("job inventory changed while paging")
        for job in batch:
            if not isinstance(job, dict) or not positive_integer(job.get("id")) or job["id"] in ids:
                raise EvidenceError("duplicate or invalid job identity")
            ids.add(job["id"])
            jobs.append(job)
        if len(jobs) == total:
            return jobs
        if not batch or len(jobs) > total:
            raise EvidenceError("incomplete job inventory")
    raise EvidenceError("job inventory page ceiling")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    if args.input.stat().st_size > MAX_TEXT:
        raise EvidenceError("input byte ceiling")
    data = strict_json(args.input.read_text())
    result = score(data["request"], data["answer"], data["evidence"])
    print(json.dumps(result, sort_keys=True))
    return 0 if result["score"] == 1 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EvidenceError, KeyError, OSError) as error:
        print(json.dumps({"error": str(error)}))
        sys.exit(2)
