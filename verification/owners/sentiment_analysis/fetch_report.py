# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""CI-only retrieval of one real successful audit artifact with an ephemeral token."""
import argparse
import io
import os
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from contracts import (
    FAMILY,
    MAX_BODY,
    REPO,
    bare_hash,
    remote_only,
    sha,
    strict_json,
    validate_ready,
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def read(url, token=None, ceiling=MAX_BODY):
    headers = {"User-Agent": "Sentiment-owner-report/1.0"}
    if token:
        if not url.startswith("https://api.github.com/repos/" + REPO + "/"):
            raise ValueError("ephemeral token may only reach the owned GitHub API")
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
        raw = response.read(ceiling + 1)
        if response.status != 200 or len(raw) > ceiling:
            raise ValueError("audit artifact response bound")
        return raw


def fetch_report(root, run_id, report_sha):
    remote_only()
    bare_hash(report_sha)
    if type(run_id) is not int or run_id < 1:
        raise ValueError("actual audit run id required")
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("CI ephemeral actions:read token missing")
    api = "https://api.github.com/repos/" + REPO + "/actions/"
    metadata = strict_json(read(api + "runs/" + str(run_id), token))
    if metadata.get("conclusion") != "success" or metadata.get("head_sha") != os.environ["GITHUB_SHA"]:
        raise ValueError("successful audit/source commit mismatch")
    listing = strict_json(read(api + "runs/" + str(run_id) + "/artifacts?per_page=100", token))
    matches = [a for a in listing.get("artifacts", []) if a.get("name") == "audit-" + FAMILY + "-" + str(run_id)]
    if len(matches) != 1 or matches[0].get("expired") or matches[0].get("size_in_bytes", 0) > 64 << 20:
        raise ValueError("exact bounded audit artifact missing")
    artifact = matches[0]
    try:
        raw = read(artifact["archive_download_url"], token, 64 << 20)
    except urllib.error.HTTPError as error:
        if error.code != 302:
            raise ValueError("audit artifact download refused") from None
        location = error.headers.get("Location", "")
        parsed = urllib.parse.urlsplit(location)
        if parsed.scheme != "https" or not (parsed.hostname or "").endswith(".blob.core.windows.net"):
            raise ValueError("untrusted audit archive redirect") from None
        # Do not forward the GitHub token or log the signed archive URL.
        raw = read(location, ceiling=64 << 20)
    if artifact.get("digest") != "sha256:" + sha(raw):
        raise ValueError("audit archive digest mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if len(archive.infolist()) > 16 or sum(i.file_size for i in archive.infolist()) > 128 << 20:
            raise ValueError("audit archive inventory ceiling")
        report_files = [i for i in archive.infolist() if i.filename.split("/")[-1] == "registration-ready.json"]
        if len(report_files) != 1 or report_files[0].file_size > MAX_BODY:
            raise ValueError("exact audit report missing")
        report_bytes = archive.read(report_files[0])
    if sha(report_bytes) != report_sha:
        raise ValueError("actual audit report SHA mismatch")
    report = validate_ready(strict_json(report_bytes))
    if report["run_id"] != run_id or report["commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("audit report/run/source identity mismatch")
    root.mkdir(parents=True, exist_ok=True)
    (root / "registration-ready.json").write_bytes(report_bytes)
    return report


def main():
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-run-id", type=int, required=True)
    parser.add_argument("--report-sha256", required=True)
    args = parser.parse_args()
    fetch_report(args.root, args.audit_run_id, args.report_sha256)
    print("BOUND_REAL_AUDIT_REPORT_FETCHED")


if __name__ == "__main__":
    main()
