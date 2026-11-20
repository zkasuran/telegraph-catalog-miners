# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Bounded Brreg observations. Source facts retain NLOD-2.0, never SAND."""

import argparse
import hashlib
import json
import re
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = "https://data.brreg.no/enhetsregisteret/api/enheter"
DOCS = "https://data.brreg.no/enhetsregisteret/api/dokumentasjon/no/index.html"
GRANT = "https://data.norge.no/nlod/no/2.0"
GRANT_SHA = "c2c60474f811395722cedfe1580f5a5d0c4d2ccee1db4b2e2293490db74d02b5"
PAGES = (7, 43, 91, 137)
SIZE = 64
MAX_BODY = 2 << 20
MAX_SECONDS = 600
MAX_RENDER_BYTES = 1200
KEYS = (
    "organisasjonsnummer",
    "navn",
    "registrertIMvaregisteret",
    "registrertIForetaksregisteret",
    "erIKonsern",
    "historiskeNavn",
    "slettedato",
    "respons_klasse",
)
ATTRIBUTION = "Inneholder data under Norsk lisens for offentlige data (NLOD) tilgjengeliggjort av Brønnøysundregistrene."


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def packed(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def normalized(text):
    # Preserve punctuation. Case and compatibility/whitespace variants stay together.
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def org_number(text):
    if not isinstance(text, str):
        raise ValueError("organisation number must be a string")
    match = re.fullmatch(
        r"(?:NO\s*)?([0-9]{3})[ .]?([0-9]{3})[ .]?([0-9]{3})(?:\s*MVA)?",
        text.strip().upper(),
    )
    if not match:
        raise ValueError("unsupported Norwegian organisation number presentation")
    number = "".join(match.groups())
    check = (
        11 - sum(int(n) * w for n, w in zip(number[:8], (3, 2, 7, 6, 5, 4, 3, 2))) % 11
    ) % 11
    if check > 9 or check != int(number[-1]):
        raise ValueError("organisation number checksum mismatch")
    return number


class LegalText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag in ("p", "div", "li", "h1", "h2", "h3", "br"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        if tag in ("p", "div", "li", "h1", "h2", "h3"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("source redirect refused")


class Collector:
    def __init__(self, raw_root):
        self.root = raw_root
        self.started = time.monotonic()
        self.probes = []

    def fetch(self, url, key, accept):
        request_started = time.monotonic()
        if time.monotonic() - self.started > MAX_SECONDS:
            raise ValueError("collection elapsed ceiling reached")
        if not (
            url in (GRANT, DOCS)
            or url.startswith(BASE + "?")
            or re.fullmatch(re.escape(BASE) + r"/[0-9]{9}", url)
        ):
            raise ValueError("non-allowlisted source URL")
        request = urllib.request.Request(
            url,
            headers={
                "Accept": accept,
                "User-Agent": "Telegraph-VAT-observation-research/1.0",
                "Accept-Encoding": "identity",
            },
        )
        with urllib.request.build_opener(NoRedirect()).open(
            request, timeout=20
        ) as response:
            chunks, total = [], 0
            while chunk := response.read1(min(65536, MAX_BODY + 1 - total)):
                total += len(chunk)
                chunks.append(chunk)
                if total > MAX_BODY:
                    raise ValueError("source body ceiling reached")
                if (
                    time.monotonic() - request_started > 20
                    or time.monotonic() - self.started > MAX_SECONDS
                ):
                    raise ValueError(
                        "source request or collection elapsed ceiling reached"
                    )
            raw = b"".join(chunks)
            if len(raw) > MAX_BODY or response.status != 200:
                raise ValueError("source body or status refused")
            media = response.headers.get("Content-Type", "")
            if (accept == "application/json" and "json" not in media) or (
                accept == "text/html" and "html" not in media
            ):
                raise ValueError("unexpected source content type")
            probe = {
                "key": key,
                "url": url,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "status": response.status,
                "bytes": len(raw),
                "sha256": sha(raw),
                "headers": {
                    k: response.headers.get(k)
                    for k in ("Content-Type", "Date", "ETag", "Last-Modified")
                },
            }
        # Exact transport bytes are private evidence, never model input or patch content.
        (self.root / (key + ".body")).write_bytes(raw)
        self.probes.append(probe)
        write(self.root / "PROBES.json", self.probes)
        return raw, probe


def projection(entity, probe, index):
    if not isinstance(entity, dict):
        raise ValueError("source entity must be an object")
    if not isinstance(entity.get("organisasjonsnummer"), str):
        raise ValueError("source organisation number must be a string")
    number = org_number(entity["organisasjonsnummer"])
    if not isinstance(entity.get("navn"), str) or not entity["navn"].strip():
        raise ValueError("missing legal identity")
    if type(entity.get("registrertIMvaregisteret")) is not bool:
        raise ValueError("missing declared VAT boolean")
    if entity.get("organisasjonsform", {}).get("kode") != "AS":
        raise ValueError("source ignored incorporated-company filter")
    aliases = entity.get("historiskeNavn", [])
    if not isinstance(aliases, list) or any(
        not isinstance(a, dict) or not isinstance(a.get("navn"), str) for a in aliases
    ):
        raise ValueError("invalid historical identity schema")
    result = {k: entity[k] for k in KEYS if k in entity}
    result["organisasjonsnummer"] = number
    result["source_binding"] = {
        "url": probe["url"],
        "raw_sha256": probe["sha256"],
        "pointer": f"/_embedded/enheter/{index}" if index is not None else "",
        "retrieved_at": probe["retrieved_at"],
        "source_licence": "NLOD-2.0",
        "grant_sha256": GRANT_SHA,
    }
    return result


def identities(row):
    return {normalized(row["navn"])} | {
        normalized(a["navn"]) for a in row.get("historiskeNavn", [])
    }


def candidates(rows, excluded):
    buckets = defaultdict(list)
    names = defaultdict(set)
    counts = Counter()
    for row in rows:
        for name in identities(row):
            names[name].add(row["organisasjonsnummer"])
    for row in rows:
        if row["organisasjonsnummer"] in excluded:
            counts["development_excluded"] += 1
        elif row.get("erIKonsern") is not False:
            counts["declared_group_or_unknown_excluded"] += 1
        elif row.get("slettedato") or row.get("respons_klasse") != "Enhet":
            counts["deleted_or_unknown_excluded"] += 1
        elif any(len(names[name]) > 1 for name in identities(row)):
            counts["shared_normalized_identity_excluded"] += 1
        else:
            buckets[row["registrertIMvaregisteret"]].append(row)
    return buckets, dict(counts)


def preference(row, style="json"):
    number, name = row["organisasjonsnummer"], row["navn"]
    observed = row["source_binding"]["retrieved_at"][:10]
    source = {
        "org": number,
        "name": name,
        "date": observed,
        "registrertIMvaregisteret": row["registrertIMvaregisteret"],
    }
    prompt = (
        "VENDOR_VERIFY. Interpret only this trusted dated Brreg VAT observation, not legitimacy. "
        + packed(source)
    )

    def answer(value):
        if style == "prose":
            return f"For NO {number} MVA ({name}) on {observed}, observed VAT membership is {str(value).lower()}."
        if style == "bare":
            return str(value).lower()
        return packed({"org": number, "name": name, "date": observed, "valid": value})

    pair = {
        "prompt": prompt,
        "chosen": answer(row["registrertIMvaregisteret"]),
        "rejected": answer(not row["registrertIMvaregisteret"]),
        "margin": 0.0,
        "label_kind": "preference_pair",
        "label": str(row["registrertIMvaregisteret"]).lower(),
        "source_id": number,
        "group_id": sha(normalized(name).encode()),
        "source_binding": row["source_binding"],
        "construction": "assessment_boolean_only",
    }
    if (
        max(
            len(("### Task:\n" + prompt + "\n\n### Answer:\n" + pair[k]).encode())
            for k in ("chosen", "rejected")
        )
        > MAX_RENDER_BYTES
    ):
        raise ValueError(
            "render exceeds conservative byte ceiling; token gate still required"
        )
    return pair


def freeze(rows, excluded, out):
    buckets, removed = candidates(rows, excluded)
    # A fixed snapshot diagnostic, never a prospective temporal split.
    sizes = {"train": 128, "validation": 32, "audit": 64}
    required = sum(sizes.values()) // 2
    if any(len(buckets[v]) < required for v in (False, True)):
        raise ValueError(
            f"insufficient distinct source entities: need {required} per VAT state, got "
            + str({str(v): len(buckets[v]) for v in (False, True)})
        )
    splits = {s: [] for s in sizes}
    for value in (False, True):
        pool = sorted(
            buckets[value],
            key=lambda r: sha(("vendor-split-v1:" + r["organisasjonsnummer"]).encode()),
        )
        offset = 0
        for split, count in sizes.items():
            splits[split].extend(pool[offset : offset + count // 2])
            offset += count // 2
    manifest = {
        "schema_version": 1,
        "source_licence": "NLOD-2.0",
        "attribution": ATTRIBUTION,
        "split_policy": "entity and all normalized current/historical names disjoint; one-day diagnostic",
        "removed": removed,
        "files": {},
        "audit_opened": False,
        "training_authorized": False,
        "model_selected": False,
    }
    for split, selected in splits.items():
        pairs = [
            preference(r)
            for r in sorted(selected, key=lambda r: r["organisasjonsnummer"])
        ]
        raw = ("\n".join(packed(p) for p in pairs) + "\n").encode()
        (out / (split + ".jsonl")).write_bytes(raw)
        manifest["files"][split] = {
            "sha256": sha(raw),
            "originals": len(pairs),
            "vat_true": len(pairs) // 2,
            "vat_false": len(pairs) // 2,
        }
    write(out / "FREEZE.json", manifest)
    return manifest


def verify_bindings(rows, raw_root, probes):
    by_url = {p["url"]: p for p in probes}
    cache = {}
    for row in rows:
        binding = row["source_binding"]
        pointer = binding.get("pointer")
        if not isinstance(pointer, str) or (
            pointer
            and re.fullmatch(r"/_embedded/enheter/(?:0|[1-9][0-9]*)", pointer) is None
        ):
            raise ValueError("noncanonical source pointer")
        probe = by_url[binding["url"]]
        if probe["key"] not in cache:
            if (
                not isinstance(probe["key"], str)
                or re.fullmatch(r"[A-Za-z0-9_-]{1,80}", probe["key"]) is None
            ):
                raise ValueError("unsafe raw source key")
            path = raw_root / (probe["key"] + ".body")
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size > MAX_BODY
            ):
                raise ValueError("unsafe or oversized raw source file")
            raw = path.read_bytes()
            if sha(raw) != probe["sha256"]:
                raise ValueError("raw source hash mismatch")
            cache[probe["key"]] = json.loads(raw)
        source = cache[probe["key"]]
        pointer = binding["pointer"]
        if pointer:
            embedded = source.get("_embedded") if isinstance(source, dict) else None
            entities = embedded.get("enheter") if isinstance(embedded, dict) else None
            index = int(pointer.rsplit("/", 1)[1])
            if not isinstance(entities, list) or index >= len(entities):
                raise ValueError("source array pointer does not resolve")
            entity = entities[index]
        else:
            entity = source
        if row != projection(
            entity, probe, int(pointer.rsplit("/", 1)[1]) if pointer else None
        ):
            raise ValueError("projection does not match exact raw source")
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--exclude", required=True, type=Path)
    args = parser.parse_args()
    raw_root, out = args.raw_root.resolve(), args.out.resolve()
    if raw_root == out or raw_root.is_relative_to(out) or out.is_relative_to(raw_root):
        raise ValueError("raw transport and projected output roots must be separate")
    if raw_root.is_symlink() or out.is_symlink():
        raise ValueError("symlink output roots refused")
    if raw_root.exists() or out.exists():
        raise ValueError("refusing to overwrite an acquisition or frozen audit")
    raw_root.mkdir(parents=True, mode=0o700)
    out.mkdir(parents=True)
    excluded = set(json.loads(args.exclude.read_text())["organisation_numbers"])
    collector = Collector(raw_root)
    plan = {
        "pages": PAGES,
        "size": SIZE,
        "listing_GET_cap": 8,
        "exact_GET_cap": 15,
        "byte_cap_per_response": MAX_BODY,
        "timeout_seconds": 20,
        "source_licence": "NLOD-2.0",
        "sort": "organisasjonsnummer,ASC",
    }
    write(out / "PLAN.json", plan)
    try:
        legal, grant_probe = collector.fetch(GRANT, "nlod", "text/html")
        if sha(legal) != GRANT_SHA:
            raise ValueError("NLOD hash drift, grant review required")
        text = LegalText()
        text.feed(legal.decode())
        (out / "NLOD-2.0.txt").write_text(
            "\n".join(
                line.strip() for line in "".join(text.parts).split("\n") if line.strip()
            )
            + "\n"
        )
        docs, docs_probe = collector.fetch(DOCS, "docs", "text/html")
        if GRANT.encode() not in docs or b"registrertIMvaregisteret" not in docs:
            raise ValueError("official API grant or VAT field binding missing")
        rows, seen = [], set()
        for state in (False, True):
            for page in PAGES:
                if (page + 1) * SIZE > 10000:
                    raise ValueError("documented pagination ceiling")
                url = (
                    BASE
                    + f"?organisasjonsform=AS&registrertIMvaregisteret={str(state).lower()}&sort=organisasjonsnummer,ASC&page={page}&size={SIZE}"
                )
                raw, probe = collector.fetch(
                    url, f"vat-{state}-{page}", "application/json"
                )
                body = json.loads(raw)
                entities = body.get("_embedded", {}).get("enheter", [])
                if len(entities) != SIZE or body.get("page", {}).get("number") != page:
                    raise ValueError("listing page incomplete or query ignored")
                for index, entity in enumerate(entities):
                    row = projection(entity, probe, index)
                    if (
                        row["registrertIMvaregisteret"] is not state
                        or row["organisasjonsnummer"] in seen
                    ):
                        raise ValueError(
                            "source state filter ignored or duplicate identity"
                        )
                    rows.append(row)
                    seen.add(row["organisasjonsnummer"])
                write(
                    out / "SOURCE-PROGRESS.json",
                    {"entities": len(rows), "listing_GETs": len(collector.probes) - 2},
                )
        consistent = 0
        for row in sorted(
            rows, key=lambda r: sha(("readback:" + r["organisasjonsnummer"]).encode())
        )[:15]:
            raw, probe = collector.fetch(
                BASE + "/" + row["organisasjonsnummer"],
                "exact-" + row["organisasjonsnummer"],
                "application/json",
            )
            exact = projection(json.loads(raw), probe, None)
            if any(exact.get(k) != row.get(k) for k in KEYS):
                raise ValueError("listing/direct-read observation disagreement")
            consistent += 1
        raw = ("\n".join(packed(r) for r in rows) + "\n").encode()
        (out / "observations.jsonl").write_bytes(raw)
        bindings = verify_bindings(rows, raw_root, collector.probes)
        frozen = freeze(rows, excluded, out)
        write(
            out / "SOURCE-REPORT.json",
            {
                "status": "snapshot_diagnostic_frozen",
                "source_licence": "NLOD-2.0",
                "attribution": ATTRIBUTION,
                "entities": len(rows),
                "vat_counts": dict(
                    Counter(str(r["registrertIMvaregisteret"]) for r in rows)
                ),
                "projection_sha256": sha(raw),
                "raw_bindings_checked": bindings,
                "same_registry_readbacks": consistent,
                "grant": grant_probe,
                "docs": docs_probe,
                "probes": collector.probes,
                "freeze_sha256": sha((out / "FREEZE.json").read_bytes()),
                "split_counts": frozen["files"],
                "independent_tax_corroboration": False,
                "acquisition_days": 1,
            },
        )
        print(
            packed(
                {
                    "entities": len(rows),
                    "bindings": bindings,
                    "readbacks": consistent,
                    "splits": frozen["files"],
                }
            )
        )
    except (ValueError, KeyError, OSError, TypeError) as error:
        write(
            out / "BLOCKER.json",
            {
                "status": "blocked",
                "reason": str(error),
                "probes_completed": len(collector.probes),
                "source_licence": "NLOD-2.0",
                "own_additions_licence": "LicenseRef-zkasuran-SAND-1.0",
            },
        )
        raise


if __name__ == "__main__":
    main()
