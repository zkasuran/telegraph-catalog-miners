# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Index primary CC BY originals and the full cached link graph. No models."""
import argparse
import hashlib
import json
import sqlite3
import tarfile
from array import array
from collections import Counter
from pathlib import Path

from source_probe import EXPORT, LINKS_SHA, MAX_ID, fetch, norm, records, sha, write

HERE = Path(__file__).resolve().parent
SPEC = HERE / "general-sources.json"


def capture(out):
    out.mkdir(parents=True, exist_ok=True)
    previous = json.loads(SPEC.read_text()) if SPEC.exists() else {}
    files = {}
    publisher_path = out / "downloads.html"
    if not publisher_path.is_file():
        fetch("https://tatoeba.org/en/downloads", publisher_path, 2 << 20)
    publisher = publisher_path.read_text()
    for lang in ("eng", "fra"):
        for kind in ("sentences", "sentences_detailed"):
            name = f"{lang}_{kind}.tsv.bz2"
            url = EXPORT + f"per_language/{lang}/{name}"
            if url not in publisher:
                raise ValueError("export not listed by the captured primary publisher")
            path = out / name
            old = previous.get("files", {}).get(name)
            if old and path.exists() and sha(path) == old["sha256"]:
                files[name] = old
            else:
                info = fetch(url, path)
                if old and info["sha256"] != old["sha256"]:
                    raise ValueError("general export snapshot changed: " + name)
                if "14 Nov 2026" not in (info["last_modified"] or ""):
                    raise ValueError("general export is not the cached links' weekly snapshot")
                files[name] = info
            print(name, files[name]["sha256"], files[name]["bytes"], flush=True)
    spec = {"schema_version": 1, "license": "CC-BY-2.0-FR", "files": files,
            "links_sha256": LINKS_SHA, "snapshot_week": "2026-11-14",
            "grant_url": "https://creativecommons.org/licenses/by/2.0/fr/legalcode",
            "publisher_url": "https://tatoeba.org/en/downloads",
            "publisher_clause": "These files are released under CC BY 2.0 FR.",
            "grant_3_2": "créer et reproduire des Oeuvres dites Dérivées",
            "grant_3_5": "lorsque l'Oeuvre est une base de données, extraire et réutiliser des parties substantielles de l'Oeuvre.",
            "attribution_4_b": "Il doit communiquer le nom de l'Auteur Original ou son éventuel pseudonyme s'il est indiqué",
            "conditions": "Retain author or credited contributor, sentence/source URI, applicable licence, original notices and construction notice. No restriction on original source rights. No CC0 relabelling."}
    if not previous:
        write(SPEC, spec)
    links = out / "links.tar.bz2"
    if not links.is_file() or sha(links) != LINKS_SHA:
        info = fetch(EXPORT + "links.tar.bz2", links)
        if info["sha256"] != LINKS_SHA:
            raise ValueError("primary link snapshot changed")
    return previous or spec


def connection(out):
    db = sqlite3.connect(out / "general.sqlite")
    db.execute("PRAGMA journal_mode=DELETE")
    return db


def index(out):
    import bz2
    spec = json.loads(SPEC.read_text())
    db = connection(out)
    try:
        exists = db.execute("SELECT name FROM sqlite_master WHERE name='sentences'").fetchone()
        if exists:
            raise ValueError("source index exists; use its checked index report, never overwrite")
        db.execute("CREATE TABLE sentences(id INTEGER PRIMARY KEY, lang TEXT, text TEXT, norm TEXT, owner TEXT, created TEXT, modified TEXT, line INTEGER, detail_line INTEGER)")
        counts = {}
        for lang in ("eng", "fra"):
            for suffix, fields in (("sentences", 3), ("sentences_detailed", 6)):
                filename = f"{lang}_{suffix}.tsv.bz2"
                if sha(out / filename) != spec["files"][filename]["sha256"]:
                    raise ValueError("source byte hash mismatch")
                count = 0
                with bz2.open(out / filename, "rb") as stream:
                    for count, parts in records(stream, fields):
                        sid, language, text = parts[:3]
                        if not sid.isascii() or not sid.isdigit() or not 0 < int(sid) <= MAX_ID or language != lang or not text:
                            raise ValueError("bad original identity or language")
                        if fields == 3:
                            normalized = hashlib.sha256(norm(text).encode()).hexdigest()
                            db.execute("INSERT INTO sentences(id,lang,text,norm,line) VALUES(?,?,?,?,?)", (int(sid), lang, text, normalized, count))
                        else:
                            changed = db.execute("UPDATE sentences SET owner=?,created=?,modified=?,detail_line=? WHERE id=? AND lang=? AND text=? AND detail_line IS NULL", (*parts[3:], count, int(sid), lang, text)).rowcount
                            if changed != 1:
                                raise ValueError("detailed ID/language/original text is not the same as the general export")
                        if count % 50000 == 0:
                            db.commit()
                db.commit()
                counts[filename] = count
                print("indexed", filename, count, flush=True)
            if counts[f"{lang}_sentences.tsv.bz2"] != counts[f"{lang}_sentences_detailed.tsv.bz2"]:
                raise ValueError("general/detailed counts differ")
        missing = db.execute("SELECT COUNT(*) FROM sentences WHERE detail_line IS NULL").fetchone()[0]
        if missing:
            raise ValueError("originals lack detailed identity bindings")
        db.execute("CREATE INDEX sentence_norm ON sentences(norm)")
        db.commit()
        report = {"source_spec_sha256": sha(SPEC), "rows": counts,
                  "missing_detailed_identities": missing,
                  "missing_credited_owner": db.execute("SELECT COUNT(*) FROM sentences WHERE owner IS NULL OR owner='' OR owner='\\N'").fetchone()[0],
                  "normalized_duplicate_originals": db.execute("SELECT COUNT(*)-COUNT(DISTINCT norm) FROM sentences").fetchone()[0],
                  "license": "CC-BY-2.0-FR", "source_text_changed": False}
        write(out / "general-index.json", report)
        print(json.dumps(report), flush=True)
    finally:
        db.close()


class Union:
    def __init__(self, ceiling=MAX_ID):
        self.parent = array("I", range(ceiling + 1))
        self.rank = bytearray(ceiling + 1)

    def root(self, sid):
        p = self.parent
        while p[sid] != sid:
            p[sid] = p[p[sid]]
            sid = p[sid]
        return sid

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def graph(out):
    if sha(out / "links.tar.bz2") != LINKS_SHA:
        raise ValueError("cached primary links hash mismatch")
    db = connection(out)
    if db.execute("SELECT name FROM sqlite_master WHERE name='pairs'").fetchone():
        db.close()
        raise ValueError("verified link graph exists; reuse its report and database instead of overwriting")
    union = Union()
    languages = bytearray(MAX_ID + 1)
    last_norm, last_id, duplicate_unions = None, None, 0
    for sid, lang, normalized in db.execute("SELECT id,lang,norm FROM sentences ORDER BY norm,id"):
        languages[sid] = 1 if lang == "eng" else 2
        if normalized == last_norm:
            union.join(sid, last_id)
            duplicate_unions += 1
        last_norm, last_id = normalized, sid
    # All-language equivalence first. No unselected language text is fetched.
    edges = {}
    total, maximum = 0, 0
    with tarfile.open(out / "links.tar.bz2", "r|bz2") as archive:
        member = next(iter(archive))
        if member.name != "links.csv" or not member.isfile():
            raise ValueError("unexpected primary links member")
        with archive.extractfile(member) as stream:
            for total, fields in records(stream, 2):
                a, b = map(int, fields)
                if not 0 < a <= MAX_ID or not 0 < b <= MAX_ID or a == b:
                    raise ValueError("invalid primary link")
                maximum = max(maximum, a, b)
                union.join(a, b)
                if languages[a] and languages[b] and languages[a] != languages[b]:
                    e, f, bit = (a, b, 1) if languages[a] == 1 else (b, a, 2)
                    key = (e << 32) | f
                    if edges.get(key, 0) & bit:
                        raise ValueError("duplicate directed English/French link")
                    edges[key] = edges.get(key, 0) | bit
        if archive.next() is not None:
            raise ValueError("extra primary link member")
    missing = sum(value != 3 for value in edges.values())
    if missing:
        raise ValueError(f"{missing} English/French links lack reciprocal identity")
    db.execute("CREATE TABLE pairs(eng INTEGER, fra INTEGER, component INTEGER, PRIMARY KEY(eng,fra))")
    db.executemany("INSERT INTO pairs VALUES(?,?,?)", ((key >> 32, key & 0xffffffff, union.root(key >> 32)) for key in sorted(edges)))
    db.execute("CREATE INDEX pair_component ON pairs(component)")
    db.execute("CREATE TABLE components(component INTEGER PRIMARY KEY, originals INTEGER)")
    counts = Counter(union.root(sid) for sid, in db.execute("SELECT id FROM sentences"))
    db.executemany("INSERT INTO components VALUES(?,?)", counts.items())
    db.execute("CREATE TABLE membership(id INTEGER PRIMARY KEY, component INTEGER)")
    db.executemany("INSERT INTO membership VALUES(?,?)", ((sid, union.root(sid)) for sid, in db.execute("SELECT id FROM sentences")))
    db.execute("CREATE INDEX membership_component ON membership(component)")
    db.commit()
    report = {"links_sha256": LINKS_SHA, "general_spec_sha256": sha(SPEC),
              "all_directed_link_rows": total, "maximum_link_id": maximum,
              "reciprocal_english_french_pairs": len(edges), "missing_reciprocals": missing,
              "normalized_alias_unions": duplicate_unions,
              "english_french_link_components": db.execute("SELECT COUNT(DISTINCT component) FROM pairs").fetchone()[0],
              "graph_scope": "Every language in primary links plus normalized visible English/French text across both languages",
              "source_text_modified": False}
    write(out / "general-graph.json", report)
    print(json.dumps(report), flush=True)
    db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("capture", "index", "graph"))
    parser.add_argument("--out", type=Path, default=HERE / "out")
    args = parser.parse_args()
    try:
        {"capture": capture, "index": index, "graph": graph}[args.phase](args.out)
    except (OSError, ValueError, sqlite3.Error, tarfile.TarError) as error:
        write(args.out / "general-failure.json", {"phase": args.phase, "error": str(error), "model_execution": False})
        raise


if __name__ == "__main__":
    main()
