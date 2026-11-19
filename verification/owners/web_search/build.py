# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Source and data preparation only. No weights, trainer imports or inference."""
import argparse
import ast
import hashlib
import io
import json
import re
import tarfile
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = "WEB_SEARCH"
ATTEMPT = "803e7c2595204a1ba05c3ea079eece73"
NODE = "https://devnode.telegraphprotocol.com"
KIT = "1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd"
TOKENIZER_SHA = "67bc76f8b9386efba1a761c1c8e5ad439fd0eaf1c2826e9dd1a205c7af4dc456"
RELEASE = "https://github.com/telegraphprotocol/telegraph-foundation/releases/download/modernbert-base-lora16-v1/"
SOURCES = {"python": ("python/cpython", "v3.14.0", "Doc/library/", ".rst", "PSF-2.0"),
           "node": ("nodejs/node", "v24.0.0", "doc/api/", ".md", "MIT")}
MAX_ARCHIVE = 128 << 20
MAX_EXPANDED = 512 << 20
MAX_PAGE = 2 << 20


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def fetch(url, ceiling=MAX_PAGE):
    if not url.startswith("https://"):
        raise ValueError("HTTPS source required")
    started = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": "WEB_SEARCH-source-proof/1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read(ceiling + 1)
    if len(raw) > ceiling or time.monotonic() - started > 120:
        raise ValueError("source byte or time ceiling")
    return raw


def capture(root):
    source_dir = root / "sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    pins = {}
    for name, (repo, tag, prefix, suffix, licence) in SOURCES.items():
        ref = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/{tag}"))
        commit = ref["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("invalid resolved source commit")
        url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
        path = source_dir / (name + ".tar.gz")
        if path.is_file():
            raw = path.read_bytes()
            with tarfile.open(fileobj=io.BytesIO(raw)) as cached:
                prefix_name = repo.split("/")[1] + "-" + commit
                if not all(m.name.rstrip("/") == prefix_name or m.name.startswith(prefix_name + "/") for m in cached.getmembers()):
                    raise ValueError("cached source revision mismatch")
        else:
            raw = fetch(url, MAX_ARCHIVE)
            path.write_bytes(raw)
        pins[name] = {"repo": repo, "tag": tag, "revision": commit, "url": url,
                      "sha256": sha(raw), "bytes": len(raw), "prefix": prefix,
                      "suffix": suffix, "licence": licence}
        write(root / "source-pins.json", {"family": FAMILY, "run_id": ATTEMPT, "sources": pins})
    for name, url in {"trainer.py": f"https://raw.githubusercontent.com/telegraphprotocol/telegraph-wasm-baseline/{KIT}/float_runtime/train_adapter.py",
                      "tokenizer.json": RELEASE + "tokenizer.json"}.items():
        raw = fetch(url, 8 << 20)
        (source_dir / name).write_bytes(raw)
        pins[name] = {"url": url, "sha256": sha(raw), "bytes": len(raw)}
        write(root / "source-pins.json", {"family": FAMILY, "run_id": ATTEMPT, "sources": pins})
    if pins["tokenizer.json"]["sha256"] != TOKENIZER_SHA:
        raise ValueError("released tokenizer pin mismatch")
    live = {"family": json.loads(fetch(NODE + "/api/wasm?resolve=WEB_SEARCH")),
            "child": json.loads(fetch(NODE + "/engine/v1/intents/WEB_SEARCH_GENERAL"))}
    if (live["family"]["resolve"]["class"] != "judgment" or
            live["family"]["resolve"]["intent"] != FAMILY or
            live["child"]["class"] != "judgment" or live["child"]["intent_id"] != FAMILY + "_GENERAL"):
        raise ValueError("live class disagreement")
    write(root / "live.json", {"at": datetime.now(timezone.utc).isoformat(), "node": NODE, **live})
    write(root / "source-pins.json", {"family": FAMILY, "run_id": ATTEMPT, "sources": pins})


def inventory(root, name, pin):
    if pin.get("scoped_snapshot"):
        raw = (root / "sources/node-pages.json").read_bytes()
        if sha(raw) != pin["scoped_snapshot"] or len(raw) > 16 << 20:
            raise ValueError("scoped documentation snapshot mismatch")
        return json.loads(raw)
    raw = (root / "sources" / (name + ".tar.gz")).read_bytes()
    if sha(raw) != pin["sha256"] or len(raw) > MAX_ARCHIVE:
        raise ValueError("source archive pin mismatch")
    pages = {}
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        members = archive.getmembers()
        expanded = sum(m.size for m in members)
        write(root / (name + "-archive-inventory.json"), {"members": len(members), "declared_expanded_bytes": expanded,
                                                       "expanded_ceiling": MAX_EXPANDED})
        if len(members) > 100000 or expanded > MAX_EXPANDED:
            raise ValueError(f"{name} archive expansion ceiling: {len(members)} members, {expanded} bytes")
        for member in members:
            path = member.name.partition("/")[2]
            if path == "LICENSE":
                if not member.isfile() or member.size > MAX_PAGE:
                    raise ValueError("invalid source licence member")
                (root / "sources" / (name + "-LICENSE")).write_bytes(archive.extractfile(member).read())
            if path.startswith(pin["prefix"]) and path.endswith(pin["suffix"]):
                if not member.isfile() or member.size > MAX_PAGE or ".." in Path(path).parts:
                    raise ValueError("invalid documentation member")
                if path in pages:
                    raise ValueError("duplicate documentation member")
                pages[path] = archive.extractfile(member).read().decode("utf-8")
    if not pages or not (root / "sources" / (name + "-LICENSE")).is_file():
        raise ValueError("missing complete source inventory or licence")
    return pages


def capture_node_docs(root):
    record = json.loads((root / "source-pins.json").read_text())
    pin = record["sources"]["node"]
    repo, commit = pin["repo"], pin["revision"]
    def tree(revision):
        data = json.loads(fetch(f"https://api.github.com/repos/{repo}/git/trees/{revision}"))
        if data.get("truncated") is not False:
            raise ValueError("documentation tree is truncated")
        return data
    top = tree(commit)
    doc = tree(next(item["sha"] for item in top["tree"] if item["path"] == "doc"))
    api = tree(next(item["sha"] for item in doc["tree"] if item["path"] == "api"))
    if any(item["type"] == "tree" for item in api["tree"]):
        raise ValueError("unexpected nested documentation inventory")
    entries = [item for item in api["tree"] if item["path"].endswith(".md")]
    def page(item):
        path = "doc/api/" + item["path"]
        raw = fetch(f"https://raw.githubusercontent.com/{repo}/{commit}/{path}")
        git_hash = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if item["type"] != "blob" or git_hash != item["sha"]:
            raise ValueError("documentation Git blob identity mismatch")
        return path, raw.decode("utf-8")
    with ThreadPoolExecutor(max_workers=4) as pool:
        pages = dict(pool.map(page, entries))
    raw = (json.dumps(pages, sort_keys=True, ensure_ascii=True) + "\n").encode()
    if len(raw) > 16 << 20 or len(pages) != len(entries):
        raise ValueError("complete documentation snapshot ceiling")
    (root / "sources/node-pages.json").write_bytes(raw)
    (root / "sources/node-LICENSE").write_bytes(fetch(f"https://raw.githubusercontent.com/{repo}/{commit}/LICENSE"))
    pin["scoped_snapshot"] = sha(raw)
    pin["scoped_pages"] = len(pages)
    write(root / "node-doc-tree.json", api)
    write(root / "source-pins.json", record)


def python_facts(text):
    lines = text.splitlines()
    result = []
    for i, line in enumerate(lines):
        match = re.fullmatch(r"([ ]*)\.\. (?:function|method|classmethod|staticmethod|class|attribute|data):: ([A-Za-z_][\w.]*)[\s(].*", line + " ")
        if not match:
            continue
        indent, entity = len(match[1]), match[2]
        end = i + 1
        while end < len(lines) and (not lines[end].strip() or len(lines[end]) - len(lines[end].lstrip()) > indent):
            end += 1
        added = []
        ambiguous = False
        for j in range(i + 1, end):
            m = re.fullmatch(r"([ ]*)\.\. versionadded:: (3\.\d+)", lines[j])
            if not m or len(m[1]) != indent + 3:
                continue
            # Subordinate text can describe a parameter, platform or alias instead.
            after = j + 1
            while after < end and not lines[after].strip():
                after += 1
            if after < end and len(lines[after]) - len(lines[after].lstrip()) > len(m[1]):
                ambiguous = True
            added.append((m[2], j + 1))
        if len(added) == 1 and not ambiguous:
            version, number = added[0]
            result.append({"entity": entity, "version": version, "line": number,
                           "raw_fact": lines[number - 1], "heading_line": i + 1})
    return result


def node_facts(text):
    lines = text.splitlines()
    result = []
    for i, line in enumerate(lines):
        m = re.fullmatch(r"#{2,6} `([A-Za-z_][\w.]*)\([^`]*\)`", line)
        if not m or i + 2 >= len(lines) or lines[i + 2] != "<!-- YAML":
            continue
        end = i + 3
        while end < len(lines) and lines[end] != "-->":
            end += 1
        block = lines[i + 3:end]
        added = [re.fullmatch(r"added: (v\d+\.\d+\.\d+)", item) for item in block]
        added = [a for a in added if a]
        if end < len(lines) and len(added) == 1:
            number = i + 4 + block.index(added[0][0])
            result.append({"entity": m[1], "version": added[0][1], "line": number,
                           "raw_fact": lines[number - 1], "heading_line": i + 1})
    return result


def render(prompt, answer):
    return f"### Task:\n{prompt}\n\n### Answer:\n{answer}"


def make_block(name, path, first, second, pin):
    versions = [first["version"], second["version"]]
    if len(set(versions)) != 2 or len(versions[0]) != len(versions[1]):
        return []
    url = ("https://docs.python.org/3.14/library/" + Path(path).stem + ".html" if name == "python" else
           "https://nodejs.org/docs/latest-v24.x/api/" + Path(path).stem + ".html")
    label = "Python" if name == "python" else "Node.js"
    rows = []
    for target, other in ((first, second), (second, first)):
        prompt = json.dumps({"query": f"Which {label} version added {target['entity']}?",
                             "evidence": f"{first['entity']}: added {versions[0]}; {second['entity']}: added {versions[1]}."}, separators=(",", ":"))
        if len(prompt.encode()) > 256:
            return []
        def answer(version):
            claim = f"{target['entity']} was added in {label} {version}."
            if len(claim) > 80:
                return None
            return json.dumps({"answer": claim, "sources": [url], "brief": claim}, separators=(",", ":"))
        chosen, rejected = answer(target["version"]), answer(other["version"])
        if not chosen or not rejected:
            return []
        rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "label_kind": "preference_pair",
                     "group_id": sha((name + ":" + path).encode()), "domain": name, "page": path,
                     "entities": [first["entity"], second["entity"]], "facts": [first, second],
                     "source_revision": pin["revision"], "source_licence": pin["licence"],
                     "preference_authorship": "programmatic version swap, AI-assisted code, not human-reviewed"})
    return rows


def validate(splits):
    seen = {key: set() for key in ("page", "entity", "answer")}
    for split, rows in splits.items():
        current = {key: set() for key in seen}
        if len(rows) < 32:
            raise ValueError("insufficient rows: " + split)
        for row in rows:
            if row["chosen"] == row["rejected"] or len(row["prompt"].encode()) > 256:
                raise ValueError("invalid contrast or request size")
            if len(row["chosen"]) != len(row["rejected"]):
                raise ValueError("length shortcut")
            current["page"].add(normalized(row["domain"] + ":" + row["page"]))
            current["entity"].update(normalized(row["domain"] + ":" + e) for e in row["entities"])
            current["answer"].update(normalized(row[s]) for s in ("chosen", "rejected"))
        if any(seen[key] & current[key] for key in seen):
            raise ValueError("normalized source/entity/answer leakage")
        for key in seen:
            seen[key].update(current[key])
        counts = Counter(row["rejected"] for row in rows)
        if len(counts) / len(rows) < .8 or max(counts.values()) / len(rows) > .02:
            raise ValueError("repeated negative gate")


def build(root):
    pins = json.loads((root / "source-pins.json").read_text())["sources"]
    trainer_raw = (root / "sources/trainer.py").read_bytes()
    if sha(trainer_raw) != pins["trainer.py"]["sha256"]:
        raise ValueError("trainer source changed")
    tree = ast.parse(trainer_raw)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render")
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "primary-render", "exec"), scope)
    if scope["render"]("a\r\nb", "{\"x\":1}") != render("a\r\nb", "{\"x\":1}"):
        raise ValueError("literal primary renderer mismatch")
    from tokenizers import Tokenizer  # tokenizer only, never a model
    tokenizer = Tokenizer.from_file(str(root / "sources/tokenizer.json"))
    if sha((root / "sources/tokenizer.json").read_bytes()) != TOKENIZER_SHA:
        raise ValueError("tokenizer identity changed")
    tokenizer.no_padding()
    tokenizer.no_truncation()
    splits = {"train": [], "validation": [], "audit": []}
    report = {}
    maximum = 0
    for name in SOURCES:
        pages = inventory(root, name, pins[name])
        facts = {path: (python_facts(text) if name == "python" else node_facts(text)) for path, text in pages.items()}
        # Repeated API identifiers are ambiguous across pages. Drop them globally
        # before assigning groups, including property vs method aliases.
        frequency = Counter(normalized(f["entity"]) for items in facts.values() for f in items)
        page_report = []
        for path, items in sorted(facts.items()):
            unique = [f for f in items if frequency[normalized(f["entity"])] == 1]
            unique.sort(key=lambda f: sha((path + ":" + f["entity"]).encode()))
            used = set()
            page_rows = []
            for i, first in enumerate(unique):
                if i in used:
                    continue
                for j, second in enumerate(unique[i + 1:], i + 1):
                    if j in used:
                        continue
                    block = make_block(name, path, first, second, pins[name])
                    if not block:
                        continue
                    visible = True
                    for row in block:
                        ids = [tokenizer.encode(scope["render"](row["prompt"], row[s])).ids for s in ("chosen", "rejected")]
                        maximum = max(maximum, *map(len, ids))
                        if max(map(len, ids)) > 256 or ids[0] == ids[1]:
                            visible = False
                    if visible:
                        page_rows.extend(block)
                        used.update((i, j))
                        break
            split = "audit" if name == "node" else "validation" if int(sha(path.encode())[:8], 16) % 5 == 0 else "train"
            splits[split].extend(page_rows)
            page_report.append({"path": path, "sha256": sha(pages[path].encode()), "facts": len(items),
                                "unambiguous_unique_facts": len(unique), "pairs": len(page_rows), "split": split})
        report[name] = {"complete_inventory": True, "pages": len(pages), "metadata_facts": sum(map(len, facts.values())),
                        "page_inventory": page_report}
    validate(splits)
    files = {}
    for split, rows in splits.items():
        # Keep complete balanced blocks. A bounded deterministic prefix after full
        # inventory is a pilot, never a representative population measurement.
        limit = {"train": 256, "validation": 64, "audit": 128}[split]
        groups = defaultdict(list)
        for row in rows:
            groups[row["group_id"]].append(row)
        selected = []
        keys = sorted(groups)
        for offset in range(0, max(map(len, groups.values())), 2):
            for key in keys:
                block = groups[key][offset:offset + 2]
                if len(block) == 2 and len(selected) + 2 <= limit:
                    selected.extend(block)
        rows = selected
        splits[split] = rows
        raw = "".join(json.dumps(r, ensure_ascii=True, separators=(",", ":")) + "\n" for r in rows).encode()
        target = root / (split + ".jsonl")
        target.write_bytes(raw)
        files[split] = {"rows": len(rows), "pages": len({r["group_id"] for r in rows}), "sha256": sha(raw)}
    validate(splits)
    gradient = Counter()
    for row in splits["train"]:
        for word in re.findall(r"[a-z0-9']+", row["chosen"].casefold()):
            gradient[word] += 1
        for word in re.findall(r"[a-z0-9']+", row["rejected"].casefold()):
            gradient[word] -= 1
    if any(gradient.values()):
        raise ValueError("exact train-only linear baseline balance failed")
    frozen = {"fit_split": "train", "length_direction": 1, "overlap_direction": 1,
              "answer_only_bow_coefficients": {}, "explanation": "Mirrored complete blocks give exact zero linear bag-of-words gradient. Coefficients are immutable zero, not refitted during reproduction."}
    frequencies = Counter()
    for row in splits["train"]:
        frequencies.update(set(re.findall(r"[a-z0-9']+", row["prompt"].casefold())))
    frozen["train_document_frequency"] = dict(sorted(frequencies.items()))
    write(root / "frozen-baselines.json", frozen)
    # Audit contents checked for structural identity only. No audit score used
    # to select data or fit baseline orientations.
    write(root / "data-report.json", {"family": FAMILY, "run_id": ATTEMPT, "sources": pins, "inventory": report,
          "files": files, "normalized_cross_split_overlap": 0, "max_checked_tokens": maximum,
          "request_byte_ceiling": 256, "token_ceiling": 256, "audit_metrics": "not measured",
          "baseline_sha256": sha((root / "frozen-baselines.json").read_bytes()),
          "selection": "bounded source-order pilot after full inventory, not a population benchmark",
          "scope": "supplied API introduction facts only, no browsing, no future-time claim"})
    print(json.dumps({"files": files, "inventory_pages": {n: r["pages"] for n, r in report.items()}, "max_checked_tokens": maximum}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("capture", "capture-node-docs", "build"))
    parser.add_argument("--root", type=Path, default=HERE / "evidence")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    {"capture": capture, "capture-node-docs": capture_node_docs, "build": build}[args.action](args.root)


if __name__ == "__main__":
    main()
