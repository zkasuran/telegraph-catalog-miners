# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Original bounded Solidity corpus. Preferences are constructed, not human labels."""
import argparse
import hashlib
import json
import re
from pathlib import Path

FAMILY = "SMART_CONTRACT_AUDIT"
ATTEMPT = "74fac904d3e14d2690ef15803146a56a"
DATA_VERSION = "self-contained-coverage-v4"
BASELINE_FILE = "frozen-baselines-v4.json"
SAMPLING_FILE = "frozen-sampling-v3.json"
RETIRED_AUDIT_SHA256 = "6cd7d2c0b11206df3c6ad04471cd53683c791f9393140e8c764101457cc31dd8"
SPDX = "// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0\n"
PLAN = {"train": (range(4), 8), "validation": (range(4, 6), 6), "audit": (range(0), 0)}
MAX_SOURCE_BYTES = 2048
MAX_ROWS = 256


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def render(prompt, answer):
    return "### Task:\n" + prompt + "\n\n### Answer:\n" + answer


def normalize(source):
    # Alpha-normalize generated identifiers and numeric constants, not keywords.
    source = re.sub(r"//[^\n]*", "", source)
    source = re.sub(r"\b(?:C|f|v|read|auth|helper|only|Authority)\d+\b", "ID", source)
    source = re.sub(r"\b\d+\b", "NUM", source)
    return " ".join(re.findall(r"\w+|[^\w\s]", source))


def source_for(template, entity, vulnerable):
    if template >= 8:
        raise ValueError("retired exposed source-audit designs cannot generate new fitting/test rows")
    c, f, v, rd = (p + str(entity) for p in ("C", "f", "v", "read"))
    owner = 100 + entity
    if template in (0, 1, 4, 6):
        guard = "" if vulnerable else f"require(msg.sender == address({owner}));"
        decl = ""
        suffix = ""
        if template == 1:
            guard = "" if vulnerable else f"if (msg.sender != address({owner})) revert();"
        elif template == 4:
            decl = f"modifier only{entity}() {{ {guard} _; }}"
            guard, suffix = "", f" only{entity}"
        elif template == 6:
            decl = f"function auth{entity}() private view returns(bool) {{ return msg.sender == address({owner}); }}"
            guard = "" if vulnerable else f"require(auth{entity}());"
        body = f"uint256 private {v}; {decl} function {f}(uint256 x) external{suffix} {{ {guard} {v} = x; }} function {rd}() external view returns(uint256) {{ return {v}; }}"
        facts = {"mechanism": "access_control", "function": f, "read_function": rd,
                 "owner": owner, "attacker": owner + 10000, "argument": 17 + entity,
                 "signature": f + "(uint256)"}
    else:
        width = {2: 8, 3: 8, 5: 16, 7: 32}[template]
        typ = f"uint{width}"
        op = "*" if template == 5 else "-" if template == 7 else "+"
        expr = f"x {op} y"
        if template == 2:
            calc = f"unchecked {{ return {expr}; }}" if vulnerable else f"return {expr};"
            body = f"function {f}({typ} x, {typ} y) external pure returns({typ}) {{ {calc} }}"
        else:
            bound = f"require(x >= y);" if op == "-" else f"require(y == 0 || x <= type({typ}).max / y);" if op == "*" else f"require(x <= type({typ}).max - y);"
            calc = ("" if vulnerable else bound) + f" unchecked {{ return {expr}; }}"
            if template == 3:
                body = f"function {f}({typ} x, {typ} y) external pure returns({typ}) {{ {calc} }}"
            else:
                body = f"function {f}({typ} x, {typ} y) external pure returns({typ}) {{ return helper{entity}(x,y); }} function helper{entity}({typ} x, {typ} y) private pure returns({typ}) {{ {calc} }}"
        bad = [2**width - 1, 2] if op != "-" else [0, 1]
        good = [2, 3] if op != "-" else [3, 2]
        result = ((bad[0] * bad[1]) if op == "*" else (bad[0] - bad[1]) if op == "-" else sum(bad)) % 2**width
        normal = (good[0] * good[1]) if op == "*" else (good[0] - good[1]) if op == "-" else sum(good)
        facts = {"mechanism": "arithmetic", "function": f, "width": width, "operation": op,
                 "bad_args": bad, "good_args": good, "wrapped_result": result, "normal_result": normal,
                 "signature": f + f"({typ},{typ})"}
    source = SPDX + f"pragma solidity 0.8.30;\ncontract {c} {{ {body} }}\n"
    facts["contract"] = c
    if len(source.encode()) > MAX_SOURCE_BYTES:
        raise ValueError("source too long")
    return source, facts


def answers(facts, vulnerable, style=0):
    f = facts["function"]
    if facts["mechanism"] == "access_control":
        detail = f"Access control in {f}: caller {facts['attacker']} is not owner {facts['owner']}."
        yes = f"{detail} The caller can set the stored value to {facts['argument']}."
        no = f"{detail} The caller cannot set the stored value to {facts['argument']}."
    else:
        detail = f"Arithmetic in {f} with " + " and ".join(f"{name}={value}" for name, value in zip("xyz", facts["bad_args"])) + ":"
        yes = f"{detail} the call returns {facts['wrapped_result']} after wraparound."
        no = f"{detail} the call reverts instead of returning {facts['wrapped_result']} after wraparound."
    if style == 1:
        yes, no = (json.dumps({"finding": s}, separators=(",", ":")) for s in (yes, no))
    elif style == 2:
        yes, no = ("Report\n" + s + "\nScope: only this call." for s in (yes, no))
    return (yes, no) if vulnerable else (no, yes)


def response_views(rows, split):
    """Construct balanced supported-format views only inside train/validation.

    These repeated views are not new source groups or human annotations.
    The independent audit is never passed into this development constructor.
    """
    if split not in ("train", "validation") or any(row["split"] != split for row in rows):
        raise ValueError("development views may use only their own train/validation split")
    views = []
    for row in rows:
        for style, name in enumerate(("prose", "json", "report")):
            chosen, rejected = answers(row["facts"], row["vulnerable"], style)
            views.append({**row, "chosen": chosen, "rejected": rejected, "response_view": name})
    return views


def check(rows):
    if not 0 < len(rows) <= MAX_ROWS:
        raise ValueError("row ceiling")
    seen, groups = {}, {}
    for row in rows:
        if row["source_sha256"] != sha(row["source"].encode()):
            raise ValueError("source hash mismatch")
        if row["chosen"] == row["rejected"]:
            raise ValueError("identical preference")
        if row["source"] not in row["prompt"] or "@vulnerable" in row["prompt"]:
            raise ValueError("source visibility or annotation leak")
        for key in (row["template_group"], row["sibling_group"]):
            if key in groups and groups[key] != row["split"]:
                raise ValueError("group crosses split")
            groups[key] = row["split"]
        norm = normalize(row["source"])
        if norm in seen and seen[norm] != row["split"]:
            raise ValueError("normalized source overlaps splits")
        seen[norm] = row["split"]
    return {"normalized_cross_split_duplicates": 0, "source_rows": len(rows),
            "template_groups": len({r["template_group"] for r in rows})}


def build(out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, (templates, count) in PLAN.items():
        for template in templates:
            for number in range(count):
                entity = 1000 * (template + 1) + number
                for vulnerable in (True, False):
                    source, facts = source_for(template, entity, vulnerable)
                    chosen, rejected = answers(facts, vulnerable)
                    if template < 8:
                        call = f"Caller {facts['attacker']} calls {facts['function']}({facts['argument']}); owner is {facts['owner']}." if facts["mechanism"] == "access_control" else f"Call {facts['function']}({facts['bad_args'][0]}, {facts['bad_args'][1]})."
                    else:
                        call = f"Caller {facts['attacker']} calls {facts['function']}({', '.join(map(str, facts['bad_args']))}); owner is {facts['owner']}." if facts["mechanism"] == "access_control" else f"Call {facts['function']}({', '.join(map(str, facts['bad_args']))})."
                    prompt = "Audit the complete Solidity source for the stated call. Describe whether that call succeeds and its effect, not universal safety.\n" + call + "\n" + source
                    rows.append({"id": f"{entity}-{int(vulnerable)}", "split": split,
                                 "template_group": f"design-{template}", "sibling_group": str(entity),
                                 "source": source, "source_sha256": sha(source.encode()), "prompt": prompt,
                                 "chosen": chosen, "rejected": rejected, "label_kind": "preference_pair",
                                 "vulnerable": vulnerable, "facts": facts,
                                 "authorship": "Original AI-assisted Solidity and constructed preferences; no human review claimed",
                                  "licence": "LicenseRef-zkasuran-SAND-1.0"})
    from training_designs import candidates
    for candidate in candidates():
        facts, source = candidate["facts"], candidate["source"]
        chosen, rejected = answers(facts, candidate["vulnerable"])
        call = f"Caller {facts['attacker']} calls {facts['function']}({facts['argument']}); owner is {facts['owner']}." if facts["mechanism"] == "access_control" else f"Call {facts['function']}({', '.join(map(str, facts['bad_args']))})."
        prompt = "Audit the complete Solidity source for the stated call. Describe whether that call succeeds and its effect, not universal safety.\n" + call + "\n" + source
        rows.append({"id": candidate["id"], "split": "train", "template_group": candidate["proposed_group"],
                     "sibling_group": candidate["sibling_group"], "source": source, "source_sha256": candidate["source_sha256"],
                     "prompt": prompt, "chosen": chosen, "rejected": rejected, "label_kind": "preference_pair",
                     "vulnerable": candidate["vulnerable"], "facts": facts,
                     "authorship": "Original AI-assisted Solidity and constructed preferences; no human review claimed",
                     "licence": "LicenseRef-zkasuran-SAND-1.0"})
    # Training-side identifier variants, not new implementations or audit inputs.
    extra = [r for r in candidates(4) if int(r["id"].split("-")[3]) >= 2]
    for template in range(4):
        for variant in range(2):
            entity = 200000 + template * 100 + variant
            for vulnerable in (True, False):
                source, facts = source_for(template, entity, vulnerable)
                extra.append({"id": f"augment-core-{template}-{variant}-{int(vulnerable)}", "proposed_group": f"design-{template}",
                              "sibling_group": str(entity), "source": source, "source_sha256": sha(source.encode()),
                              "vulnerable": vulnerable, "facts": facts})
    for candidate in extra:
        facts, source = candidate["facts"], candidate["source"]
        chosen, rejected = answers(facts, candidate["vulnerable"])
        call = f"Caller {facts['attacker']} calls {facts['function']}({facts['argument']}); owner is {facts['owner']}." if facts["mechanism"] == "access_control" else f"Call {facts['function']}({', '.join(map(str, facts['bad_args']))})."
        rows.append({"id": candidate["id"], "split": "train", "template_group": candidate["proposed_group"],
                     "sibling_group": candidate["sibling_group"], "source": source, "source_sha256": candidate["source_sha256"],
                     "prompt": "Audit the complete Solidity source for the stated call. Describe whether that call succeeds and its effect, not universal safety.\n" + call + "\n" + source,
                     "chosen": chosen, "rejected": rejected, "label_kind": "preference_pair", "vulnerable": candidate["vulnerable"], "facts": facts,
                     "authorship": "Original AI-assisted Solidity and constructed preferences; no human review claimed",
                     "licence": "LicenseRef-zkasuran-SAND-1.0", "augmentation": "training-only identifier variants of existing implementation groups"})
    # Preserve legitimate training/validation in their exact original order.
    # The exposed original audit is excluded, not regenerated as new test data.
    rows = [r for r in rows if r["split"] == "train"] + [r for r in rows if r["split"] == "validation"]
    from fresh_sources import training as broad_training, audit as clean_audit
    for split, count, start, factory in (("train", 12, 340000, broad_training), ("audit", 32, 500000, clean_audit)):
        for design in range(count):
            entity = start + design * 10
            for vulnerable in (True, False):
                source, facts = factory(design, entity, vulnerable)
                chosen, rejected = answers(facts, vulnerable)
                call = f"Caller {facts['attacker']} calls {facts['function']}({facts['argument']}); owner is {facts['owner']}." if facts["mechanism"] == "access_control" else f"Call {facts['function']}({', '.join(map(str, facts['bad_args']))})."
                rows.append({"id": f"fresh-{split}-{design}-{int(vulnerable)}", "split": split,
                             "template_group": f"fresh-{split}-design-{design}", "sibling_group": str(entity),
                             "source": source, "source_sha256": sha(source.encode()), "facts": facts, "vulnerable": vulnerable,
                             "prompt": "Audit the complete Solidity source for the stated call. Describe whether that call succeeds and its effect, not universal safety.\n" + call + "\n" + source,
                             "chosen": chosen, "rejected": rejected, "label_kind": "preference_pair",
                             "authorship": "Original AI-assisted Solidity and constructed preferences; no human review claimed",
                             "licence": "LicenseRef-zkasuran-SAND-1.0", "provenance": "New source implementation, no exposed audit source/annotation/score used"})
    report = check(rows)
    files = {}
    for split in PLAN:
        raw = b"".join((json.dumps(r, sort_keys=True) + "\n").encode() for r in rows if r["split"] == split)
        (out / (split + ".jsonl")).write_bytes(raw)
        files[split] = {"sha256": sha(raw), "rows": sum(r["split"] == split for r in rows),
                        "template_groups": sorted({r["template_group"] for r in rows if r["split"] == split})}
    # Full owned inventory, not the first page of an imported corpus.
    raw = (json.dumps(rows, sort_keys=True, indent=2) + "\n").encode()
    (out / "inventory.json").write_bytes(raw)
    report.update({"family": FAMILY, "owner_attempt": ATTEMPT, "files": files,
                   "inventory_sha256": sha(raw), "labels": "pending compiler/EVM proof, not accepted training labels",
                   "scope": "Bounded owner-only setter and integer wrapping claims. No reentrancy, bytecode-only or arbitrary audit coverage.",
                   "independent_groups": len({r["template_group"] for r in rows}),
                   "audit_used_for_selection": False,
                    "data_version": DATA_VERSION, "baseline_file": BASELINE_FILE, "sampling_file": SAMPLING_FILE,
                    "retired_exposed_audit_sha256": RETIRED_AUDIT_SHA256,
                    "independence": "Twenty-four original training implementations, two retained validation implementations and 32 clean reserved audit implementations. All siblings and variants stay grouped. Original training and validation bytes are retained; the exposed original audit is retired and excluded from new fitting/test data. All source/preferences are AI-assisted, not independent human adjudication."})
    write(out / "data-report.json", report)
    print(json.dumps(report, sort_keys=True))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    build(parser.parse_args().out)
