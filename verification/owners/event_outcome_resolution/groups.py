# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Count independent event/entity components, never just preference rows."""
def components(rows):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    events = set()
    for row in rows:
        eid = "event:" + row["event_id"]
        events.add(eid)
        for person in row["people"]:
            union(eid, "person:" + person)
    return len({find(e) for e in events})


if __name__ == "__main__":
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent
    for name in ("train", "validation", "audit"):
        rows = [json.loads(l) for l in (root / "frozen" / (name + ".jsonl")).read_text().splitlines()]
        print(name, "events", len({r["event_id"] for r in rows}), "independent components", components(rows))
