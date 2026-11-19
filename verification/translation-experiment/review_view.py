# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Readable frozen original/donor pairs for bilingual review, never new text."""
import json
from pathlib import Path

from archive_failed import ARCHIVE
from source_probe import sha

HERE = Path(__file__).resolve().parent
FROZEN_SHA = "ca65b547545bee9858f69bc73a70c30695db947c5ba80c793ca9944e428212c0"


def main():
    path = ARCHIVE / "pilot-manifest.json"
    if sha(path) != FROZEN_SHA:
        raise ValueError("frozen pilot changed")
    manifest = json.loads(path.read_text())
    lines = ["Archived failed bilingual source review. Each line is two original old links. The old audit remains failed."]
    for split, blocks in manifest["splits"].items():
        for number, block in enumerate(blocks):
            texts = []
            for e, f in block["links"]:
                originals = manifest["originals"]
                texts.append(f"{e}/{f}: {originals[str(e)]['text']} => {originals[str(f)]['text']}")
            lines.append(f"{split}:{number:03} {block['id'][:12]} | " + " || ".join(texts))
    (HERE / "out/bilingual-review.txt").write_text("\n".join(lines) + "\n")
    print("wrote", len(lines) - 1, "frozen blocks for review")


if __name__ == "__main__":
    main()
