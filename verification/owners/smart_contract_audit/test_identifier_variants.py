# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import json
import tempfile
import unittest
from pathlib import Path

import data


class IdentifierChecks(unittest.TestCase):
    def test_full_original_train_and_heldout_bytes_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = data.build(root)
            kept = [r for r in rows if r["split"] == "train" and "augmentation" not in r]
            raw = b"".join((json.dumps(r, sort_keys=True) + "\n").encode() for r in kept)
            self.assertEqual(data.sha(raw), "2e3ddb9793a001def2239bbd3350be720e225302c53228894f633408ffd50c48")
            extra = [r for r in rows if "augmentation" in r]
            self.assertEqual(len(extra), 48)
            self.assertEqual({r["split"] for r in extra}, {"train"})
            self.assertEqual(len({r["template_group"] for r in extra}), 12)
            self.assertEqual(data.check(rows)["template_groups"], 46)
            self.assertEqual(data.sha((root / "validation.jsonl").read_bytes()), "8c79f65c18479354acc40065036a7d1e7c26a6ddcb6cc2b41eb608266138aec4")
            self.assertEqual(data.sha((root / "audit.jsonl").read_bytes()), "6cd7d2c0b11206df3c6ad04471cd53683c791f9393140e8c764101457cc31dd8")


if __name__ == "__main__":
    unittest.main()
