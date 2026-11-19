# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import datetime as dt
import json
import unittest
from pathlib import Path

import contracts

HERE = Path(__file__).resolve().parent


class Variant2Tests(unittest.TestCase):
    def data(self):
        root = HERE / "data-v2" if (HERE / "data-v2").exists() else HERE / "variant2-data"
        return contracts.read_development(root, "compact-elapsed-v2")

    def test_full_frozen_variant_counts_groups_and_shortcuts(self):
        data = self.data()
        self.assertEqual(len(data["train"]), 512)
        self.assertEqual(len(data["validation"]), 128)
        for rows in data.values():
            for key in ("length", "overlap", "prefer_current"):
                self.assertEqual(contracts.margin_accuracy([contracts.basic_margins(row)[key] for row in rows]), 0.5)
            self.assertEqual(len({row["rejected"] for row in rows}), len(rows))
            self.assertTrue(all(row["observation_modified"] is False for row in rows))

    def test_elapsed_fields_derive_from_unchanged_source_and_assessment(self):
        for rows in self.data().values():
            for row in rows:
                if row["format"] == "json":
                    at = dt.datetime.fromisoformat(json.loads(row["chosen"])["as_of"].replace("Z", "+00:00"))
                    seen = dt.datetime.fromisoformat(row["source_observation"]["last_seen"].replace("Z", "+00:00"))
                    self.assertEqual(int((at - seen).total_seconds()), row["elapsed_seconds"])
                    expected = "publisher_observed_in_window" if row["elapsed_seconds"] <= 30 * 86400 else "historical_publisher_observation"
                    self.assertEqual(row["label"], expected)
                    self.assertEqual(json.loads(row["chosen"])["tier"], row["source_observation"]["tier"])
                    self.assertEqual(json.loads(row["chosen"])["current_universal_reputation"], "unknown")

    def test_development_groups_and_source_values_stay_separate(self):
        data = self.data()
        a = {row["source_asn"] for row in data["train"]}
        b = {row["source_asn"] for row in data["validation"]}
        self.assertFalse(a & b)
        self.assertFalse({row["group_id"] for row in data["train"]} & {row["group_id"] for row in data["validation"]})
        self.assertFalse({row["source_ip"] for row in data["train"]} & {row["source_ip"] for row in data["validation"]})


if __name__ == "__main__":
    unittest.main()
