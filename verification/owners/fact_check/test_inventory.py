# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Artificial transport/schema controls only, never training data."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import inventory
from inventory import QUERIES, collect, fetch, page_query, validate


def fixtures(count=2, ids=("Q1", "Q2")):
    aggregate = {"results": {"bindings": [{"count": {"value": str(count)}}]}}
    entities = {"results": {"bindings": [{"entity": {"type": "uri", "value": "http://www.wikidata.org/entity/" + ident}}
                                         for ident in ids]}}
    return json.dumps(aggregate).encode(), json.dumps(entities).encode()


class InventoryTests(unittest.TestCase):
    def test_primary_population_contract(self):
        self.assertEqual(validate(*fixtures()), ["Q1", "Q2"])
        self.assertTrue(all("OFFSET" not in query for query in QUERIES.values()))
        self.assertIn("STR(?entity) >", page_query("Q123"))
        with self.assertRaises(ValueError):
            page_query('Q1" arbitrary query')

    def test_first_page_duplicate_and_bad_identity_fail(self):
        for args in ((3, ("Q1", "Q2")), (2, ("Q1", "Q1")), (1, ("not-an-entity",)), (0, ())):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate(*fixtures(*args))

    def test_arbitrary_or_truncated_query_refused_before_network(self):
        with self.assertRaisesRegex(ValueError, "exact aggregate"):
            fetch(QUERIES["entities"] + " LIMIT 512")

    def test_timeout_resume_preserves_request_history_and_full_count(self):
        aggregate, entities = fixtures()
        with tempfile.TemporaryDirectory(dir=inventory.HERE) as temporary:
            root = Path(temporary)
            (root / "SOURCE-CC0.txt").write_bytes((inventory.HERE / "SOURCE-CC0.txt").read_bytes())
            (root / "sources").mkdir()
            (root / "sources/discovery.json").write_bytes(entities)
            def response(query):
                raw = aggregate if query == QUERIES["count"] else entities
                url = "https://query.wikidata.org/sparql?" + inventory.urllib.parse.urlencode({"query": query, "format": "json"})
                return raw, url
            calls = 0
            def transient(query):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise TimeoutError("artificial timeout control, not a source result")
                return response(query)
            with patch.object(inventory, "HERE", root), patch.object(inventory, "fetch", transient), patch.object(inventory.time, "sleep"):
                result = collect(root / "out")
            self.assertTrue(result["complete"])
            self.assertEqual(result["population"], 2)
            self.assertEqual(result["attempts"], 4)
            self.assertEqual(len(result["refusals"]), 1)
            self.assertEqual(len(result["records"]), 3)
            with patch.object(inventory, "HERE", root), patch.object(inventory, "fetch", side_effect=AssertionError("cache replay made a network request")):
                self.assertTrue(collect(root / "out")["complete"])

    def test_timeout_budget_and_invalid_source_remain_fail_closed(self):
        self.assertEqual(inventory.SPACING_SECONDS, 65)
        for error, expected_calls in ((TimeoutError("artificial timeout"), 16), (ValueError("artificial invalid source"), 1)):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory(dir=inventory.HERE) as temporary:
                root = Path(temporary)
                (root / "SOURCE-CC0.txt").write_bytes((inventory.HERE / "SOURCE-CC0.txt").read_bytes())
                with patch.object(inventory, "HERE", root), patch.object(inventory, "fetch", side_effect=error) as request, patch.object(inventory.time, "sleep"), self.assertRaises(ValueError):
                    collect(root / "out")
                self.assertEqual(request.call_count, expected_calls)
                result = json.loads((root / "out/index.json").read_text())
                self.assertFalse(result["complete"])
                self.assertEqual(result["attempts"], expected_calls)


if __name__ == "__main__":
    unittest.main()
