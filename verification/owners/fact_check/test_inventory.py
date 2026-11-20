# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Artificial transport/schema controls only, never training data."""
import json
import unittest

from inventory import QUERIES, fetch, page_query, validate


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


if __name__ == "__main__":
    unittest.main()
