# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import ast
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

from inventory import parse_table
import scorer
from split_gate import check_split, identity

ROOT = Path(__file__).resolve().parent


class ProductTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = scorer.records()

    def answer(self, product):
        return scorer.expectation(product, self.rows)

    def test_all_released_records_and_label_counts(self):
        self.assertEqual(len(self.rows), 34)
        self.assertEqual(sum(r["source_label"] == "Authentic" for r in self.rows.values()), 11)
        self.assertEqual(sum(r["source_label"] == "Counterfeit" for r in self.rows.values()), 23)
        for rid in self.rows:
            product = scorer.product_for(rid)
            correct = self.answer(product)
            self.assertEqual(scorer.score(product, json.dumps(correct), self.rows)["score"], 10000)
            wrong = copy.deepcopy(correct)
            wrong["verdict"] = "Counterfeit" if correct["verdict"] != "Counterfeit" else "Authentic"
            wrong["authentic"] = not correct["authentic"]
            self.assertEqual(scorer.score(product, json.dumps(wrong), self.rows)["score"], 0)

    def test_source_label_controls_not_our_own_recomputed_label(self):
        # Concrete expectations independently checked against the source's cells.
        self.assertEqual(self.rows["newton2008-st001-row49"]["hplc"], "48.55 artesunate")
        self.assertEqual(self.rows["newton2008-st001-row46"]["hplc"], "0 artesunate 14.76 artemisinin")
        self.assertEqual(self.rows["newton2008-st001-row17"]["batch_dates"], self.rows["newton2008-st001-row2"]["batch_dates"])
        self.assertEqual(self.rows["newton2008-st001-row17"]["source_label"], "Counterfeit")
        self.assertEqual(self.rows["newton2008-st001-row2"]["source_label"], "Authentic")

    def test_s18_cross_specimen_chemistry_is_inconclusive(self):
        product = scorer.product_for("newton2008-st001-row4")
        self.assertEqual(self.answer(product)["verdict"], "Inconclusive")
        self.assertFalse(self.answer(product)["authentic"])

    def test_low_price_absent_record_and_copied_serial_not_counterfeit(self):
        for product in ("Cheap Ozempic listing", "serial ABC123 absent from database", "Artesunate barcode 6920108100058", "Guilin batch 010401", "seller copied newton2008-st001-row49", scorer.product_for("newton2008-st001-row999")):
            with self.subTest(product=product):
                answer = self.answer(product)
                self.assertEqual(answer["verdict"], "Inconclusive")
                self.assertEqual(scorer.score(product, json.dumps(answer), self.rows)["score"], 10000)

    def test_exact_identity_echo_not_normalized_standin(self):
        product = scorer.product_for("newton2008-st001-row49")
        answer = self.answer(product)
        for echo in (" " + product, product + "\n", product.replace("Archived", "archived")):
            answer["product"] = echo
            self.assertEqual(scorer.score(product, json.dumps(answer), self.rows)["score"], 0)

    def test_literal_kit_render_bytes(self):
        # Extract only the primary source's pure render function. Do not import
        # its torch/model module or execute any other node in the source.
        tree = ast.parse((ROOT / "sources/kit-train.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render")
        self.assertTrue(all(isinstance(n, (ast.Expr, ast.Return)) for n in function.body))
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "kit-render-only", "exec"), namespace)
        prompt, answer = "\r\n X  ", '{"verdict":"Inconclusive"}\n'
        self.assertEqual(namespace["render"](prompt, answer).encode(), b'### Task:\n\r\n X  \n\n### Answer:\n{"verdict":"Inconclusive"}\n')
        self.assertNotEqual(namespace["render"](prompt.strip(), answer.strip()), namespace["render"](prompt, answer))

    def test_parser_retains_bold_identity_and_footnotes(self):
        parsed = parse_table('<table><tr><td><b>A</b><sup>1</sup></td><td>plain</td></tr></table>')
        self.assertEqual(parsed.rows[0]["cells"][0]["text"], "A")
        self.assertEqual(parsed.rows[0]["cells"][0]["bold_text"], "A")
        self.assertEqual(parsed.rows[0]["cells"][0]["footnotes"], "1")

    def test_overclaim_missing_scope_and_wrong_chemistry_fail(self):
        product = scorer.product_for("newton2008-st001-row49")
        for evidence in (["A new physical item is proven genuine"], [], ["HPLC record: 0 artesunate"]):
            answer = self.answer(product)
            answer["evidence"] = evidence
            self.assertEqual(scorer.score(product, json.dumps(answer), self.rows)["score"], 0)

    def test_strict_json_failure_controls(self):
        product = scorer.product_for("newton2008-st001-row49")
        for bad in ('{"verdict":"Authentic","verdict":"Counterfeit"}', '{"score":NaN}', "[]", "[" * 1100, "x" * 4097, "not json"):
            self.assertEqual(scorer.score(product, bad, self.rows)["score"], 0)
        answer = self.answer(product)
        answer["authentic"] = 1
        self.assertEqual(scorer.score(product, json.dumps(answer), self.rows)["score"], 0)

    def test_product_utf8_ceiling(self):
        scorer.expectation("é" * 128, self.rows)
        with self.assertRaises(ValueError):
            scorer.expectation("é" * 129, self.rows)

    def test_random_parser_bytes_do_not_escape(self):
        rng = random.Random(7301)
        product = scorer.product_for("newton2008-st001-row49")
        for _ in range(200):
            answer = bytes(rng.randrange(256) for _ in range(rng.randrange(512))).decode("utf-8", errors="replace")
            self.assertEqual(scorer.score(product, answer, self.rows)["score"], 0)

    def test_split_gate_rejects_time_and_specimen_leakage(self):
        rows = list(self.rows.values())
        with self.assertRaisesRegex(ValueError, "source:"):
            check_split({"train": rows[:5], "validation": rows[5:23], "audit": rows[23:]})
        self.assertEqual(identity(" ＧＵＩＬＩＮ\n"), identity("guilin"))

    def test_primary_cli_smoke_and_tamper(self):
        product = scorer.product_for("newton2008-st001-row49")
        answer = self.answer(product)
        for expected in (10000, 0):
            if expected == 0:
                answer["authentic"] = False
            process = subprocess.run([sys.executable, str(ROOT / "scorer.py")],
                input=json.dumps({"product": product, "answer": json.dumps(answer)}), text=True,
                capture_output=True, timeout=5, check=True)
            self.assertEqual(json.loads(process.stdout)["score"], expected)


if __name__ == "__main__":
    unittest.main()
