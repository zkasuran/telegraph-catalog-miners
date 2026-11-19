# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import ast
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from collect import Fetcher, parse_pdfsig, sitemap, variants
from scorer import FIELDS, compact, digest, make_pair, render, score, strict_json, validate_report
from check_data import checks, fit


class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.report = {"package": "BILLS-113hr12enr", "sha256": digest(b"test fixture, not a source PDF"),
                       "integrity": True, "whole_document": False, "trust": "unchecked"}
        self.row = make_pair(self.report, "integrity", self.report["sha256"], "train", "fixture")

    def test_primary_path_and_reversal(self):
        self.assertEqual(score(self.row["prompt"], self.row["chosen"])["score"], 1)
        self.assertEqual(score(self.row["prompt"], self.row["rejected"])["score"], 0)
        self.assertIsNone(score(self.row["prompt"], self.row["chosen"])["authentic"])

    def test_all_four_measured_states(self):
        for integrity in (False, True):
            for whole in (False, True):
                report = {**self.report, "integrity": integrity, "whole_document": whole}
                for field in FIELDS:
                    row = make_pair(report, field, "fixture", "train", "fixture")
                    self.assertEqual(score(row["prompt"], row["chosen"])["score"], 1)
                    self.assertEqual(score(row["prompt"], row["rejected"])["score"], 0)

    def balanced_splits(self):
        result = {}
        for split, session in (("train", "113"), ("validation", "117"), ("audit", "119")):
            rows = []
            group = digest(split.encode())
            for i, (a, b) in enumerate(((True, True), (True, False), (False, True), (False, False))):
                report = {**self.report, "package": f"BILLS-{session}hr12enr", "sha256": group if i == 0 else digest((split+str(i)).encode()), "integrity": a, "whole_document": b}
                for field in FIELDS:
                    rows.append(make_pair(report, field, group, split, "original" if i == 0 else str(i)))
            result[split] = rows
        return result

    def test_immutable_integer_baselines_and_overlap_failure(self):
        splits = self.balanced_splits()
        baseline = fit(splits["train"])
        self.assertEqual(baseline["answer_only_integer_coefficients"], {})
        result = checks(splits, baseline)
        self.assertEqual(result["normalized_cross_split_overlap"], 0)
        for split in result["splits"].values():
            for metric in split["shortcut_baselines"].values():
                self.assertEqual(metric["correct_twice"], metric["pairs"])
        splits["validation"][0]["group_id"] = splits["train"][0]["group_id"]
        with self.assertRaisesRegex(ValueError, "overlapping original"):
            checks(splits, baseline)

    def test_gate_rejects_original_overlap_shortcut(self):
        splits = self.balanced_splits()
        for rows in splits.values():
            for row in rows:
                task = json.loads(row["prompt"])
                del task["claim_values"]
                row["prompt"] = compact(task)
        with self.assertRaisesRegex(ValueError, "shortcut exceeds unchanged 65"):
            checks(splits, fit(splits["train"]))

    def test_report_swap_changes_score_not_answer_style(self):
        task = json.loads(self.row["prompt"])
        task["report"]["integrity"] = False
        self.assertEqual(score(compact(task), self.row["chosen"])["score"], 0)
        self.assertEqual(score(compact(task), self.row["rejected"])["score"], 1)

    def test_report_removal_cannot_self_certify(self):
        task = json.loads(self.row["prompt"])
        del task["report"]
        self.assertFalse(score(compact(task), self.row["chosen"])["supported"])

    def test_wrong_document_digest_and_claim_type(self):
        for key, value in (("sha256", "0"*64), ("value", "true"), ("field", "certificate_valid")):
            answer = json.loads(self.row["chosen"])
            answer["evidence"][0][key] = value
            self.assertFalse(score(self.row["prompt"], compact(answer))["supported"])
        answer = json.loads(self.row["chosen"])
        answer["document"] += " "
        self.assertFalse(score(self.row["prompt"], compact(answer))["supported"])

    def test_counterparty_url_is_literal_not_normalized(self):
        task = json.loads(self.row["prompt"])
        for replacement in (task["document"].replace("https:", "http:"), task["document"]+"?x=1", task["document"]+"\n", task["document"].replace("govinfo", "GOVINFO")):
            task["document"] = replacement
            self.assertFalse(score(compact(task), self.row["chosen"])["supported"])

    def test_unknown_boolean_not_forged(self):
        for verdict in ("Authentic", "Likely authentic", "Likely forged", "Forged"):
            answer = json.loads(self.row["chosen"])
            answer["verdict"] = verdict
            self.assertFalse(score(self.row["prompt"], compact(answer))["supported"])
        answer = json.loads(self.row["chosen"])
        answer["authentic"] = False
        self.assertFalse(score(self.row["prompt"], compact(answer))["supported"])

    def test_reports_cannot_claim_trust_or_numeric_booleans(self):
        for key, value in (("trust", "valid"), ("integrity", 1), ("sha256", "A"*64), ("package", "../../../secret")):
            report = {**self.report, key: value}
            with self.assertRaises(ValueError):
                validate_report(report)

    def test_strict_json_controls(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '[]', '{"x":Infinity}', "x"*8193):
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                strict_json(text)
        nested = '{"a":'+"["*2000+"0"+"]"*2000+"}"
        self.assertFalse(score(nested, self.row["chosen"])["supported"])

    def test_fuzz_bounded_failure(self):
        rng = random.Random(23)
        for _ in range(400):
            text = "".join(chr(rng.randrange(32, 127)) for _ in range(rng.randrange(256)))
            outcome = score(text, text)
            self.assertIn(outcome["score"], (0, 1))
            self.assertIsNone(outcome["authentic"])

    def test_pdfsig_parser_unknown_and_expired_controls(self):
        text = "Signature #1:\n  - Total document signed\n  - Signature Validation: Signature is Valid.\n  - Certificate Validation: Certificate has Expired\n"
        self.assertEqual(parse_pdfsig(text), {"integrity": True, "whole_document": True, "trust": "unchecked"})
        changed = text.replace("Signature is Valid.", "Digest Mismatch.").replace("Total document signed", "Not total document signed")
        self.assertEqual(parse_pdfsig(changed), {"integrity": False, "whole_document": False, "trust": "unchecked"})
        for bad in ("", text+text, text.replace("Signature is Valid.", "Unknown."), text.replace("Total document signed", "Unknown coverage")):
            with self.assertRaises(ValueError):
                parse_pdfsig(bad)

    def test_mutation_requires_signed_byte(self):
        raw = b"%PDF-1.7 /ByteRange [0 100 200 20]"
        views, pos = variants(raw)
        self.assertEqual(len(views["header_tamper"]), len(raw))
        self.assertNotEqual(views["header_tamper"][pos], raw[pos])
        with self.assertRaises(ValueError):
            variants(raw.replace(b"[0 100 200 20]", b"[0 1 200 20]"))

    def test_sitemap_refuses_entities_and_unknown_roots(self):
        with self.assertRaises(ValueError):
            sitemap(b'<!DOCTYPE x><urlset/>')
        with self.assertRaises(ValueError):
            sitemap(b'<unexpected/>')

    def test_source_allowlist_and_cached_mutation_fail(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix="test-source-") as directory:
            root = Path(directory)
            fetcher = Fetcher(root)
            for url in ("http://www.govinfo.gov/x", "https://evil.invalid/x", "https://devnode.telegraphprotocol.com/api/evalpool", "https://www.govinfo.gov/x?key=x"):
                with self.assertRaises(ValueError):
                    fetcher.get(url, "source", 1024)
            raw = b"original self-authored source fixture"
            (root/"source").write_bytes(raw+b"changed")
            fetcher.receipts = [{"path": "source", "url": "https://www.govinfo.gov/fixture", "sha256": digest(raw)}]
            with self.assertRaisesRegex(ValueError, "cached source hash changed"):
                fetcher.get("https://www.govinfo.gov/fixture", "source", 1024)

    def test_remote_driver_refuses_before_model_operations(self):
        result = subprocess.run([sys.executable, str(Path(__file__).parent/"remote_train.py")],
                                capture_output=True, text=True, timeout=10,
                                env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model operations require the controller", result.stderr)

    def test_exact_upstream_render_function(self):
        # Pure AST extraction only. No trainer import, tokenizer or model loading.
        path = Path(os.environ["DOCUMENT_KIT_SOURCE"])/"float_runtime/train_adapter.py"
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        for prompt, answer in (("a\r\nb", '{"authentic":false}'), ("\u00e9\n### Answer:", "\x00"), ("", "")):
            self.assertEqual(render(prompt, answer).encode(), namespace["render"](prompt, answer).encode())


if __name__ == "__main__":
    unittest.main()
