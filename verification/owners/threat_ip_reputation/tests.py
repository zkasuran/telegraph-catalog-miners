# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Standalone contract tests using licensed frozen source cards only."""
import ast
import datetime as dt
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import reference as ref

HERE = Path(__file__).resolve().parent


class PublicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = ref.selected_rules(HERE)

    def prompt(self, address="196.251.107.61", age=0):
        as_of = (ref.instant(ref.PINNED_CAPTURED_AT) + dt.timedelta(seconds=age)).isoformat()
        return ref.prompt_for(address, self.records, ref.PINNED_CAPTURED_AT, as_of)

    def answer(self, address="196.251.107.61", ids=None, state="rule_address_match"):
        return json.dumps({"ip": address, "rule_ids": [2066932, 2069054] if ids is None else ids,
                           "evidence_state": state, "current_abuse": "unknown", "basis": ref.BASIS})

    def test_exact_overlapping_source_predicates(self):
        self.assertEqual(ref.score(self.prompt(), self.answer(), self.records, ref.PINNED_CAPTURED_AT), 1.0)
        self.assertEqual(ref.score(self.prompt(), self.answer(ids=[2069054]), self.records, ref.PINNED_CAPTURED_AT), 0.0)

    def test_neighbor_and_cidr_boundaries(self):
        for address, ids in (("62.60.225.255", []), ("62.60.226.0", [2066440]),
                             ("62.60.226.255", [2066440]), ("62.60.227.0", []), ("178.16.53.8", [])):
            self.assertEqual(ref.score(self.prompt(address), self.answer(address, ids, "rule_address_match" if ids else "no_rule_address_match"),
                                       self.records, ref.PINNED_CAPTURED_AT), 1.0)

    def test_mapped_ipv6_and_unseen_ipv6(self):
        self.assertEqual(ref.score(self.prompt("::ffff:196.251.107.61"), self.answer("::ffff:c4fb:6b3d"),
                                   self.records, ref.PINNED_CAPTURED_AT), 1.0)
        self.assertEqual(ref.score(self.prompt("2001:db8::1"), self.answer("2001:db8::1", [], "no_rule_address_match"),
                                   self.records, ref.PINNED_CAPTURED_AT), 1.0)

    def test_current_abuse_and_packet_match_shortcuts_rejected(self):
        good = json.loads(self.answer())
        for bad in ({**good, "current_abuse": "malicious"}, {**good, "current_abuse": "benign"},
                    {**good, "basis": "observed_packet_match"}, {**good, "rule_ids": [True]}, {**good, "extra": 1}):
            self.assertEqual(ref.score(self.prompt(), json.dumps(bad), self.records, ref.PINNED_CAPTURED_AT), 0.0)

    def test_stale_and_future_snapshot_controls(self):
        for age, state in ((86399, "rule_address_match"), (86400, "stale_snapshot"), (86401, "stale_snapshot")):
            self.assertEqual(ref.score(self.prompt(age=age), self.answer(state=state), self.records, ref.PINNED_CAPTURED_AT), 1.0)
        with self.assertRaises(ValueError):
            ref.expected(self.prompt(age=-1), self.records, ref.PINNED_CAPTURED_AT)
        with self.assertRaises(ValueError):
            ref.expected(self.prompt(), self.records, "2026-11-20T11:52:04.418529+00:00")

    def test_complete_source_set_and_metadata_are_required(self):
        for records in (self.records[:-1], [{**self.records[0], "line": 1}] + self.records[1:]):
            with self.assertRaises(ValueError):
                ref.expected(ref.prompt_for("196.251.107.61", records, ref.PINNED_CAPTURED_AT), records, ref.PINNED_CAPTURED_AT)
        records = list(reversed(self.records))
        self.assertEqual(ref.expected(ref.prompt_for("196.251.107.61", records, ref.PINNED_CAPTURED_AT), records, ref.PINNED_CAPTURED_AT)["rule_ids"], [2066932, 2069054])

    def test_strict_parser_and_seeded_controls(self):
        for raw in ("null", "[]", "{", '{"x":1,"x":2}', '{"x":NaN}', "[" * 1500 + "0" + "]" * 1500,
                    " " * (ref.MAX_ANSWER + 1)):
            with self.assertRaises(ValueError):
                ref.decode(raw, ref.MAX_ANSWER)
        rng = random.Random(9927)
        for _ in range(500):
            raw = "{" + "".join(chr(rng.randrange(32, 127)) for _ in range(rng.randrange(1, 80)))
            try:
                self.assertIsInstance(ref.decode(raw, ref.MAX_ANSWER), dict)
            except ValueError:
                pass

    def test_literal_primary_render_bytes(self):
        tree = ast.parse((HERE / "primary-render.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "render")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "licensed-primary-render", "exec"), namespace)
        for prompt, answer in (("{  \"ip\": \"::ffff:1.2.3.4\" }\r\n", " A\n"), ("café\t", "### Answer:\n\u200b"), ("", "")):
            self.assertEqual(ref.render(prompt, answer).encode(), namespace["render"](prompt, answer).encode())

    def test_literal_cli_good_overclaim_duplicate_and_invalid(self):
        good = {"prompt": self.prompt(), "answer": self.answer()}
        overclaim = {**good, "answer": json.dumps({**json.loads(good["answer"]), "current_abuse": "malicious"})}
        cases = ((json.dumps(good), 0, 1.0), (json.dumps(good, indent=2), 0, 1.0),
                 (json.dumps(overclaim), 1, 0.0), ('{"prompt":"wrong",' + json.dumps(good)[1:], 2, None),
                 (json.dumps({**good, "extra": 1}), 2, None), ("null", 2, None), ("{", 2, None))
        for raw, exit_code, score in cases:
            with tempfile.TemporaryDirectory(dir=HERE) as tmp:
                path = Path(tmp) / "request.json"
                path.write_text(raw)
                run = subprocess.run([sys.executable, "-W", "error", str(HERE / "reference.py"), "--sources", str(HERE),
                                      "--request", str(path)], text=True, capture_output=True, timeout=10)
            self.assertEqual(run.returncode, exit_code)
            response = json.loads(run.stdout)
            self.assertFalse(response["live_qualified"])
            self.assertNotIn("Traceback", run.stderr)
            if score is None:
                self.assertEqual(response["error"], "invalid_request")
            else:
                self.assertEqual(response["score"], score)


if __name__ == "__main__":
    unittest.main()
