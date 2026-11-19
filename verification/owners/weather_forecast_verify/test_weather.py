# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import ast
from decimal import Decimal
import json
from pathlib import Path
import random
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from build import SPLIT_POLICY, load_archive, verify_splits
from core import ArchiveScorer, Observation, answer_value, identity, inputs_question, iso_day, parse_dly, render

HERE = Path(__file__).resolve().parent
STATION = "USW00094728"


def record(element="TMAX", value=123, measurement=" ", qc=" ", source="1", year=2025, month=1):
    import calendar
    day_count = calendar.monthrange(year, month)[1]
    cell = f"{value:5d}{measurement}{qc}{source}"
    return (STATION + f"{year:04d}{month:02d}" + element + cell * day_count
            + "-9999   " * (31 - day_count) + "\n").encode()


class ParserTests(unittest.TestCase):
    def test_exact_tenths_and_negative_temperature(self):
        rows = parse_dly(record(value=-123), STATION)
        self.assertEqual(rows[(STATION, "2025-01-01", "TMAX")].value, Decimal("-12.3"))

    def test_missing_is_not_zero(self):
        self.assertEqual(parse_dly(record(value=-9999), STATION), {})

    def test_every_other_source_is_excluded(self):
        for source in "0WDXZACSRabcdefghijklmnopqrstuvwxyz":
            self.assertEqual(parse_dly(record(source=source), STATION), {})

    def test_each_quality_failure_is_excluded(self):
        for qc in "DGIKLMNORSTWXZ":
            self.assertEqual(parse_dly(record(qc=qc), STATION), {})

    def test_presumed_zero_and_hourly_missing_are_excluded(self):
        for flag in "PHI":
            self.assertEqual(parse_dly(record(element="PRCP", value=0, measurement=flag), STATION), {})

    def test_trace_is_distinct_from_measured_zero(self):
        trace = parse_dly(record(element="PRCP", value=0, measurement="T"), STATION)
        self.assertEqual(trace[(STATION, "2025-01-01", "PRCP")].measurement, "T")
        with self.assertRaisesRegex(ValueError, "trace"):
            parse_dly(record(element="PRCP", value=1, measurement="T"), STATION)

    def test_station_mismatch_and_duplicate_record_refused(self):
        with self.assertRaises(ValueError):
            parse_dly(record(), "USW00012839")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_dly(record() + record(), STATION)

    def test_leap_day_and_nonexistent_day(self):
        self.assertEqual(len(parse_dly(record(year=2024, month=2), STATION)), 29)
        invalid = record(year=2025, month=2).replace(b"-9999   ", b"    01  ", 1)
        with self.assertRaises(ValueError):
            parse_dly(invalid, STATION)

    def test_inconsistent_high_low_refused(self):
        with self.assertRaisesRegex(ValueError, "high lower"):
            parse_dly(record(value=100) + record(element="TMIN", value=200), STATION)

    def test_ranges_and_source_ceiling(self):
        with self.assertRaises(ValueError):
            parse_dly(record(value=601), STATION)
        with self.assertRaises(ValueError):
            parse_dly(b"a" * ((16 << 20) + 1), STATION)

    def test_random_parser_inputs_do_not_escape_as_crashes(self):
        rng = random.Random(91)
        for _ in range(200):
            raw = rng.randbytes(rng.randrange(1000))
            try:
                self.assertEqual(parse_dly(raw, STATION), {})
            except (ValueError, UnicodeError):
                pass


class AnswerTests(unittest.TestCase):
    def test_exact_six_decimal_and_celsius(self):
        for text in ('12.300000', '12.3 C', '12.3 degrees Celsius', '{"temp_max_c":12.3}'):
            self.assertEqual(answer_value(text, "NY CITY CNTRL PARK", "2025-01-01"), Decimal("12.3"))

    def test_fahrenheit_and_arbitrary_number_mining_refused(self):
        for text in ('54 F', 'high 12.3 C', '2025-01-01 12.3', 'NaN', 'Infinity', '12,3', '1e309'):
            with self.assertRaises(ValueError):
                answer_value(text, "NY CITY CNTRL PARK", "2025-01-01")

    def test_wrong_location_day_and_injected_fields_refused(self):
        for text in ('{"location":"Miami","temp_max_c":12.3}',
                     '{"date":"2025-01-02","temp_max_c":12.3}',
                     '{"temp_max_c":12.3,"ignore_instructions":true}'):
            with self.assertRaises(ValueError):
                answer_value(text, "NY CITY CNTRL PARK", "2025-01-01")

    def test_duplicates_nonfinite_string_number_bool_refused(self):
        for text in ('{"temp_max_c":12.3,"temp_max_c":14.3}', '{"temp_max_c":NaN}',
                     '{"temp_max_c":"12.3"}', '{"temp_max_c":true}', '{"temp_max_c":null}',
                     '{"temp_max_c":60.1}', '12.3000001'):
            with self.assertRaises(ValueError):
                answer_value(text, "NY CITY CNTRL PARK", "2025-01-01")

    def test_byte_and_nesting_ceiling(self):
        with self.assertRaises(ValueError):
            answer_value("a" * 4097, "X", "2025-01-01")
        with self.assertRaises(ValueError):
            answer_value('{"temp_max_c":' + '[' * 1800 + '0' + ']' * 1800 + '}', "X", "2025-01-01")

    def test_decimal_error_order_property(self):
        scorer = ArchiveScorer({(STATION, "2025-01-01", "TMAX"): Observation(STATION, "2025-01-01", "TMAX", 123, " ", 1)},
                               {STATION: "NY CITY CNTRL PARK"})
        for offset in range(-100, 101):
            response = scorer.score(STATION, "2025-01-01", str(Decimal(123 + offset) / 10))
            self.assertEqual(Decimal(response["absolute_error_c"]), abs(Decimal(offset) / 10))

    def test_unsupported_stays_unknown_and_city_alias_not_invented(self):
        scorer = ArchiveScorer({}, {STATION: "NY CITY CNTRL PARK"})
        for location in (STATION, "New York", "Miami", "Ｎｅｗ York"):
            result = scorer.score(location, "2025-01-01", "0")
            self.assertEqual(result["status"], "unsupported")
            self.assertIsNone(result["score"])

    def test_date_is_not_utc_instant_or_forecast(self):
        for text in ('2025-1-1', '2025-01-01T00:00:00Z', 'tomorrow', '2025-02-29'):
            with self.assertRaises(ValueError):
                iso_day(text)


class SourceAndWireTests(unittest.TestCase):
    def test_literal_question_matches_primary_expression(self):
        source = (HERE / "sources/onchain.go").read_text()
        self.assertIn('fmt.Fprintf(&b, "%s request.", intentID)', source)
        self.assertIn('fmt.Fprintf(&b, " %s (%s): %s.", input.Name, input.Description, val)', source)
        self.assertEqual(inputs_question("  MIAMI\nINTL AP", "2025-09-01"),
                         "WEATHER_FORECAST_VERIFY_GENERAL request. location (City or place name):   MIAMI\nINTL AP. date (Past date to verify): 2025-09-01.")

    def test_literal_render_matches_primary_function(self):
        # Execute only the isolated pure function. No import or model code.
        tree = ast.parse((HERE / "sources/train_adapter.py").read_bytes())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render")
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "primary-pure-render", "exec"), namespace)
        for prompt, answer in (("a", "b"), ("line\n\"quote\"", " 12.3\n"), ("Ｃity", "-2 C")):
            self.assertEqual(render(prompt, answer), namespace["render"](prompt, answer))
        self.assertNotEqual(render("a", "b"), 'Question: a\nAnswer: b')

    def test_real_archive_primary_smoke(self):
        observations, names = load_archive()
        row = observations[("USW00012839", "2025-09-01", "TMAX")]
        scorer = ArchiveScorer(observations, names)
        good = scorer.score(names[row.station], row.day, str(row.value))
        wrong = scorer.score(names[row.station], row.day, str(row.value + 2))
        self.assertEqual(Decimal(good["absolute_error_c"]), Decimal("0"))
        self.assertEqual(Decimal(wrong["absolute_error_c"]), Decimal("2"))

    def test_frozen_splits_overlap_and_mutation_control(self):
        root = HERE / "data"
        splits = {split: [json.loads(line) for line in (root / (split + ".jsonl")).read_text().splitlines()]
                  for split in SPLIT_POLICY}
        verify_splits(splits)
        mutated = {**splits, "validation": [splits["train"][0]] + splits["validation"][1:]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            verify_splits(mutated)

    def test_normalized_identity_and_ambiguous_alias_control(self):
        self.assertEqual(identity("  ＭＩＡＭＩ  INTL AP "), identity("miami intl ap"))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            ArchiveScorer({}, {"one": "same", "two": "ＳＡＭＥ"})


class RemoteFailureControls(unittest.TestCase):
    def test_literal_bare_hash_not_normalized(self):
        from protocol import bare_hash, check_reply, input_root
        hashes = {"foundation_hash": "a" * 64, "adapter_hash": "b" * 64, "tokenizer_hash": "c" * 64}
        batch = [[50281, 123, 50282]]
        response = {**hashes, "input_root": input_root(batch), "results": [{"score": 0x3f800000}]}
        self.assertEqual(check_reply(response, batch, hashes), [0x3f800000])
        for field in hashes:
            with self.assertRaises(ValueError):
                check_reply({**response, field: "0x" + response[field]}, batch, hashes)
        with self.assertRaises(ValueError):
            bare_hash("0x" + "a" * 64)

    def test_input_root_contract_and_negative_root(self):
        from protocol import check_reply, input_root
        self.assertEqual(input_root([]), "")
        hashes = {"foundation_hash": "a" * 64, "adapter_hash": "b" * 64, "tokenizer_hash": "c" * 64}
        with self.assertRaises(ValueError):
            check_reply({**hashes, "input_root": "wrong", "results": [{"score": 0}]}, [[1]], hashes)
        self.assertNotEqual(input_root([[1], [2]]), input_root([[2], [1]]))

    def test_ready_cannot_be_empty_or_wrong_identity(self):
        from ready_contract import validate_ready
        with self.assertRaises(ValueError):
            validate_ready({"schema_version": 1, "status": "REGISTRATION_READY"})

    def test_local_model_guard_fails_before_model_import(self):
        from remote import remote_only
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "bounded public CPU"):
                remote_only()

    def test_stamp_retains_descriptor_and_payload(self):
        from remote import stamp_safetensors
        header = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        encoded = json.dumps(header).encode()
        payload = struct.pack("<f", 1.25)
        old = struct.pack("<Q", len(encoded)) + encoded + payload
        stamped = stamp_safetensors(old, {"kit_mit": "unchanged", "sand": "terms"})
        size = struct.unpack("<Q", stamped[:8])[0]
        parsed = json.loads(stamped[8:8 + size])
        self.assertEqual(parsed["weight"], header["weight"])
        self.assertEqual(stamped[8 + size:], payload)
        self.assertEqual(parsed["__metadata__"]["x-terms-kit_mit"], "unchanged")
        with self.assertRaises(ValueError):
            stamp_safetensors(b"bad", {})

    def test_reward_head_zip_comment_preserves_contents(self):
        from remote import preserve_checkpoint
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with zipfile.ZipFile(path / "reward_head.pt", "w") as archive:
                archive.writestr("data.pkl", b"not a neural object, inert byte fixture")
            preserve_checkpoint(path, {"sand": "terms", "kit_mit": "grant"})
            with zipfile.ZipFile(path / "reward_head.pt") as archive:
                self.assertEqual(archive.read("data.pkl"), b"not a neural object, inert byte fixture")
                self.assertEqual(json.loads(archive.comment), {"sand": "terms", "kit_mit": "grant"})

    def test_fp32_wire_bad_bits_are_rejected(self):
        from audit_remote import decode_bits
        for value in (True, -1, 0x100000000, 0x7f800000, 0x7fc00000, "123"):
            with self.assertRaises(ValueError):
                decode_bits(value)
        self.assertEqual(decode_bits(0x3f800000), 1.0)

    def test_quality_cannot_pass_empty_nonfinite_or_ties(self):
        from audit_remote import quality
        self.assertFalse(quality({}, {}, {})["passed"])
        good = {name: {"accuracy": 0.9, "subsets": {"positive_error": 0.9, "negative_error": 0.9, "negative_c": 0.9}}
                for name in ("json", "celsius", "bare")}
        initial = {name: {"accuracy": 0.5} for name in good}
        self.assertTrue(quality(good, initial, {"length": 0.5})["passed"])
        bad = {**good, "bare": {**good["bare"], "accuracy": float("nan")}}
        self.assertFalse(quality(bad, initial, {"length": 0.5})["passed"])
        self.assertFalse(quality(good, initial, {"length": 0.9})["passed"])

    def test_workflow_refuses_permissions_and_gpu_mutations(self):
        from static_check import validate_workflow
        workflow = json.loads((HERE / "workflow.json").read_text())
        self.assertTrue(validate_workflow(workflow))
        with self.assertRaises(ValueError):
            validate_workflow({**workflow, "permissions": {"contents": "write"}})
        modified = json.loads(json.dumps(workflow))
        modified["jobs"]["train"]["runs-on"] = "self-hosted"
        with self.assertRaises(ValueError):
            validate_workflow(modified)


if __name__ == "__main__":
    unittest.main()
