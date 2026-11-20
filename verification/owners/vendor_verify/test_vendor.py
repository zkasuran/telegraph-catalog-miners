# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Synthetic mechanics tests. None is counted as a real entity observation."""

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from collect import (
    Collector,
    MAX_BODY,
    NoRedirect,
    candidates,
    freeze,
    org_number,
    preference,
    projection,
    sha,
    verify_bindings,
)
from probe import assess, controls, lookup


def record(number="923609016", name="Synthetic control AS", value=True):
    entity = {
        "organisasjonsnummer": number,
        "navn": name,
        "organisasjonsform": {"kode": "AS"},
        "registrertIMvaregisteret": value,
        "registrertIForetaksregisteret": True,
        "erIKonsern": False,
        "historiskeNavn": [],
        "respons_klasse": "Enhet",
    }
    probe = {
        "key": "fake",
        "url": "https://data.brreg.no/enhetsregisteret/api/enheter/" + number,
        "retrieved_at": "2026-11-18T17:00:00Z",
        "sha256": sha(json.dumps(entity).encode()),
    }
    return entity, probe, projection(entity, probe, None)


class Tests(unittest.TestCase):
    def test_observation_not_legitimacy(self):
        for value in (False, True):
            row = record(value=value)[2]
            self.assertEqual(lookup(preference(row)), (True, False))

    def test_unknown_not_false(self):
        self.assertIsNone(assess(None, {"valid": False}))
        self.assertIsNone(
            assess({"org": "923609016", "date": "2026-11-18"}, {"valid": True})
        )

    def test_wrong_id_or_date_abstains(self):
        obs = {
            "org": "923609016",
            "date": "2026-11-18",
            "registrertIMvaregisteret": True,
        }
        self.assertIsNone(
            assess(obs, {"org": "995849364", "date": "2026-11-18", "valid": True})
        )
        self.assertIsNone(
            assess(obs, {"org": "923609016", "date": "2026-11-17", "valid": True})
        )

    def test_number_normalization_and_checksum(self):
        for text in ("923609016", "NO 923 609 016 MVA", "923.609.016"):
            self.assertEqual(org_number(text), "923609016")
        for text in (
            "923609017",
            "92360901",
            "NO 923609016 fake",
            "９２３６０９０１６",
        ):
            with self.assertRaises(ValueError):
                org_number(text)

    def test_non_string_candidate_ids_abstain_without_crashing(self):
        observation = {
            "org": "923609016",
            "date": "2026-11-18",
            "registrertIMvaregisteret": True,
        }
        for value in (None, False, 923609016, 3.14, [], {}):
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises(ValueError):
                    org_number(value)
                self.assertIsNone(
                    assess(
                        observation, {"org": value, "date": "2026-11-18", "valid": True}
                    )
                )

    def test_only_assessment_changes(self):
        pair = preference(record()[2])
        good, bad = json.loads(pair["chosen"]), json.loads(pair["rejected"])
        self.assertNotEqual(good.pop("valid"), bad.pop("valid"))
        self.assertEqual(good, bad)

    def test_schema_rejects_missing_and_coerced_boolean(self):
        entity, probe, _ = record()
        for value in (None, 0, "false"):
            with self.assertRaises(ValueError):
                projection({**entity, "registrertIMvaregisteret": value}, probe, None)

    def test_alias_and_related_groups_removed(self):
        a = record(name="CAFÉ AS")[2]
        b = record("995849364", "other AS", False)[2]
        b["historiskeNavn"] = [{"navn": " café  as "}]
        buckets, removed = candidates([a, b], set())
        self.assertEqual(sum(map(len, buckets.values())), 0)
        self.assertEqual(removed["shared_normalized_identity_excluded"], 2)

    def test_no_padding_when_insufficient(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                freeze([record()[2]], set(), Path(temp))
            self.assertFalse((Path(temp) / "FREEZE.json").exists())

    def test_raw_tamper_and_projection_tamper_fail(self):
        entity, probe, row = record()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "fake.body").write_bytes(json.dumps(entity).encode())
            self.assertEqual(verify_bindings([row], root, [probe]), 1)
            with self.assertRaises(ValueError):
                verify_bindings(
                    [{**row, "registrertIMvaregisteret": False}], root, [probe]
                )
            (root / "fake.body").write_bytes(b"{}")
            with self.assertRaises(ValueError):
                verify_bindings([row], root, [probe])

    def test_noncanonical_source_array_pointers_fail(self):
        entity, probe, _ = record()
        raw = json.dumps({"_embedded": {"enheter": [entity]}}).encode()
        probe = {**probe, "sha256": sha(raw)}
        row = projection(entity, probe, 0)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "fake.body").write_bytes(raw)
            self.assertEqual(verify_bindings([row], root, [probe]), 1)
            for pointer in (
                "/_embedded/enheter/-1",
                "/_embedded/enheter/00",
                "/wrong/0",
                None,
                "/_embedded/enheter/1",
                "/_embedded/enheter/999999999999",
            ):
                with self.subTest(pointer=pointer):
                    changed = {
                        **row,
                        "source_binding": {**row["source_binding"], "pointer": pointer},
                    }
                    with self.assertRaises(ValueError):
                        verify_bindings([changed], root, [probe])

    def test_real_record_intervention_mechanics(self):
        rows = [
            preference(record(value=True)[2]),
            preference(record("995849364", value=False)[2]),
        ]
        report = controls(rows)
        for key in (
            "missing_abstentions",
            "formatting_preserved",
            "misleading_display_name_ignored",
            "real_opposite_state_and_binding",
        ):
            self.assertEqual(report[key], 2)

    def test_redirect_cannot_escape_source(self):
        with self.assertRaises(ValueError):
            NoRedirect().redirect_request(
                None, None, 302, "", {}, "https://example.org"
            )

    def test_network_payload_ceiling_and_checkpoint(self):
        class Response:
            status = 200
            headers = {"Content-Type": "application/json"}

            def __init__(self, raw):
                import io

                self.body = io.BytesIO(raw)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read1(self, size):
                return self.body.read(size)

        class Opener:
            def __init__(self, raw):
                self.raw = raw

            def open(self, request, timeout):
                if timeout != 20:
                    raise AssertionError("request timeout changed")
                return Response(self.raw)

        with tempfile.TemporaryDirectory() as temp:
            collector = Collector(Path(temp))
            url = "https://data.brreg.no/enhetsregisteret/api/enheter/923609016"
            with patch(
                "collect.urllib.request.build_opener", return_value=Opener(b"{}")
            ):
                raw, probe = collector.fetch(url, "ok", "application/json")
                self.assertEqual(raw, b"{}")
                self.assertEqual(probe["sha256"], sha(b"{}"))
            with patch(
                "collect.urllib.request.build_opener",
                return_value=Opener(b"x" * (MAX_BODY + 1)),
            ):
                with self.assertRaises(ValueError):
                    collector.fetch(url, "overflow", "application/json")
            self.assertFalse((Path(temp) / "overflow.body").exists())
            self.assertEqual(len(collector.probes), 1)

    def test_source_object_and_number_types(self):
        entity, probe, _ = record()
        for bad in ([], None, {**entity, "organisasjonsnummer": 923609016}):
            with self.assertRaises(ValueError):
                projection(bad, probe, None)


if __name__ == "__main__":
    unittest.main()
