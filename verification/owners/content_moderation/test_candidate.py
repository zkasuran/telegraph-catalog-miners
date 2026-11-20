# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Real immutable artifact-byte controls, not neural operations or new audit data."""
import copy
import json
import unittest

from candidate_contract import validate_bytes
from contracts import FAMILY, HERE


class CandidateTests(unittest.TestCase):
    def inputs(self):
        root = HERE / "candidate"
        return ((root / (FAMILY + ".safetensors")).read_bytes(), json.loads((root / "registration-ready.json").read_bytes()),
                json.loads((root / (FAMILY + ".safetensors.parser.json")).read_bytes()), (root / "source-manifest.json").read_bytes(),
                json.loads((HERE / "support/training/licences.json").read_bytes()))

    def test_actual_primary_release_byte_contract(self):
        values = self.inputs()
        self.assertEqual(len(validate_bytes(*values)), 8)

    def test_real_package_mutation_and_source_replacement_refuse(self):
        package, report, parser, source, licences = self.inputs()
        with self.assertRaises(ValueError):
            validate_bytes(package[:-1] + bytes([package[-1] ^ 1]), report, parser, source, licences)
        with self.assertRaises(ValueError):
            validate_bytes(package, report, parser, source + b" ", licences)

    def test_actual_ready_report_identity_and_quality_mutations_refuse(self):
        package, report, parser, source, licences = self.inputs()
        for key, value in (("package_keccak", "0x" + report["package_keccak"]), ("package_sha256", "0" * 64), ("engine_sha256", "0" * 64)):
            changed = copy.deepcopy(report)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_bytes(package, changed, parser, source, licences)
        changed = copy.deepcopy(report)
        changed["quality"]["formats"]["bare_label"]["per_label"]["Flagged"] = 0.69
        with self.assertRaises(ValueError):
            validate_bytes(package, changed, parser, source, licences)


if __name__ == "__main__":
    unittest.main()
