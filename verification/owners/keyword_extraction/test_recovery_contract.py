# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Frozen identity and budget controls. No neural work or audit selection."""
import copy
import json
from pathlib import Path
import unittest

import audit_remote
from recovery_contract import EXPECTED, validate_candidate, validate_contract
from supervise import timeout_for


class RecoveryContractTests(unittest.TestCase):
    def test_actual_public_contract_parses_and_matches_frozen_identity(self):
        value = json.loads((Path(__file__).parent / 'audit-recovery-contract.json').read_text())
        self.assertEqual(validate_contract(value), EXPECTED)

    def test_any_model_or_data_contract_change_is_refused(self):
        for key in ('package_sha256', 'initial_package_sha256', 'config_sha256',
                    'data_manifest_sha256', 'original_commit'):
            value = copy.deepcopy(EXPECTED)
            value[key] = '0' * len(value[key])
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_contract(value)
        value = {**EXPECTED, 'candidate_changed': True}
        with self.assertRaises(ValueError):
            validate_contract(value)

    def test_wrong_original_candidate_or_data_refused(self):
        data = {'synthetic': True}
        candidate = {'family': EXPECTED['family'], 'commit': EXPECTED['original_commit'],
                     'run_id': str(EXPECTED['original_run_id']), 'data': data,
                     **{key: EXPECTED[key] for key in ('package_sha256', 'initial_package_sha256',
                         'frozen_baselines_sha256', 'foundation_spec_sha256', 'config_sha256')}}
        report = validate_candidate(candidate, data, EXPECTED, EXPECTED['config_sha256'])
        self.assertFalse(report['candidate_changed'])
        self.assertFalse(report['audit_used_for_selection'])
        for changed in ({**candidate, 'commit': '0' * 40}, {**candidate, 'package_sha256': '0' * 64},
                        {**candidate, 'data': {'synthetic': False}}):
            with self.assertRaises(ValueError):
                validate_candidate(changed, data, EXPECTED, EXPECTED['config_sha256'])

    def test_audit_budget_is_bounded_and_training_is_unchanged(self):
        self.assertEqual(timeout_for('train', [], {'timeout_seconds': 6900}), 6900)
        self.assertEqual(timeout_for('audit', ['python3', str(Path(audit_remote.__file__))], {}), 19800)
        self.assertLessEqual(19800, 350 * 60)
        with self.assertRaises(ValueError):
            timeout_for('audit', ['python3', 'arbitrary-model.py'], {})

    def test_no_boolean_or_unapproved_fields_are_accepted(self):
        for value in ({**EXPECTED, 'schema_version': True}, {**EXPECTED, 'unapproved': 'field'}):
            with self.assertRaises(ValueError):
                validate_contract(value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
