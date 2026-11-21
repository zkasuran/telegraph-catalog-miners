# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Byte-preserving recovery of an interrupted audit, never a model selection path."""
import hashlib
import json

EXPECTED = {
    'schema_version': 1,
    'family': 'KEYWORD_EXTRACTION',
    'original_run_id': 44838708435,
    'original_commit': '05b102a5cdaf911949f2df2b7af583df484ee905',
    'training_artifact_id': 14679982638,
    'training_archive_sha256': '66977b4246c6e3c042a211934366744edcf5aac7754954f025d9571e0657c175',
    'interrupted_audit_artifact_id': 14685357977,
    'interrupted_audit_archive_sha256': '90b80a68583527b43047aead0d94752592ba5e6edb17f7834eab820efc9c4839',
    'package_sha256': '0bb0556e8352772453149d97c669be95ba094a8c5c7de648ebd6ce01ef475e87',
    'initial_package_sha256': '7d5bdc7cbb8f5ae9dc8f34171e3a68568ce34066193f5cf505b5125e04e1051e',
    'frozen_baselines_sha256': '215f4f35cc4801a5aa1bbee76876267e1ed120e21b218092427f18d0a48c65d6',
    'foundation_spec_sha256': '5306d3dd09b7cb91c62bbfce0a6bdcc1c581646b85977fff3612406e4e30caf1',
    'config_sha256': 'f4ee4a7ac2ca14ed1b22e6381e37c8957619853becc8defd401a48f04cd7af4a',
    'data_manifest_sha256': '50b421457c6a5caf682bd57968bbfa888e7bb749702613b2ac9f24b24dad05b9',
    'failure': 'time ceiling',
    'failure_timeout_seconds': 6900,
    'candidate_changed': False,
    'audit_used_for_selection': False,
    'spdx_license_identifier': 'LicenseRef-zkasuran-SAND-1.0'
}


def validate_contract(value):
    if value != EXPECTED or any(type(value[key]) is not type(EXPECTED[key]) for key in EXPECTED):
        raise ValueError('recovery must preserve the exact interrupted audit contract')
    return value


def validate_candidate(candidate, data_report, contract, config_sha256):
    validate_contract(contract)
    if (candidate.get('family') != contract['family'] or candidate.get('commit') != contract['original_commit'] or
            candidate.get('run_id') != str(contract['original_run_id']) or candidate.get('data') != data_report or
            config_sha256 != contract['config_sha256']):
        raise ValueError('recovery candidate/source identity differs from the frozen training run')
    for key in ('package_sha256', 'initial_package_sha256', 'frozen_baselines_sha256',
                'foundation_spec_sha256', 'config_sha256'):
        if candidate.get(key) != contract[key]:
            raise ValueError('recovery candidate bytes changed: ' + key)
    raw = (json.dumps(data_report, indent=2, ensure_ascii=False) + '\n').encode()
    # The serialized manifest is checked separately using its original raw bytes.
    return {'original_training_run_id': contract['original_run_id'],
            'original_training_commit': contract['original_commit'],
            'contract_sha256': hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
            'candidate_changed': False, 'audit_used_for_selection': False,
            'data_semantics_sha256': hashlib.sha256(raw).hexdigest()}
