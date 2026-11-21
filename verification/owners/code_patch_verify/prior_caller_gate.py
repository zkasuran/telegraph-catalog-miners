# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Historical structural gate retained only as a regression control."""
import hashlib
import json
import re

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def caller_contract(path):
    if not path.is_file() or path.stat().st_size > 20000:
        raise ValueError('missing independently verified literal caller contract')
    record = json.loads(path.read_text())
    for key, size in (('caller_source_sha256', 64), ('caller_revision', 40), ('sample_sha256', 64)):
        if not isinstance(record.get(key), str) or not re.fullmatch('[0-9a-f]{' + str(size) + '}', record[key]):
            raise ValueError('missing caller source/sample binding')
    if record.get('family') != 'CODE_PATCH_VERIFY' or record.get('class') != 'judgment' or record.get('trusted_ci_observation_in_literal_prompt') is not True or (record.get('protected_rows_used') is not False) or (record.get('sample_origin') != 'independently_constructed_canonical_request'):
        raise ValueError('unverified or protected caller evidence')
    if digest(record.get('literal_sample_prompt', '').encode()) != record['sample_sha256']:
        raise ValueError('literal caller sample hash mismatch')
    sample = json.loads(record['literal_sample_prompt'])
    if not isinstance(sample.get('ci_observation'), dict) or not isinstance(sample.get('patch'), str):
        raise ValueError('literal caller does not expose fixture evidence fields')
    return record
