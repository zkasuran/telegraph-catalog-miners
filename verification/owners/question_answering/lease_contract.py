# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
import base64
import re
import struct
from datetime import datetime, timezone

WINDOW = 5 * 3600
COMMIT = re.compile(r'[0-9a-f]{40}')

def utc(value):
    if not isinstance(value, str):
        # Preserve the primary lease decoder's malformed-value contract.
        raise ValueError('timestamp must be a string')  # noqa: TRY004
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError('UTC timestamp required')
    return parsed

def now():
    return datetime.now(timezone.utc)

def public_key(value):
    if not isinstance(value, str) or not re.fullmatch('ssh-ed25519 [A-Za-z0-9+/]+={0,2}(?: [A-Za-z0-9_-]{1,80})?', value):
        raise ValueError('invalid lease public key')
    blob = base64.b64decode(value.split()[1], validate=True)
    if len(blob) != 51 or blob[:19] != struct.pack('>I', 11) + b'ssh-ed25519' + struct.pack('>I', 32):
        raise ValueError('invalid ed25519 public key wire shape')
    return ' '.join(value.split()[:2])

def lease_for(row, key, batch, run_id, commit, created=None):
    created = created or now()
    if not re.fullmatch('[A-Za-z0-9_-]{1,64}', batch) or not re.fullmatch('[1-9][0-9]{0,19}', run_id) or (not COMMIT.fullmatch(commit)):
        raise ValueError('invalid integration lease identity')
    key = public_key(key)
    expires = datetime.fromtimestamp(created.timestamp() + WINDOW, timezone.utc)
    port = row['forward_port']
    expiry = expires.strftime('%Y%m%d%H%M%SZ')
    restrictions = f'restrict,port-forwarding,permitlisten="127.0.0.1:{port}",permitopen="registration-denied.invalid:1",command="/bin/false",expiry-time="{expiry}"'
    return {'schema_version': 1, 'family': row['family'], 'owner': row['owner'], 'slot': row['slot'], 'run_id': run_id, 'commit': commit, 'batch': batch, 'public_key': key, 'authorized_keys_line': restrictions + ' ' + key, 'adapter_hash': row['keccak'], 'package_sha256': row['sha256'], 'forward_port': row['forward_port'], 'runnable_url': row['runnable_url'], 'existing_registration_id': row['existing_registration_id'], 'created_at': created.isoformat(), 'expires_at': expires.isoformat(), 'candidate_proof_sha256': row['candidate_proof']['sha256'], 'runtime_proof_sha256': row['runtime_proof']['sha256'], 'source_proof_sha256': row['source_proof']['sha256'], 'required_server_policy': {'AllowTcpForwarding': 'remote', 'AllowStreamLocalForwarding': 'no', 'PermitTunnel': 'no', 'GatewayPorts': 'no'}}

def validate_lease(lease, row, clock=None):
    clock = clock or now()
    created = utc(lease['created_at'])
    expected = lease_for(row, lease['public_key'], lease['batch'], lease['run_id'], lease['commit'], created)
    if lease != expected or created > clock or clock >= utc(lease['expires_at']):
        raise ValueError('lease identity, bounds or expiry mismatch')
    return created
