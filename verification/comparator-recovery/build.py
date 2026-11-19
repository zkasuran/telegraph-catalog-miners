#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Offline isolated source build and deterministic layered release packaging."""
import hashlib
import json
from pathlib import Path
import subprocess
from tools import Tools, arguments, clean_env

ROOT = Path(__file__).resolve().parent
RUST_PIN = '8bab26f4f68e0e26f0bb7960be334d5b520ea452'
KIT_PIN = '1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd'
VERSION = 'zkcmp-s2-20261118-1'
PREFIX = '/_zkcmp_s2_20261118/'
ORIGIN = 'https://telegraph-fin.margyn.workers.dev'
FAMILIES = ['CRYPTO_PRICE', 'CURRENCY_EXCHANGE', 'FX_NOW', 'SSL_VERIFICATION', 'STOCK_PRICE', 'WEATHER_CHECK']
ARCHIVES = {
    'libcore-01b7d99809a6084b.rlib': 'f65e247727c462608cd37864e9227977fd34ef1debc6cb91f2250fd9fba7a642',
    'libcompiler_builtins-0f081996ed7c8abe.rlib': '29ae493076037b12c6f4e10fb52ee1e836ef5cd0422f4c50867604af1c46b179',
}
NOTICES = {
    'rust-MIT.txt': 'b71bd43a069ca0641a9ecfe585ca7b3c53b5cc1608f8b68321168698e28b5ea1',
    'builtins-LICENSE.txt': 'ab6eec6caf0fa5775e411c7a8bc6a45c4ef2956b0980b157ab74fc5cd62a928b',
    'llvm-CREDITS.txt': 'a9901f47a089da41e4690682d00ce4cedaa2baf41fedbe79beee366d43ac2461',
    'specialized-LICENSE-MIT': 'd7e72b6086a884fcec0905aa957dc6c6c18aeccaab8a3ce5b91b91454f308d15',
    'libm-LICENSE.txt': '3823dda7cf046602f4b4e77ec8e227863dc4736037cc85bb33d9f19febe16bb7',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(relative, data, immutable=False):
    p = (ROOT / relative).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('write outside recovery root')
    p.parent.mkdir(parents=True, exist_ok=True)
    data = data.encode() if isinstance(data, str) else data
    if immutable and p.exists() and p.read_bytes() != data:
        raise ValueError('immutable artifact collision: ' + relative)
    p.write_bytes(data)


def record(relative, obj, immutable=False):
    write(relative, json.dumps(obj, indent=2, sort_keys=True) + '\n', immutable)


def env():
    return clean_env()


def run(label, argv, environment=None, cwd=ROOT, timeout=180):
    e = env() if environment is None else environment
    p = subprocess.run(argv, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=timeout, check=False)
    write('evidence/' + label + '.log', p.stdout)
    record('evidence/' + label + '.json', {'argv': argv, 'cwd': str(cwd), 'environment': e, 'exit_code': p.returncode})
    print(label, p.returncode, p.stdout.decode(errors='replace'), flush=True)
    if p.returncode:
        raise RuntimeError(label + ' failed')
    return p.stdout


def leb(n):
    out = bytearray()
    while n > 127:
        out.append((n & 127) | 128)
        n >>= 7
    out.append(n)
    return bytes(out)


def sections(data):
    if data[:8] != b'\x00asm\x01\x00\x00\x00':
        raise ValueError('invalid WASM header')
    def number(at):
        n = 0
        for shift in range(0, 35, 7):
            byte = data[at]
            at += 1
            n |= (byte & 127) << shift
            if byte < 128:
                return n, at
        raise ValueError('invalid LEB')
    at = 8
    result = []
    while at < len(data):
        kind = data[at]
        n, start = number(at + 1)
        end = start + n
        if end > len(data):
            raise ValueError('truncated WASM section')
        name, payload = None, data[start:end]
        if kind == 0:
            length, value = number(start)
            if value + length > end:
                raise ValueError('truncated custom name')
            name = data[value:value + length].decode()
            payload = data[value + length:end]
        result.append((kind, name, payload))
        at = end
    return result


def active_hashes(data):
    return [{'id': k, 'sha256': sha(p)} for k, _, p in sections(data) if k != 0]


def custom(name, obj):
    name = name.encode()
    payload = leb(len(name)) + name + json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return b'\x00' + leb(len(payload)) + payload


def prepare(tools):
    record('evidence/toolchain.json', tools.proof)
    version = run('rustc-version', [tools.executable('rustc'), '--version', '--verbose'], tools.env()).decode()
    if 'release: 1.97.1\n' not in version or 'commit-hash: ' + RUST_PIN not in version:
        raise ValueError('compiler pin mismatch')
    archive_proof = {}
    for name, expected in ARCHIVES.items():
        data = (tools.sysroot / 'lib/rustlib/wasm32-unknown-unknown/lib' / name).read_bytes()
        if sha(data) != expected:
            raise ValueError('linked archive pin mismatch')
        archive_proof[name] = {'sha256': expected, 'bytes': len(data)}
    originals = {}
    for name in ['Cargo.toml', 'Cargo.lock', 'src/lib.rs', 'src/tag.rs']:
        data = (ROOT / 'original-source' / name).read_bytes()
        originals[name] = sha(data)
    if originals['src/lib.rs'] != '647b79404f8ec05cf5a470495ae8583efd8150132b9e1a700fc3d682af9a23ed':
        raise ValueError('arithmetic source changed')
    locked_manifest = json.loads((ROOT / 'manifest.json').read_bytes())
    if originals != locked_manifest['original_source_sha256']:
        raise ValueError('bundled original source pin mismatch')
    original_artifacts = locked_manifest['original_artifacts']
    texts = {}
    sources = json.loads((ROOT / 'licenses/PINS.json').read_bytes())
    for name, expected in NOTICES.items():
        data = (ROOT / 'licenses' / name).read_bytes()
        if sha(data) != expected:
            raise ValueError('notice pin mismatch: ' + name)
        texts[name] = data.decode()
    mit = (ROOT / 'licenses/PRIOR-OWN-MIT.txt').read_bytes().decode()
    if mit != texts['rust-MIT.txt'].replace('Copyright (c) The Rust Project Contributors', 'Copyright (c) 2026 zkasuran'):
        raise ValueError('bundled prior MIT grant mismatch')
    sand = (ROOT / 'licenses/SAND.txt').read_bytes().decode()
    notices = {'rust_core': {'copyright': 'Copyright (c) The Rust Project Contributors',
                             'installed_attribution': 'The Rust Project Developers', 'license_text': texts['rust-MIT.txt']},
               'compiler_builtins': {'version': '0.1.160', 'license_text': texts['builtins-LICENSE.txt'],
                                    'credits_text': texts['llvm-CREDITS.txt'], 'credits_url': next(s['url'] for s in sources if s['name'] == 'llvm-CREDITS.txt')},
               'specialized_div_rem': {'version': '1.0.0', 'archive_sha256': '97c4ca84f4ea39a6194276aa8b3d50e440815ab9a1d88d96d4561abd629154b9',
                                      'copyright': 'Copyright © 2018 Aaron Kutch', 'license_text': texts['specialized-LICENSE-MIT']},
               'libm_integer_support': {'copyright': 'Copyright (c) 2018 Jorge Aparicio', 'license_text': texts['libm-LICENSE.txt']},
               'prior_own_mit': {'license_text': mit, 'source_sha256': originals}, 'source_pins': [s for s in sources if s['name'] in NOTICES]}
    notice_text = '\n\n'.join([f'{name}\n{value}' for name, value in texts.items()]) + '\n\nPrior own MIT arithmetic\n' + mit
    write('NOTICE', notice_text, True)
    scope = ('Layered comparator recovery licence, prepared 2026-11-18.\n'
             'Upstream portions retain the complete grants in NOTICE and licenses/.\n'
             'Unchanged arithmetic in src/lib.rs and original-source/ remains MIT.\n'
             'Prior obtained MIT copies and their rebuild rights remain unchanged.\n'
             'Only new family tags, packaging, planner, wrapper and campaign metadata use SAND.\n'
             'No exclusive SAND claim is made over the unchanged MIT executable program.\n'
             'This is a prospective release. Its publication commit is not yet assigned.\n'
             'The controller records the publication commit before publication.\n\n')
    write('LICENSE', scope + notice_text + '\n\nNew own additions\n' + sand, True)
    handler = (ROOT / 'worker/handler.mjs').read_bytes()
    if sha(handler) != 'b1a5e7b8792fdce3507b5529627949a307c62a48a14b041ef3706907376118f3':
        raise ValueError('handler source drift')
    original_env = tools.env()
    original_env['CARGO_TARGET_DIR'] = str(ROOT / 'scratch/target/original')
    run('build-original', [tools.executable('cargo'), 'build', '--manifest-path', str(ROOT / 'original-source/Cargo.toml'),
                          '--locked', '--offline', '--release', '--target', 'wasm32-unknown-unknown'], original_env)
    old_base = (ROOT / 'scratch/target/original/wasm32-unknown-unknown/release/zk_comparator.wasm').read_bytes()
    if sha(old_base) != 'bfab9c497f0834b275086521baa221ea55f6f761e1271c586842f3959b417c64':
        raise ValueError('historical base mismatch')
    manifest = {'spdx_license': 'LicenseRef-zkasuran-SAND-1.0', 'version': VERSION, 'prefix': PREFIX,
                'origin': ORIGIN, 'publication_state': 'prospective', 'publication_commit': None,
                'compiler': {'release': '1.97.1', 'commit': RUST_PIN, 'rustc_sha256': tools.proof['rustc']['sha256'],
                             'archives': archive_proof, 'flags': ['--locked', '--offline', '--release', '--target', 'wasm32-unknown-unknown', '-Dwarnings']},
                'kit_commit': KIT_PIN, 'original_source_sha256': originals, 'original_artifacts': original_artifacts,
                'handler_sha256': sha(handler), 'families': {}}
    for family in FAMILIES:
        source = 'source/' + family
        cargo = (ROOT / 'original-source/Cargo.toml').read_text().replace('license = "MIT"', 'license-file = "LICENSE"')
        write(source + '/Cargo.toml', cargo, True)
        write(source + '/Cargo.lock', (ROOT / 'original-source/Cargo.lock').read_bytes(), True)
        write(source + '/src/lib.rs', (ROOT / 'original-source/src/lib.rs').read_bytes(), True)
        tag = f'zkcmp-s2/{family}'
        padded_tag = tag.encode().ljust(33, b'\x00')
        rust_tag = tag + '\\0' * (33 - len(tag))
        write(source + '/src/tag.rs', '// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0\n// Copyright (c) 2026 zkasuran\n'
              f'#[link_section = "zk.family"]\n#[used]\nstatic ZK_FAMILY_TAG: [u8; 33] = *b"{rust_tag}";\n')
        write(source + '/LICENSE', (ROOT / 'LICENSE').read_bytes(), True)
        e = tools.env()
        e['CARGO_TARGET_DIR'] = str(ROOT / 'scratch/target' / family)
        run('build-' + family, [tools.executable('cargo'), 'build', '--manifest-path', str(ROOT / source / 'Cargo.toml'),
                              '--locked', '--offline', '--release', '--target', 'wasm32-unknown-unknown'], e)
        base = (ROOT / 'scratch/target' / family / 'wasm32-unknown-unknown/release/zk_comparator.wasm').read_bytes()
        old_sections = {k: p for k, _, p in sections(old_base) if k != 0}
        new_sections = {k: p for k, _, p in sections(base) if k != 0}
        old_tag = b'zk-comparator/1.1.0/WEATHER_CHECK'
        if old_sections[11].count(old_tag) != 1:
            raise ValueError('historical data tag is not unique')
        old_sections[11] = old_sections[11].replace(old_tag, padded_tag)
        if new_sections != old_sections:
            raise ValueError('non-tag active bytes changed: ' + family)
        tags = [p for k, n, p in sections(base) if k == 0 and n == 'zk.family']
        if tags != [padded_tag]:
            raise ValueError('family stamp mismatch')
        source_hashes = {n: sha((ROOT / source / n).read_bytes()) for n in ['Cargo.toml', 'Cargo.lock', 'src/lib.rs', 'src/tag.rs', 'LICENSE']}
        metadata = {'copyright': 'Copyright (c) 2026 zkasuran', 'family': family, 'version': VERSION,
                    'license_text': sand, 'license_scope': scope, 'publication_state': 'prospective',
                    'publication_commit': None, 'base_sha256': sha(base), 'source_sha256': source_hashes,
                    'original_source_sha256': originals}
        final = base + custom('zk.notices', notices) + custom('zk.license', metadata)
        keccak = run('keccak-' + family, [tools.executable('cast'), 'keccak', '0x' + final.hex()], tools.env()).decode().strip()
        if not __import__('re').fullmatch('0x[a-f0-9]{64}', keccak):
            raise ValueError('invalid Keccak result')
        filename = f'{family}-{keccak[2:]}.wasm'
        write('binaries/' + filename, final, True)
        manifest['families'][family] = {'file': 'binaries/' + filename, 'module_name': filename, 'bytes': len(final),
                                       'sha256': sha(final), 'keccak': keccak, 'base_sha256': sha(base),
                                       'source_sha256': source_hashes, 'active_sections': active_hashes(final),
                                       'binary_url': f'{ORIGIN}{PREFIX}binary/{family}/{keccak[2:]}.wasm',
                                       'runnable_url': f'{ORIGIN}{PREFIX}runnable/{family}'}
    if len({v['keccak'] for v in manifest['families'].values()}) != 6:
        raise ValueError('duplicate release hashes')
    record('manifest.json', manifest, True)
    record('PROVENANCE.json', {'spdx_license': 'LicenseRef-zkasuran-SAND-1.0', 'lineage': 'own_source_with_linked_toolchain_components',
                              'binary_wrap': 0, 'arithmetic_source_unchanged': True, 'code_sections_equal_historical_base': True,
                              'data_section_change': 'Only the fixed-width 33-byte family tag differs.',
                              'original_source_sha256': originals, 'compiler': manifest['compiler'],
                              'families': manifest['families'], 'publication_commit': None}, True)
    print('Six final recovery artifacts built. Arithmetic sections unchanged. Publication pending.')


if __name__ == '__main__':
    prepare(Tools(arguments('Build from bundled sources with pinned offline tools.')))
