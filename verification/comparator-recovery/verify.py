#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
"""Mandatory final-byte reference, Worker and preservation release gate."""
import ast
import json
import re
import sys
from build import ROOT, KIT_PIN, FAMILIES, sha, sections, run, record, prepare
from tools import Tools, arguments, public_files, scan_public_paths

KIT = ROOT / 'validation'


def validate_bundle():
    inventory = json.loads((KIT / 'PINS.json').read_bytes())
    if inventory['kit_commit'] != KIT_PIN:
        raise ValueError('bundled kit revision mismatch')
    expected = set(inventory['files'])
    actual = {str(p.relative_to(ROOT)) for p in public_files() if p.is_relative_to(KIT) and p.name != 'PINS.json'}
    if actual != expected:
        raise ValueError('validation file inventory changed')
    for name, proof in inventory['files'].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(KIT) or sha(path.read_bytes()) != proof['sha256']:
            raise ValueError('bundled validation input drift: ' + name)
    return inventory


def main():
    record('evidence/RELEASE-GATE.json', {'status': 'RUNNING', 'controller_rollout_gates': 'pending'})
    record('evidence/public-path-scan.json', scan_public_paths())
    before = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in public_files()}
    if sys.version_info < (3, 11):
        raise ValueError('Python 3.11 or newer is required')
    tools = Tools(arguments('Reproduce the bundled six-family offline release gate.'))
    inventory = validate_bundle()
    run('python-integrity-controls', [sys.executable, '-W', 'error', '-m', 'unittest', '-v', 'test_tools'], tools.env())
    prepare(tools)
    run('python-lint', [tools.executable('ruff'), 'check', '--no-cache',
        str(ROOT / 'build.py'), str(ROOT / 'verify.py'), str(ROOT / 'tools.py'), str(ROOT / 'test_tools.py')], tools.env())
    run('bash-syntax', ['bash', '-n', str(ROOT / 'verify.sh')], tools.env())
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    pins = {'pkg/scoring/comparator/reference.go': '26e9e139e690398ee751361a5bfde7c48f6d4465f83f8f6a488c2743511af017',
            'pkg/scoring/comparator/zkcomparator_test.go': '0d382b42c0c0881268defee113bdd1c27ec6969c0512a9c60486d49a165f14ed',
            'pkg/wasm/runtime/deterministic.go': '905d7856a10c8bd101e664936c01fc6445fc3a23fffdc27ec321bafb9d18d12f',
            'pkg/wasm/remote/remote.go': '9d8939b86181449d2c92a5c7f1a720ca34aee27b5ae491c84246c4e2eb2c871f'}
    for path, expected in pins.items():
        if sha((KIT / path).read_bytes()) != expected:
            raise ValueError('kit source drift: ' + path)
    record('evidence/kit-pins.json', {'commit': KIT_PIN, 'source_sha256': pins,
                                   'bundled_validation_files': len(inventory['files'])})
    results = []
    for family in FAMILIES:
        row = manifest['families'][family]
        wasm = ROOT / row['file']
        data = wasm.read_bytes()
        if sha(data) != row['sha256'] or len(data) != row['bytes']:
            raise ValueError('final artifact drift')
        customs = [(n, p) for k, n, p in sections(data) if k == 0]
        if [n for n, _ in customs] != ['zk.family', 'zk.notices', 'zk.license']:
            raise ValueError('notice order or unexpected custom sections')
        if customs[0][1].rstrip(b'\x00').decode() != 'zkcmp-s2/' + family:
            raise ValueError('incorrect family stamp')
        notices = json.loads(customs[1][1])
        license_record = json.loads(customs[2][1])
        for component, filename in [('rust_core', 'rust-MIT.txt'), ('compiler_builtins', 'builtins-LICENSE.txt'),
                                    ('specialized_div_rem', 'specialized-LICENSE-MIT'), ('libm_integer_support', 'libm-LICENSE.txt')]:
            if notices[component]['license_text'] != (ROOT / 'licenses' / filename).read_bytes().decode():
                raise ValueError('full linked notice mismatch')
        if notices['compiler_builtins']['credits_text'] != (ROOT / 'licenses/llvm-CREDITS.txt').read_bytes().decode():
            raise ValueError('credits mismatch')
        if license_record['license_text'] != (ROOT / 'licenses/SAND.txt').read_text() or license_record['family'] != family:
            raise ValueError('own license mismatch')
        for path, expected in row['source_sha256'].items():
            if sha((ROOT / 'source' / family / path).read_bytes()) != expected:
                raise ValueError('family source drift')
        e = tools.env()
        e['ZK_COMPARATOR_WASM'] = str(wasm)
        output = run('gate-' + family, [tools.executable('go'), '-C', str(KIT), 'test', './pkg/scoring/comparator',
            '-run', '^TestZKComparator$', '-v', '-count=1', '-timeout=180s'], e)
        if b'537592 cases, 0 mismatches (24 overflow inputs refused by both)' not in output or b'--- PASS: TestZKComparator' not in output or b'SKIP' in output:
            raise ValueError('missing non-skipped real reference gate')
        output = run('independent-' + family, [tools.executable('go'), '-C', str(KIT), 'test',
            './pkg/scoring/comparator', '-run', '^TestIndependent', '-v', '-count=1', '-timeout=180s'], e)
        if b'SKIP' in output or output.count(b'--- PASS: TestIndependent') != 3 or b'raw coverage: 27769 cases' not in output:
            raise ValueError('independent boundary/caller suite incomplete')
        results.append({'family': family, 'sha256': sha(data), 'reference_cases': 537592, 'mismatches': 0,
                        'reference_overflow_refusals': 24, 'independent_raw_cases': 27769, 'skips': 0})
        record('evidence/final-byte-gates.json', results)
    for path in ['planner.mjs', 'worker/handler.mjs', 'worker/runtime.mjs', 'test_wire.mjs', 'test_planner.mjs', 'test_resources.mjs', 'registrar_binding.mjs', 'test_registrar.mjs']:
        run('syntax-' + path.replace('/', '_'), [tools.executable('node'), '--check', str(ROOT / path)], tools.env())
    for name in ['test_wire', 'test_planner', 'test_resources', 'test_registrar']:
        output = run(name, [tools.executable('node'), str(ROOT / (name + '.mjs'))], tools.env(), timeout=45)
        record('evidence/' + name + '-results.json', json.loads(output.decode().splitlines()[-1]))
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text())
    for path in ROOT.glob('*.md'):
        if re.search(r'[\u2013\u2014]|,\s+(?:and|or)\b|\b(?:robust|seamless|leverage|delve)\b|it\x27s worth noting|in conclusion', path.read_text(), re.I):
            raise ValueError('voice gate failed: ' + str(path))
    after = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in public_files()}
    if before != after:
        raise ValueError('public bundled inputs changed during reproduction')
    scan = scan_public_paths()
    record('evidence/public-path-scan.json', scan)
    record('evidence/RELEASE-GATE.json', {'status': 'PASS', 'exact_final_binaries': 6, 'reference_cases': 3225552,
        'independent_raw_cases': 166614, 'skips': 0, 'bundled_inputs_unchanged': True,
        'public_files_scanned': scan['files_scanned'], 'private_path_hits': 0, 'external_source_inputs': 0,
        'empty_isolated_module_cache': True, 'controller_rollout_gates': 'pending'})
    print('ALL GREEN: local six-family prospective release. Authenticated rollout gates pending.')


if __name__ == '__main__':
    main()
