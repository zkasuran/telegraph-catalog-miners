# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Validate byte bindings, documented projection and independent Decimal arithmetic."""
import hashlib
import json
import re
from decimal import Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path
import jsonschema
import yaml

ROOT = Path(__file__).resolve().parent


def parse(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON member: ' + key)
            result[key] = value
        return result
    return json.loads(text, parse_float=Decimal, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def decimal_string(value, decimals):
    assert isinstance(value, str) and re.fullmatch(r'\d+' + (rf'\.\d{{{decimals}}}' if decimals else ''), value), 'invalid decimal string'
    result = Decimal(value)
    assert result.is_finite()
    return result


def independent_apy(rate, denominator):
    with localcontext() as ctx:
        ctx.prec = 90
        return (((Decimal(1) + Decimal(rate) / Decimal(denominator)) ** 31536000 - 1) * 10000).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)


def main():
    allow = json.loads((ROOT / 'PUBLICATION-ALLOWLIST.json').read_text())['files']
    assert len(set(allow)) == len(allow)
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(ROOT).parts)}
    assert actual == set(allow), f'Unexpected or missing publication files: {actual ^ set(allow)}'
    for file in json.loads((ROOT / 'MANIFEST.json').read_text())['files']:
        data = (ROOT / file['file']).read_bytes()
        assert len(data) == file['bytes'] and hashlib.sha256(data).hexdigest() == file['sha256'], file['file']
        assert bytes.fromhex('2f686f6d652f') not in data and bytes.fromhex('2e68712f') not in data, 'private path in ' + file['file']
        assert not re.search(rb'sk-[A-Za-z0-9]{20,}', data) and bytes.fromhex('50524956415445204b45592d2d2d2d2d') not in data, 'secret-shaped bytes in ' + file['file']
    baseline = json.loads((ROOT / 'BASELINE-HASHES.json').read_text())['files']
    assert len(baseline) == 66
    for record in baseline:
        assert hashlib.sha256((ROOT / record['file']).read_bytes()).hexdigest() == record['sha256']
    config = json.loads((ROOT / 'wrangler.json').read_text())
    before = json.loads((ROOT / 'baseline/wrangler.json').read_text())
    for key in ['name','compatibility_date','compatibility_flags','keep_vars','observability','send_metrics','no_bundle','find_additional_modules']:
        assert config[key] == before[key], key
    assert config['main'] == 'worker.mjs'
    assert not {'d1_databases','kv_namespaces','r2_buckets','vars','account_id'} & set(config)
    for rule in config['rules']:
        assert rule['fallthrough'] is False
        for glob in rule['globs']:
            assert '*' not in glob and (ROOT / glob).is_file(), glob
    schema = json.loads((ROOT / 'baseline/descriptor-schema.json').read_text())['json_schema']
    routes = json.loads((ROOT / 'CONTRACTS.json').read_text())['routes']
    assert {r['issue'] for r in routes} == {77, 136, 108, 168}
    for route in routes:
        text = (ROOT / 'truth' / (route['stem'] + '.yaml')).read_text()
        doc = yaml.safe_load(text)
        assert doc == json.loads((ROOT / 'truth' / (route['stem'] + '.json')).read_text())
        jsonschema.validate(doc, schema)
        endpoint = doc['endpoints'][0]
        assert doc['base_url'] == 'https://telegraph-s2-hashprice.margyn.workers.dev'
        assert endpoint['path'] == endpoint['external_path'] == route['path'] and endpoint['method'] == 'GET'
        assert {p['name']: p['type'] for p in endpoint['params']['query']['required']} == route['required']
        assert endpoint['intents'] == [route['intent']] and doc['auth'] == {'type':'none'}
        mapping = doc['intent_mapping'][route['intent']]
        assert mapping['endpoint'] == route['path']
        assert mapping['input'] == {p: {'from':p} for p in route['required']}
        assert mapping['output'] == {route['field']: {'path':route['field'], 'unit':route['unit']}}
        rows = (ROOT / 'truth' / (route['stem'] + '.jsonl')).read_text().rstrip('\n').split('\n')
        assert len(rows) == 1
        row = parse(rows[0])
        assert row['label_kind'] == 'truth_source' and row['intent_id'] == route['intent']
        assert row['source'] == doc or row['source'] == text
        assert not {'chosen','rejected','prompt','signature','public_key'} & set(row)
    vectors = json.loads((ROOT / 'sources/finance/arithmetic-vectors.json').read_text())
    assert len(vectors) == 400
    for vector in vectors:
        assert independent_apy(vector['rate'], vector['denominator']) == decimal_string(vector['expected'], 6)
    assert decimal_string('116144645100', 0) == Decimal('116144645100')
    for bad in ['1e3', 'NaN', 'Infinity', '1.0', 1, True]:
        try:
            decimal_string(bad, 0)
        except AssertionError:
            continue
        raise AssertionError('decimal rejection control passed')
    print(f'PASS {len(allow)} allowlisted files, 66 preserved baseline files, four descriptor projections and 400 independent Decimal APYs')
    print('Documented schema projection only. Native codec, loader and admission remain separate.')


if __name__ == '__main__':
    main()
