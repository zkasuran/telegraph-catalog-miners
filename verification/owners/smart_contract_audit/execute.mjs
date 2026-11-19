// SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
// Offline interpreter only. No runTx, RPC, private key, signer or wallet.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import solc from 'solc';
import { createVM } from '@ethereumjs/vm';
import { Account, createAddressFromString, hexToBytes, bytesToHex } from '@ethereumjs/util';
import { Common, Hardfork, Mainnet } from '@ethereumjs/common';
import { keccak256 } from 'ethereum-cryptography/keccak.js';

const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const inventory = process.argv[2];
const output = process.argv[3];
if (!inventory || !output) throw new Error('usage: execute.mjs INVENTORY REPORT');
const raw = fs.readFileSync(inventory);
if (raw.length > 2 ** 21) throw new Error('inventory ceiling');
const rows = JSON.parse(raw);
if (!Array.isArray(rows) || rows.length === 0 || rows.length > 256) throw new Error('row ceiling');
assert.match(solc.version(), /^0\.8\.30\+/);
const report = {schema_version: 1, inventory_sha256: hash(raw), compiler: solc.version(),
  runtime: 'ethereumjs-vm-10.1.0 runCode only', hardfork: 'cancun', gas_limit: '1000000',
  node_version: process.version, run_id: process.env.GITHUB_RUN_ID ?? null,
  commit: process.env.GITHUB_SHA ?? null, rows: [], outcome: 'incomplete'};
const lockBytes = fs.readFileSync(new URL('./package-lock.json', import.meta.url));
const lock = JSON.parse(lockBytes);
report.package_lock_sha256 = hash(lockBytes);
report.dependency_versions = Object.fromEntries(Object.entries(lock.packages)
  .filter(([name]) => name !== '').map(([name, info]) => [name, info.version]));

function calldata(signature, args) {
  const selector = bytesToHex(keccak256(new TextEncoder().encode(signature))).slice(2, 10);
  return hexToBytes('0x' + selector + args.map(x => BigInt(x).toString(16).padStart(64, '0')).join(''));
}
function integer(value) {
  const text = bytesToHex(value);
  if (value.length !== 32) throw new Error('expected uint ABI word, got ' + text);
  return BigInt(text).toString();
}
async function verify(row) {
  assert.equal(hash(row.source), row.source_sha256, 'source identity');
  if (Buffer.byteLength(row.source) > 2048) throw new Error('source ceiling');
  const settings = {optimizer: {enabled: false}, evmVersion: 'cancun',
    outputSelection: {'*': {'*': ['evm.deployedBytecode.object', 'abi']}}};
  const result = JSON.parse(solc.compile(JSON.stringify({language: 'Solidity',
    sources: {'Fixture.sol': {content: row.source}}, settings})));
  assert.equal((result.errors ?? []).length, 0, JSON.stringify(result.errors));
  const compiled = result.contracts['Fixture.sol'][row.facts.contract];
  const code = hexToBytes('0x' + compiled.evm.deployedBytecode.object);
  if (code.length === 0 || code.length > 8192) throw new Error('bytecode ceiling');
  const common = new Common({chain: Mainnet, hardfork: Hardfork.Cancun});
  const vm = await createVM({common});
  // No chain, block, transaction, account key or network is used.
  const to = createAddressFromString('0x0000000000000000000000000000000000001000');
  await vm.stateManager.putAccount(to, new Account());
  const f = row.facts;
  async function call(signature, args, caller = 99999) {
    const from = createAddressFromString('0x' + BigInt(caller).toString(16).padStart(40, '0'));
    const res = await vm.evm.runCode({code, data: calldata(signature, args), caller: from,
      to, gasLimit: 1000000n, value: 0n});
    return {reverted: Boolean(res.exceptionError), exception: res.exceptionError?.error ?? null,
      data: bytesToHex(res.returnValue), gas: res.executionGasUsed.toString(), bytes: res.returnValue};
  }
  let bad, normal, before, after;
  if (f.mechanism === 'access_control') {
    before = integer((await call(f.read_function + '()', [])).bytes);
    assert.equal(before, '0');
    bad = await call(f.signature, f.bad_args ?? [f.argument], f.attacker);
    assert.equal(bad.reverted, !row.vulnerable, row.id + ' unauthorized outcome');
    after = integer((await call(f.read_function + '()', [])).bytes);
    assert.equal(after, row.vulnerable ? String(f.argument) : '0', row.id + ' stored effect');
    normal = await call(f.signature, f.normal_args ?? [f.argument + 1], f.owner);
    assert.equal(normal.reverted, false, 'owner control must succeed');
    assert.equal(integer((await call(f.read_function + '()', [])).bytes), String(f.argument + 1));
  } else {
    bad = await call(f.signature, f.bad_args);
    assert.equal(bad.reverted, !row.vulnerable, row.id + ' boundary outcome');
    if (row.vulnerable) assert.equal(integer(bad.bytes), String(f.wrapped_result), row.id + ' wrapped result');
    normal = await call(f.signature, f.good_args);
    assert.equal(normal.reverted, false, 'normal input control must succeed');
    assert.equal(integer(normal.bytes), String(f.normal_result));
    // Wrong selector must be rejected, not silently accepted as a valid call.
    const wrong = await call('notAFunction(uint256)', [0]);
    assert.equal(wrong.reverted, true, 'unknown selector control');
  }
  const strip = ({bytes, ...rest}) => rest;
  return {id: row.id, split: row.split, source_sha256: row.source_sha256,
    bytecode_sha256: hash(code), settings, observed: {bad: strip(bad), normal: strip(normal), before, after},
    accepted_label: row.vulnerable, assertions: 'pass'};
}

for (const row of rows) {
  report.rows.push(await verify(row));
  fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
}
report.outcome = 'pass';
report.compiled_contracts = report.rows.length;
report.attack_or_boundary_probes = report.rows.length;
report.legitimate_controls = report.rows.length;
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({outcome: report.outcome, compiled_contracts: report.compiled_contracts,
  attack_or_boundary_probes: report.attack_or_boundary_probes, legitimate_controls: report.legitimate_controls,
  inventory_sha256: report.inventory_sha256}));
