# Six reference-compatible comparator recovery artifacts

Prospective release `zkcmp-s2-20261118-1`, prepared November 18, 2026.
Controller review and authenticated rollout remain pending.

The six-family package rebuilds our existing MIT comparator arithmetic without
changing it. Rust code sections are byte-identical to the historical base.
Only its fixed-width family tag differs in the data section. Every binary has
the correct `zk.family` tag followed by `zk.notices` and `zk.license`.
All linked-component notices are complete and precede the new own SAND grant.

## Reproduce the local release gate

From this release directory, with the pinned executables on PATH:

```bash
bash ./verify.sh
```

The gate uses installed Rust 1.97.1 at commit
`8bab26f4f68e0e26f0bb7960be334d5b520ea452`, isolated Cargo/target output,
`--locked --offline` and warnings as errors. Cargo has zero external crates.
Pinned precompiled Rust core and compiler-builtins archive hashes are checked
before compiling. It runs the actual Go 1.25.7 reference gate from the bundled
Telegraph validation subset at `1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd`.
The reference function, deterministic runtime, caller and reference test are
unchanged. Supporting constants and input helpers are recorded unchanged
excerpts. Source hashes and upstream URLs are in `validation/PINS.json`.

`python3 build.py` rebuilds and packages from the bundled source and licences.
`verify.py` performs that rebuild then tests the exact final bytes. `scratch/`
holds isolated rebuild outputs and Go caches.
`evidence/` holds full command lines, environments, logs and result records.
Neither directory belongs in a public release.

All source, complete licence text and validation dependencies required by the
gate travel with this release. No original checkout, operator packet, remote
source tree or shared module cache is required. Go uses `-mod=vendor`,
`GOTOOLCHAIN=local`, `GOWORK=off`, `GOPROXY=off` and an isolated empty module
cache. The gate does not download tools or dependencies.

Install the tools before running. `toolchain.json` pins Linux AMD64 executable
hashes and versions: Rust/Cargo 1.97.1, Go 1.25.7, cast 1.7.1, Node 22.22.2
and Ruff 0.15.14. Python 3.11 or newer and Bash are also required. The Rust
sysroot must contain the pinned wasm32 core and compiler-builtins archives.
Rustup dispatchers are not the pinned compiler binary. Select the compiler
and Cargo executables directly if a dispatcher is first on PATH.

Every executable can be selected explicitly:

```bash
bash ./verify.sh \
  --rustc /opt/zkcmp/rust/bin/rustc \
  --cargo /opt/zkcmp/rust/bin/cargo \
  --go /opt/zkcmp/go/bin/go \
  --cast /opt/zkcmp/bin/cast \
  --node /opt/zkcmp/bin/node \
  --ruff /opt/zkcmp/bin/ruff
```

Use your installed executable locations in that command. Equivalent environment
variables are `ZKCMP_RUSTC`, `ZKCMP_CARGO`, `ZKCMP_GO`, `ZKCMP_CAST`, `ZKCMP_NODE`
and `ZKCMP_RUFF`. Their values are not inherited by the tools as credentials.
Subprocess environments contain only tool/build settings and isolated scratch
directories. Executable bytes and versions are checked before compilation.

The public path scan includes every bundled source, licence and binary file.
Exact CRLF upstream licence bytes remain intact. Two wazero documentation-only
path examples were neutralized, with original and changed hashes recorded.
No validation behavior changed. Validation dependencies retain their own
licences, including wazero Apache-2.0 and x/sys BSD-3-Clause.
Git attributes use binary diffs only for the CRLF-bearing licence files.
No source-code whitespace check is disabled.
One unused platform-specific x/sys assembly file had a trailing tab removed.
Its copyright and instructions remain unchanged; both hashes are recorded.

## Verified scope

| Check | Actual result |
|---|---:|
| Final artifacts | 6 |
| Reference cases per artifact | 537592 |
| Total reference cases | 3225552 |
| Reference mismatches | 0 |
| Reference skips | 0 |
| Independent raw cases per artifact | 27769 |
| Total independent raw cases | 166614 |
| Positive repeatability cases per artifact, repeated twice | 2500 |
| Local Worker ordered wire cases | 690 |
| Local Worker per-item overflow refusals | 42 |
| Invalid wire-body refusals | 10 |
| Additional malformed/resource refusals | 12 |
| Planner negative controls | 19 |
| Prototype registrar negative controls | 24 |
| Portable-input integrity tests | 5 |

`manifest.json` names each immutable binary and records SHA-256, Keccak,
source hashes, archive hashes and prospective URLs. These URLs have not been
deployed. Their namespace is `/_zkcmp_s2_20261118/`.

## Additive rollout planner

`planner.mjs` exports pure functions. It does not read files, obtain credentials,
upload versions, change settings or promote traffic. The controller supplies
the actual complete current snapshot and release bytes to `buildPlan`.

The normalized snapshot schema is version 1:

| Field | Required contents |
|---|---|
| `schema`, `complete`, `worker` | `1`, `true`, `telegraph-fin` |
| `deployment` | Active deployment ID and one version with ID and 100 percent weight |
| `main_module` | Current root-level ES module filename |
| `modules` | Every module with name, normalized type, exact base64 bytes and SHA-256 |
| `settings` | Complete settings with compatibility date, flags and usage model |
| `bindings` | Complete bindings and resource identities, with secret names/types retained |
| `routes`, `domains` | Complete route configuration and domain enablement |
| `entry_contract` | Verified named exports, default handlers and namespace availability |
| `old_binary_routes` | All known immutable URLs bound to retained wasm module hashes, MIME and cache policy |

`test_planner.mjs` contains a synthetic full snapshot example. It is explicitly
a fixture, not a snapshot of Cloudflare. Raw provider API field normalization
and verification of snapshot completeness are controller responsibilities.
Module types normalize to `esm`, `wasm`, `text`, `data` or `commonjs`.
Root-level module names are supported. Split deployments and additional
entry handlers/exports are refused until the controller establishes a safe
representation. No export is silently discarded.

The planner retains every old module byte and type, all settings, bindings,
routes and domains. New hash-qualified module names are appended. The new
entry handles only the exact namespace prefix including its trailing slash.
All other paths pass the same Request, env and ctx to the unchanged prior
entry. Both generated entry and generated repair entry were executed in
memory with the exact new binary and runnable paths.

`assertNoDrift(plan, current)` compares the complete snapshot fingerprint.
Re-read immediately before upload and promotion. Any independent deployment,
version, module, setting, binding or route change requires re-staging.

`planRollback(plan, current, registered, expectedCurrentFingerprint)` requires
the controller's pinned promoted deployment fingerprint. It returns a
compatibility repair target with a distinct entry module. It keeps every old
module and all new binary/runnable URLs. It never selects the old version that
would remove a registered URL. This repairs the dispatch layer only. A runtime
fault needs a separately verified compatible repair that retains exact
registered binaries. There is no automatic deregistration or bond action.

## Remaining controller rollout gates

1. Review the source, exact final bytes and full local evidence. Assign the
   publication commit in the external release ledger before publication.
   Do not restamp or change these binary bytes after qualification.
2. Obtain a fresh authenticated full telegraph-fin snapshot. Retain all module
   bytes/types, active deployment/version IDs and weights, settings, bindings,
   routes and domain flags. Verify namespace absence and entry export contract.
   Preserve secret values through the provider's supported keep mechanism.
   The pure planner is not a Cloudflare multipart uploader.
3. Verify the actual account's existing free-plan headroom. Measure aggregate
   Worker startup, all WASM memory and worst-case 100-question CPU. Local Node
   elapsed times are not Cloudflare CPU measurements. The existing handler's
   1 MiB body cap bounds size but does not prove the free-plan CPU budget.
4. Build the additive plan, bind the complete snapshot fingerprint and save
   a compatibility repair plan. Upload without promotion through controller
   tooling. Verify full module/settings/binding readback and version-targeted
   preview behavior. Recheck drift before promotion.
5. Verify every old immutable binary and old health/runnable/miner route.
   Compare binary bytes, MIME and immutable cache policy. Dynamic miner data
   requires shape and question-alignment checks, not literal price equality.
6. Promote only the compatible new version to 100 percent before registration.
   Verify exact new bytes and caller precision/refusals from independent public
   vantage points. A mixed deployment can send a new URL to an old version.
7. Finish the authoritative current registration ABI/facet binding, direct
   test-USDC funding rules, gas ceilings, wallet lock and durable write-ahead
   transaction ledger. Check live chain, class, family, pool readiness, unused
   hashes and fresh balances/allowance before the FX_NOW canary.
8. Register and reconcile only through controller-owned signing. Check canonical
   safe receipt, full event tuple and same-block getter. Wait for the node's
   exact family/owner/hash/URL canonical result before proceeding. A receipt or
   active contract flag is not a slot win. Currency conversion child readiness
   and all real child jobs remain separate unresolved gates.

## Prototype receipt binding

`registrar_binding.mjs` validates already decoded transaction, receipt, event
and getter records against a deployment-scoped operation. It has no signer,
network-write entry point, persistence or ABI decoder. Its fixture uses invented
transaction/block IDs and intentionally small synthetic calldata. A passing
fixture is not proof of actual registration ABI correctness or gas feasibility.

## Licence and lineage

The unchanged arithmetic remains MIT. Prior MIT copies and rebuild rights
remain valid. SAND applies only to new own tags, wrapper, planner and packaging
metadata. Its actual text includes a competing-service restriction. It does
not make the old MIT program exclusively SAND. No rival binary was used.

Read `LICENSE`, `NOTICE`, `PROVENANCE.json` and `licenses/PINS.json`.
Rust core retains The Rust Project Contributors copyright and the installed
Developers attribution. Compiler-builtins 0.1.160 retains full MIT, Apache-2.0,
LLVM exception and compiler-rt credits. specialized-div-rem 1.0.0 retains
Aaron Kutch's MIT text. libm retains Jorge Aparicio's full supplied licence.
Telegraph/wazero notices accompany local validation sources; neither runtime
is linked into the released comparator WASM.

AI assistance: OpenCode with agentrouter/gpt-6.1-sol authored the new packaging,
planner and tests. The checks above were run locally. Human review is pending.
