# Verify a candidate registration with the real executor

This job serves a verified package through the pinned Telegraph `adapterexec`.
It provides an endpoint for the coordinator's registration integration test.
It does not register, sign or deploy an evaluator. Provider and signer
credentials stay outside CI.

## Controller inventory

The controller supplies `verification/registration/candidates.json` before
dispatch. Its schema is `candidates.schema.json`. The Python validator also
enforces immutable URL path safety, unique routing identities and the reviewed
campaign anchors in `common.py`. The server and lease implementation are generic.
This campaign currently admits only the verified sentiment candidate.

The root has `schema_version: 1` and `candidates`, an array of candidate rows.
Each row has the following fields:

| Field | Sentiment value or requirement |
|---|---|
| family | SENTIMENT_ANALYSIS |
| child | SENTIMENT_ANALYSIS_GENERAL, inferred when absent |
| class | judgment |
| file | SENTIMENT_ANALYSIS.safetensors |
| url | Immutable public raw GitHub URL for the exact unchanged package |
| sha256 | f2eefcc7b7fd807bcbd260d97c8b3b0a6b50276199b475131df968a5b60d43d9 |
| keccak | 4fba5da30996be940825450ae3abc5a438b1c582e098346469b83be42168a125, no 0x prefix |
| slot | 1 |
| forward_port | 48601 |
| runnable_url | https://tga1.zkasuran.dev/_s2sent/runnable |
| owner | Explicitly selected Telegraph identity, 0x8b224783fe5b3c52b7db0cb9b1754f8812b75287 |
| verified_run_id | String 44441475723 |
| verified_commit | a18b51ce4be30741a86e2432416b5b283a795b4d |
| candidate_proof | Object with immutable public url and sha256 of candidate-manifest.json |
| runtime_proof | Object with immutable public url and sha256 of runtime-result.json |
| source_proof | Object with immutable public url and sha256 of data-binding.json |
| existing_registration_id | Integer zero for a new registration or an explicit existing id |

Every package/proof URL must start with
`https://raw.githubusercontent.com/zkasuran/telegraph-catalog-miners/`, followed
by a full 40-character commit and a safe relative path. Branch names, query
strings, userinfo and foreign repositories fail. Publish the exact proof bytes,
without reformatting or adding fields. Their fixed hashes are:

| Proof | SHA-256 |
|---|---|
| candidate-manifest.json | 32c0e285f6726b07ced06f07b53f33a94194d373973de56bfb74f94c55ecfa7d |
| runtime-result.json | dd66981ce9936b2bf7f906a6be416ec4a3a6f3d4830bdfe5691b1778614b32a5 |
| data-binding.json | 4ee10138832d9958f6e06f6da8c19eecedd1f10936d100e95e74def299e6e119 |

This campaign requires the explicitly selected Telegraph owner. No wallet is opened here.
Adding another family requires its independently verified anchor and schema
update. A family name alone is not permission to serve an unverified package.

## Workflow and lease sequence

Manually dispatch `evaluator-candidate-verify.yml` with `family`, `batch` and
`existing`. Batch is 1 to 64 ASCII letters, digits, underscores or hyphens.
Existing is zero for a new entry or the exact registration id to resume.
The workflow uses standard public ubuntu-24.04 CPU, pinned actions, read-only
contents permission and one job per family. Its job ceiling is 350 minutes.

Preparation checks the public successful recovery run and exact commit, then
fetches and hashes the unchanged package and all three public proofs. It checks
the source boundary and all-format runtime quality evidence. It fetches the
pinned foundation, engine, tokenizer, base configuration and reference kit.
No training checkpoint or base-model weights are used.

Fresh class discovery reads exactly the family resolution URL and its selected
child discovery URL on devnode.telegraphprotocol.com. Parent and child identities,
class, evaluation kind, request/response/onchain family pointers and readiness
must agree. A GENERIC parent scorer fallback is permitted. It is not a schema
family override. The envelope is hash-bound to the static checker.

`stagecheck.go` installs only the checked classes and calls the unmodified
`CheckLoRA`. All original Stage 1 checks remain. The original
RequireTraining=false setting is retained with an additional selected-child
training-record check. No class override or foundation-load flag exists.

Preparation creates one ephemeral ed25519 forwarding key in RUNNER_TEMP. Only
the public lease JSON is published as `lease-SENTIMENT_ANALYSIS`. Its fields are
schema_version, family, owner, slot, run_id, commit, batch, public_key,
authorized_keys_line, adapter_hash, package_sha256, forward_port, runnable_url,
existing_registration_id, created_at, expires_at, the three proof hashes and
required_server_policy.

The coordinator must install the supplied `authorized_keys_line` exactly. It
uses restrict, port-forwarding, an exact loopback permitlisten, a local destination
restricted to registration-denied.invalid:1, a forced failing command and an
expiry. It permits the reverse TCP listener only
at 127.0.0.1:48601. The fixed SSH host and host key are the existing integration
host from verification/prepare.py. SSH ignores other configs and identities.
The dedicated `telegraph-verify` SSH user policy must enforce AllowTcpForwarding=remote,
AllowStreamLocalForwarding=no, PermitTunnel=no and GatewayPorts=no.
Authorized-key permitopen syntax requires a literal hostname and positive port.
The server-configuration value `none` is not valid there. The reserved `.invalid`
destination adds a key-level restriction. Server policy disables local and Unix
socket forwarding. CI does not modify that policy or claim it was applied.

Lease publication requests authorization. It does not assert public runtime
readiness. The coordinator waits for `PUBLIC_EXECUTOR_READY SENTIMENT_ANALYSIS`
before registering externally. The reverse proxy must map the HTTPS URL to
127.0.0.1:48601 and preserve either `/runnable` or the full public path.

The session starts exactly one real upstream executor and one bounded HTTP
front end. Before forwarding, it checks four fixed token vectors twice for
finite identical FP32 bits. It also checks real empty/negative-token refusals.
The same loaded process serves the node. There is no fake scorer or second
foundation process. Public fixed-vector scores must match the local preflight.

## Request bounds

| Boundary | Limit |
|---|---|
| HTTP body and response | 1 MiB |
| Headers | 16 KiB aggregate |
| Request line | 8 KiB |
| Read phase | 10 seconds absolute |
| Concurrent connections | 8 |
| Concurrent scoring batches | 1, no inference queue |
| Batch items | 100 |
| Input tokens per item | 1 through 512 |
| Total batch tokens | 51,200 |
| Token ids | Integers 0 through 50,367, no booleans |
| Scoring call | 600 seconds absolute |
| Authorization wait | At most 600 seconds within the lease |
| Lease | Five hours from created_at, including startup |

The package itself caps guest scoring at 256 tokens. The front end accepts the
node's bounded 512-token input contract and lets the unmodified runtime perform
its own documented truncation. Input roots bind the original received vectors.

POST JSON accepts `batch`, foundation_hash, adapter_hash, tokenizer_hash and
optional input_root. Each batch item is exactly an object with `tokens`.
Identity, token type, count and input-root checks run before scoring. Duplicate
JSON keys, ambiguous Content-Length, transfer/content encoding and oversized
inputs fail. Each connection closes after one request.

The proxy checks actual executor identities, finite score bits, exact positional
result count and the input-root echo. Missing scores, score/error conflicts and
extra results fail. Equal score bits can be legitimate and are not rejected.
The upstream wire format has no per-result ids. The trusted pinned executor and
the input-root binding preserve positional order. A malformed or timed-out
backend marks the session fatal so no more inference is queued.

The upstream executor's stdout/stderr are discarded because its original logs
include token lengths and scores. The proxy has no access, input or token logs.
Private state, source snapshots, static diagnostics and keys are never uploaded.

## Decisions and cleanup

The session only GETs the public wasm listing. A terminal entry must match the
exact family, owner and package Keccak. New entries must have registered_at
at or after the lease's created_at. With an explicit existing id, only that id
is eligible, including an older pending entry. Future, stale and unrelated
entries do not close the lease. Duplicate or multiple matching new entries fail
closed rather than choosing the wrong registration.

Active, rejected, superseded and deregistered stop the session. Only active exits
zero. A hard timer ends the lease at five hours even during a blocked call.
The workflow has an outer timeout and an always-run cleanup step. Process groups
are owned by this session and cleanup checks their start-time identities before
stopping them. Both SSH and the real runtime are terminated and the ephemeral
key files are removed. The coordinator removes the remote authorization entry.

The final artifact contains only `preflight.json` from the fixed source-only
vectors and sanitized `verdict.json`. It contains no node evaluation payload,
pool roots, reasons, received tokens, score arrays or runtime diagnostics.
Preflight records the real invalid-token refusals, proxy refusals, repeated-bit
check and a fingerprint of the fixed scores. Its public_identity_and_bits_match
field becomes true only after the public endpoint returns those exact scores.

## Local source-only gate

```sh
bash verification/registration/verify.sh
ruff check --no-cache verification/registration
/tmp/opencode/actionlint -shellcheck= .github/workflows/evaluator-candidate-verify.yml
```

Run Go source tests from the pinned kit directory with Go 1.25.7:

```sh
go test -mod=readonly -count=1 -v /absolute/checkout/verification/registration/stagecheck.go /absolute/checkout/verification/registration/stagecheck_test.go
```

They do not load a model. They exercise class authority, wrong hashes, invalid
static containers and the real node's input-root implementation. Preparation
runs this same command before creating a key.

To verify retained public proof files without a model or a network request:

```sh
python3 verification/registration/check_proof.py --family SENTIMENT_ANALYSIS --candidate candidate-manifest.json --runtime runtime-result.json --source data-binding.json
```

Local HTTP tests use labelled source-only fixtures. They prove the proxy's
framing, rejection, count, timeout and no-queue behavior. They are not evidence
of actual foundation loading or public node registration. Those checks run only
in the manually dispatched workflow.

## Primary contracts and terms

- [EXEC] Telegraph cmd/adapterexec/main.go at kit revision 1ed063e2f0066cfaf0e124d66ec6a17b7a6121fd defines real batch requests, actual identity echoes, positional scores and executor logging.
- [INPUT] Telegraph pkg/wasm/remote/remote.go and pkg/wasm/runtime/host.go at that revision define BatchInputRoot and canonical input JSON.
- [CHECK] Telegraph pkg/adapterbuild/lora.go at that revision defines the retained CheckLoRA parser and training-record checks.
- [LIVE] The exact public family and child GETs bind current class authority. The public wasm listing supplies decision identities and UTC times.
- [SSH] OpenSSH 9.6 auth-options.c handle_permit and sshd(8) define authorized-key port restrictions, expiry and restrict behavior. The coordinator must verify its server policy independently.

Helper source is LicenseRef-zkasuran-SAND-1.0 under this repository's terms.
The original package keeps all embedded terms. The pinned kit retains its MIT
notice and foundation assets retain their existing upstream terms. Recovery
proofs and package bytes are unchanged.
