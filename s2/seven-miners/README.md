# Ask questions and read digital stock

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

The stock miner returns current availability from the primary controlled digital
allocation store. It subtracts reservations from on-hand inventory, binds the
merchant, SKU and location and reports read time separately from mutation time.

The chat miner returns a generated reply, a verdict and an at-most-80-character
brief. It reserves each attempt in a primary D1 budget before calling the model.
At most twenty attempts per UTC day are allowed across all callers. No web
retrieval, independent fact verification or calibrated forecast is claimed.

```bash
curl --fail --max-time 55 --get \
  'https://telegraph-text.margyn.workers.dev/miners/v3/chat' \
  --data-urlencode 'message=What is the sum of 23 and 19?'
```

## Try the service

```bash
curl --fail --max-time 15 --get \
  'https://telegraph-s2-hashprice.margyn.workers.dev/truth/v2/sku' \
  --data-urlencode 'merchant=zkasuran-controlled-digital-store' \
  --data-urlencode 'sku=commerce-source-bundle-digital-allocation' \
  --data-urlencode 'location=local-digital-allocation'
```

`sku-in-stock.yaml` declares `SKU_IN_STOCK_GENERAL`. The Intent has no canonical
owner-defined response format yet. Registration, ranking, routed traffic and
on-chain answer delivery are separate states. No ranking or adoption is claimed.

## What is real

| Item | State |
|---|---|
| Inventory | Owner-operated digital allocations in primary D1 |
| Available quantity | Current on-hand minus reservations |
| Stock tests | Nine independent tests with generated states and hostile inputs |
| Chat tests | Ten tests for contracts, provider failures, budget and sibling routing |
| Runtime source | Two byte-identical modules captured from the deployed Worker |
| Physical stock or external retailers | Not supported |
| Canonical comparison | Waiting for owner-defined format |

## Verify

Node.js and Python are required. No package installation is needed.

```bash
bash verify.sh
```

The gate checks descriptor keys, a deliberately invalid descriptor, runtime syntax,
source hashes, generated allocation conservation, identity and time refusals and
the real handler against an independent primary-session fixture. Live HTTP checks
are recorded separately because they depend on current network/database state.
The same gate runs chat controls and exercises the exact quota SQL on independent
SQLite state. `chat-completion.yaml` declares `CHAT_COMPLETION_GENERAL` and maps
all three canonical fields. `chat-router.mjs` mounts the new path while delegating
existing Worker requests unchanged. Existing text data modules are not republished
as part of this package.

See [SECURITY.md](SECURITY.md), [DATA-SOURCES.md](DATA-SOURCES.md) and
[PROVENANCE.json](PROVENANCE.json). Source is under [LICENSE](LICENSE).

AI assistance was used to prepare the descriptor and independent checks. No
neural model runs locally and no submitted evaluation answers are used.
