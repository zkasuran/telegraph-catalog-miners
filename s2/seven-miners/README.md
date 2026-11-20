# Read digital allocation stock

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

This miner returns current availability from the primary controlled digital
allocation store. It subtracts reservations from on-hand inventory, binds the
merchant, SKU and location and reports read time separately from mutation time.

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
| Tests | Nine independent tests with generated states and hostile inputs |
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

See [SECURITY.md](SECURITY.md), [DATA-SOURCES.md](DATA-SOURCES.md) and
[PROVENANCE.json](PROVENANCE.json). Source is under [LICENSE](LICENSE).

AI assistance was used to prepare the descriptor and independent checks. No
neural model runs locally and no submitted evaluation answers are used.
