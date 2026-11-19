# Four additional reference routes

This source package adds WBTC reserve, Ethereum USDC supply APY, validator balance
and airport temperature routes to the existing hashprice Worker. It preserves
**all 66 baseline files**. The new routes are prepared locally, not deployed.

| Route | Required parameters | Scored value |
|---|---|---|
| `/truth/v2/finance-reserve` | `asset` | `attested_reserve`, integer micro-BTC as a decimal string |
| `/truth/v2/finance-yield` | `project`, `symbol`, `chain` | `supply_apy`, six-decimal basis points as a decimal string |
| `/truth/v2/operations/validator` | `validator_index` | `effective_balance`, gwei |
| `/truth/v2/operations/weather` | `location` | `temperature_c`, degrees Celsius; identical `temp_c` alias |

Finance covers WBTC plus Aave V3 and Compound V3 USDC on Ethereum. Validator data
cover exact indices with a one-hour delayed-epoch ceiling. Weather supports
London/EGLC, Karachi/OPKC and Tokyo/RJTT. These are airport observations.

The wrapper matches exactly four paths. Every other request goes unchanged to the
baseline default export with its original env, context and receiver. No catalog,
D1 database, preference pairs or protected evaluation samples are packaged.

Run `bash verify.sh`. It checks exact bytes, documented descriptor schema, Decimal
arithmetic, wrapper preservation, real public fixture replay and refusal controls.
The original baseline test gate also runs. No credentials are required.

`wrangler.json` keeps the existing Worker name, compatibility date, compatibility
flags and `keep_vars`. Explicit ESModule and Text lists preserve runtime and notices.
Actual deployment, current native codec, source loader and admission are separate
gates. Local numeric extraction does not prove node numeric-string acceptance.

Source facts and licences are in the component dossiers. AI assistance used OpenCode
with agentrouter/gpt-6.1-sol for integration typing and verification. No human code
review is claimed. Own source licence: `LicenseRef-zkasuran-SAND-1.0`.
