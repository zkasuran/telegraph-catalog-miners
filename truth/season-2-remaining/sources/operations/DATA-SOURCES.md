# Operations source rights

## ethPandaOps Xatu

Primary source: https://raw.githubusercontent.com/ethpandaops/xatu-data/master/README.md

> This work is licensed under
> CC BY 4.0

The Available data table includes CBT transformation tables. The selected public
`fct_validator_balance` and `dim_validator` facts are from those tables.
Attribution: ethPandaOps Xatu. Changes: exact-index selection plus epoch and public
key checks. Values remain gwei. An absent record is unavailable rather than zero.

## MET Norway

Primary source: https://api.met.no/doc/License

> Unless specified otherwise, all data and products are licensed under the Norwegian Licence for Open Government Data (NLOD) 2.0 and Creative Commons 4.0 BY International licences.

> Credit should be given to The Norwegian Meteorological Institute, shortened MET Norway, as the source of data.

Primary product description: https://api.met.no/weatherapi/tafmetar/1.0/documentation

> Meteorological Aerodrome Report (current weather at an airport)

Attribution: MET Norway / Avinor. Changes: select the latest fresh report from one
named airport and parse Celsius temperature. Verify station and raw/XML timestamp
agreement. This is an airport observation, not a forecast or city-centre estimate.

## Attribution licence

Primary source: https://creativecommons.org/licenses/by/4.0/legalcode.en

> reproduce and Share the Licensed Material, in whole or in part

> produce, reproduce, and Share Adapted Material.

Section 3 requires attribution and modification notices. Those are in NOTICE and
the responses. No Fintraffic records, Wikipedia material or preference pairs are
included. Unused functions in the unchanged module do not expose additional paths.
