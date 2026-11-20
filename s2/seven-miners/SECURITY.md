# Stock service trust model

SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0

| Input or failure | Control | Verification |
|---|---|---|
| Wrong merchant, SKU or location | Exact identity binding before returning state | Identity refusal tests |
| Invented availability | Missing row is 404 and missing primary binding is 503 | HTTP refusal tests |
| Reservation inconsistency | Integer and conservation checks | Boundary and generated-state tests |
| Old or future observation | Strict calendar and five-minute observation ceiling | Time refusal tests |
| Duplicate or oversized parameters | Fixed field allowlist and request limits | HTTP validation tests |
| Concurrent replica lag | D1 first-primary session | Primary-session HTTP test |
| Slow database | Bounded source read, six seconds | Existing runtime control, timeout not load-tested here |

The service exposes read-only availability, not buyer information or database
write access. It cannot verify another merchant's stock. Availability can change
after a read and does not reserve inventory. No physical inventory or external
customer is claimed. Canonical comparison awaits an owner-defined Intent format.

## Chat route

| Input or failure | Control | Verification |
|---|---|---|
| Runaway shared subscription traffic | Atomic primary D1 budget, twenty attempts per UTC day | Actual SQL and no-provider-call refusal tests |
| Overlong or malformed message | 256-byte UTF-8 ceiling and fixed parameter allowlist | Boundary and HTTP tests |
| Provider returns HTTP 200 with an error | Application status and finish-reason checks | Provider error tests |
| Model returns wrong format | Strict reply, verdict and brief validation | Malformed response tests |
| Provider leaks a credential into content | Exact secret-byte refusal, fixed public error strings | Credential echo test |
| New route regresses a sibling | Existing request and binding delegation | Router test and postdeploy source attestation |

Model answers are generated, not independently verified. Input is sent to the named
provider under its privacy terms. No raw prompt logging is added. Query strings can
still appear in infrastructure access logs. The budget limits attempts, not a promise
of provider availability. Failed attempts retain their reservation. No unlimited
quota, monetary savings or empirical accuracy is claimed.

## Remaining integrations

| Input or failure | Control | Verification |
|---|---|---|
| Media fetch redirects or private origin | Fixed public media origins and manual redirects | Source refusal tests |
| Large media or unsupported container | 256-KiB streamed bound and MP4 movie-header limit | Byte and container tests |
| Unsupported certainty | Strong authenticity/manipulation labels downgraded to likely assessments | Assessment tests |
| Invented X post URL | URLs only from server-search results with current oEmbed identity confirmation | Retrieval and publisher tests |
| Prohibited post-text redistribution | Public response contains IDs and links only | X privacy test |
| Unapproved paid X reads | Direct API mode needs an explicit spending flag | Missing-source test |
| Unauthorized parcel disclosure | Account permission plus primary per-shipment public-status grant | Consent refusal test |
| Wrong shipment data | Exact unique shipment-ID binding | Identity and ambiguity tests |
| Forged, wrong-port or stale port report | HMAC, semantic/unit checks and five-minute observation bound | Signature, identity and time tests |

Source records in offline tests are artificial and do not prove live parcel or port
data availability. Those routes remain fail-closed until actual sources and rights
are configured. Media model confidence is not calibrated; sampled frames miss
between-frame events. The source publisher remains a trust assumption for port
measurement truth. No untrusted source grants itself permission through a request.
