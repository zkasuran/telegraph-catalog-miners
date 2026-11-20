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
