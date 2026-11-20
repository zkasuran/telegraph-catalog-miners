-- SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
CREATE TABLE IF NOT EXISTS s2_miner_quota (
  feature TEXT NOT NULL,
  day TEXT NOT NULL,
  calls INTEGER NOT NULL CHECK(calls BETWEEN 1 AND 250),
  PRIMARY KEY(feature,day)
) STRICT;
CREATE TABLE IF NOT EXISTS s2_tracking_grants (
  tracking_sha256 TEXT PRIMARY KEY,
  public_status_consented INTEGER NOT NULL CHECK(public_status_consented IN (0,1)),
  expires_at TEXT NOT NULL
) STRICT;
