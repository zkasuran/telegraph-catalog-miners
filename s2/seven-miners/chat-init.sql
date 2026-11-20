-- SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
CREATE TABLE IF NOT EXISTS s2_chat_quota (
  day TEXT PRIMARY KEY,
  calls INTEGER NOT NULL CHECK (calls BETWEEN 1 AND 20)
) STRICT;
