# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import json
from sources import digest, fetch, save_raw
METHODS = {"eth_chainId", "eth_blockNumber", "eth_getBlockByNumber", "eth_call", "eth_getCode"}

def word(value):
    if isinstance(value, str):
        value = int(value, 16)
    if not 0 <= value < 2 ** 256:
        raise ValueError("ABI word overflow")
    return f"{value:064x}"

def words(raw):
    if not isinstance(raw, str) or not raw.startswith("0x") or (len(raw) - 2) % 64:
        raise ValueError("invalid ABI words")
    return [int(raw[i:i + 64], 16) for i in range(2, len(raw), 64)]

def address(raw):
    values = words(raw)
    if len(values) != 1 or not 0 < values[0] < 2 ** 160:
        raise ValueError("invalid ABI address")
    return f"0x{values[0]:040x}"

def decode_batch(raw, size):
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) != size:
        raise ValueError("RPC batch cardinality")
    by_id = {}
    for row in data:
        if (not isinstance(row, dict) or row.get("jsonrpc") != "2.0"
                or type(row.get("id")) is not int or row["id"] not in range(1, size + 1)
                or row["id"] in by_id or "error" in row or "result" not in row):
            raise ValueError("RPC error, duplicate/foreign ID or malformed envelope")
        by_id[row["id"]] = row["result"]
    return [by_id[i] for i in range(1, size + 1)]

class RPC:
    def __init__(self, url):
        if url != "https://ethereum-rpc.publicnode.com":
            raise ValueError("non-allowlisted endpoint")
        self.url = url
        self.receipts = []

    def batch(self, calls):
        if not 1 <= len(calls) <= 32 or any(method not in METHODS for method, _ in calls):
            raise ValueError("RPC read-only method/batch limit")
        payload = json.dumps([{"jsonrpc": "2.0", "id": i + 1, "method": method, "params": params}
                              for i, (method, params) in enumerate(calls)]).encode()
        raw, seconds = fetch(self.url, payload)
        self.receipts.append({"endpoint": self.url, "request_sha256": digest(payload),
                              "request_path": save_raw(payload), "response_sha256": digest(raw),
                              "response_path": save_raw(raw), "seconds": seconds, "calls": len(calls)})
        return decode_batch(raw, len(calls))
