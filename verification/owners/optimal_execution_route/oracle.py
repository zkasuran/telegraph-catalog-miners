# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Integer quote scorer for a controller-trusted bounded state, not a model adapter."""
import argparse
import hashlib
import json
import re
from decimal import Decimal, DecimalException
from pathlib import Path

SCALE = 1000000
MAX_BODY = 16384
MAX_ARTIFACT = 2 * 1024 * 1024
AMOUNT_IN = 10 ** 18
OBJECTIVE = "max_gross_exact_input_unsplit_v2"
QUESTION = "OPTIMAL_EXECUTION_ROUTE_GENERAL request. pair (Swap to route: sell exactly 1 ETH for USDC on Ethereum): ETH/USDC."
ADDR = re.compile(r"0x[0-9a-fA-F]{40}\Z")
HASH = re.compile(r"0x[0-9a-f]{64}\Z")


class Refused(ValueError):
    pass


def canonical_address(value):
    if not isinstance(value, str) or not ADDR.fullmatch(value):
        raise Refused("bad address")
    return value.lower()


def integer(value, upper=2 ** 256 - 1):
    if isinstance(value, str):
        if len(value) > 78 or not re.fullmatch(r"0|[1-9][0-9]*", value):
            raise Refused("bad integer string")
        value = int(value)
    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value() or value < 0 or value > upper:
            raise Refused("non-integral or out-of-bounds number")
        value = int(value)
    if type(value) is not int or not 0 <= value <= upper:
        raise Refused("bad integer")
    return value


def amount_out(amount, reserve_in, reserve_out, fee_ppm=3000):
    """Fee-inclusive constant-product floor. No transfer-tax token support."""
    amount = integer(amount)
    reserve_in = integer(reserve_in, 2 ** 112 - 1)
    reserve_out = integer(reserve_out, 2 ** 112 - 1)
    if not amount or not reserve_in or not reserve_out or fee_ppm != 3000:
        raise Refused("unsupported size, liquidity or fee")
    adjusted = amount * 997
    output = adjusted * reserve_out // (reserve_in * 1000 + adjusted)
    if not output:
        raise Refused("zero-output quote")
    return output


def state_identity(state):
    """Reserve identity intentionally omits block/time: unchanged pools are duplicates."""
    pools = sorted((canonical_address(p["address"]), canonical_address(p["token0"]),
                    canonical_address(p["token1"]), integer(p["reserve0"]),
                    integer(p["reserve1"]), integer(p["fee_ppm"])) for p in state["pools"])
    return hashlib.sha256(json.dumps([state["chain_id"], pools], separators=(",", ":")).encode()).hexdigest()


class Oracle:
    def __init__(self, policy, state):
        if (policy.get("schema_version") != 1 or policy.get("objective") != OBJECTIVE
                or policy.get("pair") != "ETH/USDC" or policy.get("chain_id") != 1
                or integer(policy.get("amount_in_wei")) != AMOUNT_IN
                or policy.get("usdc_decimals") != 6 or policy.get("fee_ppm") != 3000
                or policy.get("routes") != ["WETH/USDC", "WETH/DAI/USDC"]):
            raise Refused("wrong policy")
        if (state.get("chain_id") != 1 or not HASH.fullmatch(state.get("block_hash", ""))
                or integer(state.get("block_number")) == 0 or len(state.get("pools", [])) != 3):
            raise Refused("wrong state context")
        self.policy, self.state = policy, state
        self.weth = canonical_address(state["weth"])
        self.usdc = canonical_address(policy["usdc"])
        self.dai = canonical_address(policy["dai"])
        self.paths = [[self.weth, self.usdc], [self.weth, self.dai, self.usdc]]
        self.pool_map = {}
        seen = set()
        expected = {frozenset(path) for path in ([self.weth, self.usdc], [self.weth, self.dai], [self.dai, self.usdc])}
        for p in state["pools"]:
            addr = canonical_address(p["address"])
            t0, t1 = canonical_address(p["token0"]), canonical_address(p["token1"])
            key = frozenset((t0, t1))
            if addr in seen or key in self.pool_map or key not in expected or t0 >= t1 or p["fee_ppm"] != 3000:
                raise Refused("duplicate or unsupported pool identity")
            if not integer(p["reserve0"], 2 ** 112 - 1) or not integer(p["reserve1"], 2 ** 112 - 1):
                raise Refused("unavailable pool")
            seen.add(addr)
            self.pool_map[key] = p
        self.quotes = [self.quote(path, AMOUNT_IN) for path in self.paths]
        self.best_amount = max(q["amount_out"] for q in self.quotes)
        self.ties = [q for q in self.quotes if q["amount_out"] == self.best_amount]

    def quote(self, path, amount):
        if path not in self.paths:
            raise Refused("route outside complete declared universe")
        amounts = [integer(amount)]
        pools = []
        for a, b in zip(path, path[1:]):
            p = self.pool_map[frozenset((a, b))]
            r_in, r_out = (p["reserve0"], p["reserve1"]) if a == canonical_address(p["token0"]) else (p["reserve1"], p["reserve0"])
            amounts.append(amount_out(amounts[-1], r_in, r_out, p["fee_ppm"]))
            pools.append(canonical_address(p["address"]))
        bps = integer(self.policy["slippage_bps"], 10000)
        return {"tokens": path, "pools": pools, "fees_ppm": [3000] * len(pools),
                "amounts": amounts, "amount_out": amounts[-1],
                "minimum_out": amounts[-1] * 10000 // (10000 + bps)}

    def route_identity(self, quote):
        return {"chain_id": 1, "block_hash": self.state["block_hash"], "trade_type": "exact_input",
                "amount_in_wei": str(AMOUNT_IN), "usdc_decimals": 6, "objective": OBJECTIVE,
                "tokens": quote["tokens"], "pools": quote["pools"], "fees_ppm": quote["fees_ppm"]}

    def check_route(self, route):
        if not isinstance(route, dict):
            raise Refused("route identity absent")
        matched = dict(route)
        for key in ("chain_id", "usdc_decimals"):
            matched[key] = integer(matched.get(key))
        if not isinstance(matched.get("fees_ppm"), list) or len(matched["fees_ppm"]) > 2:
            raise Refused("route fee identity incomplete")
        matched["fees_ppm"] = [integer(f, 1000000) for f in matched["fees_ppm"]]
        for key in ("tokens", "pools"):
            if not isinstance(matched.get(key), list) or len(matched[key]) > 3:
                raise Refused("route identity incomplete")
            matched[key] = [canonical_address(a) for a in matched[key]]
        if not any(matched == self.route_identity(q) for q in self.ties):
            raise Refused("not a complete tied optimal route identity")

    def verify_router_observations(self):
        rows = self.state["router_quotes"]
        expected = {(tuple(path), a) for path in self.paths for a in (10 ** 17, 10 ** 18, 10 ** 19)}
        seen = set()
        for row in rows:
            key = (tuple(row["path"]), integer(row["amount_in"]))
            if key in seen or key not in expected:
                raise Refused("duplicate or unexpected router observation")
            quote = self.quote(row["path"], row["amount_in"])
            if quote["amounts"] != [integer(v) for v in row["amounts"]]:
                raise Refused("reserve-derived quote differs from router view")
            seen.add(key)
        if seen != expected:
            raise Refused("incomplete router observation inventory")
        return len(rows)


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise Refused("duplicate JSON field")
        obj[key] = value
    return obj


def score_wire(question, body, oracle=None):
    """Score the native amount_out field only with separately trusted reference state."""
    if question != QUESTION:
        return {"status": "refused", "reason": "literal task mismatch", "score_ppm": None}
    if oracle is None:
        return {"status": "abstain", "reason": "no controller-trusted block and objective", "score_ppm": None}
    try:
        if not isinstance(body, str) or len(body.encode()) > MAX_BODY:
            raise Refused("body limit")
        parsed = json.loads(body, parse_int=Decimal, parse_float=Decimal,
                            parse_constant=lambda x: (_ for _ in ()).throw(Refused("nonfinite number")),
                            object_pairs_hook=unique_object)
        value = parsed.get("amount_out") if isinstance(parsed, dict) else parsed
        # Native response field is a JSON number, never a quoted numeric string.
        if not isinstance(value, Decimal):
            raise Refused("amount_out is not a native number")
        value = integer(value, 100000000000)
        if value < 100000000:
            raise Refused("amount_out below live range")
        route_checked = isinstance(parsed, dict) and "route" in parsed
        if route_checked:
            oracle.check_route(parsed["route"])
        error = abs(value - oracle.best_amount)
        penalty = (error * SCALE + oracle.best_amount - 1) // oracle.best_amount
        return {"status": "bounded_quote_score", "score_ppm": max(0, SCALE - penalty),
                "absolute_error_base_units": error, "exact_amount": error == 0,
                "route_identity_checked": route_checked, "block_hash": oracle.state["block_hash"],
                "objective": OBJECTIVE, "not_execution_or_global_optimality": True}
    except (Refused, ValueError, TypeError, KeyError, RecursionError, OverflowError, DecimalException, UnicodeError) as exc:
        return {"status": "refused", "reason": str(exc)[:160], "score_ppm": None}


def load_trusted(path, expected_sha256):
    path = Path(path)
    if path.stat().st_size > MAX_ARTIFACT:
        raise Refused("artifact limit")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise Refused("controller-pinned artifact hash mismatch")
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--body", required=True, help="Non-secret raw numeric miner response")
    args = parser.parse_args()
    try:
        data = load_trusted(args.state, args.expected_sha256)
        policy = load_trusted(args.policy, args.policy_sha256)
        oracle = Oracle(policy, data["states"][args.state_index])
        oracle.verify_router_observations()
        result = score_wire(QUESTION, args.body, oracle)
    except (Refused, ValueError, KeyError, IndexError, OSError) as exc:
        result = {"status": "refused", "score_ppm": None, "reason": str(exc)[:160]}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "bounded_quote_score" and result["exact_amount"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
