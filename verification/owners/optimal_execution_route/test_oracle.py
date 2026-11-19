# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from collect_state import RPC, address, decode_batch, word, words
from oracle import (AMOUNT_IN, QUESTION, Oracle, Refused, amount_out,
                    load_trusted, score_wire, state_identity)
from sources import HERE


def fixtures():
    policy = json.loads((HERE / "policy.json").read_text())
    data = json.loads((HERE / "evidence/STATE.json").read_text())
    return policy, data["states"][0]


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.policy, self.state = fixtures()
        self.oracle = Oracle(self.policy, self.state)
        self.body = json.dumps({"amount_out": self.oracle.best_amount})

    def test_primary_router_view_replay_all_eighteen(self):
        count = 0
        for name in ("STATE.json", "STATE-OLDER.json"):
            data = json.loads((HERE / "evidence" / name).read_text())
            for state in data["states"]:
                count += Oracle(self.policy, state).verify_router_observations()
        self.assertEqual(count, 18)

    def test_native_wire_exact_amount(self):
        got = score_wire(QUESTION, self.body, self.oracle)
        self.assertEqual(got["score_ppm"], 1000000)
        self.assertFalse(got["route_identity_checked"])
        self.assertTrue(got["not_execution_or_global_optimality"])

    def test_literal_question_not_normalized(self):
        for q in (QUESTION.lower(), QUESTION + " ", QUESTION.replace("ETH/USDC", "ETH/USDC.e")):
            self.assertEqual(score_wire(q, self.body, self.oracle)["status"], "refused")

    def test_missing_trusted_state_abstains_for_every_answer(self):
        for body in (self.body, "0", "{}", "best route is Uniswap", "1799488639"):
            got = score_wire(QUESTION, body)
            self.assertEqual(got["status"], "abstain")
            self.assertIsNone(got["score_ppm"])

    def test_wrong_size_real_router_quotes_are_not_correct_one_eth(self):
        for quote in self.state["router_quotes"]:
            if int(quote["amount_in"]) == AMOUNT_IN:
                continue
            got = score_wire(QUESTION, json.dumps({"amount_out": int(quote["amounts"][-1])}), self.oracle)
            self.assertEqual(got["status"], "bounded_quote_score")
            self.assertFalse(got["exact_amount"])

    def test_one_unit_error_never_receives_exact_score(self):
        for offset in (-1, 1):
            got = score_wire(QUESTION, str(self.oracle.best_amount + offset), self.oracle)
            self.assertEqual(got["score_ppm"], 999999)
            self.assertFalse(got["exact_amount"])

    def test_slippage_floor_is_not_gross_output(self):
        q = self.oracle.ties[0]
        self.assertLess(q["minimum_out"], q["amount_out"])
        self.assertFalse(score_wire(QUESTION, str(q["minimum_out"]), self.oracle)["exact_amount"])

    def test_unit_and_shape_failures(self):
        for body in ('{"amount_out":1799.488639}', '{"amount_out":true}',
                     '{"amount_out":"1799488639"}', '{"amount_out":NaN}',
                     '{"amount_out":1e1000}', '{"amount_out":1799488639,"amount_out":1}',
                     '{"amount_out":null}', '[]', '"1799488639"'):
            with self.subTest(body=body):
                self.assertEqual(score_wire(QUESTION, body, self.oracle)["status"], "refused")

    def test_native_integral_exponents_remain_valid_numbers(self):
        body = '{"amount_out":' + str(self.oracle.best_amount) + '.0}'
        self.assertTrue(score_wire(QUESTION, body, self.oracle)["exact_amount"])

    def test_ceiling_and_deep_json(self):
        self.assertEqual(score_wire(QUESTION, " " * 16385, self.oracle)["status"], "refused")
        self.assertEqual(score_wire(QUESTION, "[" * 2000 + "]" * 2000, self.oracle)["status"], "refused")
        self.assertEqual(score_wire(QUESTION, '{"amount_out":1e999999999999999999999999}', self.oracle)["status"], "refused")
        self.assertEqual(score_wire(QUESTION, "\ud800", self.oracle)["status"], "refused")

    def test_route_complete_identity(self):
        identity = self.oracle.route_identity(self.oracle.ties[0])
        body = json.dumps({"amount_out": self.oracle.best_amount, "route": identity})
        self.assertTrue(score_wire(QUESTION, body, self.oracle)["route_identity_checked"])
        identity["tokens"] = ["0x" + t[2:].upper() for t in identity["tokens"]]
        body = json.dumps({"amount_out": self.oracle.best_amount, "route": identity})
        self.assertEqual(score_wire(QUESTION, body, self.oracle)["status"], "bounded_quote_score")

    def test_route_each_decisive_field_is_bound(self):
        route = self.oracle.route_identity(self.oracle.ties[0])
        mutations = {"chain_id": 11155111, "block_hash": "0x" + "0" * 64,
                     "amount_in_wei": str(10 ** 17), "trade_type": "exact_output",
                     "usdc_decimals": 18, "objective": "max_net_after_gas", "fees_ppm": [500],
                     "tokens": [self.oracle.usdc, self.oracle.weth],
                     "pools": ["0x" + "1" * 40]}
        for key, value in mutations.items():
            changed = copy.deepcopy(route)
            changed[key] = value
            got = score_wire(QUESTION, json.dumps({"amount_out": self.oracle.best_amount, "route": changed}), self.oracle)
            self.assertEqual(got["status"], "refused", key)
        incomplete = copy.deepcopy(route)
        del incomplete["fees_ppm"]
        self.assertEqual(score_wire(QUESTION, json.dumps({"amount_out": self.oracle.best_amount, "route": incomplete}), self.oracle)["status"], "refused")

    def test_amount_optimum_does_not_excuse_losing_route(self):
        losing = min(self.oracle.quotes, key=lambda q: q["amount_out"])
        self.assertLess(losing["amount_out"], self.oracle.best_amount)
        body = json.dumps({"amount_out": self.oracle.best_amount, "route": self.oracle.route_identity(losing)})
        self.assertEqual(score_wire(QUESTION, body, self.oracle)["status"], "refused")

    def test_tied_routes_all_accepted_on_constructed_state(self):
        # Constructed arithmetic boundary, not an observed mainnet state or preference.
        state = copy.deepcopy(self.state)
        target = 10 ** 9
        intermediate = self.oracle.quote(self.oracle.paths[1], AMOUNT_IN)["amounts"][1]
        for idx, amount in ((0, AMOUNT_IN), (2, intermediate)):
            pool = state["pools"][idx]
            rin = int(pool["reserve1"] if idx == 0 else pool["reserve0"])
            numerator = target * (rin * 1000 + amount * 997)
            rout = (numerator + amount * 997 - 1) // (amount * 997)
            pool["reserve0" if idx == 0 else "reserve1"] = str(rout)
        tied = Oracle(self.policy, state)
        self.assertEqual([q["amount_out"] for q in tied.quotes], [target, target])
        self.assertEqual(len(tied.ties), 2)
        for q in tied.ties:
            tied.check_route(tied.route_identity(q))

    def test_pool_and_policy_failure_controls(self):
        for mutation in ("duplicate", "zero", "fee", "token"):
            state = copy.deepcopy(self.state)
            if mutation == "duplicate":
                state["pools"][1] = copy.deepcopy(state["pools"][0])
            elif mutation == "zero":
                state["pools"][0]["reserve0"] = "0"
            elif mutation == "fee":
                state["pools"][0]["fee_ppm"] = 500
            else:
                state["pools"][0]["token0"] = "0x" + "1" * 40
            with self.assertRaises(Refused):
                Oracle(self.policy, state)
        policy = copy.deepcopy(self.policy)
        policy["objective"] = "max_net_after_gas"
        with self.assertRaises(Refused):
            Oracle(policy, self.state)

    def test_normalized_overlap_includes_identical_states_at_distinct_blocks(self):
        data = json.loads((HERE / "evidence/STATE.json").read_text())
        self.assertNotEqual(data["states"][0]["block_hash"], data["states"][1]["block_hash"])
        self.assertEqual(state_identity(data["states"][0]), state_identity(data["states"][1]))

    def test_mutated_snapshot_fails_router_replay(self):
        state = copy.deepcopy(self.state)
        state["pools"][0]["reserve0"] = str(int(state["pools"][0]["reserve0"]) + 1000000)
        with self.assertRaises(Refused):
            Oracle(self.policy, state).verify_router_observations()

    def test_broken_fee_implementation_fails_primary_check(self):
        def no_fee(amount, rin, rout, fee):
            return int(amount) * int(rout) // (int(rin) + int(amount))
        with patch("oracle.amount_out", no_fee):
            with self.assertRaises(Refused):
                Oracle(self.policy, self.state).verify_router_observations()

    def test_pinned_artifact_failure(self):
        with tempfile.TemporaryDirectory(prefix="route-owner-") as directory:
            path = Path(directory) / "state.json"
            raw = b'{"states": []}'
            path.write_bytes(raw)
            expected = hashlib.sha256(raw).hexdigest()
            self.assertEqual(load_trusted(path, expected), {"states": []})
            path.write_bytes(raw + b" ")
            with self.assertRaises(Refused):
                load_trusted(path, expected)

    def test_properties_integer_conservation_and_monotonicity(self):
        rng = random.Random(20261119)
        for _ in range(1000):
            rin, rout, amount = rng.randrange(10 ** 6, 10 ** 20), rng.randrange(10 ** 6, 10 ** 20), rng.randrange(10 ** 6, 10 ** 19)
            out = amount_out(amount, rin, rout)
            self.assertGreater(out, 0)
            self.assertLess(out, rout)
            self.assertLessEqual(out, amount_out(amount + 1, rin, rout))
            self.assertGreaterEqual((rin * 1000 + amount * 997) * (rout - out), rin * rout * 1000)
            self.assertLess((rin * 1000 + amount * 997) * (rout - out - 1), rin * rout * 1000)

    def test_sdk_licensed_arithmetic_fixture(self):
        # MIT upstream pair.test.ts, no tax test: reserves 10000/10000, in 100, out 98.
        self.assertEqual(amount_out(100, 10000, 10000), 98)


class TransportTests(unittest.TestCase):
    def test_abi_roundtrip(self):
        self.assertEqual(words("0x" + word(1) + word(2)), [1, 2])
        self.assertEqual(address("0x" + word(1)), "0x" + "0" * 39 + "1")

    def test_rpc_reorders_only_by_exact_integer_id(self):
        raw = json.dumps([{"jsonrpc": "2.0", "id": 2, "result": "b"}, {"jsonrpc": "2.0", "id": 1, "result": "a"}])
        self.assertEqual(decode_batch(raw, 2), ["a", "b"])

    def test_rpc_foreign_duplicate_error_and_string_id_refused(self):
        for row in ({"jsonrpc": "2.0", "id": "1", "result": "a"},
                    {"jsonrpc": "2.0", "id": 2, "result": "a"},
                    {"jsonrpc": "2.0", "id": 1, "result": "a", "error": None}):
            with self.assertRaises(ValueError):
                decode_batch(json.dumps([row]), 1)
        with self.assertRaises(ValueError):
            decode_batch(json.dumps([{"jsonrpc": "2.0", "id": 1, "result": "a"}] * 2), 2)

    def test_no_writes_even_if_requested(self):
        rpc = RPC("https://ethereum-rpc.publicnode.com")
        for method in ("eth_sendTransaction", "eth_sendRawTransaction", "personal_sign", "eth_estimateGas"):
            with self.assertRaises(ValueError):
                rpc.batch([(method, [])])
        with self.assertRaises(ValueError):
            RPC("https://example.com")


if __name__ == "__main__":
    unittest.main()
