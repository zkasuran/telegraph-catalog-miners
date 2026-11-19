# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Standalone local contracts and source-only verification. No neural imports."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import yaml

from oracle import QUESTION, Oracle

HERE = Path(__file__).resolve().parent
WORK = HERE / ".gate-work"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def run(args, name, expected=0, env=None):
    result = subprocess.run(args, capture_output=True, text=True, timeout=100, env=env, cwd=HERE)
    (WORK / name).write_text(result.stdout + result.stderr)
    if result.returncode != expected:
        raise ValueError(f"{name} exited {result.returncode}, expected {expected}")
    return result.stdout


def workflow_check(path):
    data = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    if set(data["on"]) != {"workflow_dispatch"} or data["permissions"] != {"contents": "read"}:
        raise ValueError("workflow trigger/permissions mismatch")
    if len(data["jobs"]) != 1:
        raise ValueError("source job must be bounded to one public CPU runner")
    job = data["jobs"]["source"]
    if job["runs-on"] != "ubuntu-24.04" or not 1 <= int(job["timeout-minutes"]) <= 20:
        raise ValueError("source runner/timeout bound")
    if data["concurrency"]["cancel-in-progress"] != "false":
        raise ValueError("healthy source run cancellation forbidden")
    for step in job["steps"]:
        if not 1 <= int(step["timeout-minutes"]) <= 10:
            raise ValueError("step timeout missing or over limit")
        if "uses" in step and not re.fullmatch(r"actions/[a-z-]+@[0-9a-f]{40}", step["uses"]):
            raise ValueError("action pin absent")
    if data["on"]["workflow_dispatch"]["inputs"]["mode"]["options"] != ["source_verify"]:
        raise ValueError("source pipeline cannot dispatch neural work")
    uploader = [s for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact@")]
    if (len(uploader) != 1 or uploader[0]["with"].get("path") != "verification/owners/optimal_execution_route/result/"
            or uploader[0]["with"].get("include-hidden-files", "false") != "false"
            or uploader[0]["with"].get("if-no-files-found") != "error"
            or uploader[0].get("if") != "always() && steps.source_proofs.outcome == 'success'"):
        raise ValueError("upload must use the closed non-hidden source-result staging directory")
    return {"jobs": 1, "runner": job["runs-on"], "timeout_minutes": int(job["timeout-minutes"])}


def source_lock_check(lock, gate):
    required = (
        "f742a7dbd0d9cb9ea41e9a24a918996e8170c799fa528688d40dd582c8265f4f",
        "ba1cc08a7ccde2d2ec775841541641e4548226580ab850948cbfda66a1befcdc",
        "b8bb0864c5a28024fac8a632c443c87c5aa6f215c0b126c449ae1a150412f31d",
    )
    if "pyflakes==3.4.0" not in lock or "PyYAML==6.0.3" not in lock:
        raise ValueError("reviewed passive tool versions changed")
    if any("--hash=sha256:" + value not in lock for value in required) or "--require-hashes" not in gate:
        raise ValueError("passive tool hash enforcement or supported wheel coverage missing")
    return {"pyflakes": "3.4.0", "pyyaml": "6.0.3", "python_wheels": ["cp311-linux-x86_64", "cp312-linux-x86_64"]}


def primary_wire():
    raw = (HERE / "primary/onchain.go.txt").read_text()
    function = re.search(r"func InputsQuestion\(.*?\n}\n", raw, re.S)
    executor = (HERE / "primary/adapterexec.go.txt").read_text()
    handler = re.search(r"func handler\(.*?\n}\n", executor, re.S)
    types = re.search(r"type tokensItem struct.*?\n}\n\n.*?type batchRequest struct.*?\n}\n", executor, re.S)
    if not function or not handler or not types:
        raise ValueError("primary source expressions missing")
    source = '''// SPDX-License-Identifier: MIT
// Copyright (c) 2026 telegraphprotocol
// Unchanged primary functions, with explicit source-only mock plumbing.
package main
import("fmt"; "strings"; "strconv"; "encoding/json"; "os"; "context"; "net/http"; "net/http/httptest"; "sync"; "time"; "log")
type Input struct { Name string `json:"name"`; Description string `json:"description"`; Unit string `json:"unit"` }
type Format struct { Inputs []Input }
type KV struct { Key string; Value any }
type OrderedObject []KV
func (o OrderedObject) Get(key string) (any,bool) { for _,kv := range o { if kv.Key == key { return kv.Value,true } }; return nil,false }
type identity struct { FoundationHash string; AdapterHash string; TokenizerHash string }
type mockScore uint32
func (s mockScore) Bits() uint32 { return uint32(s) }
func (s mockScore) Hex() string { return fmt.Sprintf("0x%08x", uint32(s)) }
type scorer interface { ScoreTokens(context.Context, []int64) (mockScore,error); Identity() identity }
type spy struct { Calls [][]int64 }
func (s *spy) ScoreTokens(_ context.Context, t []int64) (mockScore,error) { s.Calls=append(s.Calls,append([]int64{},t...)); return mockScore(1065353216),nil }
func (s *spy) Identity() identity { return identity{"f","a","t"} }
var remote = struct { BatchInputRoot func([][]int64) string }{func(t [][]int64) string { return "token-root-mock-not-chain-state" }}
func writeJSON(w http.ResponseWriter, status int, body any) { w.Header().Set("Content-Type","application/json"); w.WriteHeader(status); _=json.NewEncoder(w).Encode(body) }
'''
    source += function.group(0) + types.group(0) + handler.group(0)
    source += '''
func main() {
 raw,e:=os.ReadFile(os.Args[1]); if e!=nil { fmt.Fprintln(os.Stderr,e); os.Exit(2) }
 var d struct { IntentID string `json:"intent_id"`; Request struct { Params []Input `json:"params"` } `json:"request"` }
 if e:=json.Unmarshal(raw,&d); e!=nil { fmt.Fprintln(os.Stderr,e); os.Exit(2) }
 question:=InputsQuestion(d.IntentID,&Format{Inputs:d.Request.Params},OrderedObject{{"pair","ETH/USDC"}})
 s:=&spy{}; h:=handler(s)
 bodies:=[]string{`{"batch":[{"tokens":[1,2,3],"block_hash":"A","reference_state":101}],"input_root":"claimed-chain-root"}`,`{"batch":[{"tokens":[1,2,3],"block_hash":"B","reference_state":202}],"input_root":"other-chain-root"}`}
 var replies []json.RawMessage
 for _,b:=range bodies { w:=httptest.NewRecorder(); h(w,httptest.NewRequest("POST","/runnable",strings.NewReader(b))); if w.Code!=200 { os.Exit(2) }; replies=append(replies,json.RawMessage(w.Body.Bytes())) }
 if len(s.Calls)!=2 || fmt.Sprint(s.Calls[0])!=fmt.Sprint(s.Calls[1]) { os.Exit(2) }
 _=json.NewEncoder(os.Stdout).Encode(map[string]any{"question":question,"calls":s.Calls,"replies":replies,"kind":"source-only primary handler mock, not reference engine"})
}
'''
    program = WORK / "primary-wire.go"
    program.write_text(source)
    go = os.environ.get("ROUTE_GO_BINARY") or shutil.which("go")
    if not go:
        cached = Path.home() / "go/pkg/mod/golang.org/toolchain@v0.0.1-go1.25.7.linux-amd64/bin/go"
        if cached.is_file():
            go = str(cached)
    if not go:
        raise ValueError("Go 1.25.7 is required for source compilation; use ROUTE_GO_BINARY")
    env = {"PATH": "/usr/bin:/bin", "HOME": str(WORK), "GOTOOLCHAIN": "local", "GOPROXY": "off",
           "GOSUMDB": "off", "GO111MODULE": "off", "GOCACHE": str(WORK / "go-cache")}
    version = run([go, "version"], "go-version.log", env=env).strip()
    run([go, "build", "-trimpath", "-o", str(WORK / "primary-wire"), str(program)], "go-build.log", env=env)
    result = json.loads(run([str(WORK / "primary-wire"), str(HERE / "primary/live-task.json")], "primary-wire.log", env=env))
    if result["question"] != QUESTION or result["replies"][0] != result["replies"][1]:
        raise ValueError("literal question/token-only contract changed")
    render_ns = {}
    exec(compile((HERE / "primary/render_fragment.py.txt").read_text(), "primary-render-only", "exec"), render_ns)
    if render_ns["render"](QUESTION, "101") != "### Task:\n" + QUESTION + "\n\n### Answer:\n101":
        raise ValueError("literal primary renderer mismatch")
    result["go_version"] = version
    result["unchanged_functions_sha256"] = {"question": sha(function.group(0).encode()), "handler": sha(handler.group(0).encode())}
    return result


def verify():
    WORK.mkdir(exist_ok=True)
    manifest = json.loads((HERE / "source-manifest.json").read_text())
    for row in manifest["records"] + manifest["fixtures"]:
        if sha((HERE / row["path"]).read_bytes()) != row["sha256"]:
            raise ValueError("source/fixture hash mismatch")
    for path in HERE.glob("*.py"):
        compile(path.read_text(), str(path), "exec")
    for path in HERE.rglob("*.json"):
        if ".gate-work" not in path.parts and "result" not in path.parts:
            json.loads(path.read_text())
    lock = source_lock_check((HERE / "requirements-source.lock").read_text(), (HERE / "verify.sh").read_text())
    run([sys.executable, "-m", "pyflakes", *map(str, sorted(HERE.glob("*.py")))], "lint.log")
    workflow = HERE.parents[2] / ".github/workflows/evaluator-owner-optimal_execution_route.yml"
    workflow_result = workflow_check(workflow)
    wire = primary_wire()
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_*.py")
    with (WORK / "tests.log").open("w") as stream:
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise ValueError("source test suite failed; inspect .gate-work/tests.log")
    data = json.loads((HERE / "evidence/STATE.json").read_text())
    policy = json.loads((HERE / "policy.json").read_text())
    oracle = Oracle(policy, data["states"][0])
    args = [sys.executable, "-W", "error", str(HERE / "oracle.py"), "--state", str(HERE / "evidence/STATE.json"),
            "--expected-sha256", sha((HERE / "evidence/STATE.json").read_bytes()), "--policy", str(HERE / "policy.json"),
            "--policy-sha256", sha((HERE / "policy.json").read_bytes())]
    run(args + ["--body", str(oracle.best_amount)], "primary-smoke.log")
    run(args + ["--body", str(oracle.best_amount + 1)], "tamper-smoke.log", expected=2)
    blocked = json.loads(run([sys.executable, "-W", "error", str(HERE / "qualification_gate.py"), "--require-ready"], "qualification-negative.log", expected=2))
    report = {"schema_version": 1, "family": "OPTIMAL_EXECUTION_ROUTE", "status": "SOURCE_GATE_PASSED",
              "tests": tests.testsRun, "python_sources_compiled": len(list(HERE.glob("*.py"))),
              "primary_wire": wire, "workflow": workflow_result, "qualification": blocked,
              "source_tool_lock": lock,
              "source_manifest_sha256": sha((HERE / "source-manifest.json").read_bytes()),
              "implementation_sha256": {p.name: sha(p.read_bytes()) for p in sorted(HERE.glob("*.py"))},
              "no_model_operations": True, "registration_ready": False}
    (WORK / "source-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "tests": tests.testsRun, "registration_ready": False}, sort_keys=True))
    return report


if __name__ == "__main__":
    verify()
