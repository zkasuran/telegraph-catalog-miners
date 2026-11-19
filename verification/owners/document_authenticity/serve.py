# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""The identical public CPU executor with a restricted ephemeral lease only."""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from contracts import BASE, FAMILY, FOUNDATION, ready_contract, remote_only, sha, write
from lease_contract import DEFAULT_OWNER, HOST, HOST_KEY, lease_for
from remote_io import Reference, compile_runtime, fetch, full_parser, prepare


def config(value):
    required = {"package_url","package_sha256","package_keccak","report_url","report_sha256","verified_run_id","verified_commit","slot","forward_port","runnable_url"}
    if not isinstance(value,dict) or set(value) != required:
        raise ValueError("exact public serving config required")
    for key in ("package_sha256","package_keccak","report_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}",value[key]):
            raise ValueError("invalid serving hash")
    for key in ("package_url","report_url"):
        if not re.fullmatch(r"https://raw\.githubusercontent\.com/zkasuran/telegraph-catalog-miners/[0-9a-f]{40}/verification/owners/document_authenticity/[A-Za-z0-9_./-]+",value[key]) or ".." in value[key].split("/"):
            raise ValueError("immutable own package/report URLs required")
    if type(value["slot"]) is not int or not 1 <= value["slot"] <= 6 or value["forward_port"] != 48600+value["slot"]:
        raise ValueError("existing proxy reservation mismatch")
    if type(value["verified_run_id"]) is not int or not re.fullmatch(r"[0-9a-f]{40}",value["verified_commit"]):
        raise ValueError("real quality run and source commit required")
    if not re.fullmatch(r"https://tga1\.zkasuran\.dev/_s2o_document_authenticity_[0-9a-f]{8}/runnable",value["runnable_url"]):
        raise ValueError("existing proxy namespace required")
    return value


def prepare_serving(root,value):
    remote_only()
    value = config(value)
    root.mkdir(exist_ok=True)
    if (root/"forward-key").exists():
        raise ValueError("unresolved ephemeral serving lease already exists")
    package = root/(FAMILY+".safetensors")
    fetch(value["package_url"],package,value["package_sha256"],64 << 20)
    report_path = root/"registration-ready.json"
    fetch(value["report_url"],report_path,value["report_sha256"],1 << 20)
    report = ready_contract(json.loads(report_path.read_text()))
    if report["package_keccak"] != value["package_keccak"] or report["package_sha256"] != value["package_sha256"] or report["run_id"] != value["verified_run_id"] or report["commit"] != value["verified_commit"] or report["serving_commit"] != os.environ["GITHUB_SHA"]:
        raise ValueError("serving run is not the exact measured source commit")
    kit,base = prepare(root,model=False)
    compile_runtime(root,kit)
    parsed = full_parser(root,package,json.loads((base/"config.json").read_text()))
    if parsed["keccak256"] != "0x"+value["package_keccak"]:
        raise ValueError("public package differs from actual parsed bytes")
    key = root/"forward-key"
    subprocess.run(["ssh-keygen","-q","-t","ed25519","-N","","-f",str(key),"-C","document-owner-lease"],check=True,timeout=15)
    key.chmod(0o600)
    (root/"known_hosts").write_text(HOST_KEY)
    row = {"family":FAMILY,"owner":DEFAULT_OWNER,"slot":value["slot"],"forward_port":value["forward_port"],
           "runnable_url":value["runnable_url"],"keccak":value["package_keccak"],"sha256":value["package_sha256"],"existing_registration_id":0}
    for kind in ("candidate","runtime","source"):
        row[kind+"_proof"] = {"sha256":value["report_sha256"]}
    lease = lease_for(row,(root/"forward-key.pub").read_text().strip(),"doc-"+os.environ["GITHUB_RUN_ID"],os.environ["GITHUB_RUN_ID"],os.environ["GITHUB_SHA"])
    public = root/"public-lease"
    public.mkdir(exist_ok=True)
    write(public/"lease.json",lease)
    write(root/"serving-config.json",value)
    print("PUBLIC_RESTRICTED_LEASE_READY",flush=True)


def serve(root):
    remote_only()
    value = config(json.loads((root/"serving-config.json").read_text()))
    report = ready_contract(json.loads((root/"registration-ready.json").read_text()))
    if sha((root/(FAMILY+".safetensors")).read_bytes()) != value["package_sha256"]:
        raise ValueError("serving byte identity changed")
    deadline = time.monotonic()+290*60
    forward = None
    try:
        with Reference(root,root/(FAMILY+".safetensors"),value["package_keccak"]) as runtime:
            runtime.negative_controls()
            vectors = report["runtime"]["vectors"]
            expected = [v["score_bits"] for v in vectors]
            for _ in range(2):
                if runtime.score([v["tokens"] for v in vectors]) != expected:
                    raise ValueError("serving differs from the original reference-engine proof")
            arguments = ["ssh","-F","/dev/null","-N","-T","-i",str(root/"forward-key"),
                         "-o","IdentitiesOnly=yes","-o","IdentityAgent=none","-o","BatchMode=yes","-o","ConnectTimeout=15",
                         "-o","StrictHostKeyChecking=yes","-o","UserKnownHostsFile="+str(root/"known_hosts"),
                         "-o","GlobalKnownHostsFile=/dev/null","-o","ExitOnForwardFailure=yes",
                         "-o","ForwardAgent=no","-o","ForwardX11=no","-o","ServerAliveInterval=20","-o","ServerAliveCountMax=3",
                         "-R",f"127.0.0.1:{value['forward_port']}:127.0.0.1:{runtime.server.server_port}","telegraph-verify@"+HOST]
            while time.monotonic() < deadline:
                if forward is None or forward.poll() is not None:
                    forward = subprocess.Popen(arguments,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                time.sleep(10)
                if runtime.process.poll() is not None or runtime.server.fatal.is_set():
                    raise ValueError("serving engine stopped or identity changed")
    finally:
        if forward and forward.poll() is None:
            forward.terminate()
            try:
                forward.wait(timeout=10)
            except subprocess.TimeoutExpired:
                forward.kill()
                forward.wait(timeout=5)
        for name in ("forward-key","forward-key.pub"):
            (root/name).unlink(missing_ok=True)


if __name__ == "__main__":
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("prepare","run"))
    parser.add_argument("--root",type=Path,required=True)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare_serving(args.root,json.loads(os.environ["SERVING_CONFIG"]))
    else:
        serve(args.root)
