# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
"""Hash-bound remote downloads and the real pinned reference executor."""
import http.client
import json
import os
import socket
import subprocess
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

from contracts import BASE, FAMILY, FOUNDATION, KIT, KIT_SHA, RELEASE, literal_response, remote_only, sha, write
from proxy import Backend, BoundedServer, identity, input_root

HERE = Path(__file__).resolve().parent


def fetch(url,path,expected,ceiling=1 << 30):
    remote_only()
    if path.is_file() and path.stat().st_size <= ceiling and sha(path.read_bytes()) == expected:
        return
    partial = path.with_suffix(".partial")
    started,total = time.monotonic(),0
    import hashlib
    digest = hashlib.sha256()
    request = urllib.request.Request(url,headers={"User-Agent":"Telegraph-document-owner/1.0"})
    try:
        with urllib.request.urlopen(request,timeout=30) as response, partial.open("wb") as out:
            if response.status != 200 or not response.url.startswith("https://"):
                raise ValueError("public transport or status changed")
            while chunk := response.read(1 << 20):
                total += len(chunk)
                if total > ceiling or time.monotonic()-started > 600:
                    raise ValueError("download size or deadline")
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("download hash differs")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def prepare(root,model=True):
    remote_only()
    root.mkdir(exist_ok=True)
    archive = root/"kit.tar.gz"
    fetch("https://codeload.github.com/telegraphprotocol/telegraph-wasm-baseline/tar.gz/"+KIT,archive,KIT_SHA,32 << 20)
    with tarfile.open(archive) as source:
        if len(source.getmembers()) > 5000 or sum(m.size for m in source.getmembers()) > 128 << 20:
            raise ValueError("bounded kit archive")
        source.extractall(root,filter="data")
    kit = root/("telegraph-wasm-baseline-"+KIT)
    base = root/"base"
    base.mkdir(exist_ok=True)
    for name,expected in BASE["files"].items():
        if name != "model.safetensors" or model:
            fetch(f"https://huggingface.co/{BASE['id']}/resolve/{BASE['revision']}/{name}",base/name,expected,700 << 20)
    fetch(RELEASE+"tokenizer.json",base/"tokenizer.json",FOUNDATION["tokenizer.json"],2 << 20)
    return kit,base


def compile_runtime(root,kit):
    for name,source in (("adapterexec","./cmd/adapterexec"),("stagecheck",str(HERE/"stagecheck.go"))):
        subprocess.run(["go","build","-mod=readonly","-trimpath","-o",str(root/name),source],cwd=kit,check=True,timeout=600)
    subprocess.run(["go","test","-mod=readonly","-count=1","-v",str(HERE/"stagecheck.go"),str(HERE/"stagecheck_test.go")],cwd=kit,check=True,timeout=180)
    foundation = root/"foundation"
    foundation.mkdir(exist_ok=True)
    for name,expected in FOUNDATION.items():
        fetch(RELEASE+name,foundation/name,expected,1 << 30)


def full_parser(root,package,config):
    remote_only()
    hidden,intermediate = config["hidden_size"],config["intermediate_size"]
    layout = {"foundation_hash":FOUNDATION["foundation_lora_fp32.onnx"],"foundation_version":"modernbert-base-lora16-v1",
              "tokenizer_hash":FOUNDATION["tokenizer.json"],"r_max":16,"num_layers":config["num_hidden_layers"],
              "hidden_size":hidden,"targets":{"attn.Wqkv":[hidden,3*hidden],"attn.Wo":[hidden,hidden],
                                               "mlp.Wi":[hidden,2*intermediate],"mlp.Wo":[intermediate,hidden]}}
    write(root/"foundation.json",layout)
    from datetime import datetime, timezone
    snapshot = {"schema":1,"fetched_at":datetime.now(timezone.utc).isoformat()}
    for key,url in (("family","https://devnode.telegraphprotocol.com/api/wasm?resolve="+FAMILY),
                    ("child","https://devnode.telegraphprotocol.com/engine/v1/intents/"+FAMILY+"_GENERAL")):
        request = urllib.request.Request(url,headers={"User-Agent":"Telegraph-document-owner/1.0"})
        with urllib.request.urlopen(request,timeout=25) as response:
            raw = response.read((1 << 20)+1)
            if response.status != 200 or len(raw) > 1 << 20:
                raise ValueError("live class snapshot failed")
        snapshot[key] = {"url":url,"raw":raw.decode()}
    write(root/"class-snapshot.json",snapshot)
    command = [str(root/"stagecheck"),"--intent",FAMILY,"--child",FAMILY+"_GENERAL","--expected-class","judgment",
               "--package",str(package),"--sha256",sha(package.read_bytes()),"--foundation",str(root/"foundation.json"),
               "--foundation-sha256",sha((root/"foundation.json").read_bytes()),"--class-snapshot",str(root/"class-snapshot.json"),
               "--snapshot-sha256",sha((root/"class-snapshot.json").read_bytes()),"--out",str(root)]
    result = subprocess.run(command,capture_output=True,text=True,timeout=60)
    (root/"parser.log").write_text(result.stdout+result.stderr)
    if result.returncode != 0:
        raise ValueError("actual full Stage 1 parser failed")
    manifest = json.loads(result.stdout)
    if manifest["sha256"] != sha(package.read_bytes()) or manifest["manifest"]["verification"] != {"stage1":True,"loaded":False}:
        raise ValueError("full parser bytes or status differ")
    write(root/"package-manifest.json",manifest)
    return manifest


class Reference:
    def __init__(self,root,package,keccak):
        remote_only()
        self.root,self.package = root,package
        self.row = {"keccak":keccak}
        self.process,self.server,self.log = None,None,None

    def __enter__(self):
        try:
            return self.start()
        except Exception:
            self.__exit__()
            raise

    def start(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1",0))
            port = probe.getsockname()[1]
        environment = {"PATH":os.environ["PATH"],"HOME":str(self.root),"FOUNDATION_POOL_SIZE":"1",
                       "FOUNDATION_MODEL_PATH":str(self.root/"foundation/foundation_lora_fp32.onnx"),
                       "FOUNDATION_MODEL_HASH":FOUNDATION["foundation_lora_fp32.onnx"],
                       "FOUNDATION_ENGINE_WASM_PATH":str(self.root/"foundation/foundation_engine.wasm"),
                       "FOUNDATION_ENGINE_WASM_HASH":FOUNDATION["foundation_engine.wasm"]}
        self.log = (self.root/"reference.log").open("a")
        self.process = subprocess.Popen([str(self.root/"adapterexec"),"--package",str(self.package),"--listen",f"127.0.0.1:{port}"],env=environment,stdout=self.log,stderr=self.log)
        deadline = time.monotonic()+180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("reference executor failed to load")
            try:
                with socket.create_connection(("127.0.0.1",port),timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("reference executor startup timeout")
        self.server = BoundedServer(("127.0.0.1",0),self.row,Backend(port))
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        return self

    def request(self,value):
        connection = http.client.HTTPConnection("127.0.0.1",self.server.server_port,timeout=600)
        try:
            connection.request("POST","/runnable",json.dumps(value,separators=(",",":")).encode(),{"Content-Type":"application/json"})
            response = connection.getresponse()
            raw = response.read((1 << 20)+1)
            if len(raw) > 1 << 20:
                raise ValueError("reference response ceiling")
            return response.status,json.loads(raw)
        finally:
            connection.close()

    def score(self,vectors):
        batch = [{"tokens":tokens} for tokens in vectors]
        request = {"batch":batch,**identity(self.row),"input_root":input_root(batch)}
        status,value = self.request(request)
        if status != 200:
            raise ValueError("reference scoring HTTP failed")
        return literal_response(value,request)

    def negative_controls(self):
        controls = 0
        for key in ("foundation_hash","adapter_hash","tokenizer_hash"):
            request = {"batch":[{"tokens":[50281,22,50282]}],**identity(self.row)}
            request[key] = "0"*64
            status,value = self.request(request)
            if status != 400 or value != {"error":"request refused"}:
                raise ValueError("actual negative identity control was accepted")
            controls += 1
        for tokens in ([],[-1],[50368],[True]):
            status,value = self.request({"batch":[{"tokens":tokens}],**identity(self.row)})
            if status != 400 or value != {"error":"request refused"}:
                raise ValueError("actual token refusal control failed")
        return controls

    def __exit__(self,*_):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log:
            self.log.close()
