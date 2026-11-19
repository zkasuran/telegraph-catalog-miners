# SPDX-License-Identifier: LicenseRef-zkasuran-SAND-1.0
# Copyright (c) 2026 zkasuran
import os
import signal
import subprocess
from wire_common import HOST

def ssh_command(root, row, port):
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("invalid local proxy port")
    return ["ssh", "-F", "/dev/null", "-N", "-T", "-i", str(root / "forward-key"),
            "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes",
            "-o", "UserKnownHostsFile=" + str(root / "known_hosts"),
            "-o", "GlobalKnownHostsFile=/dev/null",
            "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=20",
            "-o", "ServerAliveCountMax=3", "-o", "ForwardAgent=no", "-o", "ForwardX11=no",
            "-o", "PermitLocalCommand=no", "-o", "RequestTTY=no", "-o", "LogLevel=ERROR",
            "-R", f"127.0.0.1:{row['forward_port']}:127.0.0.1:{port}", "telegraph-verify@" + HOST]

def terminate(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
