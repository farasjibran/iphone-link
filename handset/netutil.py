"""Port helpers shared by the stream servers.

Finding out which process holds a port is platform-specific: Windows needs
``netstat -ano`` plus ``taskkill``, macOS needs ``lsof`` plus ``kill``. Both
paths live here so the callers stay platform-agnostic.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from typing import Optional, Set

logger = logging.getLogger(__name__)


def port_is_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True when something accepts connections on ``host:port``."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((host, port)) == 0


def _pids_holding_port_lsof(port: int) -> Set[int]:
    """Return the PIDs listening on ``port`` according to lsof."""
    pids: Set[int] = set()
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return pids
    if result.returncode != 0:
        return pids
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.add(int(line))
    return pids


def _pids_holding_port_netstat(port: int) -> Set[int]:
    """Return the PIDs listening on ``port`` according to Windows netstat."""
    pids: Set[int] = set()
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return pids
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if f":{port} " not in stripped or "LISTENING" not in stripped:
            continue
        parts = stripped.split()
        if len(parts) >= 5:
            try:
                pids.add(int(parts[-1]))
            except ValueError:
                continue
    return pids


def pids_holding_port(port: int) -> Set[int]:
    """Return the PIDs listening on ``port`` on this platform."""
    if sys.platform == "win32":
        return _pids_holding_port_netstat(port)
    if sys.platform == "darwin":
        return _pids_holding_port_lsof(port)
    return set()


def _terminate_pid(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True,
            timeout=5,
        )
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as exc:
        logger.debug("Could not terminate PID %s: %s", pid, exc)


def kill_orphan_port_holders(port: int, host: str = "127.0.0.1") -> bool:
    """Terminate other processes listening on ``port``.

    The current process is never killed. Returns True when at least one
    foreign process was terminated.
    """
    my_pid = os.getpid()
    killed = False
    for pid in pids_holding_port(port):
        if pid <= 0 or pid == my_pid:
            continue
        logger.warning("Releasing port %d by terminating orphan PID %d", port, pid)
        _terminate_pid(pid)
        killed = True
    return killed


def find_free_port(start: int, end: int, host: str = "127.0.0.1") -> Optional[int]:
    """Return the first free port in ``[start, end]``, or ``None``."""
    import socket

    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
            except OSError:
                continue
            return port
    return None
