#!/usr/bin/env python3
"""
Client for the swarm box's GPU gate (JenAIson's lease service at http://127.0.0.1:8686, box-local only).

Every run that touches a card on the box holds a lease from the gate for its duration. The contract, as JenAIson
gave it on 2026-10-17 (the full text is API.md in the Holder's folder):

  POST /api/request   {"agent", "greeting", "task", "modality", "engine", "model_source", "model_format", "model_path",
                       "vram_mb", "minutes", "exclusive"}  ->  {"ok", "request_id", "status": "granted"|"queued",
                       "lease_id", "gpu", "position"}; a rejection is {"ok": false, "error": ...} with HTTP 200, so
                       every reply is checked for ok. The greeting must name JenAIson and the Holder to keep full
                       priority; model_path must start with /mnt/model-storage/ (the gate does not check that the
                       file exists, so the caller does).
  GET  /api/request/<request_id>  ->  {"status": queued|granted|done|expired|failed|cancelled, "position", "lease_id",
                       "gpu", "lease_status"}; polled about every 30 s while queued.
  POST /api/heartbeat {"lease_id", "extend_min"}  about every 4 minutes; a lease ends after 10 minutes without one;
                       ok false means the lease is gone and the run must stop.
  POST /api/release   {"lease_id", "failed"}; failed true on error.
  POST /api/cancel    {"request_id"} for a request still queued.
  GET  /api/status    the full snapshot.

Card numbering: the gate's gpu index follows nvidia-smi (gpu0 = RTX 3060, gpu1 = RTX 4090 on the box); the engine's
cuda:N follows the CUDA runtime order, which the Swarm benchmarks document as cuda:0 = 4090, cuda:1 = 3060. A caller
that must address the leased card through the engine maps one to the other (see dory_e2e_run.py --device-map).

Usage, as a library:
    with GpuLease(agent="...", task="...", model_path="/mnt/model-storage/...") as lease:
        ...  # lease.gpu is the card; lease.check_alive() raises once a heartbeat reports the lease gone
    run_under_lease(cmd, lease)  # a child that is stopped the moment the lease is lost

Usage, as a command wrapper (any box-side script that needs a card):
    python3 gpu_gate.py --agent "..." --task "..." --model-path /mnt/model-storage/... -- <command> [args...]
The lease is requested and heartbeated around the command, which sees GATE_LEASE_ID and GATE_GPU in its environment; a
lost lease stops it, and the lease is released as failed when the command fails. Exit status is the command's.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8686"
MODEL_STORAGE_ROOT = "/mnt/model-storage/"
GREETING = "Hello JenAIson The Holder of the GPUs, thank you kindly for your time."
HEARTBEAT_SECONDS = 240   # the gate ends a lease after 10 minutes without one
POLL_SECONDS = 30
TERMINAL_STATUSES = ("done", "expired", "failed", "cancelled")


class GateError(RuntimeError):
    """A gate reply with ok false, an HTTP failure, or a request that ended without a grant."""


def call(base, method, path, body=None, timeout=20):
    """One gate call; returns the parsed JSON and raises GateError on HTTP failure or ok false."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            reply = json.loads(resp.read().decode() or "{}")
    except (OSError, ValueError) as e:  # URLError, timeouts, a truncated body, non-JSON or non-UTF-8 text
        raise GateError(f"{method} {path}: {e}") from e
    if not isinstance(reply, dict):
        raise GateError(f"{method} {path}: unexpected reply {str(reply)[:200]!r}")
    if reply.get("ok") is False:
        raise GateError(f"{method} {path}: {reply.get('error', 'rejected')}")
    return reply


class GpuLease:
    """Requests a lease, waits for the grant, heartbeats in a background thread while the caller uses the card, and
    releases on exit (failed=True when the block raised). Reusable by any script that runs on the box."""

    def __init__(self, agent, task, model_path, *, base=DEFAULT_BASE, modality="text", engine="swarmui",
                 model_source="raid-local", model_format="gguf", vram_mb=6000, minutes=90, exclusive=False,
                 greeting=GREETING, grant_timeout_s=3600, log=print):
        if not model_path.startswith(MODEL_STORAGE_ROOT):
            raise ValueError(f"model_path must start with {MODEL_STORAGE_ROOT}: {model_path}")
        self.base = base
        self.body = {"agent": agent, "greeting": greeting, "task": task, "modality": modality, "engine": engine,
                     "model_source": model_source, "model_format": model_format, "model_path": model_path,
                     "vram_mb": vram_mb, "minutes": minutes, "exclusive": exclusive}
        self.grant_timeout_s = grant_timeout_s
        self.log = log
        self.request_id = None
        self.lease_id = None
        self.gpu = None
        self.status = None
        self.lost = False
        self._stop = threading.Event()
        self._thread = None

    def request(self):
        reply = call(self.base, "POST", "/api/request", self.body)
        self.request_id = reply.get("request_id")
        self.lease_id = reply.get("lease_id")
        self.gpu = reply.get("gpu")
        self.status = reply.get("status")
        self.log(f"[gate] request {self.request_id}: {self.status}"
                 + (f" (position {reply.get('position')})" if self.status == "queued" else f" on gpu {self.gpu}"))
        return reply

    def wait_granted(self):
        """Polls the request until granted; raises GateError when it ends any other way or the wait times out."""
        deadline = time.monotonic() + self.grant_timeout_s
        while self.status != "granted":
            if time.monotonic() > deadline:
                self.cancel()
                raise GateError(f"request {self.request_id} not granted within {self.grant_timeout_s} s")
            time.sleep(POLL_SECONDS)
            reply = call(self.base, "GET", f"/api/request/{self.request_id}")
            self.status = reply.get("status")
            if self.status in TERMINAL_STATUSES:
                raise GateError(f"request {self.request_id} ended as {self.status} before a grant")
            self.lease_id = reply.get("lease_id") or self.lease_id
            self.gpu = reply.get("gpu") if reply.get("gpu") is not None else self.gpu
            self.log(f"[gate] request {self.request_id}: {self.status}"
                     + (f" (position {reply.get('position')})" if self.status == "queued" else f" on gpu {self.gpu}"))
        return {"lease_id": self.lease_id, "gpu": self.gpu}

    def heartbeat(self, extend_min=0):
        try:
            call(self.base, "POST", "/api/heartbeat", {"lease_id": self.lease_id, "extend_min": extend_min})
        except GateError as e:
            self.lost = True
            self.log(f"[gate] heartbeat failed, lease gone: {e}")

    def _heartbeat_loop(self):
        while not self._stop.wait(HEARTBEAT_SECONDS):
            try:
                self.heartbeat()
            except Exception as e:  # noqa: BLE001 - a heartbeat that cannot be confirmed counts as a lost lease
                self.lost = True
                self.log(f"[gate] heartbeat failed unexpectedly, treating the lease as gone: {e!r}")
            if self.lost:
                return

    def start_heartbeat(self):
        self._thread = threading.Thread(target=self._heartbeat_loop, name="gate-heartbeat", daemon=True)
        self._thread.start()

    def check_alive(self):
        if self.lost:
            raise GateError(f"lease {self.lease_id} is gone (heartbeat refused); stop the run")

    def cancel(self):
        """Withdraws a request that is still queued, so an aborted acquisition never receives a card later."""
        if self.request_id and self.status == "queued":
            try:
                call(self.base, "POST", "/api/cancel", {"request_id": self.request_id})
                self.status = "cancelled"
                self.log(f"[gate] cancelled request {self.request_id}")
            except GateError as e:
                self.log(f"[gate] cancel failed: {e}")

    def release(self, failed=False):
        """Ends the heartbeat and releases the lease; a request still queued is cancelled instead."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.status == "queued":
            self.cancel()
            return
        if self.lease_id and not self.lost:
            try:
                call(self.base, "POST", "/api/release", {"lease_id": self.lease_id, "failed": failed})
                self.log(f"[gate] released lease {self.lease_id}" + (" (failed)" if failed else ""))
            except GateError as e:
                self.log(f"[gate] release failed: {e}")

    def __enter__(self):
        try:
            self.request()
            self.wait_granted()
        except BaseException:
            self.cancel()
            raise
        self.start_heartbeat()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release(failed=exc_type is not None)
        return False


def run_under_lease(cmd, lease, log_file=None, env=None, grace_s=10, poll_s=1.0):
    """Runs a command while the lease is held, polling every poll_s seconds: a lost lease terminates it (kill after
    grace_s) rather than letting it use the card until it finishes. The child's console goes to log_file when given,
    else to ours. lease may be None (nothing to lose). Returns (returncode, lease_lost)."""
    sink = open(log_file, "w") if log_file else None
    lost = False
    try:
        proc = subprocess.Popen(cmd, text=True, stdout=sink, stderr=subprocess.STDOUT if sink else None, env=env)
        while proc.poll() is None:
            if lease is not None and lease.lost:
                lost = True
                proc.terminate()
                try:
                    proc.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
            time.sleep(poll_s)
    finally:
        if sink is not None:
            sink.close()
    return proc.returncode, lost


def main(argv=None):
    global POLL_SECONDS, HEARTBEAT_SECONDS
    ap = argparse.ArgumentParser(description="Hold a GPU-gate lease around a command (see the module doc).")
    ap.add_argument("--agent", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model-path", required=True, help=f"the model the command uses, under {MODEL_STORAGE_ROOT}")
    ap.add_argument("--gate", default=DEFAULT_BASE)
    ap.add_argument("--modality", default="text")
    ap.add_argument("--engine", default="swarmui")
    ap.add_argument("--model-source", default="raid-local")
    ap.add_argument("--model-format", default="gguf")
    ap.add_argument("--vram-mb", type=int, default=6000)
    ap.add_argument("--minutes", type=int, default=90)
    ap.add_argument("--exclusive", action="store_true")
    ap.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    ap.add_argument("--heartbeat-seconds", type=float, default=HEARTBEAT_SECONDS)
    ap.add_argument("command", nargs=argparse.REMAINDER, help="the command to run, after --")
    args = ap.parse_args(argv)
    cmd = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not cmd:
        ap.error("a command is required after --")
    POLL_SECONDS, HEARTBEAT_SECONDS = args.poll_seconds, args.heartbeat_seconds
    lease = GpuLease(args.agent, args.task, args.model_path, base=args.gate, modality=args.modality, engine=args.engine,
                     model_source=args.model_source, model_format=args.model_format, vram_mb=args.vram_mb,
                     minutes=args.minutes, exclusive=args.exclusive, log=lambda m: print(m, file=sys.stderr, flush=True))
    failed = True
    try:
        lease.__enter__()
        env = dict(os.environ, GATE_LEASE_ID=lease.lease_id, GATE_GPU=str(lease.gpu))
        code, lost = run_under_lease(cmd, lease, env=env)
        failed = lost or code != 0
        if lost:
            print("[gate] lease lost; the command was stopped", file=sys.stderr, flush=True)
            return 1
        return code
    finally:
        lease.release(failed=failed)


if __name__ == "__main__":
    sys.exit(main())
