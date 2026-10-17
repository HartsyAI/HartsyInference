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

Usage:
    with GpuLease(agent="...", task="...", model_path="/mnt/model-storage/...") as lease:
        ...  # lease.gpu is the card; lease.check_alive() raises once a heartbeat reports the lease gone
"""
import json
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
    except urllib.error.URLError as e:
        raise GateError(f"{method} {path}: {e}") from e
    if isinstance(reply, dict) and reply.get("ok") is False:
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
            self.heartbeat()
            if self.lost:
                return

    def start_heartbeat(self):
        self._thread = threading.Thread(target=self._heartbeat_loop, name="gate-heartbeat", daemon=True)
        self._thread.start()

    def check_alive(self):
        if self.lost:
            raise GateError(f"lease {self.lease_id} is gone (heartbeat refused); stop the run")

    def cancel(self):
        if self.request_id and self.status == "queued":
            try:
                call(self.base, "POST", "/api/cancel", {"request_id": self.request_id})
            except GateError as e:
                self.log(f"[gate] cancel failed: {e}")

    def release(self, failed=False):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self.lease_id and not self.lost:
            try:
                call(self.base, "POST", "/api/release", {"lease_id": self.lease_id, "failed": failed})
                self.log(f"[gate] released lease {self.lease_id}" + (" (failed)" if failed else ""))
            except GateError as e:
                self.log(f"[gate] release failed: {e}")

    def __enter__(self):
        self.request()
        self.wait_granted()
        self.start_heartbeat()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release(failed=exc_type is not None)
        return False
