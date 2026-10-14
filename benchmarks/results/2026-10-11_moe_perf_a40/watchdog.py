#!/usr/bin/env python3
"""Operator-side max-lifetime watchdog for pods that pod_up.py rented.

Why it runs on the operator machine: a pod cannot stop itself without a RunPod API key on the pod, and the kit never
puts a key on a pod. So the stop is issued from here, with the key in ~/.runpod/curlrc, which is the same credential
pod_up.py already uses.

Lease lifecycle (file: ~/.runpod/podkit-leases.json, outside the repo):
  1. pod_up.py writes a PENDING lease, keyed "pending:<pod name>", BEFORE the create request. A lease-file failure
     therefore stops the bring-up before any pod is rented.
  2. The create request runs under a unique pod name. Whether or not its response arrives (a timeout can hide a
     created pod), the pending lease stays.
  3. pod_up.py binds the pending lease to the pod id once the response has it. If that write fails, the next watchdog
     pass finds the pod by its unique name and binds it there.
  4. Only leased pods are ever stopped: once RUNNING and older than their max hours, the watchdog POSTs /stop once.
     Its stop_requested mark is never cleared, so a pod restarted from the console after a stop is not re-armed; that
     is deliberate, because a restart is a new decision for the operator.

Run it on a timer, for example cron every 5 minutes:

    */5 * * * * /usr/bin/python3 /path/to/podkit/watchdog.py --once >> ~/.runpod/watchdog.log 2>&1

usage: watchdog.py --once | --loop [--interval SEC]   [--dry-run]   [--leases PATH]

A corrupt lease file is never overwritten: the pass logs the error and exits nonzero, and pod_up.py refuses to rent.
Limitation: it acts only while the operator machine is awake and cron (or the loop) runs. A machine that is off cannot
stop anything; the RunPod console remains the fallback.
"""
import argparse, contextlib, datetime as dt, json, math, os, pathlib, subprocess, sys, time

try:
    import fcntl
except ImportError:  # Windows: no advisory lock; run one writer at a time there
    fcntl = None

API = "https://rest.runpod.io/v1"
CURLRC = pathlib.Path.home() / ".runpod/curlrc"
LEASES = pathlib.Path.home() / ".runpod/podkit-leases.json"
DEFAULT_MAX_HOURS = 6.0
REQUEST_TIMEOUT_S = 60
CREATE_TIMEOUT_S = 180      # POST /pods can be slow; a timeout here still leaves a pending lease that resolves by name
VANISH_GRACE_H = 1.0        # a bound pod missing from GET /pods is dropped only after this long (listing lag)
PENDING = "pending:"


def check_max_hours(value):
    """Return the lifetime as a float, or raise ValueError. Checked before a pod is rented, not after."""
    h = float(value)
    if not (math.isfinite(h) and h > 0):
        raise ValueError(f"max hours must be a finite number above 0; got {value!r}")
    return h


def api(method, path, body=None, curlrc=CURLRC, timeout=REQUEST_TIMEOUT_S):
    """One RunPod REST call through curl, so the API key stays in the curlrc file. Returns (http_code, parsed_body_or_text).
    --max-time bounds a stalled connection: run_once holds the lease lock across its calls, so a hang would block every writer."""
    cmd = ["curl", "-s", "--max-time", str(timeout), "--config", str(curlrc), "-X", method, API + path, "-w", "\n%{http_code}"]
    if body is not None:
        cmd += ["-d", json.dumps(body)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    text, _, code = out.rpartition("\n")
    try:
        code = int(code)
    except ValueError:
        code = 0
    if text.strip().startswith(("{", "[")):
        try:
            return code, json.loads(text)
        except ValueError:
            return code, text   # truncated or malformed body: callers treat it as "not the shape I expected"
    return code, text


@contextlib.contextmanager
def _lease_lock(path):
    """Exclusive lock over a whole read-modify-write of the lease file, so pod_up.py and a watchdog pass cannot overwrite each other."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(str(p) + ".lock", "a+") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def load_leases(path=LEASES):
    """Raises ValueError on a corrupt file. Callers must not overwrite it: that would lose every lease in it."""
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    text = p.read_text() or "{}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"lease file {p} is corrupt ({e}); fix or move it aside before renting or watching") from e
    if not isinstance(data, dict):
        raise ValueError(f"lease file {p} is not a JSON object")
    return data


def save_leases(leases, path=LEASES):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(leases, indent=2, sort_keys=True))
    tmp.replace(p)


def _now():
    return dt.datetime.now(dt.timezone.utc)


def reserve_lease(pod_name, max_hours, path=LEASES, now=None):
    """Written by pod_up.py before the create request. Returns the pending key."""
    max_hours = check_max_hours(max_hours)
    key = PENDING + pod_name
    with _lease_lock(path):
        leases = load_leases(path)
        leases[key] = {
            "name": pod_name,
            "started": (now or _now()).isoformat(),
            "max_hours": max_hours,
            "stop_requested": None,
            "pod_id": None,
        }
        save_leases(leases, path)
    return key


def bind_lease(pod_name, pod_id, path=LEASES):
    """Moves a pending lease to the pod id. If this write fails, the next pass binds the pod by its unique name."""
    key = PENDING + pod_name
    with _lease_lock(path):
        leases = load_leases(path)
        rec = leases.pop(key, None)
        if rec is None:
            return leases.get(pod_id)
        rec["pod_id"] = pod_id
        leases[pod_id] = rec
        save_leases(leases, path)
        return rec


def _parse_time(s):
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _age_h(rec, now):
    return (now - _parse_time(rec["started"])).total_seconds() / 3600.0


def pods_due(leases, now):
    """Bound pod ids whose max lifetime has passed and that have not already been asked to stop."""
    due = []
    for pid, rec in leases.items():
        if pid.startswith(PENDING) or rec.get("stop_requested"):
            continue
        if _age_h(rec, now) >= float(rec["max_hours"]):
            due.append(pid)
    return due


def run_once(api_call, path=LEASES, now=None, dry_run=False, log=print):
    """One watchdog pass. api_call(method, path, body=None) -> (code, body). Returns the pod ids it stopped (or would stop).
    The whole pass holds the lease lock, so a pod recorded during the pass is never overwritten by the save at its end."""
    with _lease_lock(path):
        return _run_once_locked(api_call, path, now or _now(), dry_run, log)


def _run_once_locked(api_call, path, now, dry_run, log):
    leases = load_leases(path)
    if not leases:
        return []
    code, listing = api_call("GET", "/pods")
    if code != 200 or not isinstance(listing, list):
        log(f"watchdog: cannot list pods (HTTP {code}); no stops issued this pass")
        return []
    live = {p.get("id"): p for p in listing if isinstance(p, dict)}
    by_name = {p.get("name"): p for p in listing if isinstance(p, dict) and p.get("name")}

    # Pending leases: bind the pod that carries the unique name, or give up once the lifetime (plus grace) has passed.
    for key in [k for k in leases if k.startswith(PENDING)]:
        rec = leases[key]
        pod = by_name.get(rec["name"])
        if pod is not None and pod.get("id"):
            log(f"watchdog: bound pending lease {rec['name']} to pod {pod['id']}")
            rec["pod_id"] = pod["id"]
            leases[pod["id"]] = rec
            leases.pop(key)
        elif _age_h(rec, now) >= float(rec["max_hours"]) + VANISH_GRACE_H:
            log(f"watchdog: pending lease {rec['name']} never matched a pod; dropping it")
            leases.pop(key)

    for pid in [k for k in leases if not k.startswith(PENDING)]:
        rec = leases[pid]
        if pid not in live and _age_h(rec, now) > VANISH_GRACE_H:
            # Terminated or deleted by someone else: the lease no longer protects anything.
            if not rec.get("stop_requested"):
                log(f"watchdog: {pid} is no longer listed; dropping its lease")
            leases.pop(pid)

    stopped = []
    for pid in pods_due(leases, now):
        if pid not in live or live[pid].get("desiredStatus") != "RUNNING":
            continue
        rec = leases[pid]
        age = _age_h(rec, now)
        if dry_run:
            log(f"watchdog: DRY RUN would stop {pid} ({rec['name']}), {age:.2f} h old, max {rec['max_hours']} h")
            stopped.append(pid)
            continue
        code, body = api_call("POST", f"/pods/{pid}/stop")
        if 200 <= code < 300:
            rec["stop_requested"] = now.isoformat()
            log(f"watchdog: stop requested for {pid} ({rec['name']}), {age:.2f} h old, max {rec['max_hours']} h")
            stopped.append(pid)
        else:
            log(f"watchdog: stop for {pid} failed HTTP {code}: {body}; will retry on the next pass")
    if not dry_run:
        save_leases(leases, path)
    return stopped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=float, default=300.0, help="seconds between passes with --loop")
    ap.add_argument("--dry-run", action="store_true", help="list what would be stopped; issue no POST")
    ap.add_argument("--leases", default=str(LEASES))
    args = ap.parse_args(argv)
    ts = lambda m: print(f"{_now().isoformat()} {m}", flush=True)
    while True:
        try:
            run_once(api, args.leases, dry_run=args.dry_run, log=ts)
        except ValueError as e:   # corrupt lease file: nothing was written
            ts(f"watchdog: {e}")
            if not args.loop:
                return 2
        if not args.loop:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
