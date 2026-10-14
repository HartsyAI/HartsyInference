#!/usr/bin/env python3
"""Operator-side max-lifetime watchdog for pods that pod_up.py rented.

Why it runs on the operator machine: a pod cannot stop itself without a RunPod API key on the pod, and the kit never
puts a key on a pod. So the stop is issued from here, with the key in ~/.runpod/curlrc, which is the same credential
pod_up.py already uses.

pod_up.py records every pod it creates in a lease file (default ~/.runpod/podkit-leases.json, outside the repo) with
its start time and max lifetime. This watchdog only ever stops pods that have a lease, and only once they are RUNNING
and older than their max hours. Run it on a timer, for example cron every 5 minutes:

    */5 * * * * /usr/bin/python3 /path/to/podkit/watchdog.py --once >> ~/.runpod/watchdog.log 2>&1

usage: watchdog.py --once | --loop [--interval SEC]   [--dry-run]   [--leases PATH]

Limitation: it acts only while the operator machine is awake and cron (or the loop) runs. A machine that is off cannot
stop anything; the RunPod console remains the fallback.
"""
import argparse, datetime as dt, json, pathlib, subprocess, sys, time

API = "https://rest.runpod.io/v1"
CURLRC = pathlib.Path.home() / ".runpod/curlrc"
LEASES = pathlib.Path.home() / ".runpod/podkit-leases.json"
DEFAULT_MAX_HOURS = 6.0


def api(method, path, body=None, curlrc=CURLRC):
    """One RunPod REST call through curl, so the API key stays in the curlrc file. Returns (http_code, parsed_body_or_text)."""
    cmd = ["curl", "-s", "--config", str(curlrc), "-X", method, API + path, "-w", "\n%{http_code}"]
    if body is not None:
        cmd += ["-d", json.dumps(body)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    text, _, code = out.rpartition("\n")
    try:
        code = int(code)
    except ValueError:
        code = 0
    return code, (json.loads(text) if text.strip().startswith(("{", "[")) else text)


def load_leases(path=LEASES):
    p = pathlib.Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text() or "{}")


def save_leases(leases, path=LEASES):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(leases, indent=2, sort_keys=True))
    tmp.replace(p)


def record_lease(pod_id, name, max_hours, path=LEASES, now=None):
    """Called by pod_up.py right after POST /pods succeeds, before the wait loop, so a failed bring-up still has a lease."""
    if max_hours <= 0:
        raise ValueError("max_hours must be positive")
    leases = load_leases(path)
    leases[pod_id] = {
        "name": name,
        "started": (now or dt.datetime.now(dt.timezone.utc)).isoformat(),
        "max_hours": float(max_hours),
        "stop_requested": None,
    }
    save_leases(leases, path)
    return leases[pod_id]


def _parse_time(s):
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def pods_due(leases, now):
    """Lease ids whose max lifetime has passed and that have not already been asked to stop."""
    due = []
    for pid, rec in leases.items():
        if rec.get("stop_requested"):
            continue
        age_h = (now - _parse_time(rec["started"])).total_seconds() / 3600.0
        if age_h >= float(rec["max_hours"]):
            due.append(pid)
    return due


def run_once(api_call, path=LEASES, now=None, dry_run=False, log=print):
    """One watchdog pass. api_call(method, path, body=None) -> (code, body). Returns the list of pod ids it stopped (or would stop)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    leases = load_leases(path)
    if not leases:
        return []
    code, listing = api_call("GET", "/pods")
    if code != 200 or not isinstance(listing, list):
        log(f"watchdog: cannot list pods (HTTP {code}); no stops issued this pass")
        return []
    live = {p.get("id"): p for p in listing if isinstance(p, dict)}
    stopped = []
    for pid in list(leases):
        if pid not in live:
            # Terminated or deleted by someone else: the lease no longer protects anything.
            if not leases[pid].get("stop_requested"):
                log(f"watchdog: {pid} is no longer listed; dropping its lease")
            leases.pop(pid)
    for pid in pods_due(leases, now):
        status = live[pid].get("desiredStatus")
        if status != "RUNNING":
            continue
        rec = leases[pid]
        age_h = (now - _parse_time(rec["started"])).total_seconds() / 3600.0
        if dry_run:
            log(f"watchdog: DRY RUN would stop {pid} ({rec['name']}), {age_h:.2f} h old, max {rec['max_hours']} h")
            stopped.append(pid)
            continue
        code, body = api_call("POST", f"/pods/{pid}/stop")
        if 200 <= code < 300:
            rec["stop_requested"] = now.isoformat()
            log(f"watchdog: stop requested for {pid} ({rec['name']}), {age_h:.2f} h old, max {rec['max_hours']} h")
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
    ts = lambda m: print(f"{dt.datetime.now(dt.timezone.utc).isoformat()} {m}", flush=True)
    while True:
        run_once(api, args.leases, dry_run=args.dry_run, log=ts)
        if not args.loop:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
