#!/usr/bin/env python3
"""Rent an A40 on RunPod, wait for SSH, copy this kit over and start bootstrap.sh in the background.
usage: pod_up.py [name] [--max-hours H]
  prints POD_ID, SSH host/port; run `ssh ... tail -f /root/baseline/logs/bootstrap.log` to follow.
  The pod is recorded in the operator's lease file (~/.runpod/podkit-leases.json) with its max lifetime (default 6 h,
  or PODKIT_MAX_HOURS). Run watchdog.py on a timer (see its docstring); it POSTs /stop once a leased pod is older than that.
  The pod never gets an API key. Manual stop: curl -s --config ~/.runpod/curlrc -X POST https://rest.runpod.io/v1/pods/<id>/stop"""
import argparse, json, os, pathlib, subprocess, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from watchdog import DEFAULT_MAX_HOURS, api, check_max_hours, record_lease  # noqa: E402  (same kit dir; no key is copied to the pod)

KEY = str(pathlib.Path.home() / ".ssh/runpod_blackwell")
KIT = pathlib.Path(__file__).resolve().parent   # holds bootstrap.sh, scripts/, cfg/, prompts/

ap = argparse.ArgumentParser()
ap.add_argument("name", nargs="?", default="dsv41-perf-a40-t3")
ap.add_argument("--max-hours", type=float, default=float(os.environ.get("PODKIT_MAX_HOURS", DEFAULT_MAX_HOURS)))
args = ap.parse_args()
try:
    args.max_hours = check_max_hours(args.max_hours)   # before the create request: a bad value must not leave a pod without a lease
except ValueError as e:
    sys.exit(f"refusing to rent a pod: {e}")
name = args.name
pub = pathlib.Path(KEY + ".pub").read_text().strip()
body = {"name": name, "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", "gpuTypeIds": ["NVIDIA A40"], "gpuCount": 1,
        "cloudType": "SECURE", "containerDiskInGb": 160, "volumeInGb": 20, "volumeMountPath": "/workspace", "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": pub}}
code, pod = api("POST", "/pods", body)
if code >= 300 or not isinstance(pod, dict) or "id" not in pod:
    sys.exit(f"create failed HTTP {code}: {pod}")
pid = pod["id"]
record_lease(pid, name, args.max_hours)   # before the wait loop, so a pod that never comes up is still auto-stopped
print("POD_ID", pid, f"max_hours={args.max_hours}", flush=True)
host = port = None
for _ in range(120):
    time.sleep(10)
    code, p = api("GET", f"/pods/{pid}")
    if isinstance(p, dict) and p.get("publicIp") and (p.get("portMappings") or {}).get("22"):
        host, port = p["publicIp"], p["portMappings"]["22"]
        r = subprocess.run(["ssh", "-i", KEY, "-p", str(port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new", f"root@{host}", "echo up"], capture_output=True, text=True)
        if r.stdout.strip() == "up":
            break
else:
    sys.exit(f"pod {pid} never became reachable")
print("SSH", host, port, flush=True)
ssh = ["ssh", "-i", KEY, "-p", str(port), "-o", "StrictHostKeyChecking=accept-new", f"root@{host}"]
subprocess.run(ssh + ["mkdir -p /root/podkit"], check=True)
subprocess.run(["scp", "-i", KEY, "-P", str(port), "-r", "-o", "StrictHostKeyChecking=accept-new", str(KIT) + "/.", f"root@{host}:/root/podkit/"], check=True)
subprocess.run(ssh + ["mkdir -p /root/baseline/logs && nohup bash /root/podkit/bootstrap.sh > /root/baseline/logs/bootstrap.log 2>&1 < /dev/null & echo started"], check=True)
print(json.dumps({"pod": pid, "host": host, "port": port}), flush=True)
