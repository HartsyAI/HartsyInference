# Deploying HartsyInference.API

Unsafe/native faults can terminate the process before managed exception handlers run. Use a process
supervisor; this is recovery, not proof that model faults are isolated.

## systemd

```bash
sudo cp deploy/systemd/hartsyinference-server.service /etc/systemd/system/
sudo systemctl edit hartsyinference-server
sudo systemctl daemon-reload
sudo systemctl enable --now hartsyinference-server
```

Set WorkingDirectory/ExecStart for your installation and HartsyInference__* environment settings.
Inspect the [unit](systemd/hartsyinference-server.service) for restart and rate-limit policy;
use systemctl status and journalctl -u hartsyinference-server to diagnose failures.

## Restart wrapper

```bash
./deploy/run-with-restart.sh /path/to/HartsyInference.API.dll
```

[run-with-restart.sh](run-with-restart.sh) exposes HARTSY_RESTART_DELAY_SECS,
HARTSY_CRASH_WINDOW_SECS, and HARTSY_MAX_CRASHES_IN_WINDOW. Set ASPNETCORE_URLS and
[server options](../src/HartsyInference.API/HartsyInferenceServerOptions.cs) for the deployment.

## Health checks

- /health is process liveness (unconditional 200 while the route responds).
- /ready resolves Engine.BackendDescription, returning 503 if that throws. It does **not** check
  individual loaded-model workers or prove generation is healthy. Do not use it as that guarantee.

The source contract is [HealthEndpoints.cs](../src/HartsyInference.API/Endpoints/HealthEndpoints.cs).
Model-level readiness, draining, and fault-isolation verification remain [open work](../docs/Checklists/ROADMAP.md).
