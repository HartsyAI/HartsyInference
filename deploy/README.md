# Deploying `HartsyInference.Server`

Two files here exist for one reason: **some native/unsafe code paths in this engine can raise a
corrupted-state exception (e.g. `AccessViolationException`) that .NET Core cannot catch in-process** — the
CLR terminates the process before any exception handler runs, including the server's own global exception
middleware. That's not a bug this repo can fix in C#; the actual mitigation is process-level restart. (Every
*ordinarily* catchable failure — a bad request, a model-specific bug during decode — is already contained
in-process; see `DynamicBatchScheduler`'s per-round fault isolation. These restart mechanisms are for the
residual class of failure that genuinely can't be caught.)

Pick one:

## systemd (preferred on a Linux host/VM)

```bash
sudo cp deploy/systemd/hartsyinference-server.service /etc/systemd/system/
sudo systemctl edit hartsyinference-server   # set WorkingDirectory/ExecStart and any HartsyInference__* env vars for your deployment
sudo systemctl daemon-reload
sudo systemctl enable --now hartsyinference-server
```

`Restart=always` + `RestartSec=2` handles the corrupted-state-exception case; `StartLimitIntervalSec`/
`StartLimitBurst` stop it from crash-looping forever on something that will never recover on its own (e.g.
a bad model path in config) — after 5 restarts in 60s the unit is left `failed` for a human to look at
(`systemctl status hartsyinference-server`, `journalctl -u hartsyinference-server`).

## Bash wrapper (containers without an init system, ad-hoc use)

```bash
./deploy/run-with-restart.sh /path/to/HartsyInference.Server.dll
```

Same restart-with-backoff and crash-loop-breaker behavior as the systemd unit, configurable via
`HARTSY_RESTART_DELAY_SECS` / `HARTSY_CRASH_WINDOW_SECS` / `HARTSY_MAX_CRASHES_IN_WINDOW` env vars. Configure
the server itself the same way as systemd — set `HartsyInference__*` / `ASPNETCORE_URLS` env vars before
invoking the script.

## Either way: point your orchestrator's health checks correctly

- **Liveness** (`/health`): unconditional 200 once the process is up — restart on failure/timeout here.
- **Readiness** (`/ready`): 200 only while every loaded model's serving loop is actually alive; 503 with the
  affected model ids if one has died (see `DynamicBatchScheduler.IsLoopAlive`). Route traffic away on
  failure here, but do NOT restart the process on a 503 alone — a model with a dead loop can be recovered by
  reloading just that model (`POST /v1/models/load`) without killing every other model's traffic sharing
  the same process.
