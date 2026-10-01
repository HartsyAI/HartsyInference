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

# Deploying the phone-call voice agent

Two services on one box: [hartsyinference-voice-host](systemd/hartsyinference-voice-host.service) (the engine, the
speech models on the RTX 3060, the PhoneLink socket) and
[hartsyinference-phone-gateway](systemd/hartsyinference-phone-gateway.service) (SIP/RTP, no models). The gateway
`Requires=` the host, so `systemctl stop` or `restart` of the host stops or restarts the gateway too and drops live
calls. A host crash does not propagate: systemd restarts the host alone, and the gateway holds live calls with its "one
moment" prompt and re-attaches them. Design: [voice session](../docs/Research/VOICE_AGENT_SESSION.md#voice-host-hartsyinferencevoicehost)
and [phone gateway](../docs/Research/PHONE_GATEWAY.md); the full install and call checklist is the
[voice agent runbook](../docs/Checklists/VOICE_AGENT_VERIFICATION.md).

CPU layout on this box (i7-6900K, `lscpu -e`: CPU n and n+8 are the two threads of core n): the gateway's RTP tick
thread owns core 7 (`AllowedCPUs=7,15`, `media.tickCpu` 7, `SCHED_FIFO 50` through `LimitRTPRIO=50`); the voice host and
the API server get `AllowedCPUs=0-6,8-14`, and the host caps the engine's kernel threads to match
(`engine.cpuThreadCap: 14` in `voice.json`, which is `numerics.cpuThreads`; nothing is read from the environment).

## Install checklist (operator, sudo)

- [ ] `dotnet publish -c Release src/HartsyInference.VoiceHost -o /opt/hartsyinference/voice-host` and
      `dotnet publish -c Release src/HartsyInference.PhoneGateway -o /opt/hartsyinference/phone-gateway`.
- [ ] Secrets, root-owned, mode 0600: `/etc/hartsyinference/secrets/phone-link-token`, `phone-admin-token`,
      `sip-password` (a placeholder is fine with no registrar). The units hand them over with `LoadCredential=`.
- [ ] `/etc/hartsyinference/voice.json` (template `voice.example.json` next to the host binary) and
      `/etc/hartsyinference/phone.json` (template `phone.example.json`); their `*File` settings point at
      `/run/credentials/<unit>/<name>`.
- [ ] Review `User=`, `WorkingDirectory=` and `ExecStart=` in both units, then
      `sudo cp deploy/systemd/hartsyinference-{voice-host,phone-gateway,server}.service /etc/systemd/system/`.
- [ ] `sudo systemctl daemon-reload && sudo systemctl enable --now hartsyinference-voice-host hartsyinference-phone-gateway`
      (and `sudo systemctl restart hartsyinference-server` if it runs, for its new `AllowedCPUs=`).
- [ ] [Host tuning](#host-tuning): preview with `deploy/install-host-tuning.sh`, then
      `sudo deploy/install-host-tuning.sh --apply`; `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort |
      uniq -c` shows `16 performance`.
- [ ] Run the checks in the [runbook](../docs/Checklists/VOICE_AGENT_VERIFICATION.md), starting with the journal line
      `RTP tick thread running under SCHED_FIFO 50` on the first call.

The units were checked with `systemd-analyze verify` (no root needed):

```bash
systemd-analyze verify deploy/systemd/hartsyinference-voice-host.service \
  deploy/systemd/hartsyinference-phone-gateway.service deploy/systemd/hartsyinference-server.service \
  deploy/systemd/hartsyinference-cpu-performance.service
```

Notes that matter:

- `RuntimeDirectory=hartsyinference` is declared by the host unit only: systemd deletes a runtime directory when the
  unit that declares it stops, so a second declaration in the gateway would delete the host's socket.
- Never `CPUSchedulingPolicy=fifo` on the gateway: only the tick thread asks for FIFO.
- `DOTNET_GCgen0size=0x4000000` in the gateway unit is a .NET runtime setting (a 64 MB gen0 budget, logged at start as
  `gen0 budget=64 MB`), not one this repo reads.

## Host tuning

Two optimizations for this box, decided for the deployment and installed together by
[install-host-tuning.sh](install-host-tuning.sh):

- [hartsyinference-cpu-performance.service](systemd/hartsyinference-cpu-performance.service), a oneshot unit that sets
  every CPU's frequency governor to `performance` at boot, through
  `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` (no packages).
- [limits.d/hartsy-rt.conf](limits.d/hartsy-rt.conf), installed as `/etc/security/limits.d/hartsy-rt.conf`:
  `hartsy - rtprio 50`, for runs without systemd (the gateway unit already has `LimitRTPRIO=50`). `hartsy` is the
  units' `User=`; change both together.

```bash
deploy/install-host-tuning.sh                 # dry run: prints every file it would write and every change; no root
sudo deploy/install-host-tuning.sh --apply    # install, enable and start; running it again changes nothing
sudo deploy/install-host-tuning.sh --revert   # remove both, put every CPU on performance back on schedutil
```

- `--apply` installs this checkout's files as root, so run it from a checkout that only you or root can write. The dry
  run and `--apply` both print the sources' SHA-256, so you can check you are installing what you previewed.
- `--revert` puts back on `schedutil` only the CPUs that run `performance`, the governor the unit sets; a governor set
  by hand is left alone. A CPU that does not offer `schedutil` is an error and stays on `performance` until the next
  boot, which starts from the kernel default because the unit is gone. Stopping the unit by hand changes nothing.
- `power-profiles-daemon`, `thermald`, `tuned`, `tlp` or cpupower could rewrite the governor after boot, and a CPU
  brought online after boot starts on the kernel default. The governor check in the
  [runbook](../docs/Checklists/VOICE_AGENT_VERIFICATION.md) shows either, and running `--apply` again restarts the unit.

Why, from the voice bring-up's measurements:

- Under `schedutil`, the 2-5 s idle gaps of a conversation let the cores clock down, and the next turn pays for it:
  about 33 ms per Kokoro sentence (host-bound plan heuristics and DSP run on slowed cores) and about 0.7 ms per 20 ms
  frame of the paced voice front end. `performance` holds the clock.
- `rtprio 50` lets the gateway's RTP tick thread run under `SCHED_FIFO` (pinned to CPUs 7 and 15 in production) when
  the gateway runs outside systemd. The limit permits rather than restricts: any process of `hartsy` may then use FIFO
  up to 50, and only the tick thread asks for it.

Both are optimizations: every gate was measured and passes under the stock `schedutil` governor, and nothing depends
on them. `power-profiles-daemon` and `thermald`, both enabled on this box, did not change the governor in this
bring-up.

Considered and not adopted:

- L3 cache partitioning (Intel CAT through `resctrl`): it is system-wide, so it would squeeze SwarmUI too, and no gate
  needs it.
- C-state limits: a box-wide change to idle power and heat, and nothing depends on it.
- GPU persistence mode: the RTX 3060 stayed in P2 across the idle gaps, so there was nothing for it to keep up.

## Optional: keep SwarmUI off the gateway's CPUs

An operator decision, not part of the install: SwarmUI runs as the user unit `swarmui.service`, and with it confined
the tick thread's core is never shared. FIFO is the primary protection when something does land there.

```bash
systemctl --user edit swarmui.service
#   [Service]
#   AllowedCPUs=0-6,8-14
systemctl --user restart swarmui.service   # only in a quiet window: no generation running
```

Undo with `systemctl --user revert swarmui.service`.

## Development without systemd

The tick thread needs an `rtprio` limit to use FIFO; without one it logs the fix once and spins the last 150 µs
before each deadline instead. Grant it with `/etc/security/limits.d/hartsy-rt.conf`, which
`sudo deploy/install-host-tuning.sh --apply` installs ([Host tuning](#host-tuning)):

```
hartsy - rtprio 50
```

then open a new login session; `ulimit -r` prints 50. Run both executables with `--config`, after creating the socket
directory yourself (`sudo install -d -o hartsy -m 0750 /run/hartsyinference`).
