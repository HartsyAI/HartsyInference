# Voice agent verification runbook

Manual end-to-end checks for the phone-call voice agent on the real box: the voice host and the phone gateway under
systemd, a LAN softphone, a host killed mid-call, an idle NAT binding, and the RTP clock while SwarmUI generates.
Automated coverage (unit tests, the `[Slow]` loopback calls on the RTX 3060) is in
[voice session](../Research/VOICE_AGENT_SESSION.md#voice-host-hartsyinferencevoicehost) and
[phone gateway](../Research/PHONE_GATEWAY.md#verification); this runbook covers what only a real install shows. Record
each run's results in the table at the end.

## 1. Install (operator, needs sudo)

```bash
sudo deploy/install-host-tuning.sh --apply
sudo deploy/install-voice-agent.sh --apply
```

Preview either one first by running it without `--apply` (a dry run; needs no root). `install-host-tuning.sh`
installs the CPU governor unit (`performance` at boot) and the rtprio limit for runs without systemd; undo with
`--revert`. `install-voice-agent.sh` publishes both executables as you (never as root), installs them under
`/opt/hartsyinference`, generates the secrets under `/etc/hartsyinference/secrets` (never printed), writes
`/etc/hartsyinference/{voice.json,phone.json}` from the example templates with the LAN profile only if a file is
not already there (an existing one is never touched), copies the two units (and the server unit, but only if one
is already installed), then reloads, enables and starts the voice host and the gateway. Running either script
again changes nothing already correct; both print the SHA-256 of every file they install.

Optional, an operator decision: keep SwarmUI off CPUs 7 and 15 with the user-unit drop-in in
[deploy](../../deploy/README.md#optional-keep-swarmui-off-the-gateways-cpus).

Before installing anything, [§3](#3-lan-softphone-call) has a dev-only check that runs the same two executables as
plain user processes, no root and no systemd, to confirm the build, the models and the dial plan work on this box.

### Appendix: the manual steps `install-voice-agent.sh` automates

1. Publish both executables:
   `dotnet publish -c Release src/HartsyInference.VoiceHost -o /opt/hartsyinference/voice-host` and
   `dotnet publish -c Release src/HartsyInference.PhoneGateway -o /opt/hartsyinference/phone-gateway`.
2. Secrets, owned by root, mode 0600, under `/etc/hartsyinference/secrets/`: `phone-link-token` (any random string,
   e.g. `openssl rand -hex 32`), `phone-admin-token`, and `sip-password` (a placeholder on a LAN with no registrar; it is
   only read when `sip.registrar` is set, but `LoadCredential=` needs the file to exist).
3. `/etc/hartsyinference/voice.json` from `voice.example.json`: `llmModel: "qwen3"` now resolves on Linux (#205,
   alpha.228's model-folder case fix) without naming a checkpoint path; the host's own start-up log line confirms
   it (see "Confirm the install" below). Keep `engine.cpuThreadCap: 14` and `link.tokenFile:
   /run/credentials/hartsyinference-voice-host.service/phone-link-token`.
4. `/etc/hartsyinference/phone.json` for a LAN: `sip.registrar` empty, `sip.destinationPrefixes` empty,
   `sip.allowAnyDestination: true`, `sip.publicAddress: none`, `media.tickCpu: 7`, and the three `*File` settings
   pointing at `/run/credentials/hartsyinference-phone-gateway.service/<name>`.
5. Units: copy `deploy/systemd/hartsyinference-voice-host.service` and `hartsyinference-phone-gateway.service` to
   `/etc/systemd/system/` (adjust `User=`, `WorkingDirectory=` and `ExecStart=` if the paths differ), then
   `sudo systemctl daemon-reload && sudo systemctl enable --now hartsyinference-voice-host hartsyinference-phone-gateway`.

## 2. Confirm the install

- [ ] `systemctl status hartsyinference-voice-host hartsyinference-phone-gateway`: both `active (running)`.
- [ ] `journalctl -u hartsyinference-voice-host -b | grep '\[VoiceHost\]'`: engine on the RTX 3060, models loaded and
      warm, `pool min worker threads 22 (CPU kernel threads 14 + 8); 14 CPUs allowed`, listening on
      `/run/hartsyinference/phone.sock (mode 0660)`, and `Phone gateway connected`.
- [ ] The journal's `Engine on cuda:1 (...RTX 3060); the language model qwen3 runs on cuda:0.` and
      `whisper:..., kokoro:... and qwen3 loaded and warm in ... ms; denoise on.` name the model as `qwen3`, not a
      checkpoint path: confirms #205 (alpha.228's model-folder case fix) resolves `llmModel: "qwen3"` from
      `voice.json` as installed, unedited, by `install-voice-agent.sh` (confirmed on this box, `deploy/run-voice-agent-dev.sh`:
      `Engine on cuda:1 (NVIDIA GeForce RTX 3060); the language model qwen3 runs on cuda:0.` /
      `whisper:openai/whisper-small.en, kokoro:af_heart and qwen3 loaded and warm in 4609 ms; denoise on.`).
- [ ] `ls -l /run/hartsyinference/phone.sock` shows `srw-rw----`; `ls -ld /run/hartsyinference` shows `drwxr-x---`.
- [ ] `journalctl -u hartsyinference-phone-gateway -b | grep -E 'gen0 budget|SCHED_FIFO'`: `gen0 budget=64 MB` at start,
      and on the first call `RTP tick thread running under SCHED_FIFO 50` (not the "stays on the default scheduler"
      warning). `ps -eLo pid,tid,cls,rtprio,psr,comm | grep phone-rtp-tick` shows class `FF`, rtprio 50, CPU 7.
- [ ] `curl -s 127.0.0.1:9280/health` reports `"linkConnected": true`.
- [ ] `taskset -cp $(systemctl show -p MainPID --value hartsyinference-voice-host)` lists `0-6,8-14`.
- [ ] `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c` shows `16 performance`, after a
      reboot too, and `systemctl status hartsyinference-cpu-performance` is `active (exited)`.

## 3. LAN softphone call

Before installing anything, `deploy/run-voice-agent-dev.sh` runs the same two executables as plain user processes
(no root, no systemd, no sudo) against a LAN-profile `voice.json`/`phone.json` of their own under
`~/.config/hartsyinference/voice-agent-dev`: a fast check that the build, the models and the dial plan work on this
box before `sudo deploy/install-voice-agent.sh --apply`. It publishes both executables (`--no-build` to reuse the
last publish), checks SwarmUI is quiet before starting (`--force` to start anyway), then prints the `sip:` address
to dial and tails both logs; Ctrl+C (or `--stop` from another shell) ends both cleanly.

Either way: linphone (account "without registration") or `baresip -e "/dial sip:agent@<box-ip>:5060"`, PCMU or PCMA
enabled.

- [ ] The greeting plays; ask a question; the reply plays. The host journal shows one `[Voice] turn N (utterance): …`
      line per turn; note `voice.stt.ms`, `voice.llm.ttft_ms`, `voice.tts.first_chunk_ms` and `voice.turn.total_ms`.
- [ ] Talk over a long reply: it stops within a fraction of a second (`voice.bargein.stop_ms` in the turn line) and the
      interruption is answered.
- [ ] Ask the agent to press a key or say goodbye: `send_dtmf` reaches the far end; after the goodbye has played the
      call ends from the agent's side, and the host journal shows `the goodbye of turn N has played; hanging up` (a
      `was still going out after … ms` warning means the cap fired instead).
- [ ] Hang up from the phone: both journals show the call ending (`Call N ended RemoteHangup` on the gateway).

## 4. Host killed mid-call

During a call: `sudo kill -9 $(systemctl show -p MainPID --value hartsyinference-voice-host)`.

- [ ] The caller hears the "one moment" tones at once; the RTP clock never stops (no dead air).
- [ ] systemd restarts the host (`Restart=always`). If it is back within `link.outageHangupSeconds` (20 s), the gateway
      re-attaches the call and the caller hears `agent.resumeApology`; otherwise the caller hears "goodbye" and the call
      ends within about 1.2 s after the outage period (prompt plus grace).
- [ ] The next call is answered normally once the host is up.

## 5. Inbound call after 5 minutes idle

- [ ] Leave the box idle for 5 minutes with no call, then call in. The call is answered and audio flows both ways (the
      NAT binding and, with a provider, the registration survived the idle period). With a provider, check
      `curl -s 127.0.0.1:9280/health` shows `registered: true` before calling.

## 6. RTP clock under load

Start a 2-minute call, then queue SwarmUI generations (the 4090 and the CPUs the gateway does not own get busy).

- [ ] Every 10 s: `curl -s 127.0.0.1:9280/metrics | grep -E 'tick_late|rtp_fifo|silence'`. With FIFO granted, tick
      lateness p99 stays under 2 ms and the maximum under 10 ms; `hartsy_phone_rtp_fifo 1`.
- [ ] The caller hears no gaps or clicks; the per-call line at the end (`rtp out frames=… silence=… lateness p50=…
      p99=… max=…`) agrees.
- [ ] The host's link line at the end of the call (`sender ticks=… lateness p50=… p99=… max=… audioPathAllocated=…B`):
      lateness p99 under a few ms, and `audioPathAllocated=0B`.

## Development without systemd

`deploy/run-voice-agent-dev.sh` ([§3](#3-lan-softphone-call)) does this for you: its own `voice.json`/`phone.json`
already point `socketPath` at a directory it owns (`$XDG_RUNTIME_DIR/hartsyinference`), so it needs no
`/run/hartsyinference` of its own. Without it, by hand: `/etc/security/limits.d/hartsy-rt.conf` with
`hartsy - rtprio 50` (`sudo deploy/install-host-tuning.sh --apply` installs it), then a new login session
(`ulimit -r` prints 50); run the host and the gateway by hand with `--config`, after
`sudo install -d -o hartsy -m 0750 /run/hartsyinference` (or set both `socketPath`s to a directory you own).

## Results

| Date | Build | Check | Result | Notes |
|---|---|---|---|---|
