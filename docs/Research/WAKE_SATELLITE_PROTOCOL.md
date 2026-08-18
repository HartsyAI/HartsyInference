# Wake Satellite Protocol

> Status: Complete | Last Updated: 2026-08-16 | Implemented by: `HartsyInference.Engine/Audio/Wake/`

## Summary

The wire protocol a voice satellite speaks to HartsyInference's wake-word listener. A satellite is a small
always-on device (the reference target is a Raspberry Pi Pico W with an I2S microphone) that streams raw
microphone audio 24/7; the server runs voice activity detection, wake-word scoring, transcription and speaker
identification, and pushes events back down the same connection.

The device is deliberately dumb. A Pico cannot run a wake model — openWakeWord's chain saturates a Pi Zero,
and microWakeWord depends on ESP32-S3 vector instructions — so all detection is server-side. This is the same
tier Home Assistant's M5 ATOM Echo occupies, and it is why the protocol carries continuous audio rather than
post-wake utterances.

Framing is Wyoming-*shaped* (Home Assistant / Rhasspy): one JSON line, then an optional binary payload. That
choice is about the client, not the server — on a microcontroller it is roughly fifty lines with no handshake,
no RFC 6455 frame masking, and no varint protobuf decoder.

> **This is not wire-compatible with Home Assistant, and deliberately so.** Real Wyoming *pops* `data` out of
> the header and writes it as a separate `data_length`-prefixed block ahead of the payload; the header carries
> only `type`/`version`/`data_length`/`payload_length`. This protocol keeps `data` inline, which is simpler for
> a microcontroller to emit. Home Assistant compatibility is a **separate endpoint** with its own codec —
> `src/HartsyInference.Engine/Audio/Wake/Wyoming/` on port 10600 — so neither side has to compromise. Do not
> point HA at the satellite port.

## Transport
> **Through an HTTPS reverse proxy or tunnel (Cloudflare, nginx, Caddy).** Raw TCP does not survive those —
> Cloudflare Tunnel in particular cannot carry arbitrary TCP to a public hostname. Connect instead to the
> **WebSocket ingest** at `wss://<host>/API/AudioLabWakeIngest`, send `{"session_id":"..."}` as the first
> message (SwarmUI's own auth), then send **exactly these same frames as binary WebSocket messages**. The wire
> format inside the socket is identical, so only the transport changes. Requires TLS on the device.


- **TCP**, default port **10800**, plaintext on the LAN.
- The **device dials the server**. (Wyoming and ESPHome satellites invert this — Home Assistant dials *them*,
  discovered over mDNS. Device-dials-server is the right shape here: the server has the stable address, the
  device owns reconnect, and no mDNS responder is needed on lwIP.)
- Set **`TCP_NODELAY`** on the client. Without it Nagle coalesces 20-40 ms audio frames into bursts and adds
  latency for no benefit on a LAN.
- The server sets `TCP_NODELAY` and TCP keep-alive (30 s idle, 5 s interval, 3 probes) on every accepted socket.

## Audio format

16 kHz, 16-bit signed little-endian, **mono**, raw PCM. Not negotiable — the server rejects anything else at
`hello` rather than resampling, because resampling on an always-on path is a permanent CPU cost and a silent
change to detection accuracy.

Recommended frame size is **20-40 ms** (640-1280 bytes). At 32 KB/s a stream is ~256 kbps, about 3% of a
Pico W's measured WiFi throughput, so **do not compress**: on an RP2040 real-time Opus is not achievable, and
compressing before server-side detection only adds decode cost and artifacts.

## Frame format

```
{"type":"<event>","data":{...},"payload_length":<N>}\n
<exactly N bytes of binary payload, when payload_length > 0>
```

The header is one UTF-8 JSON object terminated by `\n`; `data` and `payload_length` are optional. Unknown
fields are ignored, so a newer client can add its own without breaking the server. The header is capped at
8 KB and the payload at 1 MB.

## Events: device → server

### `hello` — required first frame
```json
{"type":"hello","data":{"device_id":"kitchen-pico","rate":16000,"width":2,"channels":1,"firmware":"1.0.3"}}
```
`device_id` is the session key and must be **stable across reboots** — that is what lets a reconnecting device
keep its wake words and configuration. `rate`/`width`/`channels` are validated when present (omitting them
means "the documented defaults"). Any other first frame is rejected.

### `audio-chunk` — the audio stream
```
{"type":"audio-chunk","data":{"seq":41},"payload_length":640}\n
<640 bytes = 320 samples = 20 ms>
```
`seq` increments by one per frame and **must not be reused or skipped silently**. A gap tells the server audio
was lost in flight, and it resets the detection state rather than splicing across the hole — a splice presents
the model with a transient that never occurred, which is worse than a brief deafness. Reset `seq` to 0 after
every reconnect.

### `pong`
Reply to the server's `ping`. Sending it is what keeps the connection classified as alive.

### `ping`
Optional. The server replies `pong`; useful if the device wants to measure round-trip time.

### `bye`
Optional clean shutdown. The server closes the connection but keeps the session.

## Events: server → device

### `hello-ack`
```json
{"type":"hello-ack","data":{"words":["alexa","hey_jarvis"]}}
```
Confirms registration and lists the wake words active for this device.

### `ping`
Sent every 10 s while idle. **Answer it.** A device that lost its access point leaves a socket that still looks
writable to the sender indefinitely; missing pongs are what actually surfaces the loss.

### `detection` — sent immediately when the word fires
```json
{"type":"detection","data":{"name":"hey_hartsy","score":0.9812,"route":"home-agent"}}
```
Arrives within ~100 ms of the word finishing, **before** any transcription. Use it for instant feedback: light
the LED, play the chime. It deliberately carries no transcript — waiting for one would leave the device silent
for seconds after the user spoke, which reads as broken.

### `transcript` — sent when the command has been captured and transcribed
```json
{"type":"transcript","data":{"name":"hey_hartsy","score":0.9812,"route":"home-agent",
                             "transcript":"turn on the kitchen light","speaker":"kaleb"}}
```
Arrives a few seconds after `detection`. This is the one to act on. `route` is an opaque tag from server-side
configuration — the engine never interprets it, so one server can feed several agents. `speaker` appears when
speaker identification is enabled. If transcription is disabled server-side, this event still arrives, with a
null transcript, so a device can always treat it as "the utterance is over".

### `detection-rejected`
```json
{"type":"detection-rejected","data":{"name":"hey_hartsy","score":0.98,"speaker":"guest"}}
```
The word fired but was restricted to a different enrolled speaker. Cancel whatever `detection` started.

### `error`
```json
{"type":"error","data":{"text":"..."}}
```

## Client reliability contract

The server half of self-healing is already handled: device-keyed sessions, per-connection fault isolation,
ping/pong plus TCP keep-alive, and a `Restart=always` systemd unit. The following is the **device's**
responsibility, and a satellite that skips it will appear to work and then quietly stop.

1. **Reconnect with exponential backoff and full jitter.** Base 250 ms, double per failure, cap 30 s,
   `delay = random(0, min(cap, base·2ⁿ))`. Jitter is not decoration: when the server restarts, every satellite
   reconnects at once, and unjittered backoff turns a fast recovery into a synchronized stampede.
   Reset the backoff only after the connection has been healthy for several seconds, not on connect.
2. **Send `hello` on every connection**, and restart `seq` at 0.
3. **Liveness.** If no server traffic (data or `ping`) arrives for 20 s, or no send has succeeded for 10 s,
   tear the connection down and reconnect. Do not wait for TCP to notice.
4. **Hardware watchdog — non-negotiable.** The CYW43439 driver has documented lockups (`STALL`,
   `do_ioctl timeout`) that no reconnect loop can escape; only a reboot recovers. Enable the RP2040 watchdog
   (~8 s max) and feed it **only** from the main loop, gated on "the microphone produced samples AND the socket
   accepted them recently". A watchdog fed unconditionally from a timer protects nothing.
5. **Disable WiFi power save, and re-apply it after every reconnect** — the radio resets the setting, and this
   is the single most common cause of a satellite that works for an hour and then drops hourly:
   MicroPython `wlan.config(pm=0xa11140)`, C SDK `cyw43_wifi_pm(&cyw43_state, CYW43_PERFORMANCE_PM)`.
6. **Ring buffer of 0.5-2 s** for microphone audio, dropping oldest on overflow. That bridges TCP retransmits
   and brief RF fades, not real outages. Because the server resets on a `seq` gap, dropping frames is safe —
   but only if you still advance `seq` past the dropped frames rather than renumbering.
7. **Never block the sampling path** on connect or DNS. `cyw43_arch_wifi_connect_timeout_ms` can hang; run
   reconnection as a state machine and keep dropping audio into the ring while it runs.

## Constants

| Item | Value |
|---|---|
| Default port | 10800 |
| Sample rate / width / channels | 16000 Hz / 2 bytes / 1 |
| Recommended frame | 20-40 ms (640-1280 bytes) |
| Bandwidth per satellite | ~256 kbps |
| Server ping interval | 10 s |
| Client liveness timeout | 20 s no traffic / 10 s no successful send |
| Reconnect backoff | 250 ms base, 30 s cap, full jitter |
| Header limit / payload limit | 8 KB / 1 MB |
| Detection cadence | one score per 80 ms of audio |
| Warm-up before first score | ~1.3 s of audio after connect or reset |

## Reference Implementations

- **Wyoming protocol** (the framing this borrows): [spec](https://github.com/rhasspy/rhasspy3/blob/master/docs/wyoming.md),
  [library](https://github.com/rhasspy/wyoming).
- **ESPHome voice assistant** (protobuf over TCP; where the satellite ecosystem is consolidating):
  [api.proto](https://github.com/esphome/aioesphomeapi/blob/main/aioesphomeapi/api.proto).
- **wyoming-satellite** (archived Jan 2026, still the best worked example of a Linux satellite):
  [repo](https://github.com/rhasspy/wyoming-satellite). Successor:
  [linux-voice-assistant](https://github.com/OHF-Voice/linux-voice-assistant).
- **Pico microphone input**: [micropython-i2s-examples](https://github.com/miketeachman/micropython-i2s-examples)
  (`machine.I2S` on rp2 since v1.20), [pico-INMP441](https://github.com/biemster/pico-INMP441) (PIO+DMA, C SDK),
  [Arm PDM microphone library](https://github.com/ArmDeveloperEcosystem/microphone-library-for-pico).
- **CYW43 power-save behaviour**: [arduino-pico discussion #3080](https://github.com/earlephilhower/arduino-pico/discussions/3080).

## Differences From Wyoming

- Same framing, different direction: here the **device dials the server**, whereas Home Assistant dials Wyoming
  satellites after mDNS discovery.
- `audio-chunk` carries a **`seq`** field Wyoming does not define, because server-side detection needs to know
  when audio was lost in order to reset model state.
- Wyoming's `audio-start` / `audio-stop` are unused: the stream is continuous by design and its boundaries are
  the connection's.
- `detection` carries `route`, `transcript` and `speaker` in one event rather than Wyoming's separate
  `detection` and `transcript` events, so a device can act on a single message.
