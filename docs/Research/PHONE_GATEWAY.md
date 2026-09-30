# Phone Gateway

> Source snapshot: 2026-09-30. This date does not establish current build or verification status.

## Summary

`src/HartsyInference.PhoneGateway` is the SIP/RTP leg of the phone-call voice agent: a small service exe that
registers with a SIP provider (or answers LAN softphones directly), carries G.711 audio to and from the caller,
and hands 16 kHz PCM to the voice host over the [PhoneLink protocol](PHONE_LINK_PROTOCOL.md). No model runs in
it; the process exists so that the 20 ms RTP cadence is never on the same threads, GC heap or scheduler budget
as inference. Signalling is [sipsorcery](https://github.com/sipsorcery-org/sipsorcery) 10.0.16 (pure managed);
the RTP clock, the jitter buffer and the link are this repo's own code.

Package boundary: the exe references `Core` (monotonic clock, FIFO scheduling, SPSC ring, latency histogram),
`Audio` (`G711`, `StreamingResampler`) and `PhoneLink`, never `Engine`. `IsPackable=false`; it is not in any NuGet
package. Referencing `Audio` pulls the `LLM` and `ModelAssets` assemblies into the output directory; accepted for
v1 (managed-only, nothing is loaded from them) rather than moving the resamplers out of `Audio`'s public API.

## Architecture

```
SIP trunk / LAN softphone
   │  SIP over UDP (or TCP) ──────────────── SipAccount: SIPTransport, ContactHost, SIPRegistrationUserAgent
   │  RTP/RTCP, ports rtpPortStart..End       CallController: one-call state machine, screening, tools, DTMF
   ▼
phone-gateway process
   phone-rtp-tick   (ClockedAudioSource, one thread per call, SCHED_FIFO when granted)
       deadline t0 + n·20 ms on CLOCK_MONOTONIC → SpscRing<short>.Read → G711.Encode → RTPSession.SendAudio
       ring empty → comfort silence, frame still sent; late ≥ 1 period → ≤ 5 catch-up frames, else resync
       Flush(turnId) → epoch flag → SpscRing.DiscardAll on the next tick
   sipsorcery receive thread
       OnRtpPacketReceived → InboundAudioPath.HandleRtpPacket → RtpJitterBuffer.Push (one memcpy + index)
       PhoneMediaSession latches the send destination on the first packet (symmetric RTP)
   phone-rtp-pump   (InboundAudioPath, SCHED_OTHER, phased +10 ms)
       RtpJitterBuffer.Pop | PLC → G711.Decode → StreamingResampler 8k→16k → link audio lane (never blocks)
   phone-link-reader (EngineLink)
       dial UDS with backoff → Hello/HelloAck → frames: OutboundAudio(turnId) → flush-epoch drop rule →
       OutboundAudioPath (resample hostRate→8k, producer lock) → SpscRing; Flush → FlushAck; ToolRequest; CallEnd
   phone-link-writer (EngineLink)
       control lane (64 deep, see below) before audio lane (10 deep, drop-oldest) → LinkFrameWriter
       Ping every 5 s; a timer watchdog closes a connection that received nothing, or had one write stuck, for 20 s
   LinkOutageGuard  host gone mid-call → "one moment" prompt (repeat) → reconnect: CallStart(resume) │
                    after outageHangupSeconds → "goodbye" prompt → hang up
   AdminEndpoint    HttpListener on 127.0.0.1: GET /health, GET /metrics, POST /calls (bearer token)
```

Queues and who touches them:

| Queue | Producer | Consumer | Full behaviour |
|---|---|---|---|
| outbound `SpscRing<short>` (2^18 samples ≈ 32 s @ 8 kHz) | link reader (host audio) or prompt player, serialized by a producer lock | tick thread, lock-free | drop newest, counted |
| `RtpJitterBuffer` (8 × 20 ms, target depth 3) | sipsorcery receive thread | pump thread | reset on a packet beyond the window (a new SSRC is caught by its timestamp jump, not by comparing SSRCs; one stream per call) |
| link audio lane (10 frames) | pump thread | link writer | drop oldest, counted |
| link control lane (64) | SIP/tool threads, link reader (acks, pongs, and whatever its callbacks send) | link writer | sender waits up to 5 s, the link reader not at all (a reader parked on a full lane stops reading, and a host blocked writing to it stops draining); then the frame is dropped, counted in `link_control_lane_dropped_total`, and the connection restarted, which re-announces a live call with `CallStart(resume)` |

The only lock the tick thread ever sees is none: the ring has two `Volatile` indices, the flush is one `Volatile`
flag, and the histogram is written by that thread alone. Every other queue is between ordinary threads, where a
short `Monitor` is fine. `AudioRingBuffer` is deliberately not used on the tick path (it is `Monitor`-locked, and
a descheduled producer holding it would stall a FIFO consumer).

Call screening: sipsorcery's `SIPUserAgent` offers every copy of an INVITE to `OnIncomingCall` and silently
ignores a new INVITE while it holds a call, so `CallController` screens on the transport first (its request handler
is subscribed before the user agent exists). A new INVITE gets `486` while a call is up or being set up, `503`
while the host link is down and `603` when the inbound policy refuses the caller. Each refusal goes out on its own
`UASInviteTransaction`, which answers the caller's retransmissions and absorbs the ACK, and is recorded in an
`InviteRejectionLedger` keyed by Call-ID and top Via branch for 32 s (RFC 3261 Timer B): the refusal is counted
once in `calls_rejected_total`, and a copy that slips past the transaction layer gets the same answer instead of
being offered as a call.

Media faults: an exception on the tick thread ends that thread and raises `ClockedAudioSource.TickFaulted` once.
The controller logs it once with the exception, counts `calls_media_fault_total` and ends the call rather than
leaving it up in silence: an announced call gets a BYE and `CallEnd(Failed)` (the protocol's "SIP or media
failure"); a call not yet announced is never announced, and the gateway waits for the ACK to its 200 (RFC 3261
§15, at most 8 s, so a lost 200 cannot leave the caller in a call already ended) before sending the BYE. A failed
answer always leaves the caller with a final response (500) or a BYE. A pump-thread fault is logged and stops
inbound audio for that call without ending it.

Scheduling: the tick thread asks for `SCHED_FIFO` at `media.fifoPriority` (default 50) once at start through
`RealtimeScheduling.TryEnterFifo`. Refused (`ulimit -r 0` on a plain login) it logs the exact fix once and falls
back to sleeping until 150 µs before each deadline and spinning the rest. `Thread.Priority` is not used anywhere
(a no-op on Linux). The FIFO path is unit-tested for "no throw, actionable reason"; the p99 < 2 ms / max < 10 ms
bound is asserted only when FIFO was granted (`TickJitterHarnessTests`).

GC: workstation concurrent GC (csproj), `GCLatencyMode.SustainedLowLatency` and eight minimum pool threads
(`RuntimeTuning`). sipsorcery allocates one packet buffer and one `IAsyncResult` per packet, so the systemd unit
(PR9) sets `DOTNET_GCgen0size=0x4000000` to make gen0 collections rare; even a FIFO thread is suspended by a GC.
The start-up log line shows the gen0 budget the GC actually uses (`GCGen0MaxBudget`, 64 MB with that setting).

## Configuration file

`phone.json` (template: `src/HartsyInference.PhoneGateway/phone.example.json`, copied next to the binary). Every
section has defaults; `{}` is a valid LAN test file (no registrar, no tokens). Secrets are never in the file and
never in the environment: the `*File` fields name secret files, read once at start-up (see [Secrets](#secrets)).

```json
{
  "sip": {
    "listenAddress": "0.0.0.0", "port": 5060, "transport": "udp",
    "registrar": "", "username": "hartsy",
    "passwordFile": "/run/credentials/hartsyinference-phone-gateway.service/sip-password",
    "registrationExpirySeconds": 60,
    "publicAddress": "none",
    "rtpPortStart": 20000, "rtpPortEnd": 20100,
    "codec": "Any", "inboundPolicy": "AllowAll", "allowlist": [], "destinationPrefixes": [],
    "greetingPromptFile": null, "ringTimeoutSeconds": 45
  },
  "link": {
    "socketPath": "/run/hartsyinference/phone.sock",
    "tokenFile": "/run/credentials/hartsyinference-phone-gateway.service/phone-link-token",
    "outageHangupSeconds": 20
  },
  "admin": { "port": 9280, "tokenFile": "/run/credentials/hartsyinference-phone-gateway.service/phone-admin-token" },
  "media": { "fifoPriority": 50, "tickCpu": -1, "warmUpTicks": 200 },
  "recording": { "enabled": false, "directory": "/var/lib/hartsyinference/phone-recordings" },
  "logging": { "level": "Info", "sipDebug": false }
}
```

| Setting | Meaning |
|---|---|
| `sip.registrar` | Registrar host[:port]; empty means no registration (LAN, or IP-authenticated trunks). With a registrar, `sip.passwordFile` is required. |
| `sip.passwordFile` | Absolute path of the file holding the SIP password; read only when a registrar is set. |
| `sip.registrationExpirySeconds` | 60..120; sipsorcery refreshes 5 s early. |
| `sip.publicAddress` | `none` (local address), an IP literal, or `stun:host[:port]`; STUN is re-asked before every REGISTER and the Contact host rewritten. |
| `sip.rtpPortStart/End` | RTP port range (shuffled). Forward it, and `sip.port`, on the router for a provider; nothing to do on a LAN. |
| `sip.codec` | `Any` (PCMU then PCMA), `Pcmu`, `Pcma`. Only G.711 at 8 kHz is ever negotiated. |
| `sip.inboundPolicy` | `AllowAll`, `Allowlist` (caller user part in `allowlist`, else 603), `Reject` (always 603). The allowlist matches the `From` header, which anyone who can reach `sip.port` can set: on a trunk, firewall `sip.port` and the RTP range to the provider's addresses. |
| `sip.destinationPrefixes` | Number prefixes (e.g. `+1555`) that outbound calls and the `transfer` tool may dial; needs `sip.registrar`. With prefixes set, a destination must be exactly a number (digits, `+`, `*`, `#`) through the registrar: bare, `tel:<number>`, or `<number>@<registrar host>` with or without `sip:`/`sips:`. A port, a URI parameter (`maddr` overrides where the INVITE is sent, `transport` how), a header or another host refuses it (403) rather than being stripped, and the call dials `sip:<number>@<registrar>` rebuilt from the validated number. A matching number at another host would take the call, and the account's digest answer, elsewhere. Empty allows any destination. The agent can be talked into dialling by its caller, so set this on a real trunk to rule out premium-rate toll fraud. |
| `sip.greetingPromptFile` | Raw 8 kHz PCM16 file played to every answered inbound call before the host speaks. |
| `link.outageHangupSeconds` | How long a live call waits for the host before "goodbye" and hang-up. |
| `link.tokenFile` | Absolute path of the file holding the shared PhoneLink token sent in `Hello`; empty sends no token. |
| `admin.tokenFile` | Absolute path of the file holding the bearer token for `POST /calls`; empty leaves health and metrics up and `POST /calls` at 403. |
| `media.fifoPriority` | `SCHED_FIFO` priority for the tick thread; 0 never asks. |
| `media.tickCpu` | Pin the tick thread to one CPU (with `AllowedCPUs` on the unit); -1 for none. |
| `recording.enabled` | Off by default; see the consent note. |
| `logging.sipDebug` | Forwards sipsorcery's Debug/Trace output. **Never on by default**: the transport trace prints whole SIP messages, `Authorization` headers included. |

Run: `HartsyInference.PhoneGateway --config /etc/hartsyinference/phone.json`. Exit code 2 is a configuration
error; the message names the setting, and the path for a secret file.

### Secrets

The SIP password and the link and admin tokens are read once at start-up from the files `sip.passwordFile`,
`link.tokenFile` and `admin.tokenFile` name, never from the config or the environment (which leaks through
`/proc/<pid>/environ`, child processes and crash dumps). Paths are absolute. A secret file must be private to its
owner, the ssh rule: mode 0600 or 0400; any group or other permission bit makes the gateway refuse to start, naming
the file and the `chmod` that fixes it. One trailing line ending (LF or CRLF) is trimmed, an empty or missing file
is a configuration error, and no secret is ever logged.

Under systemd each secret is a credential, so the unit reads the root-owned file and hands the service a private
copy:

```ini
LoadCredential=sip-password:/etc/hartsyinference/secrets/sip-password
LoadCredential=phone-link-token:/etc/hartsyinference/secrets/phone-link-token
LoadCredential=phone-admin-token:/etc/hartsyinference/secrets/phone-admin-token
```

The service then finds them at `/run/credentials/hartsyinference-phone-gateway.service/<name>`, and `phone.json`
points at those absolute paths (as `phone.example.json` does), so not even `$CREDENTIALS_DIRECTORY` is read.
Without systemd, any file with mode 0600 works: `install -m 600 /dev/null ~/.config/hartsy/sip-password`, then
write the secret into it.

Admin endpoint (loopback only): `GET /health` (JSON, 200 when the link is up and registration holds, else 503
`degraded`), `GET /metrics` (Prometheus text: calls, rejections counted once per INVITE, media faults
(`calls_media_fault_total`), link state and RTT, audio and control lane drops, and for the live call the tick lateness histogram,
jitter-buffer counters and pump counters), `POST /calls` with
`Authorization: Bearer <token>` and `{"destination":"sip:user@host"}` (or `user@host`, or a bare or `tel:`
number, dialled through the registrar): 202 placed, 409 busy, 503 host down, 403 destination outside
`sip.destinationPrefixes`, 400 not a SIP destination or a number with no registrar, 502 not answered or media
setup failed. A failed media setup (no free RTP port, for one)
answers an INVITE with 500 and leaves the gateway idle, ready for the next call.

Telephony tools the host may request over the link: `hangup`, `send_dtmf` (`digits`, optional `gapMs`),
`transfer` (`target`, blind, refused outside `sip.destinationPrefixes`), `hold`, `unhold`, `play_prompt` (`file` = raw 8 kHz PCM16 path, or `name` =
`one-moment` | `goodbye`; `text` is answered `Unsupported`, the gateway has no TTS). Caller DTMF (RFC 4733)
arrives as `DtmfEvent` with the duration in ms.

## Manual LAN recipe (softphone, no provider)

1. Start the voice host (PR9) so `/run/hartsyinference/phone.sock` exists, or point `link.socketPath` at a test
   host. Without a host the gateway still answers the admin endpoint but rejects INVITEs with 503.
2. `phone.json`: `registrar` empty, `publicAddress: none`, `listenAddress` the LAN interface (or `0.0.0.0`),
   `admin.port` 9280, and `link.tokenFile`/`admin.tokenFile` either empty or pointing at 0600 files holding the
   tokens the host and you expect. Run the gateway; `curl 127.0.0.1:9280/health` should say `linkConnected: true`.
3. linphone: add a SIP account with no registration (or "use without account") and dial
   `sip:agent@<gateway-ip>:5060`. baresip: `baresip -e "/dial sip:agent@<gateway-ip>:5060"` (any user part;
   only the allowlist looks at the caller). PCMU/PCMA must be enabled on the phone.
4. Speak; watch `curl 127.0.0.1:9280/metrics | grep -E 'tick_late|rtp_in_(lost|late)|rtp_fifo'` during the
   call. `hartsy_phone_rtp_fifo 0` with a `SCHED_FIFO 50 refused` warning in the log is the expected state on a
   login shell; see the systemd requirements below.
5. Press keys on the phone to check `DtmfEvent` reaches the host; hang up from the phone and from the host
   (`hangup` tool); confirm the per-call log line (`Call N ended <reason> after S s: rtp out frames=… lateness
   p50=… p99=… max=…; rtp in received=… lost=…`).
6. A second phone calling during a call must hear busy (486); with `inboundPolicy: Reject` every call gets 603.

## Provider bring-up notes

- Registration: set `registrar`, `username` and `passwordFile` (a `LoadCredential=` path, see
  [Secrets](#secrets)). `registrationExpirySeconds` is clamped to 60..120; providers
  that demand longer minimums answer 423, which sipsorcery retries with the registrar's `Min-Expires`.
- NAT: `publicAddress: stun:stun.l.google.com:19302` (or the provider's STUN) writes the WAN address into
  Contact and SDP; forward UDP `sip.port` and the RTP range to the box, or the provider's media never arrives.
  With comedia/symmetric-RTP providers, `PhoneMediaSession` also re-points its send destination at the first
  packet's source, so an SDP address that is wrong is repaired as soon as the far end speaks.
- Transport: `udp` by default; `tcp` for providers that insist. TLS/SRTP are not wired in this PR (sipsorcery
  supports both; `RtpSecureMediaOption` stays `None`).
- IP-authenticated trunks: leave `registrar` empty and set `inboundPolicy: Allowlist` with the trunk's
  caller ids, or the gateway will answer anything that reaches port 5060.
- Outbound: `POST /calls` with a bare or `tel:` number dials `sip:<number>@<registrar>` with the account
  credentials; `user@host` (with or without a port) gets the `sip:` scheme.

## systemd requirements (unit lands in PR9)

- `LimitRTPRIO=50` (the `rtprio` limit the tick thread needs; `CPUSchedulingPolicy=fifo` on the whole service is
  wrong, it would make sipsorcery's threads real-time too).
- `AllowedCPUs=` a core pair the engine host does not use, and `media.tickCpu` set to one of them, so the tick
  thread is not competing with `CpuParallel` on the host or with SwarmUI.
- `Environment=DOTNET_GCgen0size=0x4000000`, `Nice=-10`, `RuntimeDirectory=hartsyinference` for the socket,
  one `LoadCredential=` per secret (see [Secrets](#secrets)), `After=`/`Requires=` the host unit.
- Development without systemd: `/etc/security/limits.d/hartsy-rt.conf` with `<user> - rtprio 50` and a new
  login session; `ulimit -r` must show 50.

## License note

sipsorcery is BSD-3-Clause with an appended usage-restriction clause in the package's `LICENSE.md`. The gateway
is an unpackaged exe, so no HartsyInference NuGet package carries the dependency; read that clause before
redistributing the gateway binary. The pin and the note live in `Directory.Packages.props`.

## Recording consent note

`recording.enabled` writes each call's caller audio (16 kHz) and sent audio (8 kHz) as raw PCM files under
`recording.directory`, named by UTC time and SIP Call-ID. It is off by default. Recording a call requires the
consent the law of every party's location demands (one-party or all-party); the greeting prompt is the place to
say so. The gateway logs a warning at start-up when recording is on.

## Embedded prompts

`Assets/one-moment.pcm` (1.2 s) and `Assets/goodbye.pcm` (1.04 s) are 8 kHz PCM16 tone patterns synthesized by a
small script (raised-cosine-faded sine notes at -12 dBFS: 440→660 Hz twice for "one moment", 660→550→440 Hz for
"goodbye"), because no TTS is available where the assets are built and silence would hide a host outage from the
caller. Replace them with spoken prompts of the same format when a voice is chosen; `play_prompt` with `file`
plays any raw 8 kHz PCM16 file.

## Verification

Tests in `tests/HartsyInference.PhoneGateway.Tests` (unit lane unless marked): `G711EquivalenceTests` (all
65 536 μ-law and A-law codes against sipsorcery's encoders; `short.MinValue` documented separately: we clip, the
NAudio-derived encoders wrap), `RtpJitterBufferTests`, `ClockedAudioSourceCadenceTests` (3 s frame count ± 1,
zero allocation on the tick thread after warm-up, flush within one tick, FIFO refusal reason, a subscriber or
start fault raising `TickFaulted` once), `EngineLinkTests` (fake host on a temporary socket: handshake, audio both
ways with sequence continuity, flush epoch, reconnect + `CallStart(resume)`, liveness, outage hang-up and
recovery, no outage for a call starting in the link's connect gap), `GatewayConfigTests`,
`InviteRejectionLedgerTests`, `CallControllerSipTests` (loopback only, no audio timing: a tick fault on an active
inbound or outbound call → BYE, `CallEnd(Failed)`, one `calls_media_fault_total`; a fault before the call is
announced → BYE after the ACK and nothing sent to the host; one INVITE sent three times → three 486, 603 or 503
answers and a counter of 1); `[Integration]` `TickJitterHarnessTests` (60 s, one burner per core) and
`LoopbackSipCallTests` (stock sipsorcery
softphone against the real controller: inbound call with audio both ways, DTMF and remote hangup; host `hangup`
tool; second INVITE → 486; outbound call; 603 policy). `HartsyInference.Phone.slnf` builds the gateway and its
tests without the GPU packages.
