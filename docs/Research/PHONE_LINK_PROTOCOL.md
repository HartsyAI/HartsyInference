# Phone Link Protocol

> Source snapshot: 2026-09-30. This date does not establish current build or verification status.

## Summary

The wire protocol between the phone gateway (SIP/RTP, no models) and the voice host (engine in-process, owns the
models) when both run on one box. It carries caller audio up, synthesized audio down, and a small set of control
messages: call lifecycle, session events, telephony tool calls and liveness. The implementation is
`src/HartsyInference.PhoneLink` (depends on `HartsyInference.Core` only); both processes link it.

Rejected alternatives: the wake-satellite codec (lives in Engine, one JSON line per frame, allocates per frame) and
WebSocket (a masking and handshake layer with no benefit between two local processes). The result is a fixed binary
header, raw PCM16 for audio and source-generated JSON only for the control messages that need structure.

## Transport

- **Unix domain socket.** The **host listens** (it owns the models and the stable lifetime); the **gateway dials**.
  Path is deployment configuration (a `RuntimeDirectory` such as `/run/hartsyinference/phone.sock`).
- **`Hello` is the first frame** on every connection, and it carries the shared token. The host validates version
  and token before anything else; on mismatch it sends `Error` (callId 0) and closes. Token comparison is
  constant-time on the host.
- **Reconnect and backoff are the gateway's job**: exponential backoff with full jitter (250 ms base, 30 s cap),
  `Hello` on every connection, `CallStart` with `resume=true` for a call that was live when the link dropped. The
  host never dials.
- **One writer thread per direction.** `LinkFrameWriter` is single-writer and throws on a concurrent call rather
  than interleaving bytes; `LinkFrameReader` is single-reader. Each process therefore runs exactly one sender and
  one receiver per connection, and the audio sender is the paced 20 ms thread.
- **Liveness**: the **gateway pings every 5 s** whether or not a call is up (it owns reconnect, so it is the side
  that must notice a dead link), and the host answers every `Ping` with a `Pong` carrying the same timestamp. The
  host may also ping. A side that receives no frame at all for 20 s closes the socket; the gateway then reconnects.
- **Unknown frame types are skipped, not fatal.** The framing stays valid, so a receiver logs the type byte once and
  reads on, which is what lets a newer peer add messages. Malformed framing (oversize length, truncated stream,
  wrong payload size for a known type) is fatal: close and reconnect.
- **The link layer validates the wire, not the policy.** `ReadHello` rejects an inbound rate other than 16 kHz and
  `ReadHelloAck` an outbound rate outside the table; token and version checks, and enforcing `maxFrameMs` on what
  the host actually sends, are the host's and gateway's own responsibility.

## Frame layout

Every frame is a 16-byte little-endian header followed by `payloadLength` bytes.

| Offset | Size | Field | Meaning |
|---|---|---|---|
| 0 | u32 | `payloadLength` | Payload bytes after the header; at most 1 MB (1,048,576) |
| 4 | u8 | `type` | `LinkMessageType` |
| 5 | u8 | `flags` | `LinkFrameFlags`; bit 0 = `Concealed` (InboundAudio only) |
| 6 | u16 | `reserved` | Written as 0, ignored on read |
| 8 | u32 | `callId` | Gateway-assigned, unique per connection, starts at 1; 0 = connection-level |
| 12 | u32 | `sequence` | Per-direction frame counter from 0 on each connection, every frame type, wraps at 2³² |

A header declaring more than 1 MB is a protocol error and is rejected before any of the payload is read. There is no
resynchronization: after any protocol error the receiver closes the socket and the gateway reconnects.

## Messages

Direction: G→H is gateway to host, H→G is host to gateway. Binary payloads are little-endian; JSON payloads use
camelCase keys, enum values by name (`"Inbound"`, `"TranscriptFinal"`, `"Unsupported"`), null fields omitted,
unknown fields ignored, UTF-8 with no HTML-style escaping.

| Type | Byte | Direction | Payload |
|---|---|---|---|
| `Hello` | 0x01 | G→H | `u16 version` (1) · `u32 inboundRate` (16000) · `u16 tokenLength` · UTF-8 token |
| `HelloAck` | 0x02 | H→G | `u32 outboundRate` ∈ {8000, 16000, 22050, 24000, 48000} · `u16 maxFrameMs` |
| `CallStart` | 0x10 | G→H | JSON `{direction, callerId?, called?, sipCallId, resume}` |
| `CallEnd` | 0x11 | either | `u8 reason`: 0 Completed, 1 RemoteHangup, 2 LocalHangup, 3 Busy, 4 NoAnswer, 5 Failed |
| `InboundAudio` | 0x20 | G→H | PCM16 at 16 kHz, exactly 320 samples (640 bytes) = 20 ms; flag `Concealed` when filled by PLC |
| `OutboundAudio` | 0x21 | H→G | `u32 turnId` · PCM16 at `outboundRate`, at most `maxFrameMs` of audio |
| `OutboundEnd` | 0x22 | H→G | `u32 turnId`: no more audio follows for this turn |
| `Flush` | 0x23 | H→G | `u32 turnId`: discard queued audio with turnId ≤ this |
| `FlushAck` | 0x24 | G→H | `u32 turnId` · `u32 msDiscarded` |
| `Event` | 0x30 | H→G | JSON `{kind, turnId?, state?, text?, latency?}` |
| `DtmfEvent` | 0x31 | G→H | `u8 digit` (ASCII `0-9 * # A-D`) · `u16 durationMs` |
| `ToolRequest` | 0x40 | H→G | `u32 requestId` · JSON `{name, arguments?}` |
| `ToolResult` | 0x41 | G→H | `u32 requestId` · JSON `{status, message?}` |
| `Ping` | 0x50 | either | `u64` sender's monotonic clock, ns |
| `Pong` | 0x51 | either | the `Ping` timestamp echoed |
| `Error` | 0x60 | either | JSON `{text}` |

Connection-level frames (`Hello`, `HelloAck`, `Ping`, `Pong`, and an `Error` that precedes a close) carry
callId 0. Everything else carries the callId the gateway assigned in `CallStart`.

### Handshake

```
G→H  Hello{version:1, inboundRate:16000, token}
H→G  HelloAck{outboundRate:16000, maxFrameMs:20}      or  Error{text} then close
```

The inbound rate is fixed at 16 kHz by the host's models (RNNoise, Silero, Whisper) and any other value in `Hello`
is a protocol error, not a negotiation; the gateway resamples from the codec rate. The outbound rate is whatever the host finds cheapest to produce (the session resamples Kokoro's 24 kHz
to 16 kHz today); the gateway resamples to the codec rate. `maxFrameMs` bounds one `OutboundAudio` frame so the
gateway's jitter buffer can size itself.

### Call lifecycle

`CallStart` opens a callId once media is flowing; `CallEnd` closes it from either side (the gateway when the far
end hangs up or SIP fails, the host after a `hangup` tool result). After `CallEnd` nothing else is sent for that
callId; a late frame is ignored. A host restart loses sessions: the gateway reconnects and re-sends `CallStart`
with `resume=true`, the host starts a fresh session and says so to the caller.

### Events (H→G)

| `kind` | Fields | Meaning |
|---|---|---|
| `State` | `state`, `turnId?` | Session state name (`Listening`, `Thinking`, `Speaking`, ...) — host vocabulary, for operators, never interpreted by the gateway |
| `TranscriptPartial` | `text` | In-progress transcript of the current utterance |
| `TranscriptFinal` | `text`, `turnId` | Final transcript that starts a turn |
| `TurnLatency` | `turnId`, `latency{sttMs?, llmFirstTokenMs?, llmFirstSentenceMs?, ttsFirstChunkMs?, transportMs?, totalMs}` | Per-turn budget breakdown in milliseconds |

Example: `{"kind":"TurnLatency","turnId":3,"latency":{"sttMs":120,"llmFirstTokenMs":90,"totalMs":800}}`.

### Tools (H→G request, G→H result)

`name` ∈ `hangup`, `send_dtmf` (`arguments.digits`, optional `gapMs`), `transfer` (`arguments.target`), `hold`,
`unhold`, `play_prompt` (`arguments.text` or `arguments.file`). `arguments` is an arbitrary JSON object owned by the
message. The gateway answers every request exactly once with `status` ∈ `Ok`, `Failed` (with `message`) or
`Unsupported`. `requestId` is host-assigned, unique per connection.

## Flush and turnId epochs

`turnId` is a per-call counter the host increments for every agent turn, starting at 1. Every `OutboundAudio` and
`OutboundEnd` carries the turn that produced it. Barge-in works entirely through this tag:

1. The host detects the caller speaking, cancels its turn, and sends `Flush(turnId = T)`.
2. The gateway sets `flushedTurn = max(flushedTurn, T)`, empties its outbound queue of frames with
   `turnId ≤ flushedTurn`, and answers `FlushAck(T, msDiscarded)` with the milliseconds of audio it threw away.
3. From then on the gateway **drops every `OutboundAudio` and `OutboundEnd` whose `turnId ≤ flushedTurn` on
   arrival**, so frames from the cancelled turn still in flight on the socket never reach the wire.
4. Audio with `turnId > flushedTurn` (the next turn) plays normally, so the host may start speaking again before
   the ack arrives.

`Flush` is idempotent and monotonic: a lower or repeated turnId is acknowledged with `msDiscarded = 0` and changes
nothing. `OutboundEnd(turnId)` lets the gateway know when a turn's audio is complete for its own playout
accounting; it is not an acknowledgement and needs no reply.

## Constants

| Item | Value |
|---|---|
| Protocol version | 1 |
| Header size | 16 bytes |
| Payload limit | 1 MB (1,048,576 bytes) |
| Inbound audio | 16 kHz PCM16 mono, 320 samples (20 ms) per frame |
| Outbound rates | 8000, 16000, 22050, 24000, 48000 Hz |
| Connection-level callId | 0; calls start at 1 |
| First turnId per call | 1 |
| Gateway ping interval | 5 s, call or no call |
| Liveness timeout | 20 s without any frame |
| Reconnect backoff (gateway) | 250 ms base, 30 s cap, full jitter |

## Implementation notes

- `LinkFrameWriter` assembles header and payload in one pooled staging buffer and hands the frame to the stream in a
  single write; audio writes allocate nothing and a write the stream completes synchronously never enters an async
  state machine. `Dispose` returns the buffer to `ArrayPool<byte>.Shared`; the stream is not owned.
- `LinkFrameReader` reassembles frames split across any read boundaries in one pooled buffer that grows only to the
  largest frame seen. `LinkFrame.Payload` is a slice of that buffer, valid only until the next `ReadAsync`; decode
  it first. The oversize check runs on the 16 header bytes, so a hostile length never costs a 1 MB allocation.
- Neither side resamples in the link layer; rates are negotiated once in the handshake and the resamplers live in
  the gateway (`StreamingResampler`) and the voice session.
