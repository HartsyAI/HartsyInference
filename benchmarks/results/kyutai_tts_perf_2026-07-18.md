# Kyutai/Moshi TTS (dsm tts-1.6b) — performance pass, 2026-07-18

**Verdict: RTF 0.51× → 1.09× realtime on the 3060 (→1.47× on the 4090), now above realtime, word-perfect + parity-verified.**
The bottleneck was the 32-codebook depformer doing its attention on host pointers (~1000 D2H drains per frame). Fixed
by moving it fully on-device (FixedKvCache + FlashAttention), the same pattern as the STT/Qwen3 passes.

## Measured (JFK-length synthesis, warm; RTF = audio ÷ wall, higher = faster than realtime)
| | 3060 RTF | 4090 RTF |
|--|--:|--:|
| Original benchmark (2026-07-16) | 0.51× | — |
| This session, before depformer fix | 0.64× | ~0.85× |
| **After depformer fix** | **1.09×** (74 ms/fr) | **1.47×** (54 ms/fr) |
| moshi reference (bf16 + CUDA graph, 07-16) | 2.25× | — |

Correctness: `KyutaiDepformerParityTests` passes (logits match the moshi reference), and a real expresso voice on
CUDA transcribes word-perfect (whisper: *"Hello There World, this is a test of the QTIE [Kyutai] Text-to-Speech
Model."*). The earlier "garbage" transcript was the **synthetic** voice (structural noise, not a speaker) — the
baseline depformer produced identical garbage with it, so it's the voice, not the change.

## Fix — depformer attention: host-pointer KV → device-resident (FixedKvCache + FlashAttention)
`MoshiDepformer.Block` ran the depth-transformer attention on host pointers: `HeadSlice` (×3, qkv column copy),
`WriteStep`/`Prefix` (×4, the depth KV cache grown by host `Buffer.MemoryCopy`), and `HeadsToFlat` — **~8 host
copies per block, and each reads/writes a device tensor → a D2H+H2D round-trip**. At 32 codebooks × 4 layers that
is ~1000 host drains per 12.5 Hz frame (the author's own comment counted "~500 syncs/frame"). Replaced with:
- **`SliceLastDim`** (device) for the q/k/v column slice + **`Permute0213`** (device) for the head split/merge —
  the identity-in-memory reshapes now stay resident instead of round-tripping.
- a per-frame **`FixedKvCache`** (Layers × Heads × DepQ × HeadDim) with `AppendStep` + **`FlashAttention`** over the
  `[0..cb]` depth prefix (`kvLen=cb+1`, `qOffset=cb`, causal) — replacing the host `WriteStep`/`Prefix`/`SDPA`.

Result: 125 → 74 ms/frame on the 3060 (1.7× this step; 2.1× vs the original benchmark).

## Rejected: run the depformer on a CPU backend
The code had a `depBackend` hook and a comment claiming a CPU backend would be "far cheaper for these small ops"
(one D2H of the context vs ~500 syncs). **Measured it: CPU depformer is 670 ms/frame — 5× *slower*.** CPU F32 GEMVs
lose badly; the right fix is device-residency, not offloading to CPU. (The hook is now moot; left in for compat.)

## Remaining gap to moshi (2.25×) — CUDA graph
After the fix the warm cost is dominated by tiny `Linear` dispatch (128 depformer blocks/frame, launch-bound — the
depformer weights are small, so it's NOT bandwidth-bound). moshi hits 2.25× with a captured CUDA graph. The
depformer's 32 codebook steps are data-dependent (each embeds the previous sample), so a graph needs the CSM
depth-decoder pattern (fixed input buffer + device position + capture-once/replay, sampling interleaved). That's the
identified next lever to match/beat moshi; not attempted here to keep the change contained + parity-verified.

File: `MoshiDepformer.cs`. Harness `<scratchpad>/kttsbench` (real voice via `KTTS_VOICE`). Engine-only, not yet
packed/deployed to Swarm.
