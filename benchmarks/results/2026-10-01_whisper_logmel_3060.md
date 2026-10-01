# Whisper log-mel parity: small.en on the RTX 3060 — 2026-10-01

Latency re-check for PR #211, which replaces Whisper's 512-point uncentered log-mel (2997 frames, 1499 encoder
positions) with the reference's 400-point centered STFT (3000 frames, 1500 positions). Gate, unchanged from #204:
small.en ≤ 350 ms per utterance for 2, 5 and 10 s of speech, in-process and warm on the 3060; JFK 11/11. The
narrowband recall gate (≥ the 16 kHz baseline − 10 pts) is evaluated on the full clip and on slices of 5 s and longer,
by the orchestrator's decision delegated by the user: the 2 s cut lands inside "Americans", and the reference model
itself drops the word's end on those narrowband samples, so recall there is informational. Numerical parity against
HF transformers is in [PARITY_VERIFICATION](../../docs/Checklists/PARITY_VERIFICATION.md#stt).

## Header

| Field | Value |
|---|---|
| Baseline build | `origin/main` `92c322af` (alpha.231), exported and built fresh |
| After build | `e9e3c376` (#211) |
| GPU | NVIDIA GeForce RTX 3060 12 GB — `GPU-b2198524-73ad-f200-16ef-6fc368d0599f` — nvidia-smi index 0; run with `CUDA_VISIBLE_DEVICES=1`, engine ordinal 0 (the bench asserts the device name). The 4090 was never opened. |
| Driver | 595.91.07 |
| Host | i7-6900K (8 cores / 16 threads), 64 GB, Linux 6.8 |
| Quiet state | `swarm-quiet-window.sh --gpu 3060` held 600 s (06:41:04–06:51:26Z): no live, waiting or loading gen, no foreign process on the card; SwarmUI's own idle context (430 MiB) on the card throughout. Bench lock taken first (`logmel 2026-10-01T06:51:30Z`), then `top` showed no foreign build or test (desktop only: Xorg ~11 %, rustdesk ~6 %). `--verify-since <arm start>` after every arm and once over the whole slot (06:51:30–06:59:02Z): no SwarmUI request, restart or queued gen. `nvidia-smi` before and after every arm: SwarmUI's idle context only. |
| Protocol | `WhisperBenchTests`: 2 warm-up + 5 timed `TranscribeAudio` calls per gate case, median / p95 / min wall; regression models 1 warm-up + 2 timed (indicative). Arms interleaved: A baseline gate (profiled), B after gate (profiled), C baseline regression, D after regression, then four arms for the separate GELU branch, then A2 baseline gate, B2 after gate. Tokens of every case compared with the baseline arm's. |

## Gate — small.en

| Slice | Variant | baseline median ms (A / A2) | after median ms (B / B2) | after p95 ms (B / B2) | ≤ 350 ms | tokens vs baseline |
|---:|---|---:|---:|---:|---|---|
| 2 s | 16k | 125.2 / 124.5 | 114.3 / 115.5 | 139.7 / 133.9 | met | identical |
| 2 s | narrowband | 97.1 / 114.7 | 111.9 / 95.5 | 125.0 / 107.2 | met | differ at step 2: "And so my fellow Ameri-", the reference's output |
| 5 s | 16k | 112.1 / 110.2 | 118.6 / 105.5 | 130.8 / 111.6 | met | identical |
| 5 s | narrowband | 113.8 / 109.5 | 105.1 / 105.3 | 111.0 / 107.6 | met | identical |
| 10 s | 16k | 193.5 / 186.3 | 177.5 / 173.2 | 181.7 / 182.4 | met | identical |
| 10 s | narrowband | 191.7 / 184.5 | 176.2 / 173.1 | 177.0 / 182.1 | met | identical |

Recall and the other paths (arm B):

| Case | Result |
|---|---|
| Full clip, 16 kHz and narrowband | recall 100 % / 100 % (Δ 0 pts, gate met), JFK 11/11 both, tokens identical to the baseline |
| 5 s / 10 s narrowband vs the same slice at 16 kHz | 100 % / 100 % (gate met) |
| 2 s narrowband vs the 2 s slice at 16 kHz | 50 % (−50 pts), informational: the cut lands inside "Americans" |
| Timestamped full clip (`SegmentAudio`) | 28 ids, identical; one segment 0.00–11.00 s |
| Streaming (`WhisperStreamingPipeline`, 1 s pushes) | final text identical |

## Attribution

Stage timer (`diagnostics.profile`, stages closed by a device sync), small.en, ms (D2H syncs were 0 for mel and
encoder in both builds):

| Stage | baseline 2 / 5 / 10 s | after 2 / 5 / 10 s |
|---|---|---|
| mel | 4.1 / 8.9 / 16.8 | 1.8–2.6 / 3.8–3.9 / 3.9–5.2 |
| frames → encoder positions | 2997 → 1499 | 3000 → 1500 |
| encoder | 50.8–51.3 | 49.5–50.0 |

The mel front end is cheaper despite the exact 400-point transform: the frames now fan out over the cores in fixed
blocks. The encoder at 1500 positions costs what it did at 1499.

## Regression — other checkpoints (1 warm + 2 timed, indicative)

| Model | Case | baseline ms (C) | after ms (D) | tokens vs baseline |
|---|---|---:|---:|---|
| whisper-tiny | full 16k / narrowband / 5 s | 165.2 / 166.8 / 72.1 | 84.7 / 99.7 / 36.0 | identical ×3 |
| whisper-small | full 16k / narrowband / 5 s | 249.8 / 182.5 / 111.3 | 197.4 / 175.1 / 110.3 | identical, identical, differ at 9 ("ask not!" → "ask not.") |
| whisper-medium | full 16k / narrowband / 5 s | 426.8 / 421.3 / 269.9 | 398.2 / 397.1 / 260.4 | identical ×3 |
| distil-large-v3 | full 16k / narrowband / 5 s | 352.5 / 352.0 / 311.1 | 331.2 / 330.7 / 301.6 | identical ×3 |
| distil-large-v3.5 | full 16k / narrowband / 5 s | 1120.7 / 1098.1 / 1059.1 | 408.0 / 411.4 / 374.2 | identical, identical, differ at 9 (final period dropped) |

distil-large-v3.5's wall clock varies between arms on this box (0.36 s in #204's session, 0.81 s in a later arm of
this one on another build); its baseline row here is not a front-end effect. Every token difference is on a 5 s or 2 s
slice, where the old input had tipped a decision. distil-large-v3.5's new 5 s output is the HF reference's on the CPU;
multilingual whisper-small was not in the CPU parity set.

## Against the HF reference, on the GPU

The bench's token dumps line up with 21 cases of the CPU parity set (small.en's six gate slices, full clip at 16 kHz
and narrowband and its timestamped decode; tiny, medium and distil-large-v3 / v3.5 on the full clip and the 5 s
slice). On the 3060 at default precision (TF32 GEMMs), **#211 decodes all 21 exactly as the HF reference does**; the
baseline matches 19 — it parts from the reference on small.en's 2 s narrowband slice (reference margin there 0.142)
and distil-large-v3.5's 5 s slice (0.436).
