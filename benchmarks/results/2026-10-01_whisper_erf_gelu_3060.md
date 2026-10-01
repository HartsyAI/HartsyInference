# Whisper exact (erf) GELU: small.en on the RTX 3060 — 2026-10-01

Latency and token check for `fix/whisper-erf-gelu`, which replaces the tanh GELU in `WhisperEncoder` and
`WhisperDecoder` with the exact erf form Whisper uses. Gate, unchanged: small.en ≤ 350 ms per utterance for 2, 5 and
10 s of speech, in-process and warm on the 3060; JFK 11/11. Measured in the same slot as #211's re-check
([results](2026-10-01_whisper_logmel_3060.md)), whose header — GPU, driver, host, quiet window, lock, `--verify-since`
after every arm — applies unchanged; the arms below ran between that slot's C/D and A2/B2 arms (06:55:48–06:58:06Z),
each followed by a clean `--verify-since`.

| Arm | Build | Compared with |
|---|---|---|
| E, F | this branch on `origin/main` `92c322af` (old log-mel): gate, then regression models | baseline arms A, C (`92c322af`) |
| G, H | #211 `e9e3c376` plus this branch's four call sites: gate, then regression models | #211 arms B, D |

## Gate — small.en

| Slice | Variant | baseline median ms (A / A2) | this branch alone median (p95) ms | #211 median ms (B / B2) | #211 + erf GELU median (p95) ms | ≤ 350 ms |
|---:|---|---:|---:|---:|---:|---|
| 2 s | 16k | 125.2 / 124.5 | 122.5 (126.2) | 114.3 / 115.5 | 118.8 (143.3) | met |
| 2 s | narrowband | 97.1 / 114.7 | 119.3 (126.0) | 111.9 / 95.5 | 117.1 (121.5) | met |
| 5 s | 16k | 112.1 / 110.2 | 114.5 (121.1) | 118.6 / 105.5 | 109.6 (119.8) | met |
| 5 s | narrowband | 113.8 / 109.5 | 111.7 (116.4) | 105.1 / 105.3 | 106.0 (112.5) | met |
| 10 s | 16k | 193.5 / 186.3 | 186.6 (194.6) | 177.5 / 173.2 | 189.4 (193.1) | met |
| 10 s | narrowband | 191.7 / 184.5 | 188.9 (194.5) | 176.2 / 173.1 | 177.4 (178.6) | met |

The erf kernel costs what the tanh one does: each pair of columns sits inside the spread of the two runs it is
compared with. Tokens: identical to the respective comparison arm in every gate case, including the full clip at
16 kHz and narrowband, the timestamped decode and the streaming pass; recall unchanged (full clip 100 %, JFK 11/11).

## Regression — other checkpoints (1 warm + 2 timed, indicative)

| Model | Case | baseline (C) | this branch (F) | #211 (D) | #211 + erf GELU (H) | tokens |
|---|---|---:|---:|---:|---:|---|
| whisper-tiny | full 16k / narrowband / 5 s | 165.2 / 166.8 / 72.1 | 165.9 / 153.9 / 68.4 | 84.7 / 99.7 / 36.0 | 85.2 / 96.5 / 51.3 | identical to C and D respectively |
| whisper-small | full 16k / narrowband / 5 s | 249.8 / 182.5 / 111.3 | 195.4 / 182.5 / 109.9 | 197.4 / 175.1 / 110.3 | 260.6 / 195.3 / 109.3 | identical |
| whisper-medium | full 16k / narrowband / 5 s | 426.8 / 421.3 / 269.9 | 414.1 / 415.2 / 268.7 | 398.2 / 397.1 / 260.4 | 398.6 / 398.8 / 260.1 | identical |
| distil-large-v3 | full 16k / narrowband / 5 s | 352.5 / 352.0 / 311.1 | 352.2 / 350.6 / 311.5 | 331.2 / 330.7 / 301.6 | 334.3 / 331.6 / 301.4 | identical |
| distil-large-v3.5 | full 16k / narrowband / 5 s | 1120.7 / 1098.1 / 1059.1 | 424.9 / 426.9 / 386.6 | 408.0 / 411.4 / 374.2 | 811.7 / 799.1 / 763.3 | identical |

distil-large-v3.5's wall clock varies between arms on this box (see #211's results); small's full-clip 16 kHz row moves
between 195 and 261 ms across arms of the same code. Neither tracks the GELU.

## Against the HF reference, on the GPU

Of the 21 bench cases the CPU parity set also covers, this branch alone decodes 19 exactly as the HF reference does —
the same 19 as the baseline, since its log-mel is still the old one — and #211 plus this branch all 21, as #211 alone
does. None of the four CPU near-ties is among these 21 cases, so the GELU shows in the CPU parity instead: with #211's
log-mel, the CPU-backend decodes equal the HF reference's on all 144 cases of the parity set (six models, 12 clips,
with and without timestamps), from 140 with the tanh form
([parity](../../docs/Checklists/PARITY_VERIFICATION.md#stt)).
