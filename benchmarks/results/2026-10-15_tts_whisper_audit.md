# TTS Whisper audit, 2026-10-15

Every local AudioLab TTS provider was generated through the live SwarmUI (`ProcessAudio`, which passes
`__model_id` and `seed` through, and `ProcessTTS` for Piper, the only provider that needs its voice resolved)
and transcribed through `ProcessSTT` with the pinned Whisper `medium` oracle. Gate sentences 01-05:

- 01 `Yes.`
- 02 `Okay, see you.`
- 03 eight words, 04 twenty words, 05 a paragraph of about 400 UTF-8 bytes.

Mean WER is Levenshtein over Whisper's English-normalized tokens. `ok` and `okay` count as the same word.
Sentence 08 (repeated `no`) is diagnostic only. Seeds 0-3 for every provider that was re-sampled (n=20 rows
per provider). Upstream comparisons used the same oracle, the same reference clip and the same seeds.

## Verdicts

| provider | model | engine gate WER (n=20) | upstream | verdict |
|---|---|---|---|---|
| CosyVoice 2 | 2-0.5b | 0.000 | not re-run | pass |
| CSM | 1b | 0.084 | HF transformers port, 0.107 | PARITY |
| Fish Audio S2 Pro | s2-pro | 0.000 | not re-run | pass |
| FishSpeech | fish-speech-1.5 | 0.000 | not re-run | pass |
| GPT-SoVITS v2 | default | 0.001 | not re-run | pass |
| IndexTTS-2 | v2_5 | 0.000 | not re-run | pass |
| IndexTTS-2 | v2_0 | 0.000 | not re-run | pass |
| IndexTTS-1.5 | v1_5 | 0.000 | not re-run | pass |
| Kokoro | default | 0.000 | not re-run | pass |
| MeloTTS | english-v3 | 0.000 | not re-run | pass |
| Piper | default | 0.000 | not re-run | pass |
| PocketTTS | default | 0.000 | not re-run | pass |
| Qwen3-TTS | 1.7B-Base | 0.005 | not re-run | pass |
| Qwen3-TTS | 0.6B-Base | 0.005 | not re-run | pass |
| Qwen3-TTS | 1.7B-VoiceDesign | 0.005 | not re-run | pass |
| Qwen3-TTS | 0.6B-CustomVoice | 0.002 | not re-run | pass |
| Qwen3-TTS | 1.7B-CustomVoice | 0.072 | qwen-tts 0.1.1, 0.095 ([follow-up](2026-10-15_tts_audit_followup.md)) | PARITY |
| Spark-TTS | 0.5B | 0.001 | not re-run | pass |
| StyleTTS2 | libritts | 0.000 | not re-run | pass |
| Zonos | transformer | 0.001 | not re-run | pass |
| AuK | base | 0.000 | not re-run | pass |
| AuK | flash | 0.007 | not re-run | FIXED (PR #419) |
| Breeze TTS 2 | breeze-tts-2 | 0.017 | not re-run | pass |
| Kyutai TTS | 1.6b-en-fr | 0.061 (12 seeds) | moshi 0.2.13, 0.118 ([follow-up](2026-10-15_tts_audit_followup.md)) | PARITY |
| Orpheus | 3b | 0.068 | not re-run | pass |
| F5-TTS | v1-base | 0.017 (after PR #419) | upstream F5, same seeds | FIXED (PR #419, #421) |
| NeuTTS | air | 0.018 on 02-05 (after PR #420) | upstream 0.003 on 02-05 | FIXED (PR #420) |
| VibeVoice | 1.5b | 0.073 (after PR #422) | upstream native 0.117 | FIXED (PR #422) |
| Bark | default | 0.240 (12 seeds) | HF `BarkModel` with the hub generation config, 0.228 ([follow-up](2026-10-15_tts_audit_followup.md)) | PARITY (stochastic) |
| Chatterbox | default | 0.200 (all from `Yes.`) | 6/16 exact on `Yes.` for both | PARITY |
| Dia | 1.6b | 0.753 (01-03 fail, 04-05 0.000) | documented 2026-10-01 upstream A/B | PARITY on in-contract text |
| ZipVoice | base | rows 02-05 0.000 (after PR #426) | k2-fsa/ZipVoice 2f7326f, rows 02-05 0.000 | FIXED (PR #426) |
| VibeVoice-Realtime | 0.5B (engine id) | not generating | n/a | known gap |

## Fixes (all merged to main)

- PR #419, merged: F5-TTS and AuK short generations use upstream's 0.3 duration speed; F5 counts the
  reference text with upstream's extra space. Measured against upstream F5: `Yes.` 0.725 s against 0.715 s
  (was `[BLANK_AUDIO]`); 3-word sentence 2.805 s against 2.795 s.
- PR #420, merged: NeuTTS prompt tokenized with the HF-exact encode and phonemes keep punctuation. Gate 02-05
  mean WER 0.231 to 0.018 (upstream 0.003).
- PR #421, merged: F5 long-text chunking and upstream cross-fade. Paragraph 21.10 s to 20.87 s (upstream
  20.82 s); one-chunk rows byte-identical to the previous build.
- PR #422, merged: VibeVoice first-frame parity. Four causes, all confirmed on fp32 CPU dumps at identical
  noise: tokenizer without the Qwen split regex, the leading space dropped by the processor, tanh GELU where
  upstream uses erf, and the DPM solver dropping to first order at the end of the schedule. Gate mean WER
  1.573 to 0.073 (upstream native 0.117). Real-time factor unchanged within noise against published alpha.368;
  peak Swarm VRAM 13,124 MiB on both builds.

## Follow-up

The Kyutai, Qwen3 CustomVoice, Bark and ZipVoice upstream runs are in
[2026-10-15_tts_audit_followup.md](2026-10-15_tts_audit_followup.md). CosyVoice 2 cloning from a long reference is
recorded in [2026-10-15_cosyvoice2_long_reference_test.md](2026-10-15_cosyvoice2_long_reference_test.md).

## Open

- VibeVoice-Realtime-0.5B is an engine id only. It cannot generate: the released checkpoint ships a decode-only
  acoustic VAE, and the upstream per-speaker voice caches need a deserializer that does not exist.
- VibeVoice segment-boundary negative-stream reset (`VibeVoicePipeline.cs` 314-338 and 399-407 against upstream
  `modeling_vibevoice_inference.py` 547-563 and 573-587) is not confirmed. No gate sentence crosses a segment.
- F5 reference seconds come from the centred mel frame count, so chunk boundaries can differ from upstream's
  `samples // hop` by a frame.
- The Qwen2 byte-map-before-regex ordering in `Qwen2Tokenizer.EncodeRawByteLevel` also affects the MiniMax H3 and
  Hunyuan text paths. Those were not part of this audit.

## Method notes

- `ProcessAudio` writes no `requested` journal line, so the sweep's quiet-window guard counts only `ProcessTTS`
  lines. A deploy script that waits on `requested` lines cannot see a `ProcessAudio` sweep.
- Oracle calibration: JFK transcribes word-perfect with small, medium, large-v3 and turbo, so JFK cannot pick an
  oracle; the audit used `medium` throughout.
- Candidate triage used the gate mean over four seeds; a candidate was not called a failure until an upstream run
  with the same oracle, seeds and reference clip was compared.
