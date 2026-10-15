# TTS audit follow-up, 2026-10-15

Four providers were run against their upstream Python after the fetches were approved (the upstream packages and small
config files; no model weights were downloaded). Same oracle as the [audit](2026-10-15_tts_whisper_audit.md) (Whisper
`medium` through `ProcessSTT`), same gate sentences 01-05, same reference clip and transcript, same seeds; mean WER over
Whisper-normalized tokens with `ok` and `okay` equal.

## Verdicts

| provider | model | engine gate WER | upstream gate WER | verdict |
|---|---|---|---|---|
| Kyutai TTS | 1.6b-en-fr | 0.061 (12 seeds, n=60) | 0.118 (moshi 0.2.13, n=60) | PARITY, engine at or below upstream |
| Qwen3-TTS | 1.7B-CustomVoice | 0.072 (4 seeds, n=20) | 0.095 (qwen-tts 0.1.1, n=20) | PARITY |
| Bark | default | 0.240 (12 seeds, n=60) | 0.228 (transformers 5.19 `BarkModel`, hub `generation_config.json`, n=60) | PARITY (stochastic) |
| ZipVoice | base | rows 02-05 0.000 on every take (after PR #426) | rows 02-05 0.000 (k2-fsa/ZipVoice 2f7326f, 12 seeds) | FIXED (PR #426) |

## Bark: the audit's reference was not the HF default

The audit's Bark row compared the engine (0.255 on four seeds) with an upstream run at 0.920 raw, 0.345 capped. That
run set the sampler by hand (suno's `generate_audio` defaults: `top_k` off, `min_eos_p` 0.2). The hub's
`generation_config.json` sets `top_k=50` for the semantic and coarse stages and no `min_eos_p`. transformers' own
`min_eos_p` is a logits processor that runs before the temperature and top-k, so it sees the raw logits, where suno's
is computed on the probability after the filters and the temperature: the same numbers do not mean the same sampler.

With the hub configuration as shipped, 12 seeds of the five gate sentences give 0.228 against the engine's 0.240; the
bootstrap 95% interval of the difference is -0.105 to +0.131. Both sides lose the same rows: the paragraph (05) is cut
at about 14 s by the model's context on both (WER 0.44 and 0.45), and 02 `Okay, see you.` fails on 3 of 12 and 4 of 12
seeds. The engine's sampler follows suno's code (top-p on the raw logits, temperature, then `min_eos_p` on the tempered
probability) and its defaults are suno's; suno's package was read, not run, because its `.pt` checkpoints are not on the
box. Neither side splits text longer than the 14 s window into sentences.

## Kyutai TTS and Qwen3-TTS CustomVoice

Upstream ran through the `moshi` and `qwen-tts` packages with the engine's conventions (Kyutai: expresso voice
`ex03-ex01_happy_001_channel1_334s`, `cfg_coef` 2.0, audio temperature 0.6, `n_q` 32; Qwen3: speaker Ryan, language
`Auto`, the checkpoint's `generation_config.json`). Both ran from the local weights with the hub config files added
beside them. The Qwen3 speech tokenizer needed `"format": "pt"` in its safetensors header, added to a copy whose tensor
data is byte-identical.

## ZipVoice

Fixed in PR #426; the full record is [2026-10-15_zipvoice_upstream_ab.md](2026-10-15_zipvoice_upstream_ab.md). The
pipeline phonemized with espeak's British `en` voice and differed from upstream's tokenizer, reference handling and
long-text path at every layer. Rows 02-05 now transcribe word-perfect on every take for engine and upstream, row 01
`Yes.` is heard as `Yeah` on both, and paired durations differ by a median 3.5%. The two diagnostic rows differ only on
`PM` and `ok`, words the espeak port reads differently from piper's build.

## Method notes

- References ran on the 3060 (`CUDA_VISIBLE_DEVICES=1`) under the shared GPU lock; the engine ran through the live
  SwarmUI on a local build of the merged branches.
- A candidate is called a failure only after an upstream run with the same oracle, seeds and reference.
