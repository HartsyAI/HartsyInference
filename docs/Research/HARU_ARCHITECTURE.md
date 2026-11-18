# Haru-4B architecture

Source snapshot 2026-11-16/17. Haru-4B is a full-duplex speech-to-speech model: one Qwen3.5 trunk reads the user's audio
and its own text and audio every 133 ms and emits the next text token and the next audio latent. This note holds the
constants, layouts and traps the port needs. Build status is in [audio status](../Checklists/MODEL_STATUS_AUDIO.md).

## Pins

| What | Pin |
|---|---|
| Weights | `higurashi-ai/Haru-4B` @ `8d6e83bcd3701a76fc27a764965665da1df7da78` (2026-11-13), ungated, Apache-2.0 |
| `model.safetensors` | 10,540,113,510 B, sha256 `ad2c3534ef6bd88525e22e5da3ac4fb739368406bf3b1c852cbafbade647c0fa` |
| `tokenizer.json` | 19,989,325 B, sha256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| Reference code | `github.com/higurashi-ai/Haru` @ `7fe6b7665f8bf035998f401f8cb779fbe53f2751`, Apache-2.0 |
| Blog | `blog.higurashi.ai/melte/haru4b/`: no paper, no benchmarks; a technical report is announced |
| Base LLM | `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, Apache-2.0 |
| Codec origin | VibeVoice acoustic tokenizer, MIT (`microsoft/VibeVoice-1.5B` @ `c00898d2`); the encoder stem and decoder head are byte-identical to it, the other tensors were not compared |

The VibeVoice model card forbids voice impersonation without recorded consent and real-time voice conversion; Haru's
README asks for rights to any reference voice. Keep both in the catalog licence note.

## Parts and parameters (5,270,000,897, all BF16, 1,022 tensors)

| Part | Params | Keys |
|---|---|---|
| Language model | 4,205,751,296 | `lm.embed_tokens.weight` [248320, 2560] (tied, 635,699,200), `lm.norm.weight`, `lm.layers.{0..31}.*` |
| Codec (encoder + decoder) | 687,392,001 | `codec.encoder.*`, `codec.decoder.*` |
| MeanFlow head | 363,407,360 | `head.*` |
| Audio projectors | 6,725,120 each | `model_audio_proj.*`, `user_audio_proj.*` |
| Latent scalars | 1 each | `latent_bias` = -0.04931640625, `latent_scale` = 0.1962890625 (BF16) |

**Language model:** `qwen3_5_text`, 32 layers, hidden 2560, FFN 9216, RMSNorm eps 1e-6. Layers 3, 7, ..., 31 are full
attention (16 query heads, 4 KV heads, head_dim 256, fused `[query|gate]` q_proj [8192, 2560], QK-norm, partial RoPE
0.25, theta 1e7, interleaved M-RoPE sections [11, 11, 10]); the other 24 are Gated DeltaNet (conv kernel 4, 16 key heads
x 128, 32 value heads x 128, `in_proj_qkv` [8192, 2560], `in_proj_z` [4096, 2560], `in_proj_a`/`in_proj_b` [32, 2560],
`A_log`, `dt_bias`, gated norm [128], `out_proj` [2560, 4096]). `attn_output_gate` is true; the engine's `Qwen35Model`
always splits the fused projection, so a checkpoint with it false would load and compute wrongly: refuse it.

**Codec:** 24 kHz, 3200 samples per 64-d latent (7.5 Hz); base 32 channels doubling per stage; ratios [8, 5, 5, 4, 2, 2]
(the encoder runs them reversed); encoder depths [3, 3, 3, 3, 3, 3, 8], decoder [8, 3, 3, 3, 3, 3, 3]; eps 1e-5;
`latent_std` 0.5; no last norm. A block is RMSNorm, depthwise causal conv k7, `gamma`, RMSNorm, `fc1` (4x), GELU, `fc2`,
`ffn_gamma`.

**MeanFlow head:** no biases anywhere. `c = silu(cond_proj(h) + t_embed(0) + dt_embed(7.0) + h_embed(1.0) +
omega_embed(1 - 1/w))`; each embed is cos/sin of 128 frequencies `exp(-ln(1e4) i / 128)`, then `fc1` (256 to 2560), SiLU,
`fc2`. `x = latent_proj(z)`; 4 layers of `shift, scale, gate = ada(c)` (7680 = 3 x 2560),
`x += gate * SwiGLU(RMSNorm(x) * w * (1 + scale) + shift)` with FFN 7680; final layer `shift, scale = ada(c)`,
`proj(RMSNorm(x) * (1 + scale) + shift)` (RMSNorm without weight). One step: `z = noise + 1.0 * u`. Guidance `w` is a network
input (default 1.5, range 1-2): one forward pass, not two-pass CFG. The raw time scalars are used, not scaled by 1000.

## Frame loop

One position is one frame. Input embedding = `embed_tokens(text_t) + model_proj(norm_latent_t) + user_proj(norm_latent_t)`
with `AudioProjector = fc2(RMSNorm(fc1(x)))` (eps 1e-6, biased) and latents normalized as `(raw + latent_bias) *
latent_scale`; a mask zeroes a stream where it does not exist. The hidden state at frame t gives frame t+1: text logits
through the tied embedding, the audio latent through the head, un-normalized by `x / scale - bias`, decoded by the codec.
`start()` runs the prompt and emits frame 0; each `step(user frame k)` runs one position and returns frame k+1.

The user latent is a sample: `mean + r * (0.5 / 0.8) * n`, one scalar `r ~ N(0, 1)` per row and `n ~ N(0, 1)` per element.
PAD (248076, `<|audio_pad|>`) and EOP (248071, `<|audio_end|>`) logits are recomputed in F32 from F32 embedding rows: the silent-or-speak decision is closer
than BF16 resolves. Text sampling: temperature 0.7, top-k 25, DRY (multiplier 0.8, base 1.75, allowed length 1, lookback
1024, max match 32, cap 32, PAD/EOP/EOS exempt), `eop_bias` added to the EOP logit, `speak_first` bans PAD/EOP/EOS for one
frame, 27,000 frames at most.

Prompt: `<|im_start|>system\n{card}<|im_end|>\n` (no audio), then the voice reference (3 s to 3 min, silence-trimmed at
-45 dBFS, last 180 s kept) on the model stream with text PAD ... PAD EOP, then for `speak_first` PAD, EOP. Positions
before generated audio carry the reference's quietest frame on the model stream and the encoded-silence latent on the
user stream. Prefill runs in chunks of 512. There is no explicit barge-in logic: the model sees the user every frame.

## Engine reuse

| Part | Engine owner | Gap |
|---|---|---|
| Trunk | `Qwen35Model.FromHuggingFace(w, cfg, cap, prefix "lm", lmHeadKey null)`; head falls back to `token_embd.weight` | none; the delta-rule scan needs equal key and value head widths (128 and 128 in the release); the embed table must stay F32/BF16/F16 |
| Config | `Qwen35HfConfig.FromTextConfig(root["lm"])`, wrapped by `Models/Haru/HaruConfig` | `attn_output_gate` is not parsed by the engine; `HaruConfig` refuses false |
| Codec | `VibeVoiceAcousticTokenizerModel` (encoder, decoder, streaming cache) | key names only (below) |
| Tokenizer | `HfTokenizerJson.LoadByteLevelBpe`, checked by `Models/Haru/HaruTokenizer` | none |
| Head | `Audio/Layers/AdaLnSwiGluStack` + `ScalarEmbed` (shared with `VibeVoiceDiffusionHead`), `Models/Haru/HaruMeanFlowHead` | none |
| DRY sampler | `LLM/Sampling/DryPenaltyStep` (new, shared), `SamplerChain.FromSteps` | none |

Codec key map (Haru to engine), implemented in `Audio/Models/Haru/HaruCodecWeights.cs` (a key outside it throws) with the
config in `HaruCodec.ConfigFrom`. Verified on the tiny checkpoint below with no change to the shared classes:

| Haru | Engine |
|---|---|
| `encoder.stem.*` | `encoder.downsample_layers.0.0.conv.conv.*` |
| `encoder.down.{i}.*` | `encoder.downsample_layers.{i+1}.0.conv.conv.*` |
| `decoder.stem.*` | `decoder.upsample_layers.0.0.conv.conv.*` |
| `decoder.up.{i}.*` | `decoder.upsample_layers.{i+1}.0.convtr.convtr.*` |
| `{encoder,decoder}.stages.{i}.{j}.conv.*` | `...stages.{i}.{j}.mixer.conv.conv.conv.*` |
| `...fc1` / `...fc2` | `...ffn.linear1` / `...ffn.linear2` |
| `{encoder,decoder}.head.*` | `{encoder,decoder}.head.conv.conv.*` |
| `norm`, `ffn_norm`, `gamma`, `ffn_gamma` | unchanged |

## Engine integration

- Catalog: `TtsCatalog` ids `haru` and `haru-4b` (`Audio/Tts/Models/HaruModel.cs`), a `ModelCatalog` entry and a `ModelIdentityCatalog` row, release files pinned by sha256 (config `ff29b2da...`, tokenizer `06b95093...`, weights `ad2c3534...`).
- The runner is an `ITtsRunner` (cache, eviction, leases and footprint unchanged) that also implements `IDuplexSpeechRunner`; `Synthesize` refuses with a pointer to the duplex API.
- Contract: `ISpeechService.OpenDuplexAsync` and `RespondAsync` as default interface members; `IDuplexSpeechSession` (`Start`, `Step`) over a lease that takes the call lock and the device gate per frame and pins the model. One conversation per loaded model.
- Precisions: `HaruWeights` converts only the language model's attention, delta-rule and feed-forward matrices (Q8_0, Q6_K, or Q4_K/Q5_K for the feed-forward with Q6_K for the rest); the tied embedding table, head, projectors and codec stay as stored. Cache: `CacheRoot/tts/_gguf-cache/higurashi-ai_Haru-4B-<precision>-v1.gguf`.
- Consumers: `hartsy duplex <recording> -m haru:q8 --reference voice.wav --card "..." --stereo`, `POST /v1/native/duplex/respond` (base64 clips in JSON, long-running queue).

## Evidence

Tiny checkpoint: `tools/haru/haru_reference.py` builds a random Haru (hidden 64, 4 layers, vocab 2048, latent 16, codec base
1, equal key and value head widths as in the release) from the unmodified upstream modules and writes fixtures under `tests/HartsyInference.Audio.Tests/Fixtures/Haru/`
(upstream `7fe6b766`, transformers 5.22.2, torch 2.6.0, CPU F32, one thread).

- 2026-11-17, codec, unmodified engine classes built by `HaruCodec.Create`, CPU F32 (`HaruCodecReferenceTests`): whole clip ending mid-frame
  (5 x 3200 + 1234 samples) encode 9.1e-7 relative, whole decode 2.1e-6, three streamed encode frames 8.3e-7, three
  streamed decode frames 3.1e-6. Streaming and whole-clip agree in the engine on a separate synthetic codec
  (`VibeVoiceCodecStreamingTests`, bit-exact on CPU). This settles the codec questions: down/up index base is 0 with a
  separate stem, mid-frame padding matches, no shared-code change is needed.
- 2026-11-17, released weights on the RTX 3060 (CUDA, `hartsy duplex`, the HartsyPhone eval question recorded as a 14 s clip, a voice clip, a card, a
  25 s reply tail so the model has room to answer; the reply Whisper-transcribed with `whisper:base`, which also transcribes the user channel as the
  question, so the check itself is sound):
  - **Q8_0**: coherent English ("A, vanilla cake. B, birthday cake. C, strawberry cake. D, chocolate cake. And, D, no, thank you."), 271 frames, the Whisper
    text matches the model's own text stream, peak VRAM 8.1 GB. 114 ms median per frame with a 167 ms maximum on an idle card (3-second tail run), 184 ms
    median when another process held the card at 43% utilisation: **not yet real time against the 133 ms budget**.
  - **Q4_K, Q5_K, Q6_K**: the replies are not intelligible (mixed-language fragments and repetition for Q4, a lone "Hm" or question marks for Q5 and Q6),
    although they run faster (median 66 to 73 ms per frame). Q4_K had already differed from upstream at frame 3 of the replayed 16-frame
    conversation (14 of 16 tokens); Q8_0 matches all 16. The earlier Q4 run that gave Chinese for an English question used a 3-second tail and the
    same weights: it was a K-quant failure, not a short tail.
  - A 3-second tail is too short for any variant to answer the 14 s question: use `--tail-seconds 20` or more.

- 2026-11-17, released codec, real weights, CPU F32 (`HaruCodecRealWeightsTests`, opt-in through `HARU_CODEC_REAL_DIR`): the
  `codec.*` byte range of `model.safetensors` @ `8d6e83bc` (1,374,784,002 B, which is 687,392,001 BF16 parameters; fetched by
  HTTP range, the whole-file sha256 not checked) against the upstream PyTorch codec loaded `strict=True` from the same
  bytes, on 6.05 s of an 8 kHz phone recording resampled linearly to 24 kHz (a clip of the HartsyPhone eval set,
  `ae91bf895267.wav`, 145,234 samples, ends mid-frame): whole encode 1.1e-5 relative, whole decode of the upstream latents
  2.3e-6, three streamed encode frames 2.7e-6, three streamed decode frames 4.2e-6. 1 m 47 s on the CPU backend.
- 2026-11-17, MeanFlow head, CPU F32 (`HaruMeanFlowHeadTests`): latents from the tiny checkpoint's hidden states and noise,
  un-normalized as the session does, 8.9e-7 relative to upstream. The same head with BF16 weights as stored stays within 1.2e-2
  of the F32 head. `VibeVoiceDiffusionHead` now runs on the same stack and scalar embedder; its output bits on a tiny random
  head are unchanged (`VibeVoiceDiffusionHeadGoldenTests`: bit-identical on the reference host, pinned before the move; the test
  keeps the values with a 1e-6 tolerance so it holds on other CPUs). Other step counts and guidance weights match upstream to the same
  bound (`haru_head_steps_expected.safetensors`, `tools/haru/head_steps_reference.py`).
- 2026-11-17, language model, CPU F32 (`HaruLmReferenceTests`): the stream-summed input (text embedding plus both projected
  latent streams, masks, `(raw + bias) * scale`), the trunk over the 22-position prompt in one call and then six single
  positions: hidden states 3.8e-6 relative to upstream, text logits of the six steps 8.6e-7, the F32 PAD/EOP logits 7.0e-7.
  The same fixture reran the codec (5e-7) and the head (6.1e-7).
- 2026-11-17, released `tokenizer.json` (sha256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`,
  `HaruTokenizerTests`, opt-in through `HARU_TOKENIZER_JSON`): the engine's loader reads it, `<|audio_pad|>` is 248076 and
  `<|audio_end|>` is 248071 as the config says, and a mixed English and Japanese system prompt encodes to the same 18 ids as
  the `tokenizers` library. The released config parses and passes `HaruConfig`'s checks.
- 2026-11-17, DRY penalty (`DryPenaltyStepTests`): the penalties equal the upstream sampler's on eight fixed histories (random,
  periodic with exempt gaps, longer than the window, allowed length 2, capped, one token, empty, only exempt tokens).
- 2026-11-17, the conversation loop, CPU F32, greedy (`HaruSessionReferenceTests`): prompt build, 512-position prefill, six user
  frames stepped, text and audio out, three runs (card and reference; empty prompt; speak-first with an EOP bias of 2). Every random
  draw of the upstream run is recorded and replayed in order (the posterior of the reference and of each user frame, the head noise),
  all draws are consumed, and **tokens and text are identical** in all three runs (7 frames each). Audio, un-clamped and speech-sized
  (peak 0.49-0.58), matches to 3.2e-5, 1.0e-4 and 2.7e-4 relative to peak. That is float noise fed back through the language model: the
  codec alone agrees to 5e-7. Reference preparation (silence trim, margin, last N seconds) is equal to upstream's, sample for sample.
  The tiny codec's output heads are scaled by 1e-3 (encoder) and 1e-4 (decoder) in the fixture so its audio is not clamped.
- 2026-11-17, **released weights**, CPU, the checkpoint verified against its pin (10,540,113,510 B, sha256 `ad2c3534...47c0fa`),
  upstream run in PyTorch with the same BF16 weights widened to F32 per call (`tools/haru/real_reference.py`), engine run on its CPU
  backend (`HaruRealWeightsTests`, opt-in through `HARU_REAL_DIR`; 1 m 24 s and 8 m 59 s):
  - fixed inputs, a 14-position prompt then 4 single positions: hidden states 5.1e-5 relative, text logits 4.8e-5, F32 PAD/EOP logits
    2.7e-5, MeanFlow head latents 9.5e-7;
  - a conversation: character card, a voice clip and a user recording from the HartsyPhone eval set (8 kHz phone speech resampled
    to 24 kHz), greedy, 16 frames, every random draw of the upstream run replayed (all 48 consumed): **tokens and text identical**
    (PAD, PAD, EOP, "Hello", ",", then silence), audio 1.9e-3 relative to peak. The model starts to speak by itself on the recording.
- Not run: a long generation (16 frames is 2.1 s; no Whisper check yet); any GPU run; the moved VibeVoice head on
  a GPU or in a real VibeVoice generation; wideband speech; GPU.

Regression net rebuilt for shared code (all CPU): `VibeVoiceCodecStreamingTests` (whole vs streamed, pinned decode),
`Qwen35HuggingFaceParityTests` (tiny Qwen3.5 trunk vs transformers, 2e-4; restored from `e9474bb06^`),
`SamplerChainGoldenTests` (pinned token digests of three configurations). Real-weight baselines (VibeVoice-1.5B speech
through Whisper, Qwen3.5 GGUF greedy digest) need weights that are not on the dev box and are still to be taken.
To rerun the real-weight codec check: fetch the `codec.*` range into a standalone safetensors file (`tools/haru/fetch_codec.py`), run the upstream codec
on a clip (`tools/haru/codec_reference.py`) to write `codec_real_expected.safetensors`, then `HARU_CODEC_REAL_DIR=<dir> HARTSY_REQUIRE_REAL_WEIGHTS=1 dotnet test
tests/HartsyInference.Audio.Tests -m:1 --filter FullyQualifiedName~HaruCodecRealWeights`.

## Costs and traps

- The tied head is BF16 `[248,320, 2,560]`. On the CPU backend `LinearTransB` widens a non-F32 weight to F32 on every call, about
  2.5 GB of allocation per projection, so a CPU run must project only the rows it needs: `HaruLm.Logits` takes the last `rows`
  rows of a hidden-state tensor, and the frame loop asks for one. PAD and EOP never go through the table; `StreamLogits` reads
  their two rows directly.
- `VibeVoiceAcousticTokenizerModel.Decode` tells channels-last `[B, T, D]` from channels-first `[B, D, T]` by `Shape[1] == VaeDim`. A
  whole-clip decode of exactly `VaeDim` frames (64 frames, 8.53 s, on the release; 16 on the tiny fixture) is read the wrong way
  with no error. The frame loop decodes one frame at a time (`Shape[1] == 1`), which is safe; a whole decode of that length must
  pass channels-first. Not fixed here: the class is shared by every VibeVoice model.
- `HaruLm` clamps its capacity to the model's `max_position_embeddings` and exposes the result as `Capacity`; the session reports
  a conversation that reaches it as done rather than failing inside the trunk cache.
- Where the session follows upstream on purpose: the DRY penalty is applied to the raw logits before the temperature divides them
  (`_emit` before `sample_token`), with PAD, EOP and EOS exempt; an EOS the model samples after the first frame is ordinary text, as
  upstream does not end the session on it. Where it differs: bytes of a half-finished character are given up when a gap starts or
  after four tokens, instead of being carried across the gap (`HaruStreamDetokenizer`); `Step` refuses to run after the last frame.

## Doc debt found

Commit `e9474bb06` ("clean house") removed every VibeVoice, Qwen3.5, Drama 3 and Clef test, `tools/fish_audio/*` and
`docs/Research/VOICE_AGENT_SESSION.md`. The Drama 3 evidence rows in `PARITY_VERIFICATION.md` and `MODEL_STATUS_AUDIO.md`
still cite tests and scripts that no longer exist, and `MODEL_STATUS_LLM.md` (Clef rows) cites `tools/clef` scripts. The
dead voice-session links and the `RealtimeScheduling` entry in `docs/Agents/AGENTS.md` were removed with this note.

## Open questions

- Whether the full 110 codec tensors equal VibeVoice-1.5B's (two compared).
- Why the K-quants lose the conversation (see Evidence, 2026-11-17 GPU): which matrices are sensitive (attention, delta-rule, first or last layers) is not
  known; a per-matrix sweep would tell whether a mixed Q8_0/K-quant file keeps coherence at less than Q8_0's size and time.
- Q8_0 decode speed on the 3060 is 114 ms median per frame with a 167 ms maximum against the 133 ms budget (`AudioLmQuant.cs` says Q8_0 takes a slow path).
- Whether `Qwen35Model` per-step decode of 4B on the 3060 meets the 133 ms frame with the codec; not measured.
- Behaviour on narrowband phone audio, behind RNNoise, and without headphones: untested upstream.
