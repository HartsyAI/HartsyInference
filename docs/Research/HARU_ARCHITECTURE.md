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
PAD (248076) and EOP (248071) logits are recomputed in F32 from F32 embedding rows: the silent-or-speak decision is closer
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
| Trunk | `Qwen35Model.FromHuggingFace(w, cfg, cap, prefix "lm", lmHeadKey null)`; head falls back to `token_embd.weight` | none expected; embed table must stay F32/BF16/F16 |
| Config | `Qwen35HfConfig.FromTextConfig(root["lm"])` | `attn_output_gate` is not parsed |
| Codec | `VibeVoiceAcousticTokenizerModel` (encoder, decoder, streaming cache) | key names only (below) |
| Tokenizer | `HfTokenizerJson.LoadByteLevelBpe` | check `tokenizer.json` loads |
| Head | `VibeVoiceDiffusionHead` is key-bound and `internal` | new shared layer stack, see the plan |
| DRY sampler | none in `src/` | new `ISamplerStep` |

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

## Evidence

Tiny checkpoint: `tools/haru/haru_reference.py` builds a random Haru (hidden 64, 4 layers, vocab 2048, latent 16, codec base
1) from the unmodified upstream modules and writes fixtures under `tests/HartsyInference.Audio.Tests/Fixtures/Haru/`
(upstream `7fe6b766`, transformers 5.22.2, torch 2.6.0, CPU F32, one thread).

- 2026-11-17, codec, unmodified engine classes built by `HaruCodec.Create`, CPU F32 (`HaruCodecReferenceTests`): whole clip ending mid-frame
  (5 x 3200 + 1234 samples) encode 9.1e-7 relative, whole decode 2.1e-6, three streamed encode frames 8.3e-7, three
  streamed decode frames 3.1e-6. Streaming and whole-clip agree in the engine on a separate synthetic codec
  (`VibeVoiceCodecStreamingTests`, bit-exact on CPU). This settles the codec questions: down/up index base is 0 with a
  separate stem, mid-frame padding matches, no shared-code change is needed.
- Not run: trunk, projectors, head and the full frame loop against the fixture; any real-weight run.

Regression net rebuilt for shared code (all CPU): `VibeVoiceCodecStreamingTests` (whole vs streamed, pinned decode),
`Qwen35HuggingFaceParityTests` (tiny Qwen3.5 trunk vs transformers, 2e-4; restored from `e9474bb06^`),
`SamplerChainGoldenTests` (pinned token digests of three configurations). Real-weight baselines (VibeVoice-1.5B speech
through Whisper, Qwen3.5 GGUF greedy digest) need weights that are not on the dev box and are still to be taken.

## Doc debt found

Commit `e9474bb06` ("clean house") removed every VibeVoice, Qwen3.5, Drama 3 and Clef test, `tools/fish_audio/*` and
`docs/Research/VOICE_AGENT_SESSION.md`. The Drama 3 evidence rows in `PARITY_VERIFICATION.md` and `MODEL_STATUS_AUDIO.md`
still cite tests and scripts that no longer exist, and `MODEL_STATUS_LLM.md` (Clef rows) cites `tools/clef` scripts. The
dead voice-session links and the `RealtimeScheduling` entry in `docs/Agents/AGENTS.md` were removed with this note.

## Open questions

- Whether the full 110 codec tensors equal VibeVoice-1.5B's (two compared).
- Q8_0 against Q4_K decode speed for the trunk on the 3060 (`AudioLmQuant.cs` claims Q8_0 takes a slow path).
- Whether the engine's `tokenizer.json` loader reads Haru's tokenizer and special ids.
- Whether `Qwen35Model` per-step decode of 4B on the 3060 meets the 133 ms frame with the codec; not measured.
- Behaviour on narrowband phone audio, behind RNNoise, and without headphones: untested upstream.
