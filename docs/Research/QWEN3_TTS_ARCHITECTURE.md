# Qwen3-TTS (12Hz) — Architecture

> Build-ready spec for `Qwen/Qwen3-TTS-12Hz-{1.7B,0.6B}-{Base,CustomVoice,VoiceDesign}` + the standalone
> `Qwen/Qwen3-TTS-Tokenizer-12Hz` codec. Sources: official `QwenLM/Qwen3-TTS` (`qwen_tts`) source,
> HF config.json files, tech report. Fetched 2026-06-19. **RVQ codec, NOT FSQ** (refuted an early claim).
> Repo note: bare `-1.7B`/`-0.6B` don't exist; "Base" = the voice-clone model; no `-0.6B-VoiceDesign`.

## Talker LM (Qwen3 decoder, `model_type qwen3_tts_talker`)

| field | 1.7B | 0.6B |
|---|---|---|
| hidden_size | 2048 | 1024 |
| num_hidden_layers | 28 | 28 |
| num_attention_heads | 16 | 16 |
| num_key_value_heads | 8 (GQA 2:1) | 8 |
| head_dim | 128 | 128 |
| intermediate_size | 6144 | 3072 |
| codec vocab_size | 3072 | 3072 |
| text_vocab_size | 151936 | 151936 |
| rope_theta | 1e6 | 1e6 |
| rms_norm_eps | 1e-6 | 1e-6 |
| attention_bias | false | false |
| num_code_groups | 16 | 16 |

**Qwen3-specific (from source, NOT config):**
- `q_norm`/`k_norm` = `RMSNorm(head_dim=128, eps)`, applied **per head after q/k proj + head reshape, before
  RoPE**. Keys `self_attn.{q_norm,k_norm}` shape `[128]`.
- **head_dim decoupled from hidden_size**: q_proj → `16*128=2048`, k/v_proj → `8*128=1024`, o_proj →
  `2048→hidden`. For 0.6B hidden=1024 but q_proj still outputs 2048 — do NOT assume `head_dim=hidden/heads`.
- **3D interleaved mRoPE** (net-new): `rope_scaling={interleaved:true, mrope_section:[24,20,20]}`,
  `position_id_per_seconds:13`. Sum 64 = head_dim/2. Like Qwen2-VL N-axis rope.
- Dual embeddings (separate `text_embedding` 151936 + `codec_embedding` 3072) + `text_projection` MLP;
  `codec_head: 2048→3072` predicts **codebook 0 only**.

**CodePredictor (MTP sub-talker, `code_predictor_config`):** predicts codebooks 1..15. hidden 1024, 5 layers,
16 heads / 8 KV, head_dim 128, intermediate 3072, vocab 2048, **plain 1D rope** (rope_scaling null), 15
separate `codec_embedding.{0..14}` + 15 `lm_head.{0..14}` + `small_to_mtp_projection`.

## Codec `Qwen3-TTS-Tokenizer-12Hz` (`qwen3_tts_tokenizer_12hz`)
**Encoder = Kyutai Mimi** (frozen; first 16 of 32 quantizers). **Decoder = custom** (net-new). RVQ
`SplitResidualVectorQuantizer`: 1 semantic (dim 256, size 4096) + 15 acoustic (size 2048, dim 512). 24 kHz
in/out, **12.5 Hz** (24000/1920). Decoder = time-domain causal-conv vocoder (NOT iSTFT):
`pre_conv` (512→1024 k3) → **8-layer causal sliding-window transformer** (hidden 512, 16 heads, head_dim 64,
RoPE θ=10000, sliding_window 72, LayerScale 0.01) → `upsample [2,2]` (CausalTransConv + ConvNeXtBlock) →
`decoder` (CausalConv 1024→1536 k7 + 4 DecoderBlocks over upsample_rates **[8,5,4,3]**, SnakeBeta +
transposed conv + 3 residual units) → SnakeBeta + conv→1ch k7 + clamp. Total upsample 8·5·4·3·2·2 = **1920**.

## Tokens / modes
Codec control ids (1.7B): pad 2148, bos 2149, **eos 2150** (stop when codebook 0 == 2150), think/nothink
2154/2155, think_bos/eos 2156/2157, language ids 2050–2074. Codec space: 0–2047 real, 2048+ control.
Text specials: im_start 151644, im_end 151645, assistant 77091, audio_start/end 151669/151670, tts_pad
151671, tts_bos/eod 151672/151673. **No `<|speech_start|>` in the text stream** — audio framing is codec
control ids only. Modes: `base`=voice_clone (ICL ref-text+audio, or x-vector via ECAPA), `custom_voice`=
9 built-in speakers (spk_id token + instruct), `voice_design`=free-form instruct. Prompt template:
`<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`. Codec prefill (auto): `[nothink,
think_bos, think_eos, codec_pad, codec_bos]`.

## Generation
Talker: do_sample, temp 0.9, top_p 1.0, top_k 50, rep_penalty 1.05. Sub-talker same. min_new 2. Stop on
codebook-0 == codec_eos (2150). max_new_tokens 2048 (code) / 8192 (1.7B-CustomVoice config — config wins).

## C# build implications (large — multi-component)
1. **Qwen3 backbone** (reusable infra): extend the Qwen2 attention with per-head q_norm/k_norm + explicit
   `head_dim` config field (decouple from hidden) + **3D interleaved mRoPE** (the hard net-new piece; reuse
   the video N-axis rope if portable). Drop QKV bias.
2. **MTP CodePredictor**: 5-layer transformer (plain rope) with 15 embeddings + 15 heads, conditioned on the
   talker hidden — similar shape to the Moshi depformer (reuse that pattern).
3. **Codec decoder (net-new vocoder)**: split-RVQ dequant + pre_conv + 8-layer causal sliding-window
   transformer (reuse the NeuCodec transformer-block pattern) + ConvNeXt upsamplers + SnakeBeta DecoderBlocks
   ([8,5,4,3]×[2,2]). No iSTFT. SnakeBeta + ConvNeXt causal-conv upsample stack are new.
4. **Mimi encoder** reused for ref-audio encoding (clone mode); **ECAPA-TDNN** speaker encoder net-new
   (x-vector clone mode) — deferrable.
5. Dual embedding tables + text_projection + codec_head; suppress control tokens at sampling.

**Assessment:** the most complex model in the AudioLab set — Qwen3 (q/k-norm + 3D mRoPE) + MTP + a full
from-scratch Snake/ConvNeXt causal vocoder + ECAPA. Build in stages: (a) Qwen3 backbone + mRoPE (reusable),
(b) MTP CodePredictor, (c) codec decoder, (d) pipeline + modes, (e) ECAPA (last).

## C# build status (2026-06-20) — stage (a) done
- [x] **`Qwen3Model`** ([`Models/LanguageModels/Qwen3/`](../../src/HartsyInference.Audio/Models/LanguageModels/Qwen3/)) — headless Qwen3 decoder: per-head **q_norm/k_norm** (RMSNorm over head_dim, pre-RoPE), **decoupled head_dim** (q/k/v sized n_heads·head_dim), GQA, SwiGLU, no QKV bias. Reuses `DiaHeads`/`RotaryEmbedding`/`WhisperOps`. **Synthetic-forward verified** (q/k-norm + GQA + RoPE finite). Reusable by Qwen3-TTS talker + code-predictor AND future Qwen3 image/text models. `Qwen3Config.Talker1_7B`/`CodePredictor` presets.
- [ ] **Staged (b–e):** the talker's dual embedding (text + codec) + `codec_head` + `text_projection`, the **3D interleaved mRoPE** (mrope_section [24,20,20] — currently 1D RoPE), the 5-layer **MTP CodePredictor** (15 codebooks), the custom **Snake/ConvNeXt causal-conv codec decoder** (split-RVQ + ConvNeXt upsample + DAC DecoderBlocks), the Mimi encoder reuse, the **ECAPA-TDNN** speaker encoder, and the 3 modes (clone/custom_voice/voice_design).
