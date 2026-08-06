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

## Checkpoint key reconciliation (verified against real weights, 2026-06-24)

Downloaded headers from `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (`model.safetensors`,
`speech_tokenizer/model.safetensors`) and its `config.json`. Config dims confirm `Qwen3TtsConfig` (text_vocab
151936, codec_vocab 3072, 16 code groups, talker hidden 2048/28L/16h-8kv/head_dim 128/inter 6144, mRoPE
[24,20,20]; codec decoder_dim 1536, upsample_rates [8,5,4,3], upsampling_ratios [2,2], num_quantizers 16,
semantic_codebook 4096, acoustic codebook 2048).

### Talker + MTP — DONE (reconciled + tests updated)
Real keys: embeddings/norm/layers under `talker.model.*`; `talker.codec_head.weight`;
`talker.text_projection.linear_fc1/linear_fc2.{weight,bias}`; MTP under `talker.code_predictor.*` with
`small_to_mtp_projection` + `lm_head.{0..14}` at that level and `model.codec_embedding.{0..14}` + `model.layers.*`
+ `model.norm` under `talker.code_predictor.model.*`. The backbones load cleanly via the headless Qwen3 loader
(HF Llama layer naming: input_layernorm, mlp.{gate,up,down}_proj, post_attention_layernorm,
self_attn.{q,k,v,o}_proj, self_attn.{q,k}_norm). `Qwen3TtsTalker`/`Qwen3MtpCodePredictor` LoadWeights updated.

### ECAPA speaker encoder — NEEDS STRUCTURAL REWORK
Real `speaker_encoder.*` is a variant, NOT SpeechBrain-standard. There is no `tdnn` stem, no `bn`, no `proj`:
- stem = `speaker_encoder.blocks.0.conv.{weight,bias}` (conv 128->512, k5).
- SE-Res2 blocks = `speaker_encoder.blocks.{1,2,3}` with inner `res2net_block.blocks.{n}.conv.*` (22 tensors each).
- `speaker_encoder.mfa.conv.{weight,bias}` (1536->1536 k1; engine reads `mfa.weight`).
- ASP: `speaker_encoder.asp.tdnn.conv.*` (128, in 4608=1536*3) + `speaker_encoder.asp.conv.*` (1536->128... ->1536).
- output = `speaker_encoder.fc.{weight,bias}` ([2048,3072,1]); enc_dim 2048. No final BN/proj.
`EcapaSpeakerEncoder` (stem `.tdnn`, `.mfa.weight`, `.bn.*`, `.proj.*`) must be reshaped to this layout.

### Codec DECODER (Qwen3-TTS-Tokenizer-V2) — DONE + REAL-WEIGHTS VERIFIED (2026-06-24)
`Qwen3TtsVocoder` reconciled to the real `decoder.*` tree and verified by loading the actual 682 MB
`speech_tokenizer/model.safetensors`: all 271 decoder keys consumed (the only 2 unread keys are the
encode-only `input_proj`s), and a decode runs to finite 24 kHz PCM of length T*1920. EMA codebooks
(`embedding_sum`/`cluster_usage`) normalized at load; `rvq_first`+`rvq_rest` each lift via their
`output_proj`; pre_conv → pre_transformer(input_proj, LayerScale layers, norm, output_proj) → 2x ConvNeXt
upsample → 4 SnakeBeta decoder blocks. The gated C# codec test was removed in the 2026-08-06 suite cleanup.
Audio-quality parity vs the Python reference is the remaining follow-up.

### Codec ENCODER (clone-mode ref audio) — NEEDS REWORK
`speech_tokenizer/model.safetensors` = `decoder.*` (271) + `encoder.*` (225). The engine `Qwen3TtsVocoder` has
the right *concept* (split-RVQ -> transformer -> pre_conv -> ConvNeXt upsample -> Snake decoder) but the real
structure differs:
- **EMA codebooks**: real stores `_codebook.embedding_sum` + `_codebook.cluster_usage` (and `initialized` on the
  encoder side). The usable codebook = `embedding_sum / cluster_usage.clamp_min(eps)`. The loader/converter must
  normalize; the engine expects a ready `codebook`.
- **RVQ wrapping**: `decoder.quantizer.rvq_first` (semantic, 1 layer) + `decoder.quantizer.rvq_rest` (15 acoustic
  layers), each with a SINGLE `input_proj`/`output_proj` over the residual stack (engine has per-codebook in_proj).
- **pre_transformer**: `decoder.pre_transformer.{input_proj,output_proj,norm}` + `layers.{0..7}` with
  `self_attn_layer_scale.scale` + `mlp_layer_scale.scale` (LayerScale) and Qwen-style q/k/v/o + gate/up/down
  (no q/k norm). Engine transformer lacks the proj wrappers + LayerScale.
- **upsample**: `decoder.upsample.{i}.{j}` (TWO indices) with ConvNeXt `dwconv.conv` + `pwconv1` + `pwconv2` +
  `gamma` + `norm` (engine uses single-index `upsample.{i}`).
- **decoder**: `decoder.decoder.{n}` with SnakeBeta `alpha`/`beta`, `block.{m}` residual units
  (`act1/act2` SnakeBeta + `conv1.conv`/`conv2.conv`), and `conv.conv` up/out convs. Plus top-level
  `decoder.pre_conv.conv.*`. All convs carry a `.conv.` wrapper.
- **encoder** (Mimi-style, ref-audio clone): `encoder.downsample.conv`, `encoder.encoder.layers.*`,
  `encoder.encoder_transformer.layers.*` (post-LN, `mlp.fc1/fc2`, LayerScale), and
  `encoder.quantizer.{semantic,acoustic}_residual_vector_quantizer.*` (EMA codebooks).

**Bottom line:** talker + MTP are done. The codec decoder, the codec encoder, and the ECAPA encoder need
structural rework to match these real layouts (not just key remaps), then real-weight audio parity on a GPU.

## Generation procedure — reverse-engineered from the pure-C reference (2026-06-24)

Source: `github.com/gabriele-mastrapasqua/qwen3-tts` (pure-C, no Python) + its MODEL.md. This is the exact
talker generation the engine's `Qwen3TtsPipeline.Generate` must reproduce (its current per-frame-text loop is
WRONG). The talker is a **dual-stream** model: every position sums a TEXT-side embed (text_embedding ->
text_projection fc1/SiLU/fc2) and a CODEC-side embed (codec_embedding), so the existing `EmbedStep(text, codec)`
sum primitive is correct, but the flow is prefill-then-free-AR, not interleaved.

### M-RoPE
`mrope_section=[24,20,20]`, theta 1e6, interleaved. For TTS (text-only input) all three position sections are
identical, so **it reduces to standard interleaved RoPE with sequential positions 0,1,2,...** No 3D logic needed.

### Special tokens
Text vocab: im_start 151644, assistant 77091, "\n" 198, im_end 151645, tts_pad 151671, tts_bos 151672,
tts_eos 151673. Codec vocab: codec_pad 2148, codec_bos 2149, codec_eos 2150, think 2154, no_think 2155,
think_bos 2156, think_eos 2157. Language ids (codec vocab): English 2050, Chinese 2055, Japanese 2058,
Korean 2064, German 2053, French 2061, Russian 2069, Portuguese 2071, Spanish 2054, Italian 2070. Preset
speakers (codec tokens, CustomVoice model only): Ryan/en 3061, Serena/zh 3066, Ono-Anna/ja 2873, Sohee/ko 2864
(9 total; remaining ids TBD from the CustomVoice checkpoint).

### Codec control prefix (built once, paired into the prefill)
- CustomVoice + language: `[think, think_bos, language_id, think_eos, speaker, codec_pad, codec_bos]`
- CustomVoice no language: `[no_think, think_bos, think_eos, speaker, codec_pad, codec_bos]`
- VoiceDesign: same but WITHOUT the speaker token.
- VoiceClone: speaker slot is a continuous ECAPA/ref embedding (norm-matched to a preset, e.g. ryan 3061).
The engine's `CodecPrefill()` and `SynthesizeCustomVoice` put the speaker AFTER pad/bos — wrong order.

### Dual-stream prefill sequence (text token, codec token) summed per position
0. Instruct tokens (text-only; system instruct, often empty).
1. Role prefix `[im_start, assistant, "\n"]` (text-only).
2. For the codec prefix minus its last element: text = tts_pad (last of this section = tts_bos), codec =
   prefix[i]. (length = codec_prefix_len - 1)
3. Text content tokens then a trailing tts_eos: text = token/tts_eos, codec = codec_pad each.
4. Final single position: text = tts_pad, codec = codec_bos.
Run all of the above as one prefill building the KV cache.

### Autoregressive generation
Each frame: input = tts_pad (text) + **sum of all 16 codebook embeddings of the previous frame** (cb0 via
`talker.model.codec_embedding`, cb1..15 via `code_predictor.model.codec_embedding.{0..14}`). Talker step ->
codebook-0 via `codec_head` (sample: temp 0.9, top_k 50, top_p 1.0, rep_penalty 1.05 over the WHOLE generated
cb0 sequence, once per unique id). **suppress_tokens (official `modeling_qwen3_tts.py` generate): every id in
`[vocab_size-1024, vocab_size)` = [2048, 3072) is masked to -inf EXCEPT codec_eos 2150** — without this the
talker samples codec_pad/control ids and EOS never fires (600+ frame runaway). min_new_tokens=2 masks EOS for
the first 2 frames. The code-predictor **SAMPLES** cb1..15 (subtalker_dosample=true, temp 0.9, top_k 50,
top_p 1.0, no rep penalty) — the pure-C reference's greedy passes were a simplification, NOT upstream. Stop
when cb0 == codec_eos (2150); cap at max_new_tokens (8192). Map control cb0 (>= 2048) to silence 0 before the
codec decoder.

### Engine deltas to implement (the remaining work to make custom_voice/voice_design produce speech)
1. `Qwen3TtsPipeline.Generate`: replace the per-frame-text loop with the dual-stream prefill above + free AR.
2. Per-frame codec input must sum all 16 codebook embeds of the previous frame (needs the talker to read the
   code-predictor's `codec_embedding` tables, or the MTP to expose them).
3. Config: fix the codec-prefix order/contents, speaker ids, language ids, MTP sampled (not greedy) decoding.
4. Loader: the talker ships SHARDED (`model.safetensors.index.json` + shards, BF16) + the codec; text via the
   embedded Qwen3 BPE tokenizer with the ChatML template. Validate end to end against the C reference output.

### 0.6B checkpoint dims (verified from Qwen3-TTS-12Hz-0.6B-CustomVoice headers, 2026-07-02)
`text_hidden_size` stays **2048** while hidden drops to 1024: `text_embedding [151936, 2048]`,
`text_projection.linear_fc1 [2048, 2048]`, `linear_fc2 [1024, 2048]` — the text row stride and fc1 in-dim must
use text_hidden, NOT hidden (this exact 1.7B-ism was the 0.6B `CUDA_ERROR_INVALID_VALUE`). Talker-width tables:
`codec_embedding [3072, 1024]`, `codec_head [3072, 1024]`, MTP `codec_embedding.{k} [2048, 1024]`.
`small_to_mtp_projection` is ABSENT on 0.6B (talker hidden == MTP hidden 1024 → identity); q_proj stays
[2048, 1024] (decoupled head_dim 16×128). lm_head.{k} [2048, 1024] on both sizes.
