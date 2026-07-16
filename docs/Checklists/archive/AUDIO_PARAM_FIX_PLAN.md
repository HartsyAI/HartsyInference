# Audio Model Parameter Fix Plan

Companion to [AUDIO_PARAM_AUDIT.md](AUDIO_PARAM_AUDIT.md). The audit found, across 44 audio model
families: **224 missing params, 88 wrong values, 39 unverified, 49 extra**. This plan sequences the
fixes so shared dependencies land first, cheap high-confidence corrections land before risky rewrites,
and each phase ends in a buildable, checkpoint-loadable state.

## Guiding principles

- **Validate against references.** A value change is not "done" until the affected model still builds
  and (where weights are available) loads the real checkpoint without a shape mismatch. True numerical
  parity for the heavier items runs in the bare-terminal env with the model cache, per the existing
  parity workflow. See [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md).
- **Shared configs first.** Qwen2Config, Qwen3Config, GptConfig, and the Mimi codec are reused by many
  models. Fix and extend them once (Phase 0) so per-model work in later phases just sets fields.
- **Config before code.** Most gaps are a missing field or a wrong default (Phases 1 to 3). Only the
  items in Phase 4 require new model-code paths or architecture rewrites.
- **No silent scope creep.** Items explicitly out of scope are listed at the bottom. Adding a field for
  a variant we will not wire is fine (records the truth) only when called out as a no-op.
- **Commits are the user's.** I edit code and report; the user does all git.

Status key per item: `[ ]` todo, `[~]` in progress, `[x]` done + builds, `[V]` validated on real weights,
`[G]` validation-gated (do NOT apply blind, see below).

## CRITICAL: the audit conflicts with real-weight validation in places

Cross-checking against [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md) shows the research audit (HF/GitHub
config.json) disagrees with several models that are already validated **bit-exact on real weights**. The
audit was wrong in at least one provable case, so its value changes are NOT authoritative where they touch
a validated model:

- **Qwen3-TTS talker** is validated bit-exact at `HiddenSize=2048`; the audit said "fix to 1024". The audit
  guessed the wrong checkpoint. **Do not change the talker dims.**
- **Spark-TTS** (validated) and **CosyVoice 2** (validated) both consume the shared `Qwen2Config.Qwen25_0_5B`
  preset, so editing that shared preset (as the audit suggested for VibeVoice) would regress them. The
  VibeVoice-Realtime fix belongs on `VibeVoiceDecoderConfig`, not the shared Qwen2 preset.
- **MeloTTS** (audio corr 0.9993), **GPT-SoVITS v2**, **Kyutai LM**, **CosyVoice flow** are all validated;
  their flagged value changes (`NumLanguages`, `FlowFlows`, `PhonemeVocab`, `IntermediateSize`,
  `UnetChannels`) risk regression and must be re-validated on weights before applying.
- Many changes are **coupled** (e.g. CSM `NumCodebooks` must move in lockstep with the Mimi codec codebook
  count) and cannot be verified without loading the checkpoints.

**Rule:** value changes are applied directly only for models/subcomponents that are NOT real-weight
validated. Value changes that touch a validated model are marked `[G]` and handed to the user to apply +
re-run parity in the bare-terminal env. Additive changes (new optional fields defaulting to current
behavior, new variant presets) are safe everywhere.

Validated audio models (per PARITY_VERIFICATION.md), treat value changes as `[G]`: GPT-SoVITS v2,
Chatterbox, Qwen3-TTS (talker LM), Piper/VITS, Kokoro, CosyVoice 2, F5-TTS, Spark-TTS, MeloTTS, Whisper,
Moonshine, OpenVoice, MusicGen (+ its EnCodec-32k), ACE-Step. Partially validated (LM only, codec not):
Kyutai TTS, HeartMuLa, FishSpeech, Dia, YuE Stage-1.

## Progress log

### 2026-06-28 batch 1 (31 files, +573 lines, Audio builds clean: 0 warn / 0 err)

Direct value fixes (isolated, non-validated, high confidence): Bark `CoarseInferToken`; YuE Stage-2 dims;
RMVPE `VoicingThreshold`/`FirstFreqHz`/`NFft`/`WinLength`; SNAC 32k/44k rates/strides/dims.

Fan-out batch (26 agents, one model each, disjoint files):
- Value corrections to non-validated models/subcomponents: CSM (codebooks 32 + Llama-3 RoPE + token ids,
  `[G]`-coupled to Mimi), WavTokenizer, Qwen3-TTS vocoder/ECAPA (talker LEFT untouched), Orpheus
  (pad + Llama-3 RoPE), BiCodec `GlobalFsqLevels`, VibeVoice realtime-0.5B decoder, StyleTTS2, NeuTTS
  (+ Nano preset), Zonos (`PadVocabToMultipleOf` + lang count), HuBERT (+ presets).
- Additive only (validated or shared, no regression): F5-TTS, Moonshine, Whisper (+.en), Bert (+JP/ZH),
  Dia, Vocos (+Encodec24k), EnCodec (+48k), MusicGen (+stereo/melody), VITS (+Ljs/Mms/Vctk), Spark-TTS,
  HeartMuLa, FishSpeech (+1.4), NeuCodec, HtDemucs (+ft/6s), Resemble-Enhance, Bark/GptConfig.

Wiring is NOT done: every new field is config-only until a model consumer reads it (Phase 2/4 follow-up).
Behavior changes that take effect immediately and need a parity check: BiCodec FSQ encode now reads 6
levels; SNAC 32k/44k presets; RMVPE front-end; Bark coarse token. CSM/Mimi codebook coupling is unwired.

Excluded as provably regressive (kept current validated values): Qwen3-TTS talker dims, shared
`Qwen2Config.Qwen25_0_5B` tie/maxpos. Still `[G]` (your validation env): MeloTTS, GPT-SoVITS, CosyVoice,
Kyutai LM value changes.

### 2026-06-29 batch 2 (SNAC codec wiring, first Phase 4 item, Audio builds clean)

Fetched the official hubertsiuzdak/snac `layers.py` and `snac_24khz/config.json` (`depthwise=true,
noise=true, attn=null`) and reconciled the codec, which previously could not load the real weights:
- `SnacDecoder` rewritten: depthwise initial (k7-depthwise + k1-pointwise), `NoiseBlock` at `block.2`
  (residual units shifted to `block.3+`), residual `groups`, final `Tanh`.
- `SnacResidualUnit` gained a `groups` parameter (depthwise first conv).
- `SnacEncoder` aligned to source (residual `groups`, final conv `groups`, removed the spurious final Snake
  our code expected at `block.{N+1}`).
- `Snac.LoadWeights` now loads the encoder tolerantly; `Encode` throws clearly if the encoder is unloaded,
  so the Orpheus decode path is unaffected by any remaining encoder reconcile.
PARITY-TODO throughout (no real-weight numeric check here). LocalMHA for 32/44 kHz still pending.

### 2026-06-29 batch 3 (SNAC real-weight parity VERIFIED)

This box has a torch (CPU) + 3060, so parity is run here, not just deferred. Downloaded the real
`hubertsiuzdak/snac_24khz`, built a torch oracle (`tests/python-reference/snac_reference/`), and verified
the SNAC decode at **corr 1.000000 / maxAbs 9.8e-6** (test `SnacParityTests`). Confirmed the real checkpoint
ships the torch>=2.1 `parametrizations.weight.original0/1` weight-norm format and that the encoder/decoder
key layout (depthwise split, NoiseBlock at `block.2`, residual units at `block.3+`, final conv with no
Snake) matches the official source. `Snac.LoadWeights` now normalizes that key format via `CodecKeyUtils`,
so real Orpheus SNAC loading works. Method going forward: for each remaining model that fits the 3060,
download weights + build a torch oracle + verify corr/maxAbs; only items too big for 12 GB get a
verify-on-larger-GPU TODO.

### 2026-06-29 batch 4 (codec verification sweep)

- **DAC 44.1 kHz**: decode corr **1.000000**, maxAbs **6.4e-7** vs `descript-audio-codec` real weights
  (test `DacParityTests`, oracle `tests/python-reference/dac_reference/`). Standalone confirmation of the
  codec under Spark/Dia/IndexTTS/Higgs (previously only indirect via Spark BiCodec). No code changes needed.
- **Mimi**: investigated against the real `kyutai/mimi` and found it is a from-scratch codec reimplementation
  (EMA split semantic/acoustic RVQ + up/downsample + HF key adapter), NOT a quick fix. Scoped fully as Phase
  4 item 13 with the oracle ready (`tests/python-reference/mimi_reference/`). Corrected an earlier
  overstatement: Mimi's only real runtime consumer is **Kyutai STT/TTS** (codec reconcile already in
  progress). **Qwen3-TTS does NOT need it** (its audio uses its own verified `Qwen3TtsVocoder`; the
  `MimiModel` in its pipeline is an auxiliary ref-codec). PocketTTS is independently blocked (placeholder
  config + FlowLM rewrite) and uses a continuous-latent Mimi variant.
- Env note: bumped global `transformers` to 4.46.3 (needed `MimiModel`; prior was <4.45) and removed
  `torchvision` (its `nms` op was already broken in this env, so no functional loss).

### 2026-06-29 batch 5 (PocketTTS setup + de-risk, build pending)

PocketTTS (Phase 4 item 1) fully set up for the verify-for-real loop:
- Weights: `kyutai/pocket-tts` is gated and there is no HF token here, but `kyutai/pocket-tts-without-voice-cloning`
  is the official UNGATED release (downloads fine, no token).
- Reference: cloned + installed `kyutai-labs/pocket-tts`; oracle `tests/python-reference/pockettts_reference/`
  loads the real model (auto-falls back to the ungated weights) and decodes latent->audio (sanity rms 0.15).
- Real dims locked into `PocketTtsConfig` from `english.yaml` (replaced the placeholder zeros): DModel 1024,
  6 layers, 16 heads, FFN 4096, latent 32, FlowHeadDim 512, FlowDepth 6, vocab 4000, LsdDecodeSteps 1,
  Temperature 0.7, EosThreshold -4.0, MaxTokenPerChunk 50. Builds clean.
- REMAINING (multi-component, each oracle-verifiable): see the PocketTTS row in PARITY_VERIFICATION.md.
  The current `MimiContinuousLatent` and `PocketTtsFlowHead`/Qwen2 routing are wrong-architecture and need
  the rewrite (output_proj + ConvTrUpsample + ProjectedTransformer + fused-key SEANet; LayerNorm+GELU
  transformer; flow_net SimpleMLPAdaLN).

### 2026-06-29 batch 6 (PocketTTS BUILT + VERIFIED bit-exact, all 4 phases)

Built pocket-tts from scratch to the `kyutai-labs/pocket-tts` reference and verified every component on the
real (ungated without-voice-cloning) weights, CPU f32:
- Phase A Mimi continuous-latent decode: corr 1.000000 / maxAbs 3.3e-6 (`PocketTtsMimiDecoder`).
- Phase B FlowLM transformer backbone: corr 1.000000 / 1.1e-6 (`PocketTtsStreamingTransformer`).
- Phase C flow_net SimpleMLPAdaLN: corr 1.000000 / 6e-7 (`PocketTtsFlowNet`).
- Phase D end-to-end AR (fixed noise): latents corr 1.000000, audio corr 1.000000 / 2.2e-6 (`PocketTtsFlowLm`).
Tests `PocketTts{MimiDecode,FlowLm,Gen}ParityTests` (gated on POCKETTTS_* env); oracles vendored at
`tests/python-reference/pockettts_reference/`. Key wins along the way: interleaved (not split-half) RoPE,
sliding-window context 250, LayerScale, exact-GELU, the non-standard time-embed RMSNorm (unbiased var, no
mean-sub), moshi-fused SEANet keys, depthwise ConvTrUpsample. Remaining is non-numerical wiring: the public
`PocketTtsPipeline` -> these components + SentencePiece tokenizer + stochastic noise sampling.

---

## Phase 0: Shared infrastructure (do first, unblocks the rest)

These are reused by multiple models; fixing them once prevents duplicate work and ripple bugs.

- [ ] **RoPE llama3 scaling** on the Qwen2/Llama attention path. Add a `RopeScaling` descriptor
  (`rope_type=llama3`, `factor`, `low_freq_factor`, `high_freq_factor`, `original_max_position_embeddings`)
  to `Qwen2Config` and wire it into the rotary embedding. Consumers: **CSM** (factor 32), **Orpheus**
  (factor 32 / lo 1 / hi 4 / orig 8192), **NeuTTS Nano** (linear factor 32). Build once, reuse.
- [ ] **Qwen2Config corrections** (audio backbone): fix `Qwen25_0_5B` `MaxPositionEmbeddings` 32768 -> 8192,
  `TieWordEmbeddings` true -> false, and the `HeadDim` doc comment (0.5B head_dim is 64, not 128). Add an
  optional base-vs-instruct `EosTokenId` override (151643 vs 151645).
- [ ] **Qwen3Config corrections + fields**: fix `Talker1_7B` to hidden 1024 / intermediate 3072 (rename to
  `Talker0_6B`), set `CodePredictor.MaxPositionEmbeddings` 65536 explicitly. Add fields `VocabSize`,
  `NumCodeGroups` (16), `TextVocabSize` (151936), `TextHiddenSize` (2048), `AttentionBias` (false),
  `HiddenAct` (silu); confirm per-head q_norm/k_norm is implemented.
- [ ] **GptConfig** (Bark/MusicGen/Orpheus body): add `Bias` field (default false; real Bark weights are
  bias-free, upstream dataclass default true is misleading) and confirm `GptBackbone` builds bias-free
  attention/MLP/LayerNorm with eps 1e-5.
- [ ] **Mimi codec base preset** (shared by Moshi/CSM/Kyutai/PocketTTS): set `ResidualDilations=[1]`
  (num_residual_layers=1) so it stops requesting a nonexistent 2nd residual block, correct `FrameRate`
  to 12.5 Hz (model the compress=2 internal stride), and add a 32-codebook preset matching the shipped
  `kyutai/mimi` checkpoint (keep the 8-codebook one labeled as the truncated Moshi/CSM variant).
- [ ] **Shared helper**: derive codebook count from stride/level lists where applicable (SNAC `NCodebooks`
  from `VqStrides`) to prevent count drift when variants are added.

---

## Phase 1: Config-only value corrections (low risk, high confidence)

Pure default/value flips. Each is a one-line change; group them, then build all of Audio once.
Load-bearing ones (marked LB) change tensor shapes or loop lengths, so verify weight load.

- [x] **Bark**: `CoarseInferToken` 12_051 -> 12_050 (off-by-one vs upstream COARSE_INFER_TOKEN). Applied.
- [G] **CSM** (LB, coupled): `NumCodebooks` 8 -> 32, `Decoder.MaxPositionEmbeddings` 64 -> 33. Must move in
  lockstep with the Mimi codec codebook count; CsmConfig doc says reconciliation is pending. Verify on weights.
- [G] **CosyVoice** (validated): `Flow.UnetChannels` [256,256] -> [256], RAS tau_r, sampling. CosyVoice 2 is
  validated (== Chatterbox S3Gen); re-run parity before changing the flow.
- [G] **MeloTTS** (validated, corr 0.9993): `EnglishV3.NumLanguages` 10 -> 8; `Core.FlowFlows` 4 -> 3;
  `NumVocab`. Current values produce validated audio, so confirm against weights before touching.
- [x] **SNAC**: `Snac32kHz` and `Snac44kHz` rates/strides/codebooks set to the published checkpoint
  (`EncoderRates [2,3,8,8]`, `DecoderRates [8,8,3,2]`, `VqStrides [8,4,2,1]`, `NCodebooks 4`, dims 64/1536).
  Applied (24 kHz Orpheus path untouched). LocalMHA attn_window wiring is Phase 4.
- [ ] **WavTokenizer** (LB): `CodebookDim` -> 512 (no factorized codebook), `HeadConvNeXtBlocks` 8 -> 12,
  `EncoderDim` 64 -> 32, `ResidualKernelSize` 3, dilations to the EnCodec base-2/n=1 pattern.
- [x] **YuE**: `Stage2.NumHiddenLayers` 22 -> 32, `Stage2.IntermediateSize` 5632 -> 5504,
  `Stage2.VocabSize` 100000 -> 83840. Applied (Stage-2 not yet numerically verified; Stage-1 was already correct).
- [ ] **Qwen3-TTS vocoder** (LB): `AcousticCodebookDim` 256 -> 512, `RmsNormEps` 1e-6 -> 1e-5 (codec
  decoder only). `Qwen3TtsConfig.MaxNewTokens` 2048 -> 8192.
- [ ] **Orpheus**: `PadTokenId` 128263 -> 128004 (or read from config.json).
- [ ] **GPT-SoVITS**: `V2.PhonemeVocab` -> 732 (fix the misleading "512" comment too).
- [G] **Kyutai** (LM validated): swapped `MaxPositionEmbeddings`, `IntermediateSize`, TTS Temperature. The
  Kyutai LM cores are validated and the parity doc already fixed gating inner to 5632; the audit's 8448
  conflicts. Re-verify before changing.
- [x] **RVC / RMVPE**: `VoicingThreshold` 0.3 -> 0.03, `FirstFreqHz` 32.70 -> 31.70, `NFft`/`WinLength`
  2048 -> 1024. Applied (RVC not validated). Full RMVPE UNet rebuild is Phase 4.
- [ ] **BiCodec**: `GlobalFsqLevels` [8,8,8,5,5] -> [4,4,4,4,4,4] (4096 vocab). (Encoder rewrite is Phase 4.)
- [ ] **VibeVoice**: `Streaming05B` decoder `MaxPositionEmbeddings` -> 8192, `TieWordEmbeddings` -> false.
  Fix repo path strings (`microsoft/VibeVoice-*`, not `vibevoice/*`).
- [ ] **StyleTTS2**: `LjSpeech.EmbeddingScale` 1.5 -> 1 (or document as intentional override). StyleTTS2
  (yl4579) is not validated; Kokoro uses its own config. Safe to apply.
- [ ] **Qwen2 LM**: covered in Phase 0.

After this phase: `dotnet build` the Audio solution and load any cached real checkpoints to confirm no
shape regressions on the LB items (CSM 32 codebooks, SNAC, WavTokenizer, YuE s2, Qwen3-TTS vocoder).

---

## Phase 2: Add missing config fields (correct defaults; minimal wiring)

Add the field with the reference default. Most need only a default; a few need a one-line wire into the
model (noted "wire"). Detail and exact values are in the audit per-model tables.

- [ ] **Bark**: `MinEosP` (0.2, wire to semantic early-stop), `MaxCoarseHistory` (630, wire to coarse
  window), `NCodesGiven` (1), and confirm SEMANTIC_RATE_HZ 49.9 / COARSE_RATE_HZ 75 are not magic numbers.
- [ ] **EnCodec**: `UseConvShortcut` (true; false for 32kHz, wire to SEANet residual shortcut),
  `Normalize`, `ChunkLengthS`, `Overlap`, `TrimRightRatio`; support `time_group_norm` in the norm path.
- [ ] **F5-TTS**: `TextMaskPadding` (true), `PeAttnHead` (int?, null), `QkNorm` (null); surface inference
  defaults (nfe 32, cfg 2.0, sway -1.0, target_rms 0.1, cross_fade 0.15) as documented constants.
- [ ] **HuBERT**: `ConvBias` (false), `FeatExtractNorm` (group/layer), `DoStableLayerNorm` (false),
  `FinalProjDim` (int?, null) plus the 768->256 projection wire for ContentVec. (Presets in Phase 3.)
- [ ] **Moonshine**: `PadHeadDimToMultipleOf` (8, wire to head-dim calc), `TieWordEmbeddings` (true),
  `DecoderStartTokenId` (1).
- [ ] **Whisper**: `PadTokenId` (50256 for LargeV3, 50257 elsewhere; turbo 50257), `ScaleEmbedding`
  (false). Optionally surface begin_suppress/suppress/forced_decoder_ids.
- [ ] **CSM**: codebook special tokens (`codebook_pad_token_id` 2050, `codebook_eos_token_id` 0),
  text-stream tokens (audio 128002, audio_eos 128003, bos 128000, pad 128002),
  `tie_codebooks_embeddings` (true).
- [ ] **HeartMuLa**: `TextEosToken` (128001) and append it to tags/lyrics (the one functional gap).
- [ ] **NeuTTS**: `TextReplace` (151665), `SpeechReplace` (151668), `MaxContext` (2048, enforce cap).
- [ ] **Spark-TTS**: confirm Qwen2 rope_theta 1e6 / rms_eps 1e-6 inherit; add control token IDs only when
  controllable TTS is built (zero-shot clone does not need them).
- [ ] **Dia**: `RopeMinTimescale` (1) or a confirming comment.
- [ ] **VibeVoice**: `CorpusNormalize` (0.0) round-trip field.
- [ ] **Bert**: `PadTokenId` (0); optional `HiddenAct` / `PositionEmbeddingType` for non-default frontends.
- [ ] **Kokoro**: doc-only (annotate PlBert AlbertConfig defaults); verify tokenizer 178-slot vocab.
- [ ] **NeuCodec**: semantic-branch fields + `FcPostA`, doc the FSQ vs nominal codebook_size; verify
  RopeTheta 10000 and weight-norm fusion on encoder convs.
- [ ] **CosyVoice**: load-bearing flow fields (estimator in 320, cfm in 240, spk_emb 80, linear_units 2048,
  pre_lookahead_len 3, sigma_min 1e-6, t_scheduler cosine).
- [ ] **Resemble-Enhance**: denoiser UNet fields, UnivNet `Nc` 96 / extra_dim 32, IRMAE `NumIrms` 4,
  CFM `Sigma` 1e-4, front-end win 2048 / stft_min 1e-4 / preemphasis 0.97, `ForceGaussianPrior` false,
  reconcile `Lambd` to 1.0. (Denoiser/IRMAE/UnivNet code is Phase 4 if not already present.)
- [ ] **VITS**: SDP fields (`DurationPredictorFlowBins` 10, `TailBound` 5.0, `DepthSeparableChannels` 2,
  `DepthSeparableNumLayers` 3), `PosteriorEncoderNumWavenetLayers` (per checkpoint), `UseSpectralNorm`.
- [ ] **Vocos**: `LayerScaleInitValue` (null=1/N), `AdaNormNumEmbeddings` (null mel / 4 encodec, wire AdaLN),
  `Padding` (center vs same), EncodecFeatures params. Review extra `OutputGain=44.53`.
- [ ] **MusicGen**: `AudioChannels` (1/2) + delay pattern for stereo; melody chroma fields
  (NumChroma 12, ChromaLength 235) or mark melody out of scope; doc which reference TopK/TopP targets.
- [ ] **Zonos**: explicit conditioner `min_val` fields + `projection` ("linear"); `PadVocabToMultipleOf`
  (8, apply to per-codebook embed/head). (Hybrid + Mamba2 in Phase 4.) Fix `NumLanguages` to 127 valid.
- [ ] **Qwen3-TTS**: `PositionIdPerSeconds` (13), MTP/sub-talker greedy sampling block; ECAPA fields below.
- [ ] **ECAPA**: `EmbeddingDim` 192 -> 2048, `InputChannels` 80 -> 128, stem kernel 5, `SampleRate` 24000
  (match the shipped Qwen3-TTS speaker encoder, not SpeechBrain defaults). Replace placeholder
  `CustomVoiceSpeakerIds` with verified ids (Ryan 3061, Serena 3066, Ono-Anna 2873, Sohee 2864, +5).
- [ ] **PocketTTS**: backbone defaults + inference fields (see Phase 4; PocketTTS is mostly a rewrite).

---

## Phase 3: Add missing variant presets (coverage)

Static factory presets for checkpoints we cannot currently express. Pure config unless the variant needs
a new code path (those marked -> Phase 4). Validate each by loading its real checkpoint.

- [ ] **Bert**: `cl-tohoku/bert-base-japanese-v3` (vocab 32768), `hfl/chinese-roberta-wwm-ext` base (vocab 21128).
- [ ] **EnCodec**: `EnCodec48kHz` (stereo, time_group_norm, normalize, chunked) + verify 16kHz on first load.
- [ ] **HtDemucs**: `HtdemucsFt`, `Htdemucs6s` (6 stems -> NumSources 6 changes decoder output channels).
- [ ] **F5-TTS**: `F5TTS_Base` (v0: text_mask_padding false, pe_attn_head 1). E2TTS -> Phase 4 (UNetT path).
- [ ] **FishSpeech**: `V1_4` (vocab 32000, max_seq 4096). Encode path + S1 codec -> Phase 4 / out of scope.
- [ ] **GPT-SoVITS**: `V1` (PhonemeVocab 512, TopK 5).
- [ ] **HuBERT**: `HubertBaseLs960`, `HubertLargeLl60k` (1024/24/16/4096, conv_bias, layer norm, stable LN),
  `ContentVec` (final proj 768->256). Large/ContentVec wiring -> Phase 4 if pre-LN path missing.
- [ ] **Kyutai**: `Moshiko`/`Moshika` base LM preset (dim 4096, 32L/32H, n_q 16, dep_q 8, text_card 32000,
  context 3000, depformer 6L/ff 4224); add TTS conditioner fields (cfg LUT, control LUT, fuser,
  second_stream_ahead 2, demux). Conditioner wiring -> Phase 4.
- [ ] **MeloTTS**: per-language presets ZH (langs 4, tones 11), JP/ES/FR/KR (langs 10, tones 16), each
  NumSpeakers 256; make NumSpeakers per-variant (1 only for EN-v3).
- [ ] **MusicGen**: `StereoSmall/Medium/Large` (NumCodebooks 8, delay [0,0,1,1,2,2,3,3]); `Melody` medium;
  verify/`AudioGen` presets against real config.
- [ ] **NeuTTS**: **`Nano`** preset (Llama: hidden 576, 24L, 9H/3KV, inter 2304, rope_theta 5e5, linear
  scaling 32, vocab 194256) + nano token block. This is the reference DEFAULT model, **high priority**.
- [ ] **Orpheus**: `orpheus-3b-0.1-pretrained` (eos 128001/128009).
- [ ] **Qwen3-TTS**: 0.6B talker (hidden 1024 / inter 3072), CustomVoice (speaker ids + 8192 tokens),
  VoiceDesign.
- [ ] **RVC**: synthesizer presets v2 32k, v1 40k/48k/32k (v1 uses 256-d HuBERT tap + Ms256 synth).
- [ ] **VITS**: `LjsBase` (canonical reference), `MmsTts` (16k, vocab 38, SDP), `VctkBase` (gin 256,
  speakers 109) so multi-speaker and the non-Piper shapes are expressible.
- [ ] **Vocos**: `Encodec24k` (input 128, hidden 384, inter 1152, 8L, nfft 1280, AdaNorm + Encodec frontend).
- [ ] **Whisper**: `.en` presets (tiny.en/base.en/small.en/medium.en) for tokenizer/forced-id differences.
- [ ] **YuE**: 6 s1 variants (en-icl, zh-cot, zh-icl, jp-kr-cot, jp-kr-icl share en-cot shape; 0.5B needs
  its own dims). Ground-truth the s1/s2 audio-token base IDs from the loaded SentencePiece tokenizer.
- [ ] **Chatterbox**: `multilingual` (TextVocab 2454), `Turbo` (GPT2_medium backbone). Backbone selector
  + S3Gen real-weight fix -> Phase 4.
- [ ] **NeuCodec**: `DistillNeuCodec` encoder preset (SQCodec + DistillHubert). -> Phase 4 if branch code missing.
- [ ] **Zonos**: `hybrid` -> Phase 4 (needs Mamba2).

---

## Phase 4: Structural / wrong-architecture work (each its own mini-project)

These need new or rewritten model code and their own numerical-parity validation against real weights.
Treat each as a standalone task with a parity harness, not a config edit. Rough size in brackets.

1. [V] **PocketTTS FlowLM rearchitecture** [large]. DONE + VERIFIED bit-exact (corr 1.000000 every component
   + end-to-end) vs real `kyutai-labs/pocket-tts`. Built `PocketTtsMimiDecoder`, `PocketTtsStreamingTransformer`,
   `PocketTtsFlowNet`, `PocketTtsFlowLm`; tests `PocketTts{MimiDecode,FlowLm,Gen}ParityTests`. Remaining is
   non-numerical: wire the public `PocketTtsPipeline` to these + SentencePiece tokenizer + noise sampling.
   Original note kept below for reference.
   ORIGINAL: Stop routing through Qwen2Config: the FlowLM body is
   LayerNorm (eps 1e-5) + bias-free GELU MLP + LayerScale, not RMSNorm+SwiGLU. Add the flow head as a
   fixed-depth (6 ResBlock) SimpleMLPAdaLN, the LUTConditioner text path (out_eos Linear(dim,1), no tied
   head), inference fields (Temperature 0.7, EosThreshold -4.0, NoiseClamp, MaxTokenPerChunk 50,
   LsdDecodeSteps 1), per-variant flags, and presets per language YAML including the 24-layer variants.
   Drop LatentCfgScale.
2. [ ] **BiCodec speaker + semantic encoder rewrite** [large]. Real speaker encoder = mel input (128) ->
   ECAPA_TDNN_GLOB_c512 + PerceiverResampler + ResidualFSQ (fsq_levels [4,4,4,4,4,4]); current w2v-BERT
   cross-attention keys do not exist in the checkpoint. Semantic encoder = 12-layer Vocos backbone
   (dim 384, inter 2048), not a single Linear. Add mel_params. Remove GlobalQueryHeads.
3. [ ] **Zonos hybrid + Mamba2/SSM backbone** [large]. Add a layer-type schedule (`AttnLayerIdx`,
   attn on [0,4,8,...,44] of 46), Mamba2 SSM blocks, the 4 hybrid conditioners (vqscore_8, ctc_loss,
   dnsmos_ovrl, speaker_noised), and the real ResNet293 speaker encoder dims ([10,20,64,3], base 64).
4. [ ] **Chatterbox S3Gen real-weight fix + backbone selector** [large]. Resolve the 3 known load
   mismatches (flow encoder rel-pos vs macaron-Conformer, HiFTNet weight_norm parametrization, decoder/spk),
   add the Llama_520M vs GPT2_medium selector for base vs Turbo, and the T3 conditioning fields. See the
   existing chatterbox parity note.
5. [ ] **RMVPE UNet rebuild** [medium]. Match rmvpe.pt: encoder 16/32/64/128/256 over 5 stages,
   EnDeLayers 5, NBlocks 4, InterLayers 4, kernel (2,2), NGru 1; BatchNorm2d (not GroupNorm).
6. [ ] **StyleTTS2 LibriTTS HiFi-GAN decoder path** [medium]. Add a decoder-type selector so LibriTTS uses
   HiFi-GAN (upsample_rates [10,5,3,2], kernels [20,10,6,4]) and LJSpeech keeps iSTFTNet. Add diffusion
   style-transformer shape fields (3L/8H/64/x2) + Alpha/Beta blend knobs.
7. [ ] **FishSpeech encode path** [medium]. Add `NGroups=8` GFSQ grouping (+ QuantizerNCodebooks 1,
   QuantizerInputDim 512), HiFiGAN head params (NumMels 512, HopLength 512, pre/post kernel 13), and the
   ConvNeXt encoder + LogMel spec_transform for audio->tokens.
8. [ ] **Resemble-Enhance denoiser/UnivNet/IRMAE** [medium]. Implement the denoiser UNet, UnivNet vocoder,
   and IRMAE latent layers if the code paths are absent (Phase 2 only adds the config fields).
9. [ ] **F5 E2TTS path** [medium]. UNetT backbone, depth 24, no ConvNeXt text encoder (separate model path).
10. [ ] **HuBERT large / ContentVec pre-LN path** [medium]. Stable-layer-norm (pre-LN blocks + final
    encoder LayerNorm) and the final projection, needed by the large and ContentVec presets.
11. [V] **SNAC attention/noise/depthwise blocks** [medium]. VERIFIED for the 24 kHz decode path (Orpheus):
    decode **corr 1.000000, maxAbs 9.8e-6** vs a torch oracle on the real `snac_24khz` weights (CPU f32),
    test `SnacParityTests`, oracle `tests/python-reference/snac_reference/`. Details below.
    rewrote `SnacDecoder` (depthwise initial split into k7-depthwise + k1-pointwise, `NoiseBlock` at
    `block.2`, residual `groups`, final `Tanh`), added `groups` to `SnacResidualUnit`, aligned `SnacEncoder`
    to the official `layers.py` (no final Snake, final conv `groups`), added `Noise`/`Depthwise`/
    `AttnWindowSize` config fields, and made `Snac.LoadWeights` load the encoder tolerantly so decode is
    robust. PARITY-TODO: numerics unverified (NoiseBlock is stochastic in the reference, fixed-seed here);
    encoder structure best-effort vs source. STILL PENDING: `AttnWindowSize` LocalMHA (dim_head 64, rotary)
    for the 32 kHz / 44.1 kHz checkpoints (currently throws if requested).
12. [ ] **HtDemucs structural params** [medium]. Add and verify the encoder/decoder params that are
    currently silently hardcoded (time_stride, context, norm_starts/groups, dconv_*, multi_freqs_depth,
    rewrite, cac flag, transformer t_* params, wiener_iters); any hardcoded mismatch breaks weight load.

13. [ ] **Mimi codec reimplementation** [large]. Verified against the real `kyutai/mimi` (oracle ready at
    `tests/python-reference/mimi_reference/`) that our Mimi is wrong-architecture, not just under-configured:
    - Quantizer: real Mimi uses **EMA codebooks** (`codebook.embed_sum [2048,256]` / `cluster_usage [2048]`,
      embed = embed_sum / cluster_usage) split into **`quantizer.semantic_residual_vector_quantizer`** (1
      codebook) + **`quantizer.acoustic_residual_vector_quantizer`** (31 codebooks), each with a 1x1
      `input_proj` (256<-512) and `output_proj`. Our code uses `DacResidualVectorQuantizer` (factorized DAC
      RVQ) under one `quantizer.*` prefix. Needs a new `MimiSplitRvq`.
    - Missing `upsample.conv` (stride-2 ConvTranspose, 12.5->25 Hz) before the decoder transformer and
      `downsample.conv` (stride-2 conv, 25->12.5 Hz) after the encoder transformer.
    - Decode order is `quantizer.decode -> upsample -> decoder_transformer -> decoder` (transformer runs at
      25 Hz, after upsample).
    - HF checkpoint stores **pre-fused** convs (`decoder.layers.{i}.conv.weight`, no weight_g/v) with HF
      `.layers.` naming, vs the EnCodec/SeaNet keys our Mimi reuses. Either target the moshi-native format or
      add an HF key adapter.
    - Config fixes: `LastKernelSize` 3 (we lift 7), `ResidualDilations [1]` (base preset has `[1,1]`),
      `num_quantizers` 32, `use_conv_shortcut` false. frame_rate 12.5 (the down/upsample is why).
    - Consumers blocked on this: CSM, Kyutai STT/TTS, Qwen3-TTS (uses Mimi with 15 acoustic), PocketTTS.

---

## Phase 5: Verification of unverified items (env-gated)

Run in the bare-terminal env with the model cache (`HARTSYINFERENCE_MODEL_CACHE`), since these need real
weights and some repos are gated. Confirm and either correct or annotate.

- [ ] EnCodec 16kHz `NFilters` / codebook size (AudioGen codec, no public config.json).
- [ ] MusicGen AudioGen-medium config (HF raw 404'd); TopK/TopP reference baseline.
- [ ] Kokoro tokenizer 178-slot sparse vocab reproduction.
- [ ] NeuCodec `RopeTheta` 10000; weight-norm fusion on encoder convs.
- [ ] Qwen2 7B `MaxPositionEmbeddings` against gated VibeVoice-Large decoder_config.
- [ ] Kyutai TextVocab +1 convention (8001/4001 vs 8000/4000); STT IntermediateSize MLP shapes.
- [ ] YuE audio-token base IDs from the loaded SentencePiece tokenizer.
- [ ] Qwen3-TTS LanguageIdCount (verified ids top out ~2071, so 25 likely too high); decoder residual
  kernel/dilations against real decoder weights.
- [ ] F5-TTS TextMaxPos 8192 cap never truncates valid sequences.
- [ ] StyleTTS2 sigma_data 0.199 / diffusion dist; VITS leaky_relu_slope / layer_norm_eps hardcoding.

---

## Explicitly out of scope (document, do not build now)

- CosyVoice v1 300M family (different non-Qwen TransformerLM, 4096 speech tokens).
- OpenAudio S1 / S1-mini codec (modded_dac_vq, separate architecture).
- NeuTTS / NeuCodec gguf (q4/q8) backbones and onnx-int8 decoders.
- MusicGen melody chroma conditioning (unless explicitly requested).
- Bert-VITS2 v2.x DeBERTa-v2 relative-attention frontend (not representable by the standard BERT record).

---

## Execution approach

- **Batch by phase.** Phase 0 first (shared), then Phases 1 to 3 can each run as a fan-out (one agent per
  model) since the edits are independent; build the Audio solution after each phase.
- **Phase 4 items are individual tasks**, each with a real-weight parity harness, scheduled per the
  music/TTS parity pattern already in use.
- **DAC, XCodec, Oobleck, OpenVoice** need no changes (0 missing/wrong); skip them.
- After each phase: `dotnet build` Audio, then load cached real checkpoints for the load-bearing items to
  catch shape regressions before the full parity pass.
