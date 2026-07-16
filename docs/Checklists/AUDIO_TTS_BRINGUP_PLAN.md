# Audio TTS/STT Bring-up & Fix Plan (2026-07-13)

Fix every local TTS/STT that is broken (runs → gibberish/silent) or not-runnable (install throws). Verified
runnable+correct already (do not touch): Kokoro, Piper, MeloTTS, F5-TTS, Bark, Chatterbox*, VibeVoice, FishSpeech
(TTS) + Moonshine, Whisper (STT). *Chatterbox = default voice only; clone path is Tier-2.

Method for every model (same rigor as the MeloTTS/F5 fixes): install → `GenerateText2Image` → transcribe with
**whisper `medium.en`** (never `base.en`) + compare to a Python reference (A/B stage dumps) → root-cause → fix →
re-verify word-correct. Cloud-API providers (ElevenLabs/Azure/OpenAI/etc.) are out of scope (not engine models).

---

## Tier 1 — Runs but broken (closest to done, start here)

### 1a. Orpheus — ✅ FIXED 2026-07-14 (prompt-format bug)
- **Root cause:** our prompt omitted the **BOS token (128000)** and the trailing **StartOfAi 128261 +
  StartOfSpeech 128257**. Correct frame (matches `orpheus_tts._format_prompt`):
  `[StartOfHuman, BOS, textTokens, EndOfText, EndOfHuman, StartOfAi, CodeStart]`.
- **Fix:** `AudioTextFrontend.OrpheusText` prepends `Llama3Bos`; `OrpheusPipeline.Synthesize` appends
  `StartOfAi + CodeStart`; added `OrpheusConfig.StartOfAi = 128261`. Verified word-perfect through Swarm
  (medium.en) on multiple sentences. Weights were never the issue (canopylabs fp32 == unsloth bf16, relMean 0.0014).
  Slow (~66 s for 3 s audio, 3B LM). Debug instrumentation removed.
- **Perf pass 2026-07-14 (findings):** decode is **GPU-compute-bound on the F32 backbone** (~245 ms/token on a
  3060), NOT host- or launch-bound. Two proportionate, Orpheus-only optimizations were tried and MEASURED FLAT:
  (a) restricting the sampler to the audio-token range `[CodeStart, vocab)` (skips ~128k text logits) — kept (safe,
  ~4%, no quality change); (b) a CUDA-graph decode of the single-token backbone step (CSM's tested path) — **no
  measurable win → reverted**. Confirms the bottleneck is the 28-layer F32 matmul (we're ~7× off even the F32
  memory floor / GPU sat ~70% util).
- **CORRECTION 2026-07-14 (F16 project premise was WRONG):** the shared `GenericTransformer` does NOT run the
  projections in F32 — `EnsureF32` is applied only to norms + the embed/tied-lm_head. The q/k/v/o/gate/up/down
  projection weights are kept in their loaded dtype, which for Orpheus (unsloth) is **BF16 on disk (verified via the
  safetensors header)** → they already run BF16 tensor-core GEMMs. On the 3060 (Ampere SM 8.6) BF16 and F16 tensor
  cores run at the same rate, so there is **no F16 speedup to take** on the projections. The only F32 hot op is the
  tied **lm_head** (3072→156,940 per token) — but that's ~2% of the step.
- **PERF WIN 2026-07-14: Orpheus ~6.5× (62 s → ~10 s for a 3 s clip; ~245 → ~30 ms/token), word-perfect, DEFAULT-ON.**
  An env-gated Stopwatch profiler on the 3 decode sections (`HARTSY_ORPHEUS_PROF=1`) found it in one gen:
  backbone-fwd 24 ms/tok, **lm_head 221 ms/tok (90%!)**, sample 3 ms. The lm_head (tied embed, 3072→156,940 vocab)
  ran as an **F32 GEMM at M=1** (~40× off the mem floor) because `GenericTransformer` only kept the raw tied embed
  for the head when *quantized*. Fixes: (1) built a fused BF16/F16 M=1 decode GEMV kernel
  (`native/cuda/lm/mul_mat_vec_f16_bf16_f32.cu`, one warp/row, F32 accumulate; NVRTC → PTX; default-on
  `HARTSY_BF16_GEMV`); (2) broadened `GenericTransformer.LoadWeights` to keep the BF16/F16 tied embed for the
  lm_head → its GEMV now hits the fused kernel: **221 → 3.8 ms/tok (58×)**. Regression: FusedGemvGroundTruth
  (avg_err<1e-4) + GenericTransformerParity 2/2 + GgufEndToEnd 1/1 + FishSpeech/Chatterbox/Bark word-perfect (first
  two on the exact Qwen2Model/GenericTransformer path). **LESSON:** the lm_head dominates decode for LARGE-vocab
  models; profile the sections before scoping (mis-diagnosed the bottleneck 3× before the 1-gen profiler nailed it).

--- (historical diagnosis; superseded by the fix above) ---
### 1a-history. Orpheus — SILENT output  (earlier mis-diagnosis)
- **Symptom:** LM emits `EndOfSpeech` after only 4 SNAC frames → 0.34 s near-silent clip (`rms 0.0006`).
- **Arch:** Llama-3.2-3B (`unsloth/orpheus-3b-0.1-ft`) via `Qwen2Model` → SNAC-24k decode.
- **DIAGNOSIS 2026-07-13 (via `HARTSY_ORPHEUS_DEBUG=1` token dump + a Python `snac` decode A/B):** everything
  downstream of the LM is CORRECT and ruled out —
  - Tokenizer ✓ (`tara: The morning sun…` → sensible Llama-3 ids; asset embedded).
  - Prompt frame ✓ (`[SOH 128259, text, EOT 128009, EOH 128260]` — matches canonical Orpheus; the model correctly
    emits `128261 start_of_ai, 128257 start_of_speech` then audio tokens).
  - Redistribution ✓ (`OrpheusCodeFrames.Redistribute` matches canonical `[0]→L1, [1,4]→L2, [2,3,5,6]→L3`).
  - **SNAC decode ✓ — bit-identical to Python `snac`**: feeding the exact 28 generated codes to `snac_24khz` in
    Python gives `rms 0.00055`, same as the engine. So the codec is not the bug.
  - **ROOT CAUSE = the LM generates DEGENERATE audio codes** (redistributed codes show heavy repetition, e.g.
    `l3=[…429,429,429,429…]`, `l2=[…3418,3418…]`) then collapses to EOS at frame 4. Config is correct (bias off,
    θ=500000, Llama3 rope-scaling IS applied — the docstring "not yet applied" is stale).
- **RESOLVED-AS-NOT-AN-ENGINE-BUG 2026-07-13 (transformers reference A/B):** built a transformers
  `LlamaForCausalLM` reference (needed `pip install -U transformers` → 5.13.1 for `llama3` rope; safe for the
  whisper/snac ref tools) on the exact downloaded weights.
  - **Tokenizer A/B:** engine ids == HF ids EXACTLY (`tara`→`t`+`ara`, `[83,5169,25,578,…]`).
  - **LM forward A/B (greedy):** engine matches the F32 reference **bit-exact for the first 16 generated tokens**,
    then drifts (expected GPU-precision vs F32-CPU) — so the engine LM forward is CORRECT.
  - **The reference itself produces GIBBERISH/silence:** reference greedy stops at ~3 frames then hallucinates
    TEXT (`"I'm not sure I can handle…"`); reference sampling (temp 0.6/topP 0.95) gives 56 frames of fluent-but-
    WRONG words (`"The dozen have a gave you…"`, unrelated to the prompt); the canonical
    `end_tokens=[128009,128260,128261,128257]` format gives 4 frames silent. **None produce the target sentence.**
  - **CONCLUSION: NOT an engine bug.** Our engine faithfully reproduces the transformers reference; the reference
    itself (these weights) doesn't do correct TTS. Root = **model weights / sourcing**: `unsloth/orpheus-3b-0.1-ft`
    (used as a non-gated mirror of the license-gated `canopylabs/orpheus-3b-0.1-ft`) does not produce correct
    text-conditioned speech in official transformers.
- **NEW FIX PATH (model-sourcing, not engine code):** (1) verify the correct Orpheus-TTS weights — try the gated
  `canopylabs/orpheus-3b-0.1-ft` (needs HF_TOKEN) and confirm it produces the sentence in transformers; (2) if the
  canonical weights work, switch the extension's `ResolveRepo`/download to them (or a verified mirror) — no engine
  change needed. (3) If even the gated weights need a special generation procedure (vLLM stop-token handling), port
  that. Debug instrumentation is gated (`HARTSY_ORPHEUS_DEBUG` / `_GREEDY` / `_NOPENALTY`) and harmless; can stay
  for the weights verification.

### 1a-2. Dia-1.6B — ✅ FIXED 2026-07-15 (WRONG CHECKPOINT, not an engine bug)
- **Symptom:** loops "Hello there" / non-verbal garbage ("screaming", "crying") across seeds; never reaches `[S2]`.
- **Root cause:** the extension downloaded the **old** `nari-labs/Dia-1.6B`. The current **`nari-labs/Dia-1.6B-0626`**
  release (drop-in — identical 343 keys + shapes, only weight *values* differ) makes the engine produce the **full
  3-turn dialogue** and **emit EOS to stop at 11.44 s** (985 frames). Proven by a full **layer-diff A/B vs the nari
  `dia` package** (which hardcodes `-0626`): forward/sampling/EOS/RoPE/masking all matched — the apparent "input-embed
  divergence" was just base-vs-0626 weights. **The engine was correct the whole time.**
- **Fix:** `SwarmUI-AudioLab` `TtsModels.cs` + `DiaTTSProvider.cs` repo `Dia-1.6B`→`Dia-1.6B-0626` (ships
  `pytorch_model.bin` → `PytorchPickleLoader(recursiveFlatten:false)`, no engine change). Rebuild + restart Swarm
  (AudioLab caches the loaded pipeline in memory) → `GenerateText2Image` transcribes **10/10** (medium.en).
- **LESSON:** when a model degenerates but the forward looks faithful, **check the checkpoint version/source first** —
  don't deep-layer-diff until the reference and the engine load the *same* weights. Also: **`base.en` was even more
  misleading here** — verify with `medium.en`.
- **Follow-ups:** RTF ≈ 0.036× (slowest TTS — dual-CFG 18-layer AR F32) → perf pass. Restore `HARTSY_AUDIO_CUDA_DEVICE`
  (the restart moved audio 3060→4090). Fix the AudioLab **eviction threshold** (3 GB free is too low → 6 GB+ models OOM
  the 3060 on a provider switch).

### 1b. Quick wins — default-voice paths of "clone-gated" models
NeuTTS, Qwen3-TTS, Chatterbox only gate the **voice-clone** path; the **default voice** may already work. Verify
each default-voice gen (no reference) with the oracle — likely fast passes that expand the verified set before the
harder clone work.

---

## Tier 2 — Clone / synth path gated (medium; each a focused feature)
- **Chatterbox clone** — needs a PCM→40-bin-mel front-end for the voice encoder.
- **NeuTTS clone** — needs the X-Codec2 encoder (`CodecEnc.*`) key mapping.
- **Qwen3-TTS clone** — ICL/ECAPA speaker path weight-validation.
- **Kyutai TTS synth** — per-frame text-stream state machine (SentencePiece 8k + PAD/EPAD/WORD) + delayed-coordinate
  / speaker-conditioning path. (Larger than the others in this tier.)

---

## Tier 3 — Full engine bring-up (large; each a real port + weight recipe)
- **StyleTTS2** — 🚧 **build started 2026-07-15.** Recon done: `yl4579/StyleTTS2-LibriTTS` `epochs_2nd_00020.pth`
  downloaded; structure = `net.{bert, bert_encoder, predictor, decoder, text_encoder, predictor_encoder,
  style_encoder, diffusion}` (+ training-only aligner/pitch/discriminators to ignore). Config dims confirmed
  **Kokoro-compatible** (hidden 512, style_dim 128×2, n_token 178, n_mels 80, 24 kHz, decoder 8h·3L) →
  bert/text_encoder/predictor/decoder reuse Kokoro's submodule loaders (flatten `net.{c}.module.…`→`{c}.…`).
  **The two style submodules are scaffolds that must be reconciled to the real checkpoint before they load
  correctly:**
  - *StyleEncoder* (clone path): checkpoint ResBlk = StarGAN-v2 with `conv1` dim_in→**dim_in** + a **learned
    depthwise stride-2 downsample** (`downsample_res.conv`, groups=dim_in) on the residual, avgpool only on the
    shortcut; the engine's `ResBlk2D` has `conv1` dim_in→dim_out + avgpool on both → **rewrite ResBlk2D** (+ needs
    a grouped/depthwise `Conv2D`). Keys `shared.0/1-4/6`→`stem/blocks.0-3/tail`. Spectral-norm σ-fold
    (`weight_orig`/`weight_u`/`weight_v` → `W/σ`) currently stubbed (uses `weight_orig` raw).
  - *StyleDenoiser / diffusion* (random/perturbed): checkpoint = archinetai `Transformer1d` (`diffusion.net.blocks.N`
    with fused `to_kv`, `attention.attention.to_out`, `norm.fc`/`norm_context.fc` AdaLN, `feed_forward.0/2`); the
    engine's `StyleTransformerLayer` expects `unet.blocks`, separate `to_k/to_v`, `to_scale_shift` and loads
    *leniently* (missing keys → no-op → silent garbage) → reconcile the transformer arch + write the sampler.
  - Then `StyleTts2Pipeline.LoadFromCheckpoint` (Kokoro submodules + reconciled style path) + verify via Swarm
    (Clone with a `reference_audio.zip` clip is the most bounded first target; Random needs the diffusion done).
  - **Progress 2026-07-15:** ✅ **StyleEncoder reconciled + VERIFIED — corr 1.000000** vs the Python `yl4579/StyleTTS2`
    reference (both acoustic + prosodic; norms exact). Fixes: real StarGAN-v2 ResBlk (`conv1` in→in, learned depthwise
    stride-2 `downsample_res`, `conv2` in→out, avgpool shortcut with the odd-width **replicate-pad**) + spectral-norm
    σ-fold + `StyleTts2Weights.Adapt` (`shared.N`→`stem/blocks/tail`). ✅ **All 6 submodules load** from LibriTTS
    (dims Kokoro-compatible). ✅ `LoadFromCheckpoint` + 178-symbol tokenizer + engine espeak-IPA phonemizer + clone
    pipeline wired. **BLOCKER found:** the LibriTTS decoder is **`type: hifigan`** (upsample_rates `[10,5,3,2]`=300,
    Snake `alphas`, 9-harmonic `m_source`, `conv_post`→1-ch waveform + tanh) — **NOT Kokoro's iSTFTNet** (2-stage +
    iSTFT), so `KokoroIStftNetDecoder` can't drive it. **Next: implement a StyleTTS2 HiFiGAN generator** (reuse
    `SnakeResBlock` + `NsfVocoderDsp.GenerateHarmonicSource` + `ConvTranspose1d`, AdaIN-conditioned on style; 4 upsample
    stages + per-stage noise-conv injection), verify vs the Python ref, then clone→whisper. Diffusion (Random mode)
    still a scaffold.
  - **HiFiGAN generator DONE + VERIFIED 2026-07-15 → clone e2e word-intelligible.** `StyleHifiGanGenerator`
    (corr **0.999999** vs the Python ref) + `StyleSineGen` (source corr **1.0**, StyleTTS2's frame-rate-cumsum +
    phase×upscale linear-interp) + wired via a gated `KokoroIStftNetDecoder(useHifiGan)` (Kokoro path unchanged).
    Clone e2e (`StyleTts2CloneEndToEndTests`) → Whisper 5/7. **Shared bug fixed:** `NormKernels.AdaInstanceNorm1d`
    single-pass `E[x²]−E[x]²` variance NaN'd via catastrophic cancellation on the ~30 k-sample HiFiGAN stages →
    stable two-pass double variance (Kokoro regression clean — the tests pass and the two formulas agree at
    Kokoro's short lengths). ✅ **Swarm extension wired 2026-07-15:** `StyleTts2Model.Descriptor` (provider
    `styletts2_tts`, registered in `AudioEngine`) → `LoadFromCheckpoint` (in-engine 178-symbol tokenizer +
    reference-mel front-end) + `SynthesizeCloneFromAudio(req.ReferenceMono24k)` + espeak `en` IPA. ✅ **Swarm e2e
    clone-VERIFIED 2026-07-15:** deployed via local-engine pack `alpha.48.2-local` (both extension pins →
    `~/.local/share/hartsy-local-nuget`), installed via `AudioLabInstallEngine`, generated through
    `/API/GenerateText2Image` with jfk.wav reference → Whisper medium.en 12/13 words, metadata sidecar present;
    Kokoro regression word-perfect on the same live engine. ✅ **Shared espeak bug fixed:** `EspeakTranslator.MatchRule`
    RULE_PRE branch indexed `buf[prePtr]` out of bounds when a pre-context scanned past the per-word buffer start
    (words like "Americans") → guard the OOB read as a space boundary (espeak's clause buffer is space-padded);
    purely additive (that branch previously always threw), fixes ALL espeak TTS. **Remaining:** Random-mode
    diffusion (no-reference synthesis) still a scaffold.
- **Spark-TTS** — reconcile `SparkTtsConfig` token offsets + BiCodec decoder keys to the real checkpoint (parity
  harness already ✅; runtime weight-valid load is the gap).
- **Zonos** — build the conditioning-prefix `[1,P,hidden]` (espeak phonemes + speaker emb + emotion/pitch/rate/lang).
- **PocketTTS** — reconcile placeholder config dims from the checkpoint + wire the SentencePiece tokenizer asset.
- **CosyVoice** — in-process-engine support is factory-blocked ("not yet supported"); needs the runtime wiring
  (parity harness ✅ via shared S3Gen).
- **CSM** — no runtime model descriptor at all; full wiring.

---

## Execution order
Tier 1a (Orpheus) → 1b (default-voice quick wins) → Tier 2 → Tier 3. Verify + doc-update after each model.
Perf follow-up (separate): Bark/Chatterbox/VibeVoice are correct but slow (host/AR-bound) → host-glue→GPU pass.
