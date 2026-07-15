# Audio Models — status

Concise status for every audio model: TTS, STT, and the codec / voice-conversion / music / separation
family. Build detail lives in [PHASE_5_AUDIO.md](PHASE_5_AUDIO.md); the music-specific completion plan is
[MUSIC_MODELS_COMPLETION_PLAN.md](MUSIC_MODELS_COMPLETION_PLAN.md). Parity evidence (maxAbs, bugs found)
lives in [PARITY_VERIFICATION.md](PARITY_VERIFICATION.md). Legend: [MODEL_STATUS.md](MODEL_STATUS.md).

> ## 🏁 First e2e TTS/STT speed benchmarks (2026-07-12) — RTF on 3060 **and** 4090
> Measured through the SwarmUI+AudioLab path: [`benchmarks/results/audio_tts_stt_2026-07-12.md`](../../benchmarks/results/audio_tts_stt_2026-07-12.md).
> Piper 10.4×/7.7×, Moonshine 6.5×/6.5×, Whisper-base 5.1×/5.4×, MeloTTS 1.7×/1.8× (3060/4090). **These small
> models are host/launch-bound — the 4090 barely helps; the lever is CUDA-graph capture, not a bigger GPU.**
> **Runtime outliers found (parity ✅ ≠ runnable):** ~~Kokoro install 401~~ **FIXED 07-13** (canonical-`.pth`
> download fallback); ~~Whisper `/API/ProcessSTT` rejects the default `en-US`~~ **FIXED 07-13**
> (`WhisperTokenizer.LanguageToTokenId` normalizes locale codes `en-US`→`en`); Spark-TTS install errors
> "checkpoint-reconciliation-pending" (not wired for runtime despite ✅ test parity — still open).
>
> ## ✅ TTS correctness + F5 perf pass (2026-07-13)
> Verified word-correct through the canonical `GenerateText2Image` path with **whisper `medium.en`** as the oracle
> (`base.en` dropped — it hallucinated the pangram onto broken audio) + RMS-envelope match vs a Python reference,
> across short/long/numbers/punctuation prompts: **Kokoro, Piper, MeloTTS, F5-TTS** all fixed to word-correct.
> MeloTTS root cause was a `PytorchPickleLoader` stride bug (transposed BERT weights → gibberish) + missing number
> normalization. **F5-TTS given a perf pass: 174.6 s → 6.4 s (34×)** — the per-forward `F5ConvPosEmbed` grouped
> Conv1D was a host loop; routed to `backend.Conv1d` (GPU), output bit-parity. Remaining perf target: MeloTTS
> (1.4×, BERT+VITS host-flat) + a CUDA-graph pass on the host-bound small models.

> ## 🗺️ Full local-TTS Swarm runtime scoreboard (2026-07-13)
> AudioLab declares ~19 **local** engine-backed TTS providers (+ 20 cloud-API providers — ElevenLabs/Azure/OpenAI/
> etc. — which proxy to third parties and aren't engine models) and 6 local STT. "Has an install button" ≠ "engine
> runtime is wired." Verified via install → `GenerateText2Image` → medium.en. Actual runnable status:
>
> | Status | TTS |
> |---|---|
> | ✅ **verified word-correct** | Kokoro, Piper, MeloTTS, F5-TTS, **Bark**, **Chatterbox**, **VibeVoice**, **FishSpeech**, **Orpheus** |
> | ⏳ **runnable, verify in progress** | Dia (slow AR, gen pending) |
> | 🚧 **partially wired** (loads; clone/synth path throws) | Kyutai TTS, NeuTTS (clone gated), Qwen3-TTS (voice_clone gated) |
> | ⛔ **not wired** (install throws a clear "not runnable yet") | CosyVoice ("not yet supported by the in-process engine"), StyleTTS2 (no unified LoadWeights), Spark-TTS (config/BiCodec reconcile), Zonos (needs conditioning prefix), PocketTTS (placeholder dims), CSM (no runtime model) |
>
> STT (6 local): ✅ Moonshine, Whisper verified word-perfect on real (JFK) speech; Distil-Whisper / Kyutai STT /
> RealtimeSTT / Whisper Streaming not yet installed/verified. **Slowness note:** Bark 85 s, Chatterbox 219 s,
> VibeVoice 206 s — all correct but host/AR-bound; a perf pass (host-glue→GPU, like F5) is the follow-up.

> ## ⚠️ STT reality-check (2026-07-08) — parity ✅ does NOT mean intelligible speech
> The ✅/🔬 marks below are **numeric-parity** verdicts (corr 1.0 vs a Python reference on random/tap inputs).
> A real-weight end-to-end pass — generate audio → resample → Whisper-base STT → content-word recall, then
> a human listen — tells a very different story. Results so far (each writes a WAV to
> `{TmpPath}/hartsyinference_tts_to_stt/`; tests: `*EndToEndSttTests` + `DiaEndToEndTests` + `TranscribeWavFileTests`):
>
> | Model | Doc mark | Whisper heard | Real verdict |
> |---|---|---|---|
> | **Kokoro** | ✅ | "Hello world. This is a test." (4/4) | ✅ **genuinely works** |
> | **MeloTTS** | ✅ | "Hello World, this is a test of the speech synthesizer." (5/5) | ✅ **genuinely works** |
> | **F5-TTS** | ✅ "bit-exact" | (07-08) "(laughs)" → (07-13, with a real voice ref) word-perfect | ✅ **works 2026-07-13** — the 07-08 run had no voice reference; given a reference clip + transcript through Swarm it transcribes word-perfect (medium.en) and clones the voice. Also 34× faster (host-conv→GPU). |
> | **Dia-1.6B** | 🔬 | "(crickets chirping)" (0/7) | ✗ **not intelligible** — gen-loop/DAC bug (transformer parity is real, output isn't) |
> | **Qwen3-TTS 0.6B** | ✅ "bit-exact" | — (RMS 0, silent) | ⚠️ **inconclusive** — probably driven wrong (voice-design mode on a CustomVoice ckpt) |
>
> Lesson: the whole audio suite's "verified" status rests on parity tests that are blind to whether the
> assembled pipeline (sampling, delay, codec decode, vocoder) makes speech. **A model is not "working" until
> Whisper recovers its words and a human confirms the WAV.** Debugging the ✗ models + STT-verifying the rest
> (Chatterbox/VibeVoice/Bark/NeuTTS/FishSpeech + the download-blocked set) is the open work.
>
> Engine changes made during this pass: `DiaPipeline.Generate` now preloads weights to **VRAM**
> (`PreloadWeights`/`FreeWeights`, like YuE) instead of streaming F32 from host RAM per op — a 6.4 GB model now
> lives on the GPU (VRAM 1.7→8.2 GB) with host RAM free, which is also what stops it OOM-crashing a
> RAM-constrained box; the Dia DAC `.pth` state-dict-unwrap load fix; bert-base-uncased converted to
> safetensors (loader can't read legacy pre-1.6 pickle). Heavy runs go through a RAM-watchdog script that
> hard-kills below 1.5 GB free.

## TTS

| Model | Status | Notes |
|---|---|---|
| **GPT-SoVITS v2** | ✅ | HuBERT 1.07e-5, s1 GPT + s2 SoVITS verified, EN end-to-end → 32 kHz on real `lj1995` weights. |
| **Chatterbox** (ResembleAI) | ✅ | Full S3Gen rewrite (== CosyVoice2); enc 2.6e-6 / dec 4.4e-5 / vocoder 1.6e-5; end-to-end on CUDA. |
| **CosyVoice 2** | ✅ | Validated via the shared Chatterbox S3Gen. |
| **Qwen3-TTS** | ✅ | Bit-exact (RoPE split-half + byte-level tokenizer fixes). |
| **Piper** (VITS) | ✅ | corr 0.9998 vs onnxruntime; 7 VITS bugs fixed (affect all VITS). **Swarm e2e word-correct 2026-07-13** — fixed the espeak language default (`en` British → the voice's `en-us` American; it was mispronouncing vowels). |
| **Kokoro** (StyleTTS2) | ✅ | ~1e-4 on the CUDA path (added `audio_leaky_relu` / `audio_adain1d` kernels). **Swarm e2e word-correct 2026-07-13** — misaki-phoneme g2p + punctuation fix (was silently dropping words); canonical-`.pth` download fallback (was install-401). |
| **F5-TTS** (v1 Base) | ✅ | Flow-matching DiT verified bit-exact: velocity corr 1.0, full CFM sample loop (generated mel) corr 1.0, Vocos corr 0.9999. 4 bugs fixed (ConvNeXt filler-mask, ×1000 timestep scale, erf/tanh GELU split, cond-anchored CFG + end-only ref-clamp). **Swarm e2e word-correct + perf pass 2026-07-13:** with a real voice ref it transcribes word-perfect (medium.en); **174.6 s → 6.4 s (34×)** by routing the `F5ConvPosEmbed` grouped Conv1D off the host loop to `backend.Conv1d` (GPU), output bit-parity (RMS-envelope corr 1.0000). |
| **Kyutai TTS** (tts-1.6b-en_fr) | 🔬 | All numerical cores verified (backbone 1.3e-4, depformer 32/32, conditioner ~1e-8). Greedy e2e diverges by argmax cascade (not a bug); Mimi decode reconcile in progress. |
| **ResembleEnhance** | 🔬 | Modules synthetic-verified + converter built; real-weight mel→mel parity pending. |
| **MeloTTS** (English-v3) | ✅ | Real-weight e2e in pure C#. **Swarm e2e word-correct 2026-07-13** — earlier "corr 0.9993 noise-0" was stale: the real e2e produced gibberish from a `PytorchPickleLoader` **stride bug** (bert-base-uncased Linear weights, saved as `.t()` views, loaded transposed → garbage BERT features), fixed with a stride-gather (`MakeRowMajor`, no-op for contiguous — helps all `.pth` models). Also added **number normalization** (`normalize_numbers`: years/currency/ordinals/decimals were dropped). `MeloTts` facade + gated parity test. |
| **Spark-TTS-0.5B** | ✅ | Real-weight e2e bit-exact, fully in-engine (controllable mode): LM logits corr 1.0 (top-1 100%), greedy tokens 32/32 global + 179/179 semantic match Python, BiCodec wav corr 1.0 (factorized VQ, FSQ d-vector, AdaLN PreNet all corr 1.0). `SparkTtsPipeline.LoadFromDirectory`/`LoadAsync` + `SynthesizeControllable(text, gender, pitch, speed)`; `SparkTtsTokenizer` reuses the shared BPE + ByteLevelCodec. Zero-shot cloning would need the BiCodec encoder side (wav2vec2 + ECAPA), not built. |
| **FishSpeech 1.5** | 🔬 | DualAR LM verified: slow (24-layer) corr 1.0, fast depth-LM (4-layer) corr 0.9999. fused-key adapter + interleaved RoPE + no embed-scale + pre-norm fast input. Only the firefly-gan-vq codec remains. |
| **Dia-1.6B** | 🔬 | Full transformer verified bit-exact (corr 1.0): encoder (12L) + decoder (18L, cross-attn/9-ch/fused head). DenseGeneral adapter + split-half RoPE + attn scale 1.0 + KV-cache AdvanceLength fix. Only DAC wiring (shared/✅) + delay-AR remain. |
| **VibeVoice / NeuTTS / Orpheus / Bark / StyleTTS2** | 🔧 | Built (varying completeness); no real-weight parity yet. Orpheus/NeuTTS are phoneme-id-blocked (caller supplies ids). |
| **Zonos** | ⛔ | Blocked: espeak phonemes + ResNet293 speaker encoder + NovelAI sampler. Deferred. |

## STT

| Model | Status | Notes |
|---|---|---|
| **Whisper** (tiny → large-v3) | ✅ | JFK clip transcribes correct content words (`WhisperEndToEndTests`). **Swarm e2e word-perfect 2026-07-13** on the real JFK clip; fixed the `en-US` default-language crash (locale-code normalization). |
| **Whisper streaming** (RealtimeSTT) | ✅ | LocalAgreement-2 + JFK streaming. |
| **Moonshine** | ✅ | Tests pass. **Swarm e2e word-perfect 2026-07-13** on real (JFK) + synthetic clips; ~2 s for 9 s audio on the 3060. |
| **Kyutai STT** (stt-1b / 2.6b) | 🔧 | Shares the moshi backbone; parity pending (no depformer). |

## Codec / voice conversion / music / separation

| Model | Status | Notes |
|---|---|---|
| **OpenVoice** (tone-color VC) | ✅ | Conv2d + GRU + speaker encoder validated. |
| **CAM++ / CamPlus** (speaker) | ✅ | From `funasr/campplus_cn_common.bin`. |
| **S3Tokenizer** | ✅ | From the `s3tokenizer` package. |
| **Vocos / vocoders** | ✅ | Test passes. |
| **GPT-SoVITS HuBERT / CosyVoice sub-encoders** | ✅ | Validated above. |
| **ACE-Step v1** (music DiT 3.5B) | ✅ | DiT ~1e-8 + DCAE decoder corr 1.0 + vocoder corr 1.0; full e2e on CUDA/3060 (bf16 + `HighPrecisionGemm`) writes finite audio. |
| **ACE-Step v1.5 turbo** (music DiT 2B) | ✅ | DiT/cond-encoder/8-step loop all corr 1.0 (~1e-6) vs torch oracle on the real Comfy-Org turbo weights; Oobleck VAE corr 0.9999999999; e2e finite tonal stereo on CUDA. **Perf 2026-07-12:** DiT rewritten host-orchestrated → GPU-resident (device modulation/gated-residual/RoPE/KV-repeat, no per-op D2H sync); bit-identical to the pre-rewrite path (CPU golden maxAbs 0), **measured 55.3 ms/step = 0.44 s for the 8-step turbo DiT at 10 s audio on a 3060** (real weights, `AceStep15DitGpuBench`). Applies to all 9 variants. Follow-ups: F16 activations (needs a split-half F16 RoPE kernel), CUDA step-graph, XL quant. |
| **Mimi** (codec) | 🔬 | SeaNet composed-weight load fixed (DSM checkpoint); DSM 32-cb decode reconcile in progress. Shared with CSM. |
| **MusicGen / AudioGen** | ✅ | T5-base corr 1.0 + decoder logits corr 0.999999 + EnCodec-32k decode corr 1.0; e2e on CUDA writes music-like audio. 5 bugs fixed (T5/EnCodec). |
| **YuE** (music, Stage-1) | ✅ | Stage-1 7B LM corr 1.0 (argmax 8/8) + XCodec (SoundStream) decode corr 1.0 → generates 16 kHz vocal audio. Stage-2 multi-codebook out of scope. |
| **HeartMuLa** (oss-3B) | ✅ | LM corr 0.9996–0.9999 + HeartCodec rewritten: flow-match estimator corr 1.0 + ScalarModel corr 1.0 → generates 48 kHz audio (CPU + CUDA). **Perf (RTX 3060, 3b-base):** ~91 ms/frame ≈ 11 fr/s bf16 (~0.9× realtime, memory-bandwidth-bound). CUDA-graph decode of the backbone + depth steps (`HARTSY_CSM_GRAPH`, default on) is bit-identical + ~5% (launch overhead is only ~8/91 ms). Disk-cached weight quant (`HARTSY_HEARTMULA_QUANT=q8_0`) is **1.41× faster** (64.8 ms/frame ≈ 15.4 fr/s, past real-time) + ~1/2 VRAM — the fix was pinning the quant weights GPU-resident (`PreloadWeights`, quantized-only); the Q8 fused GEMV is faster than bf16 when resident. |
| **RVC** (voice conversion) | 🔧 | RMVPE front-end built; parity pending. |
| **Demucs** (separation) | 🔧 | Built; parity pending. |
| **CSM** (Sesame) | 🔧 | Uses Mimi; parity pending. |
| **Stable Audio Open / DiffRhythm / AudioLDM 2 / ACE-Step XL** | ❌/🔧 | Music roadmap; see [MUSIC_MODELS_COMPLETION_PLAN.md](MUSIC_MODELS_COMPLETION_PLAN.md) for the per-model build state and ROI order. |
| **PocketTTS** (continuous-latent) | ⛔ | Gated `kyutai/pocket-tts`; config dims are placeholders. Reuses the moshi backbone. |

## Notes

- Music models have their own definition of "production-ready" and a sequenced completion plan in
  [MUSIC_MODELS_COMPLETION_PLAN.md](MUSIC_MODELS_COMPLETION_PLAN.md); the universal missing piece there is
  the audio parity harness, now proven on ACE-Step's DiT.
- Build audio with `-m:1`; the Audio test suite crashes under xunit parallel, run it sequentially
  (`-- xUnit.ParallelizeTestCollections=false`) and reuse a model cache via `HARTSYINFERENCE_MODEL_CACHE`.
