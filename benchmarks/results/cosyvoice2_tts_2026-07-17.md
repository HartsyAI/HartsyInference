# CosyVoice 2 (0.5B) — end-to-end bring-up on real weights (2026-07-17)

Text + reference clip → 24 kHz zero-shot voice clone, pure-C# `CosyVoicePipeline`: Qwen2.5-0.5B speech-token LM
(`llm.pt`) → OT-CFM flow-matching mel decoder (`flow.pt`, 10 Euler steps, CFG 0.7) → HiFTNet vocoder (`hift.pt`).
Speaker identity from a CAM++ x-vector + S3 prompt speech-tokens off the reference clip. GPU = RTX 4090
(engine `CudaBackend`, **F32** — the AR LM degenerates under TF32, so `HighPrecisionGemm` is a deliberate
correctness choice, set inside `CosyVoicePipeline.Synthesize`).

This is a **correctness / bring-up** milestone, not a perf pass. First run on the real `FunAudioLLM/CosyVoice2-0.5B`
checkpoint; every component's weight-load was reconciled against the real files and the whole pipeline verified
end-to-end by transcribing the output.

## Verification (Whisper medium.en, word-perfect)

| Path | Text in | Whisper out |
|---|---|---|
| Gated e2e test (`CosyVoiceE2eTests`) | "The quick brown fox jumps over the lazy dog. Text to speech is now working." | *"The quick brown fox jumps over the lazy dog. Text-to-speech is now working."* |
| Gated e2e test (seed 99) | "Cosy Voice two now runs natively in pure C sharp with zero shot voice cloning." | *"Cozy, Voice 2 now runs natively in pure C sharp with zero shot voice cloning."* |
| **Swarm gallery** (`GenerateText2Image`, `Audio Models/CosyVoice/2-0.5b`) | "This clip was generated through the Swarm gallery by the native Cosy Voice engine." | *"This clip was generated through the Swarm Gallery by the native Cozi voice engine."* |

Timbre clones the reference clip. The prompt-mel prefix is trimmed (`CosyVoiceFlow.Inference`) so the vocoder no
longer replays the reference sentence before the target — before the fix, whisper heard the reference transcript
prepended to every clip.

## Timing + the one perf fix applied (weight preload)

The first profile (`HARTSY_PROFILE_SYNC`) was dominated by **~62k `H2D_MISS` transfers (~3.7 s)** — the pipeline
never called `backend.PreloadWeights`, so every Linear/Conv re-uploaded its weight from CPU each call (the exact
anti-pattern AGENTS.md warns about). Added a one-time idempotent `CosyVoicePipeline.PreloadWeights` (bulk-uploads
LM + flow + vocoder + S3 + CAM++; the GPU cache keys by tensor reference so it's free after the first call).

| Metric | Value |
|---|---|
| Swarm gallery gen — **cold** (first gen: 1.9 GB LM disk-load + preload) | 35.5 s |
| Swarm gallery gen — **warm** (resident pipeline, preload done) | **28.9 s** |
| Gated e2e `Synthesize`-only (6.0 s audio, includes first-call preload) | 26.7 s (RTF 4.48) |

Warm RTF ≈ 3× slower than real time. The remaining cost is the **autoregressive speech-token LM decode**:
`CosyVoiceQwenLm.GenerateSpeechTokens` drives ~400 eager `Qwen2Model.ForwardEmbeds` steps (host per-step embed
write + slice + eager per-op attention). The real lever is switching that loop to the GPU-resident CUDA-graph decode
path (`ForwardGraphDecodeStepEmbeds` + device positions/cos-sin tables/`ArgMaxInto`, the same route the LLM sampler
uses and Zonos mirrored for its 203 → 32 ms/frame ~6× win). That is a focused follow-up (target ~6–10 s/clip);
it touches `CosyVoiceQwenLm`'s decode loop, not the already-correct math.

## Weight-load recipe (the actual bring-up work)

The 12 components + parity tests + pipeline facade already existed; the blocker was purely weight loading:

- **LM / flow / vocoder** — CosyVoice2's own `llm.pt` / `flow.pt` / `hift.pt`. Their **default** engine key maps are
  correct against the real checkpoints (`llm.model.model.*` backbone + `speech_embedding`/`llm_decoder`/`llm_embedding`;
  flow top-level `input_embedding`/`encoder.*`/`encoder_proj`/`spk_embed_affine_layer`/`decoder.estimator.*`; HiFTNet
  weight-norm via `WeightNorm.Compose`). No reconciliation needed.
- **S3 speech tokenizer + CAM++ speaker** — loaded from **ResembleAI/chatterbox `s3gen.safetensors`**
  (`tokenizer.*` / `speaker_encoder.*`). These two are FROZEN pretrained models identical across CosyVoice2 &
  Chatterbox. CosyVoice2 ships them ONLY as ONNX, whose export **fuses Conv+BN** in the CAM++ FCM head (the
  `head.bn*` params vanish → the clean-name `CamPlusSpeakerEncoder` can't bind them) and **mangles the S3 names**
  (`quantizer.project_in` vs the expected `quantizer._codebook.project_down`, biases split into anon `onnx::Add_*`).
  Chatterbox ships the same frozen weights clean-named, so they load with zero component changes.

## Deploy notes (SwarmUI-AudioLab)

- `CosyVoiceProvider` had `.WithRequiresDocker()` → `engineBacked` was false → install hit the generic "not yet
  supported by the in-process engine" branch. Dropped it; use `.WithEngineGroup("main")` (like Zonos).
- `UseLocalHartsy` HintPath deploy doesn't copy transitive NuGet deps → `Qwen2Tokenizer` throws
  "Could not load Microsoft.ML.Tokenizers". Run swarmctl `copydlls` after each rebuild.
- `CosyVoiceModel` loads S3/CAM++ from chatterbox `s3gen.safetensors` and tokenizes via `EncodeRawByteLevel`.
