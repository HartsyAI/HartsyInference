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

## Perf pass — GPU-resident CFM flow decoder (5.7× warm)

Phase timing (`backend.Sync()` boundaries, 6.0 s clip) located the cost — **not** the LM as expected:

| Phase | Before | After |
|---|---|---|
| prep (S3 + CAM++ + mels) | 1332 ms | 1259 ms |
| **LM** (149 speech tokens, AR) | 968 ms | 945 ms |
| **flow** (CFM: conformer + 10 Euler × 2 CFG estimator) | **24 645 ms** | **5 237 ms** |
| vocoder | 549 ms | 527 ms |
| **`Synthesize` total** | 27 495 ms (RTF 4.61) | **7 969 ms (RTF 1.34)** |

The LM was already fast (~1 s) — it reuses the LLM package's `GenericTransformer` + `FixedKvCache`. The bottleneck
was the **CFM velocity estimator** (`CausalConditionalDecoder`), run 20× per gen (10 Euler × 2 CFG), each forward
saturated with host-glue that read `(float*)DataPointer` on GPU tensors → a device→host sync every call: host `Mish`
(29×/fwd), host `ExactGelu` (56×/fwd), host `ToHeads`/`FromHeads` head reshapes (112×/fwd), host residual/time-emb
adds, host `PackInput`/`ConcatChannels`. The sync profile screamed **~62k `H2D_MISS` + host loops**.

Fix — moved the whole estimator on-device, reusing existing `IBackend` ops where they existed and adding one kernel:
- **`Mish`** — new `audio_mish_f32` PTX kernel (`native/cuda/audio/audio_activations_f32.cu`) + `IBackend.Mish`
  (default host impl, CUDA override). This is the only new kernel.
- `ExactGelu` → `backend.GeluErf` (erf kernel, already existed).
- `ToHeads`/`FromHeads` → `backend.Permute0213`.
- time-emb broadcast add → `backend.BroadcastAdd`; residual/attn adds → `backend.Add`.
- `PackInput` (spk time-broadcast + 4-way channel concat) → `backend.RepeatTime` + `backend.Concat`; `ConcatChannels`
  → `backend.Concat`.

Plus a one-time idempotent `CosyVoicePipeline.PreloadWeights` (the pipeline never preloaded → per-op weight
re-upload). Correctness preserved: e2e whisper stays word-perfect; the shared estimator's other consumer
(**Chatterbox**) re-verified word-perfect through the gallery. Math unchanged — pure residency refactor.

| Swarm gallery gen | Before | After |
|---|---|---|
| **cold** (first gen: LM disk-load + preload) | 35.5 s | **11.3 s** |
| **warm** (resident pipeline) | 28.9 s | **5.1 s (5.7×)** |

Remaining headroom (not pursued): the flow is now launch-overhead-bound (~32k small `Linear` launches across the 20
estimator forwards); a CUDA-graph capture of the estimator step would cut host launch cost further, at graph-capture
complexity. RTF 1.34 (warm ~5 s/clip) is a good stopping point.

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
