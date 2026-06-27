# Model Parity Verification — master checklist

> One living checklist for **every** model HartsyInference targets (all modalities), tracking the
> **numerical parity verification loop**: load the real checkpoint → dump a reference from the upstream
> implementation → diff the C# output → fix bugs → check it off. This is distinct from the per-modality
> status docs indexed in [`MODEL_STATUS.md`](MODEL_STATUS.md) (which track whether a model is *built*);
> this tracks whether it has been *proven correct against real weights*.
>
> **Per-type status docs:** [Image](MODEL_STATUS_IMAGE.md) · [Audio](MODEL_STATUS_AUDIO.md) ·
> [Video](MODEL_STATUS_VIDEO.md) · [World](MODEL_STATUS_WORLD.md) · [3D](MODEL_STATUS_3D.md) ·
> [Vision](MODEL_STATUS_VISION.md) · [LLM](MODEL_STATUS_LLM.md).

---

## FOR THE AGENT — how to verify a model (read this first)

You are continuing a long-running effort to fix and numerically verify every model in this engine. Pick the
next unchecked model (or one the user names) and run the **parity loop**. When done, check it off here and add
the bugs you found + how you found them.

### The parity loop (the method that has worked for every model below)
1. **Map the C# side.** Read the model class(es) + `LoadWeights` + the pipeline. Note the exact weight-key
   strings it expects and the config dims. (An `Explore`/general-purpose subagent is good for this.)
2. **Get the real checkpoint.** `hf download <repo> <file> --local-dir <scratchpad>`. Many are gated — ask the
   user to accept terms / provide an HF token.
3. **Inspect the real keys vs what C# expects.** Dump the safetensors/pickle header
   (`struct.unpack('<Q', f.read(8))` then the JSON). The #1 source of bugs is a key-layout / architecture
   mismatch (the C# scaffold guessed wrong). Reconcile before anything else — `LoadWeights` hard-throws on the
   first wrong key.
4. **Extract the upstream reference.** Prefer running the real upstream library (HF `transformers`,
   `diffusers`, `moshi`, the model's repo) as an oracle. `pip install` it into the sandbox; if its env is
   broken, write a standalone torch reimplementation from the source. Dump fixed-input → fixed-output IO to a
   `.safetensors` the C# test reads.
5. **Diff per-component, then end-to-end.** Validate the *risky* pieces in isolation first (novel attention /
   RoPE / norm / fused-weight layouts), each against its own reference dump, THEN the full forward. Bugs are
   far easier to localize per-component.
6. **Fix → re-run → tighten tolerance.** See tolerance table below.
7. **Check it off here** with the maxAbs achieved, the bugs found, and how.

### Reference-dump gotchas (learned the hard way)
- **Memory safety — CRITICAL.** Loading a multi-B model cast to f32 (`EnsureF32`) can be 5–13 GB RAM. Doing
  this while VSCode + another agent run **OOM-killed the box and crashed VSC**. Rules: **never `.float()` a
  whole large model in the Python reference** — cast only the submodule under test (`lm.transformer.float()`).
  Guard every C# test run: abort if `free < 2500 MB`; run **one heavy test at a time**; prefer **CUDA**
  (weights stay host-resident, ~1.5 GB VRAM, RAM only holds the f32 cast).
- **Sandbox torch is CPU-only.** It can't bf16-matmul → cast the reference to f32 (this is the reference, not
  the 3060). Disable torch.compile: `TORCHDYNAMO_DISABLE=1 TORCHINDUCTOR_DISABLE=1` (inductor's C++ compile
  fails in the sandbox).
- **bf16 reference noise** accumulates ~1e-2 over many layers; for a clean f32-vs-f32 check, cast the
  reference submodule to f32 and you'll see ~1e-4.
- **CPU is slow / dangerous for big AR models.** An O(n²) full-prefix re-run on a 1.6 B model never finished on
  CPU and pegged the CPU. Use the GPU; add a KV cache for AR generation.

### Tolerance conventions (PHASE_5_AUDIO §9 + practice)
- Linear / attention F32 vs f32 reference: **~1e-4** (≤5e-3 passes). bf16 reference: ~1e-2.
- Logits over many layers with large magnitude: **~1e-2 abs** can be expected (depformer logits hit 6.9 peak).
- **Argmax/codes (greedy):** prefer **exact token match** as the functional check, but know that bit-exact
  greedy over a *long* sequence is fragile — a validated 1e-4 backbone error eventually flips one argmax and
  greedy cascades it. For long AR, validate **logits under teacher-forcing**, not sampled codes.
- Codec round-trip: STOI > 0.95.

### How to record results here
Mark the **Parity** column: ✅ verified vs real weights · 🔬 partially verified (some components) ·
🔧 built, parity-pending · 🚧 scaffold only · ❌ not started · ⛔ blocked (gated weights / external dep).
Add a row to **§ Bugs found** for every real bug, with how it was caught.

---

## AUDIO

### TTS
| Model | Parity | Components verified (maxAbs) | Notes |
|---|---|---|---|
| **Kyutai TTS** (tts-1.6b-en_fr) | 🔬 | backbone 1.3e-4 · depformer 32/32 tokens · conditioner ~1e-8 · forward_text 2.6e-6 · scheduler exact · KV-cache StepForward 1.3e-4 | All numerical cores + loop logic verified. Greedy end-to-end codes diverge after cb-0 by **numerical argmax cascade** (not a bug). Mimi DSM decode = separate codec reconcile (in progress). See [[kyutai-tts-reconcile]]. |
| **GPT-SoVITS v2** | ✅ | HuBERT 1.07e-5 · s1 GPT (golden-fixed) · s2 SoVITS (3 goldens) · EN end-to-end → 32 kHz | English path end-to-end on real `lj1995` weights. See [[gptsovits-v2-validated]]. |
| **Chatterbox** (ResembleAI) | ✅ | S3Gen enc 2.6e-6 · dec 4.4e-5 · vocoder 1.6e-5 · perceiver 1.5e-6 · T3/VE/tokenizer | Full S3Gen rewrite (== CosyVoice2); end-to-end on CUDA. Supersedes the old "not real-weight validated" note. |
| **Qwen3-TTS** | ✅ | bit-exact | RoPE split-half + byte-level tokenizer fixes. See [[qwen-tokenizer-bytelevel-bug]]. |
| **Piper** (VITS) | ✅ | corr 0.9998 vs onnxruntime | 7 VITS bugs fixed (affect all VITS). See [[piper-vits-parity]]. Phonemization (espeak) is the only gap. |
| **Kokoro** (StyleTTS2) | ✅ | ~1e-4 (CUDA path) | Added `audio_leaky_relu`/`audio_adain1d` CUDA kernels. Vocoder DSP is pure C#. See [[kokoro-cuda-path-complete]]. Loader 404s until repacked — see [[audio-repack-tool]]. |
| **CosyVoice 2** | ✅ | == Chatterbox S3Gen (shared) | Validated via the Chatterbox rewrite. |
| **F5-TTS** | 🔧 | — | Built + wired in SwarmUI; parity dump pending. |
| **HeartMuLa** | 🔧 | — | Built + wired; parity pending. |
| **ResembleEnhance** | 🔬 | modules synthetic-verified; checkpoint converter built | Mel→mel structural green; real-weight parity pending. |
| **VibeVoice / SparkTTS / NeuTTS / Orpheus / MeloTTS / Bark / Dia / FishSpeech / StyleTTS2** | 🔧 | — | Built (varying completeness); no real-weight parity yet. Orpheus/NeuTTS are phoneme-id-blocked (caller supplies ids). |
| **Zonos** | ⛔ | — | Blocked: espeak phonemes + ResNet293 speaker encoder + NovelAI sampler. Deferred. |

### STT
| Model | Parity | Components verified | Notes |
|---|---|---|---|
| **Whisper** (tiny→large-v3) | ✅ | JFK clip transcribes correct content words | `WhisperEndToEndTests`. |
| **Whisper streaming** (RealtimeSTT) | ✅ | LocalAgreement-2 + JFK streaming | Built this milestone. |
| **Moonshine** | ✅ | tests pass | — |
| **Kyutai STT** (stt-1b/2.6b) | 🔧 | — | Shares the moshi backbone; parity pending (no depformer). |

### Codec / VC / music / separation
| Model | Parity | Components verified | Notes |
|---|---|---|---|
| **Mimi** (codec) | 🔬 | SeaNet composed-weight load fixed (DSM checkpoint) | DSM 32-cb decode needs the transformer-of-codecs reconciled to the DSM layout (in progress). Shared with CSM. |
| **OpenVoice** (tone-color VC) | ✅ | Conv2d + GRU validated; speaker encoder | See [[audiolab-engine-blockers]]. |
| **CAM++ / CamPlus** (speaker) | ✅ | from `funasr/campplus_cn_common.bin` | — |
| **S3Tokenizer** | ✅ | from `s3tokenizer` pkg | — |
| **Vocos / Vocoders** | ✅ | test passes | — |
| **ACE-Step** (music DiT 3.5B) | 🔬 | DiT parity ~1e-8 | E2E gen env-gated (13 GB F32 cast — bare terminal only). See [[ace-step-build]], [[music-models-plan]]. |
| **MusicGen** | 🔧 | — | 32 kHz EnCodec scope corrected; parity pending. |
| **YuE** (music) | 🔧 | — | Built; pending. |
| **RVC** (voice conversion) | 🔧 | — | RMVPE front-end built; pending. |
| **Demucs** (separation) | 🔧 | — | Built; pending. |
| **CSM** (Sesame) | 🔧 | — | Uses Mimi; pending. |
| **GPT-SoVITS HuBERT / CosyVoice sub-encoders** | ✅ | (validated above) | — |
| **PocketTTS** (continuous-latent) | ⛔ | — | Gated `kyutai/pocket-tts`. Config dims are placeholders (`DModel=0` etc.) — Step 0 = read the real checkpoint header. Reuses the moshi backbone. |

---

## IMAGE (diffusion) — see [`MODEL_STATUS_IMAGE.md`](MODEL_STATUS_IMAGE.md) for build detail
| Model | Parity | Notes |
|---|---|---|
| **SD 1.5 / SDXL / SD3.5 / Flux (Dev/Schnell/Krea) / Z-Image / Flux.2 Klein** | ✅ | Clean visual output verified end-to-end. Many plumbing bugs fixed (SD3 #31-35, Z-Image #25-30 in PHASE_3_DEVIATIONS). |
| **Ideogram 4** (9.3B DiT) | 🔬 | ~1e-7 parity on 3060 after the GPU-residency rewrite (`dit_f32.ptx`). A100 timing pending. See [[ideogram4-gpu-residency]]. |
| **Qwen-Image / Chroma / ChromaRadiance / ZetaChroma / ERNIE-Image / Hunyuan-Image / Lumina-2 / HiDream / Kandinsky-5 / Anima / OmniGen-2 / F-Lite / AuraFlow / Krea-2 / Boogu-Image / Microsoft Lens / Lance-image / Flux.2 Dev (32B)** | 🔧 | Built end-to-end, numerics validation-pending (need Python-parity dumps + checkpoint download; several gated on VRAM). Status detail per model in [`MODEL_STATUS_IMAGE.md`](MODEL_STATUS_IMAGE.md). |

---

## VIDEO / WORLD MODELS — see [`PHASE_9_VIDEO.md`](PHASE_9_VIDEO.md), [`PHASE_10_INTERACTIVE.md`](PHASE_10_INTERACTIVE.md)
| Model | Parity | Notes |
|---|---|---|
| **Wan 2.2 TI2V-5B (T2V/I2V) / LTX-Video / LTX-2 / WanAnimate / WanS2V / WanVace / Lance-video / Kandinsky-5-video** | 🔧 | Built end-to-end, structural tests pass; numeric parity pending. See [[wan-build]], [[ltx-build]], [[ltx2-build]], [[lance-build]]. |
| **Matrix-Game 3.0 / Matrix-Game 2.0 / Oasis-500m / Hunyuan-GameCraft** | 🔧 | World models built (structural). See [[matrix-game-build]], [[matrix-game-2-build]], [[oasis-build]], [[gamecraft-build]]. |
| **Cosmos-Predict1 V2W** | ❌ | Not started (FSQ tokenizer + AR transformer substrate). |

---

## 3D — see [`PHASE_11_THREED.md`](PHASE_11_THREED.md)
| Model | Parity | Notes |
|---|---|---|
| **Hunyuan3D-2 / TripoSR** | 🔧 | image→mesh built (structural); marching cubes + glTF/OBJ/PLY foundation. See [[threed-foundation]]. |

---

## LLM / TEXT ENCODERS — see [`PHASE_12_LANGUAGE.md`](PHASE_12_LANGUAGE.md)
| Model | Parity | Notes |
|---|---|---|
| **GenericTransformer** (Qwen2/Qwen3/Llama-3) | ✅ | Has parity tests; backs the text encoders + native LLM (Phase 12). Llama-3 RoPE NTK-by-parts validated. |
| **T5 / UMT5 / Pile-T5 (AuraFlow) / BERT / SigLIP / Qwen3-VL vision tower** | 🔬 | Text/vision encoders with diff tests (`T5EncoderDiff`, `BertModel`, `Siglip`, `Qwen3VlVisionTower`). Pile-T5 = UMT5 (per-layer rel-attn-bias). See [[pile_t5_is_umt5]]. |
| **Native LLM (Phase 12)** | 🚧 | Plan: one config-driven transformer backing both LLM + text encoders; CUDA-first quant; GPU-resident decode is the gating blocker. See [[llm-language-package-plan]]. |

---

## § Bugs found (and how they were caught)

Real bugs surfaced by the parity loop — kept here so the patterns repeat-catch across models.

| Model | Bug | How it was caught | Fix |
|---|---|---|---|
| Kyutai TTS | Cross-attention applied RoPE | Backbone diverged 2e-2 (bulk matched 1e-4); moshi `apply_rope` asserts `q.T==k.T` which fails for T≠S | moshi forbids rope in cross-attn — removed it. cb-0 backbone → 1.3e-4 |
| Kyutai TTS | RMSNorm eps 1e-8 (scaffold) vs **1e-5**; gating inner 8448 vs **5632** (⅔ trick); interleaved (not NeoX) RoPE; norm stored as `alpha` | Reading the moshi source while reconciling keys | Corrected all in `MoshiTransformer` |
| Kyutai TTS | Depformer weight layout wrong (per-(set,layer) q/k/v vs **packed** `in_proj_weight [33792,1024]` sliced by schedule) | Real safetensors key dump vs C# expectation | Rewrote `MoshiDepformer` with packed slicing |
| Kyutai TTS | `EmbedFrame` indexed `_emb[k][-1]` for the warmup zero-token (garbage) | Greedy generation parity: frame-0 cb-0 wrong (943 vs 759) | `if(code==_zeroToken\|\|code<0) continue` (moshi ScaledEmbedding `is_zero`) → cb-0 exact |
| Kyutai TTS | `MoshiDepformer.TextEmbed` didn't demux the multiplexed text token | Greedy parity (depformer test only used a small token 42, so out2 path untested) | Demux main/second via out1/out2 |
| Mimi (DSM) | SeaNet loader required `weight_g/weight_v`; DSM ships composed `conv.weight`; and `NResidualLayers` over-counted blocks per stage | `LoadWeights` KeyNotFound on `encoder.model.0.conv.conv.weight_g` then `model.2.block.1` | `LoadFusedConvWeight` accepts both formats; SEANet residual-count + decoder seqIdx fixed |
| GPT-SoVITS | s1 golden logits were **stale/wrong** (C# was right) | Independent reimpl + real upstream module both gave -6.16 | Fixed the test golden |
| GPT-SoVITS | HuBERT pos_conv weight-norm (dim=2) loader; GroupNorm needs rank-4; tanh→exact-erf GELU | HuBERT output ~zeros; diff vs PyTorch | `ComposePosConvWeightNorm`, rank-4 reshape, `ExactGelu` → 1.07e-5 |
| Qwen3-TTS | RoPE interleaved vs **split-half NeoX**; tokenizer dropped leading spaces (no GPT-2 byte-level) | bit-exact diff failed | `RopeStyle.SplitHalf` + `ByteLevelCodec.Encode` |
| Piper / all VITS | 7 VITS bugs (rel-pos attn zero-pad, folded EA `exp(-logs)`, resblock conv naming, …) | corr vs onnxruntime | fixed → 0.9998 |
| Engine-wide | CPU SDPA rank-2 causal mask mis-indexed for heads > 0 (OOB → flaky NaN) | Found while validating Boogu-Image's `LlamaStyleEncoder` | `AttentionKernels` branches on mask rank |
| Engine-wide | `WhisperOps.ProjectLinear`/`BatchedMatMul` heap-corrupt on rank-2 input | ECAPA/Zonos stat-pooling crashes | Always pass `[1,seqLen,inDim]`. See [[projectlinear-needs-rank3]] |
| Engine-wide | CPU MatMul/Linear heap-corrupt on bf16/f16 weights | Audio loaders | Cast to F32 first (`RequireF32`). See [[cpu-kernels-f32-only]] |

---

## Conventions / where things live
- **Reference dumps + scratch:** per-model under the session scratchpad; the moshi/audio refs live in
  `scratchpad/kyutai/`. Reference `.py` scripts are kept beside their dumps.
- **Parity tests:** `tests/HartsyInference.Audio.Tests/*ParityTests.cs` (gated on env vars pointing at the real
  checkpoint + the dumped reference). Image/video parity uses the per-model `*DebugDump` + diff harness.
- **Build:** audio with `-m:1`; tests crash under xunit parallel → run sequentially
  (`-- xUnit.ParallelizeTestCollections=false`). Reuse a model cache via `HARTSYINFERENCE_MODEL_CACHE`.
- **Never** commit/push — the user does all git manually.
