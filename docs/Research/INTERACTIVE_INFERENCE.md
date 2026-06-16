# Interactive Inference — Research Notes

> Status: Foundational design doc for Phase 9 (shared infra) and Phase 10 (world models). | Last Updated: 2026-05-24 | Needed Before: any world-model pipeline lands.
> Related: [`MATRIX_GAME_3_ARCHITECTURE.md`](MATRIX_GAME_3_ARCHITECTURE.md), [`MATRIX_GAME_2_ARCHITECTURE.md`](MATRIX_GAME_2_ARCHITECTURE.md), [`OASIS_ARCHITECTURE.md`](OASIS_ARCHITECTURE.md), [`HUNYUAN_GAMECRAFT_ARCHITECTURE.md`](HUNYUAN_GAMECRAFT_ARCHITECTURE.md), [`COSMOS_PREDICT1_VIDEO2WORLD_ARCHITECTURE.md`](COSMOS_PREDICT1_VIDEO2WORLD_ARCHITECTURE.md), [`LANCE_ARCHITECTURE.md`](LANCE_ARCHITECTURE.md), [`FLOW_MATCHING_AUDIO.md`](FLOW_MATCHING_AUDIO.md).

## Summary

Interactive inference is a distinct mode of generative model serving where the model **emits a frame per user input** at real-time rates (25–40 FPS) rather than producing an offline clip. The model is *driven* by an action stream — keyboard, mouse, gamepad, camera pose, or arbitrary control vectors — and maintains a persistent rolling state (KV-cache, frame history, memory bank) across steps. The classic example is "world models" for games (Matrix-Game, Oasis, Hunyuan-GameCraft), but the same machinery serves any application where a model must respond to live input: real-time animation, agent simulation, AR overlays, controllable music generation.

This document defines the **cross-cutting infrastructure** that has to land in HartsyInference before any interactive world model can run. The model-specific architecture details live in the per-model research docs; this doc is about the shared substrate: action encoders, the streaming session API, KV-cache for video tokens, discrete video tokenizers, distilled few-step schedulers, streaming VAE decode, memory-augmented attention, license-acceptance plumbing, and the package boundary between `HartsyInference.Video` (shared infra) and `HartsyInference.Interactive` (user-facing world-model pipelines).

Phase 9 (Video) is the right home for everything an offline AR-video-continuation model also needs (3D causal VAE, packed/varlen attention, distilled schedulers, streaming VAE decode, discrete video tokenizer abstraction). Phase 10 (Interactive / World Models) is the right home for the strictly real-time-and-action-conditioned pieces (`IInteractiveSession`, `IActionEncoder`, history-mask channel, memory-augmented cross-attention) and for the model pipelines themselves. We intentionally **do not** build all 10 foundational gaps up front — only the ones the chosen v1 models (Matrix-Game 2/3, Oasis, Hunyuan-GameCraft) actually need, with the rest documented as a deferred backlog.

## Detailed Findings

### 1. What makes interactive inference different from offline diffusion

| Property | Offline diffusion (SDXL, Flux, Lance image) | Offline video diffusion (LTX, Wan, Lance video) | Interactive world model (Matrix-Game, Oasis, GameCraft) |
|---|---|---|---|
| Trigger | Single prompt at submission | Single prompt at submission | One action per step, indefinitely |
| Duration | Bounded (1 image) | Bounded (one clip, ≤ 121 frames) | Unbounded (streams forever) |
| Latency target | Seconds-to-minutes | Tens of seconds | **25–40 ms per frame** |
| State | Stateless | Stateless | **Persistent rolling state** (KV-cache, history latents) |
| Sampling | 8–50 steps | 8–50 steps | **3–8 steps per frame** (distilled) |
| VAE decode | Once at the end | Once at the end | **Every frame** (or every chunk) |
| Output | `byte[] rgb` | `IAsyncEnumerable<VideoFrame>` (whole-clip) | `IAsyncEnumerable<VideoFrame>` (live, with action input loop) |
| Cancellable mid-run | Yes (drop the result) | Yes (drop pending frames) | Yes (just stop pumping actions) |
| Output reuse | None | None | **Past frames feed the next prediction** |

The defining trait is the **action input loop**: a frame can't be generated until an action arrives, and the engine has to keep up. This forces every component on the critical path — text/action encoder, transformer, scheduler, VAE — to be either skippable, cacheable, or fast enough to run every frame.

### 2. Action conditioning — the abstraction

Every interactive model takes some flavor of action. The action vocabularies vary wildly, so we define a thin generic interface and let each model implement its own encoder.

```csharp
// HartsyInference.Diffusion (Phase 9 — shared so video-cond models can use it too)
namespace HartsyInference.Conditioning;

public readonly record struct ActionInput(
    ReadOnlyMemory<byte> Payload,       // model-defined bytes (key state, camera pose, etc.)
    long FrameIndex,                    // monotonic, lets the encoder do positional things
    long TimestampNanos);               // wall-clock when produced

public interface IActionEncoder : IDisposable
{
    int EmbeddingDim { get; }            // output dim per action token
    int TokensPerAction { get; }         // most models emit 1 token per action; some emit N
    void Encode(in ActionInput action, Span<float> outputTokens); // shape [TokensPerAction, EmbeddingDim]
}
```

**Per-model implementations** (post-research; details confirmed by per-model docs):

- `Matrix-Game 3.0 IActionEncoder`: 6-dim keyboard one-hot + 2-dim mouse (Δx, Δy yaw/pitch) per RGB frame. Downsampled 4× to latent rate. Plus a separate `CameraConditioner` that integrates the action stream into per-frame SE(3) extrinsics → **Plücker 16-channel ray embeddings** at full latent resolution. The action tokens drive a per-block `ActionModule` (mouse=self-attn, keyboard=cross-attn) attached to a *subset* of the 40 DiT blocks; Plücker is added/concat into patch embeddings. **NOT** a single up-front conditioning vector.
- `Matrix-Game 2.0 IActionEncoder`: similar pattern, but **per-variant action vocabs** — Universal (4 kbd + 2 mouse), GTA (2 kbd + 2 mouse), TempleRun (7 kbd, no mouse). Past-latent windows of `windows_size=3` with 12-frame padding. Same per-block `ActionModule` injection on first 15 of 30 SkyReels-V2 (Wan2.1) DiT blocks for the distilled variant, all 30 for foundation.
- `Oasis IActionEncoder`: 25-dim Minecraft VPT action vector (23 binary keys + 2 normalised camera floats), projected by `nn.Linear(25, 1024)` and **added to the per-frame timestep embedding** (simplest pattern of the bunch — no separate cross-attn stream).
- `Hunyuan-GameCraft IActionEncoder`: 4-dim keyboard `(w/a/s/d)` + 1-dim `speed ∈ [0, 3]` → integrated to continuous `(d_trans, d_rot, α, β)` → 33 camera poses → **Plücker 6-channel ray maps** at full resolution. Plus a dedicated `CameraNet` module (`PixelUnshuffle(8) → Conv 384→192 → GN → ReLU → temporal pool → Conv 192→96 → GN → ReLU → Conv 96→16 → PatchEmbed → learnable scale`) that token-adds into the image stream.
- `Cosmos-Predict1 V2W` does **not** implement `IActionEncoder` — V2W has no action input. The Cosmos AR codebase has dormant action-conditioning hooks (`use_action_condition`, `action_dim=8`, `concat_action_to_context`) used for NVIDIA's robotics post-training — these are the seam where a future Cosmos-AR world-model variant would plug in.

**Takeaway for the abstraction:** `IActionEncoder.Encode()` returning a single `Span<float>` of tokens is *not enough*. Real interactive models need multiple typed conditioning streams per frame — at minimum (keyboard tokens, mouse tokens, camera Plücker maps) — and inject them at different depths in the transformer. The right interface is:

```csharp
public interface IActionEncoder : IDisposable
{
    // Returns multiple named streams; the pipeline routes each to its consumer block.
    IReadOnlyDictionary<string, ActionStream> Encode(in ActionInput action);
}

public readonly record struct ActionStream(
    Memory<float> Data,
    ReadOnlySpan<int> Shape,           // e.g. [1, T_lat, 1024] for mouse stream
    ActionStreamRole Role);            // PerBlockSelfAttn | PerBlockCrossAttn | PluckerMap | TimestepAddon | ...
```

Per-model encoder defines which stream names it produces; per-model pipeline knows which streams its blocks consume. Pipelines without an action encoder (Lance video, Cosmos V2W) simply pass no streams.

**Why generic instead of `KeyboardActionEncoder` / `MouseActionEncoder` / etc. baked in:** every model defines its own vocab and tokenization. A generic byte payload + per-model encoder keeps HartsyInference out of the business of unifying action spaces (which is what game engines do).

**Why not a single `Tensor` per action:** allocations on the hot path. The 25–40 FPS budget is too tight to allocate a new `Tensor` per step. `Span<float>` is filled into a pre-allocated buffer the session owns.

### 3. The interactive session — `IInteractiveSession`

The streaming loop is the highest-level abstraction. It lives in the new `HartsyInference.Interactive` package (Phase 10).

```csharp
// HartsyInference.Interactive
namespace HartsyInference.Interactive;

public interface IInteractiveSession : IAsyncDisposable
{
    int FrameWidth { get; }
    int FrameHeight { get; }
    int TargetFps { get; }
    long CurrentFrameIndex { get; }

    // Producer side — the application pushes actions in.
    ValueTask SubmitActionAsync(in ActionInput action, CancellationToken ct = default);

    // Consumer side — the application reads frames out.
    IAsyncEnumerable<VideoFrame> ReadFramesAsync(CancellationToken ct = default);

    // Inspect / tune.
    InteractiveSessionStats GetStats();    // p50 / p99 step latency, dropped frames, queue depth
    void SetQualityProfile(QualityProfile profile);
}
```

**Internal pipeline (per step):**

```
[ActionInput] → IActionEncoder → action tokens
                                  │
[history latent ring buffer]  ────┤
                                  ▼
                            Transformer.Step(noisy_latent, action_tokens, history, kv_cache)
                                  │
                                  ▼
                            scheduler.Step(velocity, dt)
                                  │
                                  ▼
                            (after N steps — typically 3-8 for distilled)
                                  │
                                  ▼
                            VAE.DecodeFrame(latent)
                                  │
                                  ▼
                            VideoFrame
```

**Bounded queues with backpressure:** input action queue + output frame queue, both bounded. If the consumer falls behind on frames, the session drops the oldest unread frame to keep latency low. If the producer overruns (impossible if user-driven, but possible if a script is replaying), the action queue blocks the submitter. Both behaviors are configurable.

**Thread model:** one dedicated compute thread per session. CUDA stream pinned to that thread; the application threads only touch the lock-free queues. Avoids spinning up `Task.Run` per step (allocation + scheduler overhead).

**Why not `IAsyncEnumerable<VideoFrame>` on its own:** the consumer-only `IAsyncEnumerable` shape (as used by Lance video) is offline-shaped. You ask for frames and get a fixed-length stream. Interactive sessions are bidirectional and indefinite — actions in, frames out, forever. The `IInteractiveSession` shape is honest about that.

**Why a separate package (`HartsyInference.Interactive`):** the streaming session has different threading and lifecycle semantics from any other pipeline. Bundling it into `HartsyInference.Video` would pull `IInteractiveSession` into every video pipeline's surface even though offline pipelines don't need it.

### 4. KV-cache for video tokens (`DenoiseKvCache`)

For models that do multi-step denoising per frame (Matrix-Game, GameCraft) or autoregressive token generation per frame (Cosmos AR, future MineWorld), per-step recomputation of the *whole* transformer is wasteful. Three cache strategies, depending on the model:

- **Diffusion prefix-cache.** The text/action/history conditioning doesn't change across denoising steps within a frame; only the noisy slot does. Cache K/V for the prefix on step 0, recompute only the noisy slot on steps 1..N. ~2-3× speedup, standard SD3/Flux trick adapted for video. Used by Lance video and GameCraft. *Not* used by Matrix-Game (which uses the sliding-window cache below).
- **Sliding-window video KV-cache** (Matrix-Game 2/3 pattern). The model attends to a bounded window of past *latent frames* (Matrix-Game 2.0: `local_attn_size=6`, `num_frame_per_block=3`, `sink_size=0`; Matrix-Game 3.0: 5 memory slots + 4 past-overlap latents + current segment). When a new frame is generated, the oldest cached K/V is evicted (or, in Matrix-Game 3.0, re-selected by camera-frustum overlap). Multiple parallel caches per layer — Matrix-Game 2.0 keeps **three** separate caches (main self-attn, mouse ActionModule, keyboard ActionModule) plus a one-shot CLIP image-conditioning cache that lives for the entire session. This is closer to StreamingLLM than to the LLM prefix-cache.
- **AR KV-cache for video tokens.** For AR-token models (Cosmos-AR-13B, future MineWorld / Solaris), the past video tokens + past actions are the K/V being attended to. The cache grows monotonically (or via sliding window) as frames are generated. This is the same pattern as LLM inference (which dotLLM owns), adapted to the multimodal token stream (interleaved text-tokens + action-tokens + video-tokens). **Note: Oasis-500m is *not* AR-over-tokens** — it's a DiT diffusion model over a continuous latent grid with action conditioning added to the timestep embedding, so it uses the diffusion prefix-cache, not the AR cache.

```csharp
// HartsyInference.Diffusion/Utilities/ (shared between Lance, Matrix-Game, GameCraft, Cosmos AR)
public sealed class DenoiseKvCache : IDisposable
{
    public DenoiseKvCache(IBackend backend, int numLayers, int hiddenDim, int maxSeq, DType dtype, KvCacheMode mode);

    // Diffusion-style: cache the prefix once per frame, reuse across denoise steps.
    public void CachePrefix(int layer, ReadOnlySpan<TensorRef> k, ReadOnlySpan<TensorRef> v);
    public void GetPrefix(int layer, out TensorRef k, out TensorRef v);
    public void ClearPrefix();

    // Sliding-window video-frame KV cache (Matrix-Game pattern).
    // `cacheId` lets a single instance hold the multiple parallel caches a Matrix-Game
    // block needs (main self-attn, mouse ActionModule, keyboard ActionModule, CLIP-cond).
    public void AppendFrame(int layer, int cacheId, in TensorRef k, in TensorRef v);
    public void EvictOldestFrame(int layer, int cacheId);
    public void SnapshotMemorySlots(int layer, ReadOnlySpan<int> srcFrameIndices, int dstCacheId);

    // AR-token append with sliding-window eviction (Cosmos AR pattern).
    public void AppendTokens(int layer, in TensorRef k, in TensorRef v);
    public void Trim(int keepLastTokens);
}

public enum KvCacheMode { DiffusionPrefix, SlidingWindowVideoFrames, AutoregressiveTokens }
```

**For v1 (Lance video + Matrix-Game 2/3 + GameCraft):** ship `DiffusionPrefix` and `SlidingWindowVideoFrames`. The `AutoregressiveTokens` mode is deferred until a model that needs it (Cosmos AR / future MineWorld) is selected for implementation.

### 5. Discrete video tokenizers (Cosmos DV / VQ-GAN)

AR world models operate on **discrete** video tokens, not continuous latents. The tokenizer is an encoder that maps a frame (or chunk of frames) to a sequence of integer codebook indices, plus a decoder that maps indices back to pixels.

```csharp
// HartsyInference.Video (Phase 9 — first cut lands with Cosmos-Predict V2W)
public interface IDiscreteVideoTokenizer : IDisposable
{
    int CodebookSize { get; }
    int EmbeddingDim { get; }
    (int Spatial, int Temporal) DownsampleFactor { get; }

    Tensor Encode(Tensor rgbFrames);     // [B, 3, T, H, W] -> [B, T/dt, H/ds, W/ds] int32 indices
    Tensor Decode(Tensor indices);       // round-trip
    Tensor EmbeddingsOf(Tensor indices); // for transformer input
}
```

**Implementations (in order of arrival):**
- `CosmosDvTokenizer` — Cosmos-Predict1 V2W's discrete tokenizer (Phase 9, alongside the V2W pipeline). **Cosmos is currently our only real discrete-video user.** Uses **FSQ** (Finite Scalar Quantization, levels `[8,8,8,5,5,5]` → product 64,000), 2-level Haar wavelet front-end, causal 3D conv body, ships as TorchScript `encoder.jit` / `decoder.jit`. Compression `[8, 16, 16]` (T × H × W = 2,048×).
- `VqGanTokenizer` / `MagViTv2Tokenizer` — placeholders for future world models that ship with VQ-GAN / MagViT-v2 tokenizers (MineWorld, Solaris). **Oasis-500m does *not* use a discrete tokenizer** — it ships a continuous Gaussian VAE with patch_size=20, scaling_factor=0.07843137255; see [`OASIS_ARCHITECTURE.md`](OASIS_ARCHITECTURE.md). Defer these implementations until a model that needs them lands.

**FSQ vs VQ-GAN.** FSQ is codebook-free — every token index is a coordinate in a fixed product grid of small integer levels, so encoding is just rounding. VQ-GAN keeps an explicit learned codebook with lookup + commitment loss at train time. From an inference perspective both reduce to "index → embedding" lookup, so the `IDiscreteVideoTokenizer` interface above is sufficient for both — the distinction matters only when authoring new codebooks (we don't, in v1).

These are **codec wrappers, not models** — the encoder/decoder weights are small and load via the existing safetensors path. The runtime cost is dominated by the encoder/decoder forward passes; both should be GPU-routed via `IBackend.Conv2D`/`Conv3D`.

### 6. Memory-augmented attention via concatenated sequence (Matrix-Game 3.0)

Matrix-Game 3.0 introduces a memory bank — stored past-frame latents with camera-aware positional encodings — but **not** as a separate cross-attention stream. Instead, the memory frames are patch-embedded by the same `Conv3d` as the current noisy latents and **concatenated along the temporal axis** before the standard joint self-attention runs:

```
x_curr   = patch_embedding(noisy_latents)                  # [B, dim, T_lat, h, w]
x_past   = patch_embedding(past_latents)                   # [B, dim, 4,     h, w]
x_mem    = patch_embedding(memory_latents)                 # [B, dim, 5,     h, w]
x        = concat([x_mem, x_past, x_curr], dim=temporal)   # T_total = 5 + 4 + T_lat
                                                            # 3D-RoPE: mem slots get *historical* t-indices
self_attn(x)                                                # one joint attention call
```

After the block stack, only the *current* slot's tokens are read out for the velocity prediction (memory + past slots are conditioning, not output). Plan: this is **not** a new DiT block class — it's a sequence-construction step that lives in `MatrixGame3Transformer.Forward()` before the standard block loop. The DiT block itself is the vanilla Wan2.2 block.

**Memory selection:** `MemoryRetrieval.SelectByFovOverlap(history, currentCamera, k=5)` runs each candidate past frame's camera frustum against the current segment's frustum via GPU-vectorized point-in-frustum sampling. Top-5 by visible overlap win. **No learned retrieval — pure geometric selection.** The session owns the rolling buffer; the retrieval helper is stateless.

**3D-RoPE for the concatenated sequence** assigns each slot a `(t, h, w)` index where memory-slot `t`s are the original historical frame indices (non-contiguous gaps OK — the model was trained with σ_θ=0.8 RoPE-θ perturbation specifically to handle this).

### 7. History-mask channel (Hunyuan-GameCraft)

GameCraft injects history as a **33-channel composite latent input**: `[noisy(16) + ref_history(16) + mask(1)]` concatenated along the channel axis. The mask is `1` for slots that contain history and `0` for slots being predicted, but it's just one channel of a much larger composite — not a free-standing tensor. The full input shape to `patchify` becomes `(B, 33, T, H, W)` instead of the base 16-channel HunyuanVideo VAE latent.

Training mixture from the paper: 70 % standard, 5 % history-only, 25 % action-only. The 5 % / 25 % paths reuse the same 33-channel layout with zero-padding in the missing component.

Plan: per-model `BuildLatentInput(noise, history, mask, ...) -> Tensor` helper at the pipeline boundary (the latent input shape is model-specific anyway). No shared abstraction needed beyond what `IActionEncoder` already provides.

### 8. Distilled few-step schedulers (DMD, CM, PCM, UniPC, Lightning)

Interactive models distill the full 30-50 step flow-matching schedule down to 3–8 steps. This is critical for the 25-40 ms per-frame budget. Five families surface across our chosen models:

- **DMD (Distribution Matching Distillation)** — used by Matrix-Game 2/3. 3-4 step inference. Compatible with existing `FlowMatchEulerDiscreteScheduler` infrastructure but with a tighter shift schedule. Matrix-Game 2.0 uses `warp_denoising_step=true` with discrete step lists `[1000, 666, 333]` (3-step) or `[1000, 750, 500, 250]` (4-step).
- **PCM (Phased Consistency Model) + CFG distillation** — used by Hunyuan-GameCraft's distilled variant. 8 inference steps at CFG=1.0 (10-20× speedup over 50-step base).
- **FlowUniPC (UniPC adapted for flow-matching)** — used by Matrix-Game 3.0 (both base 50-step and distilled 3-step). **First UniPC variant in HartsyInference's lineup** — not present today. The simplest correct port is the diffusers `FlowUniPCMultistepScheduler` algorithm with `shift=5.0` applied to the timestep grid SD3-style.
- **DDIM v-pred + sigmoid β-schedule** — used by Oasis-500m. 10 inference steps with **Diffusion Forcing** (context frames held at fixed noise level 14, target at full noise). This is genuinely different from flow-matching and gets its own scheduler class (`DdimVPredScheduler`).
- **Consistency Models (CM) / Lightning** — used by some distilled Wan variants and SD-Lightning. 1-4 step direct denoise. Already partially supported by our `LcmScheduler` for SD/SDXL; needs porting to flow-matching for Wan-family.

Plan: extend `SchedulerFactory` with `DistilledFlowMatchEuler(shift, numSteps, distillationKind)` covering DMD/CM/PCM, add a separate `FlowUniPCMultistepScheduler` for the Matrix-Game 3.0 path, and add `DdimVPredScheduler` + a `DiffusionForcing` helper for Oasis. Reuse existing flow-match plumbing where possible.

### 9. Streaming VAE decode

At 40 FPS @ 720p, the VAE decode has to run **once per frame** on a separate compute queue from the denoiser. Two strategies:

- **Per-frame decode** (Matrix-Game, GameCraft) — VAE is invoked after each frame's denoising completes. Decoder runs on a secondary CUDA stream so it overlaps with the next frame's denoising. Requires the VAE to fit alongside the transformer in VRAM (since we can't evict-and-reload between frames at 25 ms).
- **Chunked decode** (Lance video) — VAE is invoked once per chunk of frames. Acceptable for offline; not for interactive.

For interactive: keep the VAE permanently resident in VRAM (don't `FreeWeights` on the VAE between frames). Plan: a `VideoVaeStreamDecoder` helper that owns a dedicated CUDA stream and a per-frame double-buffered output staging buffer (so the previous frame can be uploaded to the application while the next is decoding).

### 10. Long-context spacetime RoPE (deferred)

AR world models (Cosmos AR 13B, future bigger Oasis variants) attend over hundreds of thousands of (frame, action) tokens. The RoPE has to be cheap to compute and consistent across spatial + temporal dimensions. Lance / Z-Image use multi-axis RoPE which already handles `(t, h, w)`; extending to AR token streams is mostly bookkeeping (track absolute token index → (t, h, w) coords).

Deferred until an AR world model is selected for implementation.

### 11. License-acceptance plumbing — **NOT BUILT (superseded by owner decision, 2026-06-15)**

> ⚠️ **This section is retained as reference only.** The owner decided the engine applies **no license gate**:
> HartsyInference is MIT and ships no weights or model code; the user supplies weights into `/Models` like every
> other model, and weight-license compliance is the user's responsibility. There is no `Licensing/` framework,
> no `LicenseAcceptance`, no `LicenseNotAcceptedException`, and no acceptance endpoint. GameCraft (and any other
> restricted-*weight* model) loads exactly like SD/Flux. The original plan below is kept for context.

Some models we want to support (Hunyuan-GameCraft) have non-permissive licenses. HartsyInference must **not** bundle these weights and must require explicit license acceptance at load time. Plan:

```csharp
// HartsyInference.ModelHandler/Licensing
public abstract record ModelLicense
{
    public abstract string Name { get; }
    public abstract string Url { get; }
    public abstract bool RequiresAcceptance { get; }
    public abstract string AcceptanceText { get; }
}

public sealed record TencentHunyuanCommunityLicense : ModelLicense { ... }
public sealed record NvidiaOpenModelLicense : ModelLicense { ... }
public sealed record ApacheLicense2 : ModelLicense { RequiresAcceptance = false; }
public sealed record MitLicense : ModelLicense { RequiresAcceptance = false; }

public static class LicenseAcceptance
{
    public static void Accept(ModelLicense license, string acknowledgmentToken);
    public static bool HasBeenAccepted(ModelLicense license);
}
```

Checkpoint converters for restricted models throw `LicenseNotAcceptedException` until `LicenseAcceptance.Accept(...)` has been called with the required token. The token is captured in a user-local file so the user only accepts once. Lance, Matrix-Game, Oasis (all permissive) skip this entirely.

## Key Numbers / Constants

| Constant | Value | Source |
|---|---|---|
| Target frame rate | **25-40 FPS** | Matrix-Game 2 = 25, Matrix-Game 3 = 40 |
| Per-frame budget | **25-40 ms** | Inverse of above |
| Default denoise steps per frame (distilled) | **3-8** | DMD: 3-4; CM: 1-4; Lightning: 1-4; GameCraft distilled: 8 |
| Action queue depth (default) | 8 | Avoids stall on micro-jitter without adding noticeable input lag |
| Output frame queue depth (default) | 4 | Drop-oldest policy beyond this |
| KV-cache hidden width | model-specific | See per-model docs |
| Memory bank max frames (Matrix-Game 3) | TBD | Per-model doc |
| History latent ring-buffer length (Matrix-Game) | TBD | Per-model doc |
| `IActionEncoder.EmbeddingDim` | model-specific | Matches transformer hidden dim or is projected |
| `IDiscreteVideoTokenizer` codebook size | model-specific | Cosmos DV: TBD; VQ-GAN: TBD |
| Streaming VAE decoder secondary stream count | **1** | One extra CUDA stream per session for decode overlap |

## Data Layouts / Formats

### Per-frame compute layout (Matrix-Game / GameCraft, schematic)

```
input:
  action_input  : Span<byte>           # raw payload from app
  history       : Tensor               # [B, z_ch (+1 mask), T_hist, H_lat, W_lat]
  prev_kv_cache : DenoiseKvCache       # session-owned

step:
  1. action_tokens = action_encoder.Encode(action_input)               # [B, n_act, hidden]
  2. noise = sample_noise()                                              # [B, z_ch, 1, H_lat, W_lat]
  3. latent_input = build_latent_input(history, noise, mask_channel)     # model-specific
  4. for step in 1..N_distill:
        cond = concat([text_emb, action_tokens, latent_input])
        velocity = transformer.Step(cond, kv_cache=prev_kv_cache)
        latent_input = scheduler.Step(latent_input, velocity, dt)
  5. frame = vae_stream_decoder.DecodeFrame(latent_input)                # [B, 3, H, W]
  6. history.push(latent_input); history.evictOldest()
  7. yield frame
```

### Session lifecycle (Phase 10 HartsyInference.Interactive)

```
[App] ─create─> InteractiveSessionFactory.Create(model_loader, options)
                                            │
                                            ▼
                              IInteractiveSession (background thread starts)
[App] ──SubmitActionAsync──> queue ──> compute thread step loop ──> queue ──> ReadFramesAsync ──> [App]
                                            │
[App] ──DisposeAsync──> graceful drain (finish current step, return)
```

## Algorithm Steps

### `IInteractiveSession.RunLoop()` (compute-thread main)

```
while not cancelled:
  action = action_queue.TryDequeue(timeout: target_frame_time / 2)
  if action is None:
    action = repeat_last_action()                    # interactive: never block
  velocity_or_token = step_model(action, history, kv_cache)
  if model.is_diffusion:
    latent = scheduler.Step(latent, velocity, dt)
    if step_count % N_distill == 0:
      frame = vae_decoder.DecodeFrame(latent)
      output_queue.Enqueue(frame, drop_oldest_if_full=true)
      history.push(latent); history.evictOldest()
  else: # AR
    next_token = velocity_or_token
    history.append(next_token)
    if history.frame_complete():
      frame = ar_tokenizer.Decode(history.last_frame_tokens())
      output_queue.Enqueue(frame, drop_oldest_if_full=true)
```

## Reference Implementations

The per-model research docs are the canonical references. This doc just defines the substrate.

- Matrix-Game 2/3: [`Skywork/Matrix-Game-2.0`](https://huggingface.co/Skywork/Matrix-Game-2.0), [`Skywork/Matrix-Game-3.0`](https://huggingface.co/Skywork/Matrix-Game-3.0)
- Oasis: [`Etched/oasis-500m`](https://huggingface.co/Etched/oasis-500m)
- Hunyuan-GameCraft: [`tencent/Hunyuan-GameCraft-1.0`](https://huggingface.co/tencent/Hunyuan-GameCraft-1.0)
- Cosmos-Predict1 V2W: [`nvidia/Cosmos-Predict1-5B-Video2World`](https://huggingface.co/nvidia/Cosmos-Predict1-5B-Video2World)
- Lance video: [`bytedance-research/Lance`](https://huggingface.co/bytedance-research/Lance) (offline video reference for shared VAE / packed attention)

## Open Questions

1. **`SubmitActionAsync` API shape under back-pressure** — block, throw, or drop oldest? Lean toward "block by default, configurable." Confirm with one application integration before locking the contract.
2. **Per-session vs shared CUDA stream** — every interactive session wants its own stream for overlap. Does the existing `CudaBackend` (one stream per backend) hold up, or do we need per-session stream pools? Probably the latter; defer to first implementation.
3. **License acceptance UX in `HartsyInference.Server`** — the headless server can't pop an interactive dialog. Plan: license-acceptance must happen as an explicit `POST /v1/licenses/accept` before the first model load that needs it. The Server package owns this endpoint; ModelHandler enforces.
4. **Whether `IInteractiveSession` belongs on `HartsyInference.Video` rather than a new `HartsyInference.Interactive`** — pro: avoids a new package. Con: `IInteractiveSession`'s threading model spills into every video pipeline that includes the type. Going with separate package; revisit if it adds friction.
5. **Action replay / record** — for testing and reproducibility, sessions should be able to record their action stream and replay it deterministically. Plan: stub `IActionLogger` interface in Phase 10, real implementation when a debugger or QA process actually needs it.

## Implementation Notes

### Phase 9 deliverables (Video) — shared infra

- `HartsyInference.Conditioning/ActionInput.cs`, `IActionEncoder.cs` — lives in Diffusion for cross-domain reuse.
- `HartsyInference.Diffusion/Utilities/DenoiseKvCache.cs` — first user is Lance video; reused everywhere.
- `HartsyInference.Diffusion/Schedulers/DistilledFlowMatchEuler.cs` — DMD / CM / Lightning support added to flow-match scheduler family.
- `HartsyInference.Video/Tokenizers/IDiscreteVideoTokenizer.cs` + first impl `CosmosDvTokenizer.cs` (with Cosmos-Predict V2W).
- `HartsyInference.Video/Streaming/VideoVaeStreamDecoder.cs` — secondary-stream per-frame VAE decode helper.
- `IBackend.PackedAttention` and `IBackend.Conv3D` — implemented across CPU / CUDA / Vulkan (Phase 9 § 3 of [PHASE_9_VIDEO.md](../Checklists/PHASE_9_VIDEO.md)).
- `HartsyInference.ModelHandler/Licensing/ModelLicense.cs`, `LicenseAcceptance.cs` — restricted-license plumbing for Hunyuan-GameCraft and similar models.

### Phase 10 deliverables (Interactive) — model pipelines

- `HartsyInference.Interactive/Sessions/IInteractiveSession.cs` and the default `BackgroundComputeSession.cs`.
- `HartsyInference.Interactive/Pipelines/MatrixGame2Pipeline.cs`, `MatrixGame3Pipeline.cs`, `OasisPipeline.cs`, `HunyuanGameCraftPipeline.cs`.
- `HartsyInference.Interactive/Models/Denoisers/DiTBlocks/MemoryAugmentedBlock.cs` — Matrix-Game 3 memory bank cross-attention.
- `HartsyInference.Interactive/ActionEncoders/KeyboardMouseEncoder.cs`, `CameraPoseEncoder.cs`, `MinecraftActionEncoder.cs`, `GamepadEncoder.cs` — per-model action encoder implementations.
- `HartsyInference.Interactive/Tokenizers/VqGanTokenizer.cs` — placeholder for future VQ-family world models (MineWorld, Solaris). Not needed for Oasis (continuous VAE).

### Deferred-foundation backlog (documented, not built v1)

| Item | Trigger to build |
|---|---|
| AR KV-cache (`DenoiseKvCache.Append/Trim`) | First AR world model lands (Oasis if it uses AR; Cosmos AR 13B) |
| Long-context spacetime RoPE | Same trigger |
| `IActionLogger` record/replay | First test that needs deterministic action playback |
| Per-session CUDA stream pool | First time a single backend instance is shared by 2+ interactive sessions |
| Server-side license-acceptance endpoint | First restricted-license model is wired into Server |
| Multi-user session manager (queueing) | Multi-tenant interactive serving scenario |
| Streaming network protocol (WebRTC / WebTransport) | First user wants browser-side interactive playback |

### Performance budget (RTX 3060 12 GB target, Matrix-Game 2.0 540p @ 25 FPS)

- Action encode: < 0.5 ms
- Transformer step (×4 distilled): ~25 ms total (~6 ms/step)
- VAE decode (overlap with next frame): ~10 ms in parallel
- Frame copy out to app: < 1 ms
- **Total per-frame critical path: ~30 ms** — matches the 25 FPS budget with headroom. Target hardware: RTX 3060 12 GB or better.

For Matrix-Game 3.0 720p @ 40 FPS the budget tightens to 25 ms total; needs at minimum RTX 4090 24 GB. Documented constraint, not blocking.

### Reuse opportunities

- **Wan2.2 3D causal VAE** (built for Lance video in Phase 9) — reused verbatim by Matrix-Game 3.0 (finetuned from Wan2.2-TI2V-5B) and Matrix-Game 2.0 (SkyReels-V2/Wan lineage).
- **`IActionEncoder` abstraction** — designed once, used by every world model and by any future controllable music / animation model.
- **`DenoiseKvCache` (prefix-cache variant)** — built for Lance video, reused by every interactive diffusion model.
- **`DistilledFlowMatchEuler`** — built for distilled video models, reused by world models and by any future fast-image distillation (Flux-Lightning, SDXL-Lightning ports).
- **`VideoVaeStreamDecoder`** — built for offline streaming (Lance video frames out as they decode), trivially extended to interactive.
- **`Cosmos DV tokenizer`** (Phase 9) — Cosmos-Predict1 V2W lands the discrete tokenizer abstraction; Oasis's VQ-GAN slots in via the same interface.
