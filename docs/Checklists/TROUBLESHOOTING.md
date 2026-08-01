# Model Bring-up Troubleshooting — consolidated reference

> The single place to look when a **new** model port is wrong, crashes, or is slow. Every entry is a
> real symptom → cause → fix distilled from bring-up work across the diffusion, LLM, audio, video,
> world, 3D, and vision stacks. Organised by theme so you can jump to the class of bug you're seeing.
> Ephemeral run-logs (dates, wall-clock, seeds) were dropped on purpose — only the reusable lesson is
> kept. Deep original journals live in git history (the old `PHASE_*_DEVIATIONS`, `*_GRIND`,
> `*_HANDOFF` files) if you ever need the blow-by-blow.

**How to use:** find the symptom (black image, word-salad, NaN, "GPU idle", garbled-but-plausible
output), read the matching section, then confirm with the [parity-debugging methodology](#parity-debugging-methodology)
at the bottom. When math is proven but output is wrong, suspect **encoder tap/order → tokenizer →
real-weight converter**, *not* the DiT/backbone.

---

## Parity-debugging methodology

**The layer-by-layer bisect (the workhorse):**
1. Build a Python reference (venv) that dumps: initial noise, text embeddings, per-step latents
   (binary + mean/std/min/max JSON), final latent. Run C# with Python's **saved** noise/embeddings —
   compare with **shared noise tensors, never by matching seeds** (C# Box-Muller ≠ PyTorch RNG).
2. Isolate a single forward pass; if it diverges the bug is in the model, not the scheduler.
3. Hook every layer in Python, step C# one layer at a time, find the **first divergent layer**.
4. Sub-layer decompose that layer (GroupNorm / proj_in / LayerNorm / QKV / reshape / logits / softmax
   / out-proj / FFN).
5. Re-run to confirm all layers match (`avg_err < 1e-3`, ideally `< 1e-4`).

**Expected FP32 tolerances** (exceed by 10×+ = real bug, not FP noise):
element-wise `<1e-7`; GroupNorm/LayerNorm `<1e-6`; Linear/Conv GEMM `<1e-5`; full attention block
`<1e-4`; full UNet/DiT pass `<1e-3`.

**Vision bisect variant:** write a Python forward using the **same folded weights** (isolates C# impl
vs weights/converter); dump the preprocessed input to disk so C# reads identical bytes; run C#
layer-by-layer; **check actual checkpoint tensor shapes against assumed shapes** before debugging
numerics (channel-width assumptions are where it breaks).

**Higher-order lessons:**
- A passing **synthetic-weight** parity harness proves the backbone MATH, not the encoder or the
  real-weight load. Math-proven but image wrong ⇒ suspect encoder concat/tap order → tokenizer/chat
  template → real-weight converter, in that order.
- **Backend A/B (CUDA vs Vulkan)** instantly eliminates "model logic is wrong" as a hypothesis class
  (they share the model, not the kernels).
- **All-step tracing for any "works once, fails on iteration N"** — accumulation/threshold bugs are
  clean at step 0; don't trust the previous step's clean trace.
- Read what the trace SAYS: `INF` (not NaN) ⇒ an operand went Inf going in ⇒ points at an F32→F16
  cast. A **step-dependent** NaN is a threshold crossing (F16 65504, F32 3.4e38), not a logic bug —
  chart the suspect tensor's magnitude across steps. One bad element per token in an attention output
  localizes to one `(head,dim)` column of V (`softmax·V` smears one Inf V element across every query
  row).
- For asymmetric/dual-network CFG, **`cfgΔ` proves nothing** (nonzero from the weight difference alone
  even if text is ignored). Build a real conditioning probe: same network, real vs zeroed text.
- Diffing directly against **ComfyUI's `comfy/text_encoders/*.py` / `comfy.sd.VAE`** is one of the
  fastest ways to localize a porting bug.
- **Never trust "tests pass" = "output correct" — visually inspect.** Numerically-plausible output
  (right ranges, no NaN) can be completely wrong. Keep known-good reference images.
- **Conditioning paths are stricter than denoising paths** — img2img re-projects latents onto the
  manifold and hides encoder errors that Kontext/Fill/IP-Adapter amplify. Pick test images with large
  flat regions plus fine texture (content-dependent triggers make a systematic bug look flaky).

---

## Tokenizer & text-encoder gotchas

- **CLIP/OpenAI BPE needs the regex pre-tokenizer + `</w>` suffix + EOS-padding.** The 2-arg
  `Microsoft.ML.Tokenizers.BpeTokenizer.Create(vocab,merges)` runs generic byte-level BPE and produces
  character-level garbage. Use the long-form overload with `RegexPreTokenizer` (CLIP `pat`),
  `LowercaseNormalizer`, `endOfWordSuffix="</w>"`, and **pad with EOT (49407), not 0** (CLIP pad==EOS).
  Smoke test: encode `"a photograph of an astronaut riding a horse"`, assert first 10 IDs match HF.
  SDXL tolerated corrupted tokens; SD3 did not.
- **Apply the chat template for LLM text encoders.** Z-Image/Qwen3: raw `Encode(prompt)` vs
  `apply_chat_template(..., add_generation_prompt=True, enable_thinking=True)` conditions the model on a
  distribution it never saw. The `<think>` block varies by model (Qwen3-VL-Instruct emits none;
  Flux.2-Klein's Qwen3-text template does) — it's causally appended so it can't corrupt prompt-token
  hiddens; minor vs other bugs.
- **Penultimate vs final hidden state matters a lot.** Z-Image wants `hidden_states[-2]` (raw, no final
  RMSNorm): post-norm `abs_mean≈1.25` vs penultimate `abs_mean≈6.4`. Feeding post-norm sends every
  downstream layer off-spec.
- **Multi-layer tap concat ORDER (tap-major vs hidden-major) is load-bearing and survives every math
  diff.** Ideogram 4 (13-layer Qwen3-VL): upstream/ComfyUI build **hidden-major** (`c = hidden*K + tap`);
  a **tap-major** default (`c = tap*H + hidden`) transposes the projection's input columns → "coherent
  image, ignored prompt." For single-tap (K=1) the layouts are identical, which hides it.
- **Filter padding tokens out of caption embeddings.** Diffusers slices `embeds[mask]`; passing full
  padded `[1,64,2560]` feeds garbage pad-position hiddens (LLM hallucinations at causally-blind pads).
  Also unmasked T5 padding (~120 PAD tokens) dilutes the caption — truncate to real tokens.
- **A `ProjectionDim=0` preset silently disables pooled output.** SD3 CLIP-L needs `ProjectionDim=768`;
  reusing SDXL's preset (falls back to SD1.5's `ProjectionDim=0`) returns null pooled → NRE in
  `ConcatPooled`. (SD3.5 CLIP-L `text_projection` is 99.9999% zero by design — still must project.)
- **Moonshine-streaming HF reference silently skips the sliding-window mask when no `attention_mask` is
  passed** — parity looks broken (cosine ~0.86) despite a correct port. Fix: pass an explicit all-ones
  mask.
- **`Qwen2Tokenizer.Decode` leaks the GPT-2 byte-level marker `Ġ`** — strip it.

---

## Attention / RoPE / multi-head parity

- **diffusers `attention_head_dim` is a TRAP — it means head COUNT, not head dim** when
  `num_attention_heads` is unset. SD1.5 `attention_head_dim=8` = 8 heads (head_dim=40), not 40 heads of
  dim 8. Wrong split → 2.24× too-large attention scale and ~56% signal dampening. Print `model.attn.heads`
  and check actual Q/K/V reshape dims before writing config.
- **Self-attention must norm K/V from the SAME normed tensor as Q.** Bug: Q from `norm(hidden)`, K/V from
  raw `context`; for self-attn (`context==hidden`) both must use normed. Fix: `ReferenceEquals(hidden,context)`.
- **GQA needs SEPARATE Q and K head counts.** A rotate/kernel taking one `numHeads` for both writes OOB
  on K and under-rotates Q (OmniGen2: 21 vs 7 heads).
- **RoPE that a block applies internally can't be "applied externally."** RoPE acts on Q/K after
  projection; if the block computes Q/K, the block must rotate them. A "None/external" mode silently
  means *no* positional encoding.
- **Same reused block class → don't split into separate C# classes that silently drop plumbing.**
  Z-Image reused one block for context_refiner/noise_refiner/layers; splitting the `modulation=False`
  path dropped RoPE from context_refiner → 7-orders-of-magnitude layer error, glitch banding.
  Image-only attention with a shared frame index is rotation-invariant (offset cancels in softmax) so
  noise_refiner masked it; caption tokens with per-token frames did not.
- **PSA/specialized attention may not be standard SDPA.** YOLO11 C2PSA: Q/K have
  `key_dim = head_dim*attn_ratio`, V has full `head_dim`; spatial axis is second-to-last; a depthwise
  3×3 positional encoding is added to V. Check QKV output channels against `3*total_head_channels`.
- **Recurring RoPE-family bug patterns** (each surfaced as fluent word-salad or instant EOS, not a
  crash): interleaved vs split-half RoPE; wrong attention scale (`1/√head_dim` vs `1.0` vs `1/√8`);
  `up·silu(gate)` MLP half-order; DenseGeneral/`[in,h,k]→[out,in]` transposes; missing
  `cache.AdvanceLength()`. Concrete cases: Zonos (interleaved RoPE + MLP half-order + `1/√head_dim` +
  channels-first vs channels-last prefix → instant EOS); Dia (split-half RoPE + scale 1.0 + prefix +
  missing cache advance); Qwen3.5 Gated DeltaNet (missing `q *= 1/√head_dim` before the recurrence).
- **Un-normalized V + an F16 attention fast path = threshold NaN.** Microsoft Lens RMS-norms Q/K but not
  V; its residual stream is architecturally huge (`max|V|` 18940→71583 across forwards), crossing F16's
  65504 → solid black. SageAttention's INT8 path casts V to F16 and is **default-on regardless of
  `allowF16`** (only the cuDNN branch honors the flag). Fix: scale joint V by 1/256 before SDPA, scale
  output back by 256 (attention is linear in V; power-of-2 = exponent-only).
- **YaRN RoPE needs HF's dimension-index ramp + `attention_factor` mscale** (Lens: 1.3466×), not a
  wavelength ramp.
- **Phi head_dim=96** (non-power-of-two) hit an infinite CUDA fallback recursion; pad the flash-attn
  block to next power of two, and extract the CPU reference to `AttentionReference` so the GPU fallback
  can't re-dispatch into itself. **Phi-4-mini fused-QKV split** must use real `head_dim=128`, not
  `rope.dimension_count=96` (they coincided on Phi-3.5, hiding the bug).

---

## Weight layout / transpose / converter parity

- **PyTorch `nn.Linear.weight` is `[out, in]`; forward is `x @ W.T`.** Raw-pointer GEMM must index
  `w[o*in + i]`, not `w[i*out + o]`. A transposed square matrix (CLIP `text_projection` 768²/1280²)
  silently "looks fine" (right norm, no NaN) but is 100% relative error (SD3.5: 270× error drop after
  fix). Verify against a **non-square 2×3/3×2** reference, or route through `backend.Linear` (cuBLAS
  `OP_T`) to make the bug structurally impossible.
- **GGUF `nn.Linear` weights need a shape *relabel* to `[out,in]`, NOT a data transpose** (bytes already
  row-major). Only raw params like `mm.input_projection` need an actual transpose. Recurs across every
  VLM.
- **BFL vs diffusers `.chunk()` order differs — verify BOTH frameworks.** BFL Flux final-layer AdaLN
  chunks `[shift, scale]`; diffusers chunks `[scale, shift]`. Loading a BFL checkpoint without swapping
  the AdaLN weight rows along dim 0 applies shift-as-scale → magenta/purple cast, per-channel velocity
  growing monotonically. Use the official `convert_*` `swap_scale_shift()`. Recurs on
  Flux.2/AuraFlow/HunyuanImage/QwenImage/Chroma/Lens.
- **Patchify in-patch dimension order is load-bearing — read the einops string.** diffusers uses
  channel-**innermost** `'B C (H p)(W q) -> B (H W)(p q C)'` → loop `for py,for px,for c`. Z-Image uses
  `[pH,pW,C]` channel-last via `permute(1,3,5,2,4,6,0)`. Wrong order silently misaligns
  `patch_embed.weight` columns. Same class: Oasis DiT unpatchify vector order `[py,px,c]`.
- **Adapter per-layer weight lists follow diffusers `attn_processors` registration order:
  down → up → MID LAST**, NOT forward traversal (down→mid→up). IP-Adapter mid-onward got wrong-shaped
  K_ip/V_ip → Linear under-filled/overran → NaN → black. Match per-index row-dims against UNet block
  widths; fail-fast on mismatch at the injection site.
- **llama.cpp SigLIP mmproj naming is swapped:** `ffn_down` is actually fc1 (up-projection),
  `ffn_up` is fc2 (down); bias sizes give it away. Wire by role, not name. (Same swap for mllama clip
  MLP.)
- **Lens MXFP4 codec was missing upstream `transpose(1,2)` on expert weights** — silently garbles every
  MoE layer.
- **`Fp8ScaleFactor` (per-tensor scale) must propagate through EVERY tensor-copying converter site** —
  QKV/single-linear splits, transposes, half-swaps, permutations. Missing propagation surfaces only at
  that layer's GEMM. Audit: grep `new Tensor(` inside converters after adding per-tensor quant metadata.
  Fold the scale into cuBLAS `alpha` (zero cost) rather than dequanting a 12B transformer (would OOM).
- **Drive conversion off patterns, not named layer indices.** A hardcoded `"model.22."` head check
  missed YOLO11's layer-23 head. Two-pass: Pass 1 fold BN siblings on `*.conv.weight`; Pass 2 copy any
  remaining `*.weight`/`*.bias` as-is → architecture-agnostic.

---

## Mixed-dtype checkpoints & dtype-sensitive kernels (recurring)

- **`(float*)weight.DataPointer` is a latent dtype bug.** Any CPU-fallback op reading a weight as
  `float*` bit-reinterprets BF16/F16 storage (two BF16 → one garbage F32). Z-Image-Base RMSNorm did
  this → clean 16-px grid of tinted blobs. Turbo hid it (F32 norms); Base stores norms BF16. Fix: cast
  non-F32 norm scales to F32 at load. Audit every `DataPointer` cast against the **actual** checkpoint
  dtypes, not the previous variant's. Same class: SD3 `pos_embed` F16, Z-Image.
- **Safetensors dtypes are NOT uniform per file.** Juggernaut XL: conv mostly F32, linear mostly F16,
  norms mixed. For F16 inference, explicitly `CastWeightsToF16` on ALL weights before loading.
- **PTX kernels are load-width-sensitive** (`ld.global.b16` vs `.b32`). When dispatching by
  `input.DType`, verify ALL operands (weights, biases) match; cast before launch (temp buffer +
  `LaunchCastF32ToF16`, free in finally). F32 norm weights read by an F16 kernel → pure noise.
- **cuBLAS `GemmEx` requires A and B the SAME dtype.** Supported: F16×F16→F16/F32, F32×F32→F32. NOT
  A=F32,B=F16 (`CUBLAS_STATUS_NOT_SUPPORTED`, err 15). Cast the mismatched operand first.
- **When narrowing an F32 activation for a mixed-precision GEMM, use BF16, not F16.** F16's ±65504
  ceiling overflows on gated activations (SwiGLU/GeGLU multiply two large activations): Z-Image SwiGLU
  produced +Inf → RmsNorm sees Inf → `Inf×(1/√Inf)=NaN` → all-black. **Dynamic range matters, not just
  declared dtype.** BF16 has F32's range, same byte count, Tensor-Core-fast on Ampere+. Bug only
  appeared step 1+ (step 0 stayed under 65504 by luck).
- **cuBLASLt bias epilogue reads bias in the OUTPUT dtype** — F16-out GEMM + F32 bias buffer = silent
  NaN, no error.
- **pos_embed / norm weights read as F32 via `(float*)ptr` when the checkpoint stores F16** → auto-cast
  to F32 at load.

---

## Quant / FP8 / W8A8 numerical findings

- **Recognize ComfyUI `fp8_scaled` format:** raw FP8 E4M3 (±448, mean abs ~14) + `.scale_weight` (F32)
  + `.scale_input` (F32) + a `scaled_fp8` marker. Plain Comfy Dev FP8 is pre-scaled (mean abs ~0.014,
  no companions). Z-Image single-file uses `.weight_scale` + `.comfy_quant` (U8). **Always inspect the
  safetensors header before writing the loader** — single-file checkpoints fuse QKV, drop patch-size
  suffixes, rename scale companions. 10 sec to inspect vs hours of KeyNotFound.
- **fp8_scaled companions must fold into `Tensor.Fp8ScaleFactor` (cuBLAS alpha)**; the `.scaled_fp8`
  zero-byte marker is skipped; `MmapHandle.PointerAt` bound check loosened `>=`→`>` for past-the-end
  zero-byte tensors.
- **W8A8 real-checkpoint activations are 3–5× uglier than synthetic** (relL2 3–5.5e-3 synthetic →
  1.35–2.52e-2 real). **Local per-layer relL2 improvement does NOT predict e2e SSIM** — SmoothQuant
  dropped aggregate local relL2 29% yet **regressed** SSIM 0.9210→0.9144. Error terms add in
  **quadrature**, so with activation error dominating ~2:1, driving weight error to zero caps local
  reduction at ~10%. SmoothQuant fails the 0.95 SSIM gate because the weight-side quant floor is
  binding, not activation outliers.
- **cuBLASLt int8 TN routes to IMMA with plain column-major layouts on cuBLAS 13.1** — the
  COL32/COL4_4R2_8C interleave plumbing is a Turing-era caveat, obsolete here. Raw int8 GEMM 3.2–3.7×
  over F16 at DiT shapes (~78–94% of INT8 TOPS peak).
- **fp8 (E4M3) common-scale requant is exact for power-of-two scale ratios**, within ½ ulp for normal
  values; only weights below `s*·2⁻⁶` degrade — negligible at real ratios (≤~8×). Enables QKV/w1w3
  fusion but measured **neutral at ≥4k-token image shapes** (GPU is GEMM-saturated; launch savings are
  noise).
- **Q6_K GEMV was latency-bound, not bandwidth-bound** — interleaved load-then-use serialized on each
  load. Fix: issue all loads up front into named locals before any unpack/FMA (bit-identical, +6.4%).
  Q4_K/Q5_K already use upfront `uint2`/`float4` vectorized loads.
- **R4 row-interleave repack rejected on evidence:** decode GEMV is compute/memory co-limited (68.65%
  DRAM / 74.6% ALU), not bandwidth-bound; input-vector-reuse shared-mem staging measured an **11%
  regression** (`__syncthreads()` cost > redundant-read savings).
- **GGUF K-quant sign/rescale bakes (recurring):** Mamba `ssm_a` — GGUF already stores `A = −exp(A_log)`
  (llama.cpp bakes it), use directly, do NOT re-apply `−exp` (caught by ssm_a cosine −0.18→1.0).
  RWKV-6 — GGUF pre-divides deep-layer output weights by `2^(layer//rescale_every_n_layers)` → apply
  runtime `x *= 0.5` every 6 layers.

---

## Gated activations (GEGLU / SwiGLU / GLU)

- **Split along the LAST dimension, never the flat midpoint.** For `[..., 2D]`: first D per row = value,
  next D = gate. In a flat-indexed PTX kernel decompose `outerIdx = i/D, d = i%D`, then
  `inputX = outerIdx*2D + d`, `inputGate = inputX + D`. Flat split (`input[i]`, `input[i+totalOutput]`)
  is WRONG for any multi-row tensor → garbled vertical banding with **correct tensor statistics**
  (invisible to 1-row unit tests). Test with `[2,2,2D]`. PTX is especially dangerous (no shape metadata).
  Same bug appeared in the Vulkan GeGlu (`inputX=outerIdx*2*D+d` fix; `[2,2,2*D]` regression guard).

---

## Norm-layer architecture quirks

- **Gemma 2 has 4 norms per block** (pre-attn, post-attn sandwich, pre-FFN, post-FFN sandwich),
  **GeluTanh** (not SiLU), and **offset-from-1 RMSNorm scale** (stored = scale−1, apply `(1+weight)` —
  fold `+1` at load). Name-collision warning: Gemma's `post_attention_layernorm` is a real post-attn
  sandwich norm, not the pre-MLP norm.
- **Final-layer norm epsilon/type is exact:** Z-Image `norm_final` is
  `LayerNorm(elementwise_affine=False, eps=1e-6)` — literally 1e-6, not RMSNorm, not the config's 1e-5.
- **Final-layer modulation can be scale-only:** Z-Image outputs one chunk → `norm(x)*(1+scale)`, no
  shift/gate (unlike Lumina2's NextDiT 2×hidden scale+shift).
- **`AdaInstanceNorm1d` single-pass variance** `E[x²]−E[x]²` NaN'd via catastrophic cancellation on
  ~30k-sample HiFiGAN stages → use two-pass double-precision variance.
- **CLIP encoder missing final LayerNorm** → embeddings std ~5 instead of ~1 → 5× amplified
  conditioning → abstract patterns. Apply `final_layer_norm` per `CLIPTextTransformer.forward()`.

---

## VAE / latent scaling & encoder parity

- **VAE stride-2 downsample: asymmetric vs symmetric padding is a DIFFERENT operator, same output
  size.** diffusers/BFL/Comfy use `F.pad((0,1,0,1))` (right/bottom) + `Conv2d(padding=0)`; symmetric
  `Conv2d(padding=1)` samples the opposite pixel parity. Three stages compound → ~1-latent-pixel phase
  error → Flux Kontext maze/speckle on flat regions. **A same-engine encode→decode round trip CANNOT
  reveal this** (decoder is misalignment-agnostic); diff the latent against ComfyUI's `comfy.sd.VAE`. A
  correlation that *improves* under a (+1,+1) shift is the smoking gun. Conditioning paths
  (Kontext/Fill) are stricter than denoising.
- **VAE latent denormalization:** the decoder must apply `raw = latent·std + mean` from checkpoint
  `per_channel_statistics`; skipping → lattice/blotches (hit LTX-Video 0.9 AND LTX-2 22B).
- **VAE decode OOM at the pipeline stage boundary:** im2col temp buffers (300MB–4.8GB at 512ch/1024²)
  can't fit while the transformer/UNet is still resident. Fix: `backend.Sync()` then
  `backend.FreeWeights(model.EnumerateWeights())` after denoise, before VAE decode. (Recurring — applied
  in SD3, AuraFlow, Chroma, Lens.)
- **VAE attention: 3D→4D reshape before SDPA.** Passing `[B,seq,C]` to a 4D-expecting kernel makes
  `Shape[3]=0` (uninitialized) → head dim D=0 → zero-iteration loops → all zeros. Reshape to `[B,1,seq,C]`.

---

## Scheduler / flow-match / sampling

- **Flow-match Euler: negate the transformer output** (`noise_pred = -noise_pred`) before the step, else
  integration ascends in the noise direction → pure RGB static.
- **Timestep inversion:** flow-match `t = 1 - sigma`; forgetting conditions the model on the opposite
  schedule point every step.
- **Gates need `tanh()`** (`gate_msa = mod[1].tanh()`) or they blow up at init and modulated residuals
  dominate.
- **Euler scheduler needs `scale_model_input`:** divide latent by `sqrt(sigma²+1)` before each UNet call.
- **Euler step div-by-zero at final timestep (sigma→0):** for epsilon prediction the algebra simplifies
  to `derivative = model_output`; guard sigma for v-prediction.
- **Timestep embedding:** SD1.5 uses `flip_sin_to_cos=True` → `[cos, sin]` layout (not `[sin,cos]`), and
  frequency divisor `/(half_dim - 1)` not `/half_dim`. Hunyuan3D `max_period = time_factor = 1000`, not
  10000. F5-TTS uses a **×1000 timestep sinusoid scale**.
- **Position IDs for DiT caption vs image tokens:** caption gets per-token frame indices `1..capPaddedLen`
  (h=w=0); image tokens share one frame `capPaddedLen+1` with varying h/w; pad slots stay (0,0,0). Use
  padded length, not real length (synthetic tests with real==padded mask the diff).
- **Sequence concat order** (`[image, caption]` vs `[caption, image]`) affects both the attention pattern
  and post-block slicing — verify against the reference line.
- **Step-cache gate signature is per-model** — check indicator schedule-informativeness before fitting a
  poly gate: Qwen informative (poly 1.20×), Ideogram uninformative (needs late-window gate, 1.39×),
  Flux.2 V-shaped (poly 2.49×), Wan/video negative under any reuse (use FVD not SSIM). Poly + late-window
  gates are substitutes, not additive.
- **CFG-interval paper result does NOT transfer** — skipping guidance on early high-noise steps (t>0.85)
  flips prompt category on text-prompt models (SSIM 0.35); only late-only bands (`0.15,1`) are
  quality-safe but marginal.

---

## GPU memory / caching / lifetime (CUDA)

- **After enabling GPU weight caching, ALL weight/bias access must route through the cache-aware
  `CopyToDevice`.** Direct `weight.DataPointer` for computation crashes (pointer=0) after CPU disposal.
  Audit all model code for direct `DataPointer` on weights.
- **Don't evict the GPU weight cache between pipeline stages.** Vestigial `EvictBackendCache("CLIP"/"UNet")`
  destroyed preloaded weights → `ObjectDisposedException` on the next stage.
- **In-place GPU ops must clear `_gpuSyncCallback`/`_gpuDisposeCallback` before `CacheActivation`.** The
  old callbacks close over the previous GPU pointer; re-caching fires them and `FreeAsync`'s the pointer
  you just modified → intermittent corruption depending on allocator reuse.
- **`cuMemFreeAsync` is stream-ordered — memory is NOT immediately reclaimed.** At stage boundaries
  explicitly `Sync()`+`FreeWeights()`; add an OOM-retry that syncs the stream (flushes pending frees) then
  retries `cuMemAlloc`.
- **Never mix synchronous `cuMemAlloc` with the deferred-free (`cuMemFreeAsync`) world** — a sync alloc
  can receive a VA whose earlier async free hasn't executed; the late free destroys the new buffer and the
  eventual free double-frees. Route all caches through the stream-ordered pool.
- **A raw `DataPointer`/`AsSpan` does NOT root a Tensor.** An undisposed tensor whose pointer is being read
  can be finalized mid-read; `~Tensor→AlignedFree` frees the live buffer and glibc `free()` writes tcache
  metadata (value `addr>>12`, high dword ≈ `0x7`) into bytes 0–7. Manifests as a "parallel-only native
  stomp" where class-set bisection never converges (co-running classes are only GC pressure, not writers).
  Prove it: forced `GC.Collect()+WaitForPendingFinalizers()` between write and read reproduces
  deterministically single-threaded. Fix: `using` declarations + `GC.KeepAlive` around raw-pointer copy
  loops. Same class: `ConvWeightSlices` cast-weight freed mid-loop → `GC.KeepAlive(wf)` after the write
  loops.
- **Pass-through helpers that MIGHT allocate are disposal traps.** `PadCaption`/`PadImage` returning the
  input unchanged when already aligned → caller disposes an aliased tensor → `ObjectDisposedException`.
  Guard `if (!ReferenceEquals(result,input)) input.Dispose();` or always allocate.
- **DeepSeek/large-MoE memory:** expert split must be a zero-copy view over mmap (copying ~7GB → host
  OOM-kill); keep the untied embedding table host-only (not uploaded to GPU) to save ~0.8GB.
- **`TextService.EnsureRamHeadroomFor` refuses to load a GGUF unless free RAM ≥ 2.5× file size** (dequant
  headroom) — large models fail here before touching GPU; this is the loader protecting from OOM-kill,
  not a bug.

---

## GPU residency / throughput (CUDA)

- **A model is GPU-resident only if EVERY hot-path op is a backend op.** One `.DataPointer`/`AsSpan` read
  on a GPU tensor fires `cuStreamSynchronize`+D2H copy (lazy-sync). Ideogram 4's CPU "glue" (modulation
  split, gating, QK-norm, RoPE, slice/reshape, CFG/Euler) caused ~370 syncs/forward → 57 min/gen. Fix: a
  PTX kernel family for all glue; assert `GetD2hSyncCount()==0` per forward. `nvidia-smi` shows GPU idle.
- **GPU idle during a gen (low `nvidia-smi` util) = host/dispatch-bound — diagnose before tuning
  kernels.** `Tensor`'s ctor eagerly did `NativeMemory.AlignedAlloc + Clear` (zeroing page-faults the
  whole buffer) for every tensor, including device-only intermediates (a `[1,4121,12288]` F32 ≈ 202MB ×
  many × steps of host malloc+memset+free the host never reads). Fix: lazy host-buffer allocation
  (`EnsureHostBuffer()` on first CPU access); `CacheActivation` must not read `DataPointer` up front.
  Result: 45s→2s/step. Decisive probe: `nvidia-smi dmon -s u`.
- **Host `DataPointer` reads mid-forward drain the compute stream every block/step**, serializing GPU
  work. Recurring root cause across GEGLU loops, LayerScale, SwiGLU gates, per-step CFG/Euler. Fix: port
  the glue op to device so activations stay resident. Health assert: ~0 mid-decode D2H syncs; any per-token
  sync is a residency bug (MoE routing readback is a current offender at 2193–3225 syncs/rep).
- **`CudaBackend.Concat` `dim>0` pathology:** issued one `cuMemcpyDtoDAsync` per outer element → ~280k
  tiny memcpy nodes/forward captured into a CUDA graph (hid ~300ms/fwd; Hunyuan3D DiT-loop 27.7→7.5s,
  bit-identical). Fixed with a single-launch `dit_concat2_f32/_f16` kernel. Shared win for any
  non-leading-dim concat (video/audio/LLM).
- **`ConvTranspose2d` silently ran on CPU** (`CudaBackend` never overrode it) — 1549ms for a 32²→64²
  upsample → 3ms with a gather-form kernel. Shared by ClipSeg/YOLO/Demucs/RVC/ResembleEnhance. Grouped
  `ConvTranspose1d` (`groups=768`, BigVGAN) was likewise rejected — add `groups` to the kernel.
- **F16/CUDA-graph only help when host-launch-bound:** `HARTSY_GEMM_F16=1` moving DiT time 0% proves it's
  per-op-launch-bound, not GEMM-bound; graph capture is then the real lever.
- **The lm_head dominates decode for large-vocab models.** Orpheus: the tied lm_head (3072→156,940 vocab)
  ran as an F32 GEMM at M=1 = 90% of the step. Fused BF16/F16 M=1 GEMV → 221ms→3.8ms/tok. Same class:
  `Qwen2Model` re-transposing weights per call via `WhisperOps.ProjectLinear` defeated the weight cache
  (~123× decode slowdown when non-resident).
- **Method rule (hard-won, stated twice):** instrument EVERY device op before concluding a bottleneck is
  "many small kernels / occupancy-bound" — an uninstrumented op (Concat, host FourierEmbed) hid the real
  wall repeatedly. Also **measure the premise/headroom of an optimization before building it** — the
  Concat fix silently invalidated the "batched CFG saves 3s" premise.
- **TripoSR triplane density decode:** the whole 25s wall was a host per-point loop over a 16.7M-point
  256³ grid; a GPU `triplane_grid_sample_f32` kernel → 354ms (beats Python's 469ms).

---

## CUDA / PTX kernel pitfalls & toolchain

- **PTX ISA version trap (recurring):** pip `nvidia-cuda-nvcc` floats `nvidia-nvvm` to 13.3 which emits
  PTX ISA 9.3; driver 580.x JIT caps at 9.0 → `CUDA error 222: Unsupported .version 9.3`. Fix: pin
  `nvidia-cuda-nvcc==13.0.88` **and** `nvidia-nvvm==13.0.*` (nvvm is the ISA-determining piece, not nvcc)
  into an isolated `pip --target` dir; **verify every emitted PTX starts `.version 9.0` before shipping.**
- **Integer overflow in im2col at 1024²+.** `channels×kH×kW×outH×outW` exceeds uint32 (512ch/3×3/1024² →
  `CUDA_ERROR_ILLEGAL_ADDRESS`; 256²/512² fit in 32-bit and pass). Cast to `long` before all GPU
  buffer-size multiplications in C#; use `cvt.u64.u32`+`mul.lo.u64`+`setp.ge.u64` in PTX.
- **CausalConv3d batched fast-path OOB write** (Wan/HunyuanVideo/LTX/Kandinsky5 3D-causal VAE): with
  `_spatialReplicatePad`, `padded` was allocated at unpadded spatial size (`hp=h`) but `BuildPaddedFrames`
  padded internally → wrote ~7.5 KB past the buffer. Manifested as `CUDA_ERROR_ILLEGAL_ADDRESS` on the
  24GB card, OOM on the 12GB card (same bug, different symptom). Fix: compute post-pad `hp=h+2·_padH` for
  the `spatialPre` case too.
- **2D right-operand in BatchedMatMul silently produced zeros.** `N = b.Shape[2]` returns 0 for a `[K,N]`
  2D tensor (uninitialized `_dim2`) → all-zero matmul → CLIP/VAE-attn became residual passthroughs
  ("brownish blob"). Detect rank-2, use `Shape[1]`. Same uninitialized-`Shape[3]` class as the VAE 3D→4D
  bug.
- **`MatMulKernels.LinearTransB` heap corruption (0xC0000374):** a wrong-shaped weight wrote past the
  output buffer with no stack trace. Now fails fast on `output.ElementCount != M·N`.
- **`UnidirectionalLstm`:** multi-layer with `InputDim < HiddenDim` reshaped the per-step buffer too small
  (only worked when input dim == hidden). Allocate per-layer at `dimIn`.
- **Exception-masking on CUDA crash:** a `using CudaBackend` disposes during stack-unwind on a broken
  context; Dispose throws its own error, .NET reports the *later* exception and discards the real one.
  Construct explicitly + try/catch/log-real-exception + Dispose in a finally that separately catches.

---

## Profiling pitfalls

- **`CUDA_VISIBLE_DEVICES=0` alone does NOT pin a specific GPU** — CUDA defaults to `FASTEST_FIRST`,
  silently selecting the fastest card. Always set `CUDA_DEVICE_ORDER=PCI_BUS_ID` too; verify by name via
  `nvidia-smi -L`. (Caused a phantom "we're faster than llama.cpp" reading that evaporated on re-run.)
- **Isolated-kernel `ncu` "Est. Speedup: X%" overstates** the achievable win on a co-limited kernel —
  it's against that one metric's ceiling, not the actual co-limiting bottleneck. Price against the
  SM/DRAM SoL balance, then confirm end-to-end tok/s.
- **`nsys --trace=cuda` undercounts kernels inside CUDA-graph replay** (260 vs ~33,000). Verify graph-only
  counts against an eager-inclusive trace.
- **`nsys` host/API overhead must be cross-checked against a clean unprofiled wall-clock** — instrumentation
  tax scales with graph-node count; an allocation-arena "win" was a pure measurement artifact.
- **`ncu` on GeForce:** `ERR_NVGPUCTRPERM` is a driver policy (admin-only perf counters). Bypass per-invocation
  with `sudo env HOME=… LD_LIBRARY_PATH=… CUDA_VISIBLE_DEVICES=n dotnet test …` (sudo resets env; engine
  resolves native libs relative to `$HOME`); `chown` report files back afterward.
- **`compute-sanitizer` pip wheel ships an incomplete layout** — needs manual symlinks for
  `libTreeLauncher*.so`/`TreeLauncher*` from the Nsight Compute bundle, else it silently no-ops.
- **Profile decode *sections* before scoping** (Orpheus mis-diagnosed 3× before a 1-gen env-gated
  Stopwatch nailed it). Always verify TTS with whisper `medium.en`, never `base.en` (actively misleading).

---

## Vulkan (SPIR-V / GLSL) pitfalls

- **Kernel-level dtype selection must be a function of the OUTPUT tensor, not the inputs.** FP8→F16 GEMM
  writing into a caller's F32 buffer (2 bytes written, 4 read) → every other element garbage → NaN within
  1–2 ops → all-black Flux. Cast FP8 inputs up to F16 for the multiply but land the result in an
  output-dtype buffer.
- **Tensor-core/coopmat paths must store to whatever dtype the pipeline allocated.** An F16-only output
  guard made the coopmat path inert on Flux (Linears allocate F32 output) — 1.4% "improvement" = never
  launched. Relax to F16-or-F32 with an `OUTPUT_F32` spec-const. Gate coopmat on actual shape enumeration
  (`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`), not extension presence.
- **"Every-N-dispatches" auto-flush must respect op atomicity — drain only at op boundaries.** Mid-op
  flush freed shared Q/K/V upload buffers still in use by later SDPA heads → garbage from head 2 onward.
  Fix: an `OpScope`/`EnterOp()` RAII nesting counter; wrap public ops.
- **Deferred-free timeline tag must be `_value + 1`** (the NEXT tick the recording op will signal), not
  `_lastSubmitted`. Tagging with the already-submitted tick lets the next reclaim destroy a still-pending
  buffer → latent UAF.
- **Cross-subgroup reductions drop partials when `gl_NumSubgroups > subgroupSize`.** With `local_size=256`
  on small-subgroup devices (LLVMpipe/older Intel, subgroup 4–8 → 32–64 subgroups), the single-subgroup
  final combine reads only `subgroupSize` partials → wrong GroupNorm/LayerNorm/RmsNorm/Softmax. NVIDIA (32)
  and AMD (32/64) unaffected. Fix: strided fold `for k=SubgroupInvocationID; k<NumSubgroups; k+=SubgroupSize`;
  size `warp_*` shared arrays `[64]`. Alternative: pin `requiredSubgroupSize`≥16.
- **NVIDIA Linux blob under-reports promoted core features.** 535-series advertises `apiVersion 1.3.x` but
  returns zeros for `shaderFloat16`/`timelineSemaphore`/`bufferDeviceAddress`/`synchronization2`/
  `subgroupSizeControl`/`shaderIntegerDotProduct` through the **consolidated** v1.2/v1.3 feature structs
  (original extension structs return the truth). Fix: trust `apiVersion` for guaranteed-promoted features,
  gated by vendorID (NVIDIA 0x10DE / AMD 0x1002 / Intel 0x8086). Re-check on driver upgrades.
- **Slab granularity matters more than allocation count near VRAM limit.** 256MB slabs OOM'd mid-load on a
  12GB 3060 (Flux ~12GB); 64MB/8MB slabs fit into post-deferred-free gaps. Add an `OnOutOfMemory`
  drain+retry.
- **Cache-miss upload buffers must be tracked and drained** or each step leaks one `VkBuffer` per miss
  (Mesa validation catches it, the NVIDIA blob doesn't).
- **GLSL has no built-in `erf`** — use Abramowitz & Stegun 7.1.26 (max err 1.5e-7) for exact GELU.
- **matmul C buffer can't be `writeonly` when `beta≠0`** (residual-add fusion reads C).
- **Older `glslangValidator` (Ubuntu 22.04) rejects `-O`** — ship unoptimized SPIR-V, let the driver
  optimize (~100ms colder first-pipeline build).
- **`VkAccessFlags2` sync2 constants are easy to get wrong** and NVIDIA over-flushes so tests pass anyway:
  `UniformRead=0x08`, `ShaderStorageRead=0x200000000`, `ShaderStorageWrite=0x400000000`,
  `ShaderSampledRead=0x100000000`. A wrong compute-to-compute barrier silently drops the write→read hazard.
  Pin with CPU-only constant regression tests. Same for coopmat sType values
  (`COOPERATIVE_MATRIX_PROPERTIES_KHR=1000506000`, `..._FEATURES_KHR=1000506001` — reverse of intuitive
  order).
- **`nonCoherentAtomSize` flush/invalidate must align both ends:** `start=AlignDown(off,atom);
  end=AlignUp(off+size,atom)` — computing size independently can miss the buffer tail (silent stale data on
  non-coherent/ReBAR drivers).
- **`Span<T>(ptr,(int)(Size/sizeof(T)))` truncates 64-bit element counts** above int.MaxValue (Vulkan
  analogue of the CUDA im2col overflow) — throw instead of returning a partial view.
- **Push descriptors regressed 7% on NVIDIA** (pool-ring is highly tuned there) — not every "best
  practice" generalizes; kept default-off, opt-in `HARTSYINFERENCE_VK_PUSH_DESCRIPTORS=1` for AMD/Intel.
- **INT8 `dotPacked4x8` overload is selected by operand type:** `uint[]` → unsigned, `int[]` → signed.
  Same bytes, different interpretation.
- **Env-var opt-in flags read into `static readonly` fields aren't unit-testable within one test process**
  (xUnit runs many tests in the same process, so the field is already frozen from whatever the env var was
  — or wasn't — set to at first touch of the type). `TryDispatchInt8Linear`'s gate started this way; fixed
  by moving to a settable *instance* property (`EnableInt8Linear`, defaulted from the env var in the
  constructor) matching `CudaBackend.EnableW8A8`'s existing pattern — tests flip it directly
  (`backend.EnableInt8Linear = true`) with no env var or fresh process needed. Prefer this pattern for any
  new opt-in switch that correctness tests need to exercise both on and off.
- **GeGlu flat-midpoint bug (same as the CUDA one):** decompose `outerIdx=i/D; d=i%D` then
  `inputX=outerIdx*2*D+d`; a multi-row `[2,2,2*D]` test is the regression guard.
- **A host-side write into a small "control" buffer (decode-graph token-id/position/history/counter) can
  race ahead of an already-recorded-but-unsubmitted dispatch that still needs the OLD value.**
  `VulkanBackend.Dispatch()` batches multiple dispatches into one command buffer, submitted only at
  `FlushThreshold` or on an explicit sync — a mapped-memory host write bypasses that queue entirely and
  takes effect immediately from the CPU's perspective, regardless of whether the GPU has actually consumed
  the buffer's previous contents yet. Found via `ApplyRepetitionPenaltyStep_MatchesHfConventionWithRepeats`
  failing ONLY on llvmpipe (passed on the 3060): 5 token ids written in a loop, each immediately followed
  by a dispatch reading the CURRENT id — every dispatch ended up reading the LAST id written, since llvmpipe
  deferred all 5 dispatches' actual execution until later, by which point every write had already
  happened. This is a genuine race, not an llvmpipe-only quirk — it happened to not manifest on the 3060
  under this test's specific batch size/timing, which is not something to rely on. Fixed by having
  `VulkanBackend.WriteScalarBuffer` call `Sync()` (submit pending work + host-wait) before writing —
  correct but synchronous; a future optimization would record the write as a `vkCmdUpdateBuffer` into the
  same batched command buffer instead, matching CUDA's async-copy-on-an-in-order-stream approach with no
  host wait needed. General lesson: any op that writes host-visible memory a PRIOR, not-yet-flushed
  dispatch might still read must sync first, or its "before" write can retroactively become "after."
- **`KvCacheAppendDev`/`FlashAttentionDev`'s device-position variants must actually READ the device
  buffer, not just accept and ignore it while forwarding the caller's other arguments.** Real callers
  (`GenericTransformer.ForwardGraphDecodeStep`) pass literal placeholder values for the plain
  `offset`/`kvLen`/`qOffset` parameters (e.g. `KvCacheAppendDev(kCache, k, 0, devicePos)` — the `0` is not
  a real offset, it's a "this doesn't matter, devicePos is authoritative" marker) — a passthrough that
  forwards those placeholders to the non-`Dev` method (`KvCacheAppend(buffer, newKv, offset)`) would
  silently append/attend at the WRONG position every step (e.g. always slot 0) the instant
  `GraphDecodeSupported` is on, corrupting the KV cache with no exception or visible symptom short of wrong
  model output. Caught in code review (an advisor pass), not by a test — the bug was invisible with
  `GraphDecodeSupported` left at its default OFF. Fixed by giving both a real device-buffer-reading kernel
  variant (`kv_cache_append_dev.comp.glsl`, `sdpa_flash.comp.glsl`'s `HAS_DEVICE_POS` compile flag) that
  reads the actual slot/length from `Pos_[1]`/`Pos_[0]` (the same `{kvLen, qOffset}` buffer
  `rope_decode_step` already reads). Regression tests deliberately pass a wrong/stale placeholder alongside
  the real devicePos value specifically to prove the placeholder is ignored, not merely that a plausible
  value happens to be correct by coincidence.
- **Ubuntu's `glslang-tools` package (15.1.0-2~ubuntu0.24.04.2, Noble) cannot compile
  `matmul_int8.comp.glsl`**: its GLSL frontend has no `GL_EXT_integer_dot_product` extension entry at all
  (confirmed via `strings` on `/usr/bin/glslang` — it knows the SPIR-V-level `SPV_KHR_integer_dot_product`
  opcodes but never registers the GLSL `#extension` pragma), so `#extension GL_EXT_integer_dot_product :
  require` fails with "extension not supported" regardless of `--target-env`. The LunarG Vulkan SDK's
  bundled glslang (1.4.357.0 / glslang 16.4.0) compiles it correctly. Fix: don't rely on the distro
  package for this shader; extract `glslangValidator`/`glslc`/`spirv-val` from the LunarG SDK tarball
  (`https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.xz`, no install/sudo needed — the
  `x86_64/bin/*` + `x86_64/lib/*` subtree runs standalone) and point `GLSLANG`/`SPIRVVAL` env vars at it
  when running `src/HartsyInference.Vulkan/Shaders/build.sh`. All 36 other kernels compiled identically (byte-for-byte) with
  either toolchain, so this is an isolated extension-table gap, not a broader compatibility problem.
- **A full `src/HartsyInference.Vulkan/Spirv/*.spv` rebuild (2026-07-28, LunarG glslang 16.4.0) reproduced all 37
  previously-committed files byte-for-byte identical**, and additionally produced `maxpool2d_{f32,f16}.spv`
  and `depthwise_conv2d_{f32,f16}.spv`, which `VulkanBackend.cs` (`MaxPool2D`/`Conv2dDepthwise`, dispatching
  `"maxpool2d"+DtypeSuffix`/`"depthwise_conv2d"+DtypeSuffix`) had been dispatching against with **no
  compiled artifact on disk** — calling either threw a shader-load failure. Root cause: no SPIR-V compiler
  was available on the dev box, so `build.sh` had never actually been run there; the C# dispatch code and
  GLSL source already agreed, only the build step was missing. Fixed by rebuilding; regression-pinned in
  `VulkanBackendSmokeTests.Backend_MaxPool2D_Matches_Cpu` / `Backend_Conv2dDepthwise_Matches_Cpu`. The
  `Tanh`/`Elu` "dispatches to a nonexistent op-code, produces silent zeros" comment on
  `VulkanBackend.Tanh`/`.Elu` (ops 8/9) turned out to be **stale** — the committed `elementwise_{f32,f16}.spv`
  already contains both ops (confirmed via `spirv-dis` showing the `Tanh` ExtInst, and via
  `Backend_Tanh_Matches_Cpu`/`Backend_Elu_Matches_Cpu` passing on real hardware) — the comment describes a
  state before someone rebuilt and committed the artifact but never updated the code comment.
- **Multi-GPU dev boxes: `VulkanBackend`'s default `deviceOrdinal: 0` is NOT nvidia-smi's index 0.**
  Vulkan's own enumeration order put the RTX 4090 at ordinal 0 and the RTX 3060 at ordinal 1 on this box
  (confirmed via `vulkaninfo --summary`), independent of `nvidia-smi`'s ordering. If another process (e.g.
  a live SwarmUI/ComfyUI session) has the ordinal-0 GPU's VRAM mostly claimed, Vulkan tests silently target
  the busy card and can spuriously `ErrorOutOfDeviceMemory` on workloads that would fit easily on an idle
  card — this is not a shader or backend bug. Mesa's `VkLayer_MESA_device_select` implicit layer (already
  present via `mesa-vulkan-drivers`) accepts `MESA_VK_DEVICE_SELECT=<vendorID>:<deviceID>` (list candidates
  with `MESA_VK_DEVICE_SELECT=list vulkaninfo --summary`) to pin ordinal 0 to a specific physical device —
  works even for pure-NVIDIA multi-GPU boxes, not just mixed-vendor ones. Useful for CI/benchmark scripts
  that need a deterministic, known-idle target.
- **A `.comp.glsl` file existing is not sufficient — it must also be listed in `build.sh`'s kernel
  arrays.** `conv1d.comp.glsl`, `conv_transpose1d.comp.glsl`, and `snake.comp.glsl` had real GLSL source
  and matching `VulkanBackend.cs` dispatch code (`GetKernel("conv1d"+DtypeSuffix, ...)` etc. — see the
  three throw sites) but were never wired because `build.sh`'s `DTYPE_KERNELS` array simply never listed
  them; `build.sh` silently produces exactly what its arrays name and nothing else, so a shader can sit
  unbuilt indefinitely with no error. Always cross-check a new/renamed shader against `build.sh`'s arrays,
  not just against its own file existing.
- **`snake.comp.glsl` never actually compiled — `flat` is a reserved GLSL keyword** (the interpolation
  qualifier), not usable as a variable name; `uint flat = gl_GlobalInvocationID.x;` is a syntax error on
  any real compiler. Because no SPIR-V compiler was available on the dev box, this had never been caught.
  Renamed to `idx`.
- **`snake.comp.glsl`'s `USE_BETA` binding gap:** `Beta_` (binding 2) was declared only inside `#if
  USE_BETA == 1`, but `Out_` was hardcoded to binding 3 unconditionally — so the vanilla (`USE_BETA=0`)
  module has bindings `{0,1,3}` with a gap at 2. `VulkanDescriptorManager.CreateSetLayout` always builds a
  *sequential* `0..N-1` layout sized by buffer count (it has no SPIR-V reflection), so a 3-buffer vanilla
  dispatch would bind a real buffer to slot 2 while the shader's `Out_` still expects slot 3 — silent
  wrong output, not a validation error. General rule for any `#if`-gated binding: every variant's *used*
  bindings must independently be contiguous from 0, since the C# side has no idea which bindings a given
  preprocessor branch kept. Fixed by making `Out_`'s binding itself branch on `USE_BETA` (2 in the vanilla
  arm, 3 in the beta arm) instead of only gating `Beta_`. `USE_BETA` is also a `#define`, not a
  `layout(constant_id=...)` spec constant like `conv1d`/`conv_transpose1d`'s `HAS_BIAS` — it needs its own
  compiled module (`snake_beta_{f32,f16}.spv`, added to `build.sh` as explicit `compile_one` calls since
  it doesn't fit the array-driven dtype-only loop), not a spec-constant toggle on the vanilla module.
- **`ConvTranspose1d`'s shipped shader has no `groups` field at all** — its inner loop sums over the full
  `cIn` range assuming dense `[cIn, cOut, K]` weights, unlike `conv1d.comp.glsl` which does support
  `groups`. Dispatching a depthwise (`groups>1`) call through it would silently read the wrong weight
  slice. `VulkanBackend.ConvTranspose1d` now throws `NotSupportedException` for `groups != 1` instead of
  producing wrong output; extending the shader for grouped transposed conv (needed for BigVGAN-style
  anti-aliased upsampling) is open work.
- **llvmpipe gaps surfaced by a full-suite sweep (2026-07-28), not yet root-caused — tracked, not fixed
  here:** `Backend_Linear_FP8Weight_CachedCast_Matches_Cpu` fails on llvmpipe with large numeric errors
  (maxAbsErr ~7128) rather than gracefully throwing/skipping — `cast_f8e4m3_f16.spv` is unchanged by the
  2026-07-28 rebuild (byte-identical), so this is pre-existing llvmpipe FP8-cast behavior, not a
  regression. Separately, `PipelineCache_PersistsToDisk_AcrossBackends` fails on llvmpipe — the reloaded
  on-disk pipeline cache is 32 bytes (header-only) instead of >1KB, suggesting llvmpipe's
  `vkGetPipelineCacheData` doesn't retain pipeline blobs the same way the NVIDIA driver does. Both pass on
  the 3060 and 4090. Neither blocks NVIDIA-target work; flag for a dedicated llvmpipe investigation before
  treating llvmpipe as a full correctness oracle for FP8/pipeline-cache-dependent paths.
- **`VulkanMemoryAllocator` used to destroy a "dedicated" (>= 16 MB, `DedicatedThreshold`) block the
  instant its one allocation was freed** (`Free()` called `Dispose()` + `_blocks.RemoveAt(i)` whenever
  `IsDedicated && LiveAllocations == 0`), instead of pooling it like slab blocks. Any op producing a
  transient output >= 16 MB (`Silu`/`Gelu`/`RmsNorm`/`LayerNorm` all allocate a fresh `outBuf` per call)
  paid a real `vkAllocateMemory`/`vkFreeMemory` round trip on **every single dispatch** — measured
  (2026-07-28, RTX 4090) at a sharply non-linear ~30x cliff: 956 μs at 1.31M elements (5.24 MB, under
  threshold) vs 29,652 μs at 5.24M elements (20.97 MB, over threshold) for the SAME `Silu` op, a 31x time
  jump for only a 4x element-count increase. `BroadcastAdd` (writes in place, no fresh allocation) didn't
  show this at the same sizes — the tell that pointed at the allocator rather than the compute shaders.
  Fixed by removing the special-casing: dedicated blocks are now scanned for reuse exactly like slabs
  (any request that fits a block's free-list is served from it, not just an exact-size match — see
  `VulkanMemoryAllocator.Allocate`'s unified scan loop) and are only actually destroyed under memory
  pressure via the existing `ReleaseEmptySlabs()` OOM-retry path, matching how slabs already behaved.
  Result: the 5.24M-element case dropped from 29,652 μs to 4,420 μs (~7x). Regression-guarded by
  `VulkanLeakTests.Vulkan_100Iter_LargeTransient_PoolsInsteadOfReallocating`, which asserts BOTH that VRAM
  usage plateaus (no leak from keeping blocks alive) and that the live block count stays flat after a
  short warm-up (proving reuse, not just absence-of-leak) — note the warm-up needs ~20 iterations to reach
  steady state for a dual-large-buffer op (upload + output), not 1, because the transient-upload list only
  drains every `FlushThreshold` dispatches.
- **`SliceLastDim`, `ApplyRope`, `KvCacheAppend` had NO `VulkanBackend` override at all** — every call
  fell through to `IBackend`'s CPU-loop default, which reads `.DataPointer` directly. On a GPU-resident
  input that forces a full device sync + host readback (`GetD2hSyncCount`), and the next GPU op needing
  the result pays a fresh H2D re-upload on top (`GetTransferCacheStats`'s miss count) — invisible unless
  you're specifically instrumenting for it, which is exactly why Phase 2 built that instrumentation
  first. A synthetic LLM decode-step measurement
  (`VulkanLinearProfileMeasurement.Measure_LlmDecodeStep_ResidencyVsDispatchOverhead`) showed
  `SliceLastDim` alone accounted for 2 of 5 D2H syncs/step (the QKV split, and `GluActivate`'s own
  internal `SliceLastDim`-based gate/up split — fixing the shared primitive fixes both call sites via
  ordinary interface dispatch, no `GluActivate` override needed). Fixed by adding real GPU dispatches
  (new shaders `slice_last_dim.comp.glsl`, `apply_rope.comp.glsl`, `kv_cache_append.comp.glsl`) —
  2026-07-29, dropped the decode step from 2.581 ms/5.0 syncs to 1.461 ms/1.0 sync per step (see
  `benchmarks/scoreboards/VULKAN.md`). `ApplyRope`/`KvCacheAppend` mutate their buffer in place, so the
  dispatch re-caches the SAME tensor via `CacheOutput` after writing — without that, a later reader would
  either see stale pre-write data (if it happened to already be host-resident) or force a needless
  re-upload (if `GetBuffer`'s cache-miss path re-uploaded the tensor's now-stale host copy). `ApplyRope`'s
  CPU reference assumes q and k have the SAME `numHeads` (it loops `q.Shape[2]` for both) — GQA models
  with different Q/K head counts use `ApplyRopeSingle` per-tensor instead; the Vulkan shader matches that
  same MHA-only contract, it doesn't add GQA support IBackend's own default doesn't have.
- **`CopyTo` was unconditionally a host-side `Buffer.MemoryCopy`**, forcing a D2H sync on `source`
  whenever it happened to be GPU-resident (the common case — `CopyTo` is mostly used as shape-view glue
  between two GPU ops), then an H2D re-upload of `destination` the next time a GPU op read it. Fixed by
  adding `VulkanGpuTransferHelper.TryGetCached` (peeks the weight/activation cache **without** uploading
  on a miss, unlike `CopyToDevice`) so `VulkanBackend.CopyTo` can take a `vkCmdCopyBuffer`
  device-to-device fast path when `source` is already cached, falling back to the original host memcpy
  only when it genuinely isn't (a truly host-only source is cheaper to memcpy directly than to promote to
  the GPU). Closed the decode step's last D2H sync: 1.461 ms/1.0 sync → 1.382 ms/**0.0 syncs** per step.
- **No fused/online-softmax attention existed on Vulkan at all** — `ScaledDotProductAttention`
  materialized the full `[Sq,Skv]` score matrix (3 dispatches: matmul, softmax, matmul), which is the
  documented root cause of the Wan-video full-resolution OOM (a single self-attention call there needs a
  ~19-25 GB score matrix), and `FlashAttention`/`FlashAttentionDev` fell through to the CPU reference
  entirely. Fixed (2026-07-29) with `sdpa_flash.comp.glsl`: one workgroup per (batch, head, query row),
  each of 32 threads (`TILE`) owns one key within the current 32-key KV tile (so scoring a tile needs no
  cross-thread reduction, only a tile-level online-softmax combine done by thread 0 over 32 values) —
  standard tiled online-softmax flash attention with Br=1. Shared-memory budget: `TILE*MAX_D*4 bytes * 2
  arrays (K,V) = 32*128*4*2 = 32 KiB`, safely under the common 48 KiB limit; `MAX_D=128` bounds real head
  dims, the rare >128 case (some SDXL variants use 160) falls back to the old materialized path rather
  than corrupting shared memory. `VulkanBackend.ScaledDotProductAttention`/`FlashAttention` both route
  through one shared `DispatchFlashAttention` helper. Supports causal masking, GQA (query head `h` reads
  kv head `h/(hq/hkv)`), a sliding window, and (a separate `HAS_MASK=1`-compiled module, same convention
  as `snake`'s `USE_BETA`) an optional additive `[Sq,Skv]` mask broadcast across batch/head. Does NOT
  support softcap/sink/ALiBi (Gemma-2/GPT-OSS/MPT-class) — those fall through to `AttentionReference`'s
  CPU reference, a documented scope boundary (none of these appear at Wan-video scale, so it doesn't
  block the OOM fix). Sets `FlashDecodeSupported => true`, unblocking Zonos onto the GPU-resident path.
  Measured (RTX 4090): the exact previously-unrunnable shape (B=1,H=24,S=16384,D=128) now completes in
  9.8 s using only the Q/K/V/O tensors' own memory — see `benchmarks/scoreboards/VULKAN.md` for the full
  before/after and the honest ceiling (this is a correctness-first design, ~30-430× slower than CUDA's
  cuDNN-fused kernel depending on shape; Phase 5 tunes it).
- **Real bug the new tests caught before shipping**: the first `sdpa_flash` version indexed the K/V
  buffer's per-head stride using `skv` (the number of VALID kv positions to attend over), which is only
  correct when the buffer is EXACTLY that size. Any real KV-cache buffer — always over-allocated to a
  max sequence length, with only a prefix currently valid — would have silently read the wrong memory for
  every position past the first head's block. Every test using an untruncated K/V tensor (`skv ==
  key.Shape[2]`) passed regardless, which is exactly why this needs an explicit test shaped like a real
  KV cache (`Backend_FlashAttention_GqaAndKvLenLessThanBuffer_MatchesCpu`: `kvLen=9` into a
  `maxSeq=32`-capacity buffer) rather than trusting same-size tests to generalize. Fixed by adding a
  separate `kvCapacity` push-constant field (the buffer's real `Shape[2]`) distinct from `skv` (the loop
  bound), and using `kvCapacity` for the per-head stride.
- **The Wan-video-scale smoke test (`Backend_FlashAttention_WanVideoScale_CompletesWithoutOom`) takes
  hours on llvmpipe** (measured: killed after 2.85 CPU-hours, still running) — software-rasterized
  scalar execution of 16384×24 = 393,216 workgroups × ~512 serial KV-tile iterations each has no
  business running on a CPU emulator. The test now skips itself when `Vk.DeviceName` contains
  "llvmpipe"; it proves no-OOM/finite-output on real hardware, which is not something llvmpipe's
  correctness-only role needs to cover — don't remove the guard to "get more coverage."
- **`VulkanGpuTransferHelper.ByteSize` computed `ElementCount * DType.SizeInBytes` — silently ZERO for
  every GGUF-quantized dtype** (Q4_0/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K all have `SizeInBytes = 0`; their real size
  is packed as `BlockByteSize`/`BlockElementCount`, exposed via `DType.ComputeByteCount`). This is the
  single helper every upload/download/cache-size computation in the class routes through, so **any
  quantized tensor Vulkan was ever asked to upload has always either thrown
  (`ArgumentOutOfRangeException: Allocation size must be > 0`) or, on an unlucky code path, allocated a
  zero-byte buffer instead of erroring** — undetected because nothing had exercised a quantized tensor
  end-to-end on Vulkan before the Phase 5 GGUF dequant shader tests (`QuantizedMatMul`, the other
  quantized-tensor consumer, is CUDA-only per `IBackend`'s own contract, so it never touched this path).
  Fixed: `ByteSize` now delegates to `DType.ComputeByteCount(ElementCount)`, the same API CUDA's
  equivalent already uses correctly. The ~20 `output.ElementCount * output.DType.SizeInBytes` call sites
  in `VulkanBackend.cs` were NOT touched — `output` there is always a freshly-allocated GEMM/kernel
  result (F16/F32/BF16 by construction; no op in this codebase ever produces a quantized output), so
  those aren't live instances of the same bug, just the same textual pattern used safely.
- **GGUF k-quant/legacy-quant dequant shaders added** (2026-07-29): `dequant_{q4_0,q5_0,q8_0,q4_k,q5_k,
  q6_k}.comp.glsl`, mirroring `native/cuda/dequant/*.cu`'s bit-packing exactly (block/super-block layouts,
  `get_scale_min_k4`, Q6_K's 4-outputs-per-thread pattern) — closes the explicit open `ROADMAP.md` §3
  item. Read as raw `uint[]` words (no 8-bit storage extension needed) via a `readByte(byteOffset)`
  helper that extracts one byte at an arbitrary — including unaligned — offset, and `unpackHalf2x16` to
  reinterpret two bytes as an FP16 scale without needing `float16_t` STORAGE support on the input side
  (core GLSL, no extension). Wired into `VulkanBackend.CastIfNeeded` — the SAME generic weight-dtype-cast
  path already used for FP8→F16 — so dequant happens lazily on first use exactly like
  `CudaBackend.CastOnGpu`'s identical quantized-source handling; there is no separate "model loading"
  step on either backend. Validated against `HartsyInference.ModelAssets.Gguf.GgufDequantizer` (the
  engine's own production CPU codec, not a hand-rolled test reimplementation) — all 6 formats matched on
  the first real run once the `ByteSize` bug above was fixed.
- **llvmpipe: a workgroup with most threads early-returning (`if (gid >= total) return;`, common when a
  dispatch's element count is smaller than one workgroup) produces wrong output for the threads that DON'T
  return** — observed as several dequant-shader output elements reading back as exactly `0`/`-0` instead
  of the correct value, specifically on the FIRST test pass which used a small (96-256 element) buffer
  that left the sole workgroup mostly idle. Bumping the test to an element count that's an exact multiple
  of the workgroup size (256) made it pass everywhere — which also happens to be the realistic case: real
  GGUF weight tensors are always whole numbers of quant blocks, so a real dequant dispatch only has a
  partially-occupied workgroup on its very last one (if at all), not throughout. Not root-caused (no
  `barrier()`/subgroup ops in these shaders, so this shouldn't matter architecturally) — tracked
  alongside the FP8-cast and pipeline-cache llvmpipe gaps above as a known, undiagnosed llvmpipe quirk,
  not something blocking NVIDIA-target work.
- **Five real Vulkan backend gaps surfaced by an actual Krea2 e2e run, invisible to prior static call-site
  analysis** (2026-07-30, RTX 4090 + 3060 + llvmpipe) — each was found by re-running the same real-weight
  Krea2 Turbo/NoCfg/1024×1024 test after fixing the previous one, exactly the "static analysis can't rule
  out runtime issues" caveat the Vulkan bring-up plan itself carried:
  - `WanRopeInterleaved` had NO `VulkanBackend` override — `FluxRope.ApplyGpuGqa` (Krea2's GQA rope) fell
    through to `IBackend`'s CPU-loop default, which read/wrote the GPU-resident Q/K tensors' raw
    `.DataPointer` directly and crashed the TEST HOST PROCESS with `AccessViolationException` (not a
    catchable managed exception — the worst failure mode seen this session). Fixed with a new
    `wan_rope_interleaved.comp.glsl` (single-tensor GPT-J-style interleaved-pair rotation, F32/F16) +
    override; regression-tested via `Backend_WanRopeInterleaved_MatchesCpu` (3060 + llvmpipe).
  - `RepeatKvHeads` had no override — the F32-only CPU default threw `NotSupportedException` on Krea2's
    F16 GQA K/V (`Krea2Attention.Forward`). Fixed with `repeat_kv_heads.comp.glsl` (pure gather/broadcast,
    no arithmetic) + override; `Backend_RepeatKvHeads_MatchesCpu`.
  - `GatedResidualLastDim` had no override — same F32-only throw on Krea2's F16 activations
    (`Krea2Block.Forward`'s `h + gate·attn`/`h + gate·ff` residual gates). Fixed with
    `gated_residual_last_dim.comp.glsl` (same broadcast convention as `affine_broadcast_last_dim`: F16/F32
    activation, always-F32 gate) + override; `Backend_GatedResidualLastDim_MatchesCpu`.
  - `SliceRows` had no override — same F32-only throw on Krea2's F16 joint img+txt sequence
    (`Krea2Transformer.SliceTail`, and `Krea2Block.SplitModulation`'s `[6,hidden]`-table row extraction).
    Fixed with `slice_rows.comp.glsl` (contiguous offset copy) + override; `Backend_SliceRows_MatchesCpu`.
  - `Conv2D`'s im2col path materialized the FULL `[gemmK, outH·outW]` column matrix in ONE allocation —
    correct for the small convs every other model on Vulkan has exercised, but at Krea2's 1024×1024
    VAE-decode resolution (`QwenImageVaeDecoder` → `QwenImageResample.Forward`, ~192 input channels, 3×3
    kernel) that's `1728 × 1,048,576 × 4 bytes ≈ 7.25 GB` for a SINGLE `vkAllocateMemory` call — OOM'd even
    with the transformer's ~13 GB of fp8 weights already freed beforehand (`Backend.FreeWeights` runs
    before `_vaeDecoder.Decode` in `Krea2Pipeline`). This is the architectural gap the Vulkan bring-up plan
    named up front (CUDA offloads conv to cuDNN's implicit-GEMM, which never materializes the full im2col
    matrix; Vulkan hand-writes im2col+GEMM). Fixed by tiling `Conv2D` over output COLUMNS instead of
    materializing the whole thing: `im2col.comp.glsl` gained `colOffset`/`tileCols` push constants (tile
    width defaults to reproducing the untiled single-tile behavior exactly when the image is small, so
    every other Conv2D call site on Vulkan is byte-for-byte unaffected), and the matmul side reuses the
    GEMM kernel's ALREADY-EXISTING `bOffset`/`cOffset` batched-dispatch offsets (no shader change needed
    there) with `ldc` kept at the REAL full output width so each tile writes into the correct column range
    of the one real output buffer. `VulkanBackend.Conv2DMaxColTileBytes` (default 256 MiB) is a settable
    instance property purely so `Backend_Conv2D_TiledPath_MatchesUntiled` can force the multi-tile path
    deterministically at small, fast-to-verify scale (every output element checked against the CPU
    reference, not just probe points) instead of needing a multi-GB shape in the test suite.
  All five are genuinely closed and regression-tested (3060 + llvmpipe, full 112-test Vulkan suite green
  after all fixes); they are NOT the reason the Krea2 e2e output is still invalid — see the next entry.
- **Krea2 Vulkan e2e: F16 activation blowup — ROOT CAUSE FOUND AND FIXED** (2026-07-30, RTX 4090). After
  fixing the five real backend gaps above, the pipeline ran end-to-end without crashing/OOMing but produced
  an invalid (NaN/all-black, or noise once F16 was worked around) image, while the identical CLI path on
  CUDA produced a correct astronaut-on-horse photo. A prior pass at this investigation forced F32
  activations on Vulkan (`DitDtype.ActFor`, since removed) after observing that eliminated the NaN — this
  was a **false fix**: the resulting image, when actually opened and viewed (not just checked for
  "not all-black"), was pure noise/static. No-NaN is not the same as correct; every e2e claim needs the
  PNG viewed, not just a statistical sanity check.
  **The real bug:** `VulkanBackend.DispatchMatmul` derived `M`/`N` from `output.Shape`'s RANK STRUCTURE
  ("flatten all-but-last dims into `M`, last dim is `N`") instead of from the weight tensor — unlike
  `CudaBackend.LinearImpl`, which derives `n = weight.Shape[0]`, `k = weight.Shape[1]` and never even reads
  `output.Shape`. This is silently wrong whenever a caller shapes a Linear's output as
  `[B, S, heads, headDim]` — exactly what `Krea2Attention.Forward` does for Q/K/V, so that the downstream
  per-head `RmsNorm`/RoPE can normalize over `headDim` without a reshape. The true output width is
  `heads·headDim`, spanning TWO trailing dims, not just the last one; the old code computed `M` far too
  large (folding `heads` into it) and `N` far too small (just `headDim`), so the GEMM kernel read `x`'s
  buffer out of its actual bounds (undefined/zero-initialized VRAM for most "rows") and used only a
  `headDim`-wide slice of the real `hidden`-wide weight matrix. Real-data validation against the actual
  Krea2 checkpoint at production scale (seqLen≈4109) showed `to_q`/`to_k`'s Linear output was **~90% exact
  zero** against the correct CPU dot-product reference. Every downstream op (`RmsNorm`, RoPE, `Sigmoid`,
  `ScaledDotProductAttention`) independently validated bit-correct against ITS OWN real input — because each
  op correctly computed its formula on already-corrupted data. **This is the methodological lesson: per-op
  real-data validation cannot catch an upstream shape-contract violation, because a wrong-shaped GEMM still
  produces internally-consistent (if wrong) numbers that every downstream op faithfully propagates.** What
  broke the case open was comparing the SAME stage across backends (Vulkan vs. CUDA, both forced F32 for a
  fair comparison) at the `to_q`/`to_k` Linear boundary specifically — `x` (the shared input) matched almost
  exactly between backends, but `q`/`k` (the Linear output) diverged by 2× in magnitude, which is what a
  wrong-shaped GEMM looks like and a precision/rounding difference does not.
  **The fix** (`VulkanBackend.DispatchMatmul`): derive `N` from the weight operand
  (`transposeB ? b.Shape[0] : b.Shape[b.Shape.Rank - 1]`), then `M = output.ElementCount / N` — a strict
  generalization that leaves every currently-correct call site (rank-2, and rank-3 outputs where the last
  dim already IS the true width, e.g. Krea2's FFN/gate Linears) byte-identical, while fixing the rank-4
  split-head case. Regression-tested via `Backend_Linear_SplitHeadOutputShape_MatchesCpu` (a rank-4 output
  shape with `heads·headDim` split across two dims — the exact shape class that exposed the bug); full
  124-test Vulkan suite green after the fix (3060). `DitDtype.ActFor` (the F32-on-Vulkan workaround) has
  been REMOVED — it masked a symptom, cost an ~18× activation-precision perf hit, and is unnecessary now
  that the actual bug is fixed: Krea2 on Vulkan runs its normal F16 hot path and produces a coherent image,
  verified via the CLI (`HartsyInference.Cli`, not the xUnit test harness) with the identical
  prompt/seed/steps/cfg as the CUDA reference — both produce the same recognizable astronaut-on-horse photo.
  **Speed, once correctness was fixed** (RTX 4090, `HARTSY_LOG_LEVEL=Verbose` breakdown): text encode ~2.4s,
  DiT preload ~2.2s, denoise loop **~34.4s/step × 8 steps ≈ 276s** (remarkably stable step-to-step, within
  1.1s across all 8 — a steady-state cost, not an anomalous stall), VAE decode ~39s, total ~322s. CUDA runs
  the identical config in ~11s end-to-end. This gap was checked for a fixable cause (stray D2H sync,
  un-batched submit-per-dispatch) before concluding anything — dispatches already batch by default
  (`HARTSYINFERENCE_VK_SUBMIT_PER_OP` defaults off, `FlushThreshold=8`), and the per-step timing's tight,
  uniform stability across 8 independent steps is inconsistent with a stray-sync/one-off explanation. This
  is the known, already-scoped Vulkan dispatch-overhead ceiling (hand-written GLSL kernels + per-dispatch
  submit/barrier overhead vs. CUDA's cuBLAS/cuDNN + graph capture) — see `benchmarks/scoreboards/VULKAN.md`
  and Phase 5–7 of the Vulkan bring-up plan (core-primitive perf ceiling, denoise-loop graph capture) for
  the scoped path to closing it. Closing it is a substantial, separate perf-engineering effort, not a bug
  fix; it was NOT attempted in this session beyond confirming it isn't a quick fix. All debug instrumentation
  used during this investigation (temporary prints/validators in `Krea2Attention.cs`, `FluxRope.cs`, an env
  override in `Krea2Recipe.cs`) was fully reverted; the only permanent code changes are the `DispatchMatmul`
  fix itself, the `Krea2Recipe.CacheWeightCasts=false` production OOM fix (a separate, genuine bug found
  earlier in this investigation), and the five backend-gap fixes above.
- **Denoise-loop graph capture (`HARTSY_DIT_GRAPH=1`) attempt on Krea2 (2026-07-30): mechanism built and
  proven correct in isolation, does not fit this model's memory profile.** Following on from the correctness
  fix above, `VulkanStepGraph` (a new file — persistent `VkCommandPool` + one reusable `VkCommandBuffer` +
  one `VkFence`, submitted via `vkQueueSubmit2` + fence-wait, replayable without re-recording) was built as
  the Vulkan-native CUDA-Graph analog and wired into `VulkanBackend.Dispatch()`/`CopyInto()` via a capture
  branch. Four real CLI runs (`HARTSY_DIT_GRAPH=1`, identical prompt/seed/steps/cfg), each hitting a
  different capture-illegal call site than the last, each caught cleanly by the model's own
  `catch (Exception ex) when (capture)` fallback (`StepGraphReset()` + `_graphDead=true` + re-run the step
  eagerly) — every run still completed with a correct, coherent image:
  - **Run 1:** `CopyToDevice: cache miss during step-graph capture` for the fixed-latent tensor. Root cause
    wasn't the copy itself — `CfgEulerStep` (the CFG/Euler-combine step) had **no `VulkanBackend` override at
    all**, so `IBackend`'s CPU-loop default was reading/writing `.DataPointer` directly every single denoise
    step, evicting the fixed latent from the device and showing up on the NEXT step as a capture-time cache
    miss. Fixed with a real GPU-resident override (`cfg_euler.comp.glsl`, in-place
    `z += (guidance*pos + (1-guidance)*neg)*delta`, F32-only matching the CPU reference and CUDA's
    `dit_cfg_euler_f32`) — this closes a real per-step D2H sync in the EAGER path too, not just a
    capture-only issue.
  - **Run 2:** "a tensor was read from host ... during step-graph capture" traced to `VulkanBackend.Concat`,
    which was a pure CPU `Buffer.MemoryCopy` loop (no GPU dispatch at all). Rewritten device-resident via
    `vkCmdCopyBuffer`, capture-aware. **Ordering pitfall found while fixing this:** `GetBuffer()`'s
    cache-miss upload path can internally call `_stream.SubmitAndAdvance()`, which invalidates an
    already-acquired recording command buffer — resolving all of `Concat`'s N input buffers via `GetBuffer()`
    must happen BEFORE acquiring the capture command buffer, not interleaved with it, or a cache-miss on
    input 2 silently invalidates the `cb` already used for input 1's barrier.
  - **Run 3:** same host-read error, traced to `IBackend.AddScalar` (via `DiTUtils.Modulate`) — no override
    existed. Fixed with a new elementwise op-code (10, `add_scalar`) + `VulkanBackend.AddScalar` override.
  - **Run 4:** a genuinely different failure class — `Vulkan error -2 (ErrorOutOfDeviceMemory):
    vkAllocateMemory(size=134643712, typeIdx=1)` inside `DispatchMatmul` → `Linear` →
    `Krea2Block.SwiGlu`, mid-block-3 of the capture. Not a capture-illegal op; a resource-exhaustion
    conflict between two correct designs: capture retains every transient buffer touched during recording
    (`VulkanGpuTransferHelper.CapturingStepGraph` redirects `FreeDevice`/`DrainTransients` to a retain-list
    instead of actually freeing, since a replayed command buffer's bindings reference fixed device
    addresses baked in at record time) — but Krea2 runs with `CacheWeightCasts=false` (`Krea2Recipe.cs`)
    specifically because its 13GB fp8 DiT dequants each Linear's weight to F16 transiently and frees it
    immediately after, to avoid ~doubling VRAM. Retaining instead of freeing all ~224 (28 blocks × 8
    Linears) of those per-GEMM casts across one capture pass exhausts VRAM even on the 4090. **Not fixed
    this session** — needs an intra-capture bump-pointer arena with slot reuse for weight-cast buffers
    specifically, i.e. building the "CUDA Graph allocation-node semantics" this plan's Non-Goals section
    already flagged as having no direct Vulkan equivalent. Before building it, measure whether it's worth
    it: capture a single block (not all 28) and time its replay against eager — peak retained VRAM drops
    ~28× (should fit easily) and the result tells you the actual payoff, since capture only removes
    host dispatch/submit overhead, not GPU kernel time (the GEMM scoreboard's 30–160× per-op kernel gap
    caps what capture alone can buy).

  **Two cross-submission subtleties found while building the mechanism itself** (before any Krea2-specific
  run), pinned by `StepGraph_TrivialCapture_ReplaysWithFreshDataAcrossThreeSteps`: (1) **Vulkan gives no
  automatic ordering between independent queue submissions** — the normal `VulkanCommandStream`'s
  timeline-semaphore batching and the separate step-graph command buffer's `vkQueueSubmit2` are unrelated
  submissions; a capture buffer replayed right after a normal-stream `CopyInto` refresh can read stale data
  because the refresh hasn't actually completed on the GPU yet. Fixed by `_stream.WaitIdleHost()` at the
  start of both `StepGraphBegin()` and `StepGraphLaunch()`. (2) **`vkCmdPushDescriptorSet` is not statically
  exported by this NVIDIA driver** — a direct `[LibraryImport]` throws `EntryPointNotFoundException` even
  though the extension is supported; it must be resolved dynamically via `vkGetDeviceProcAddr`, trying the
  `KHR`-suffixed extension name first and falling back to the unsuffixed Vulkan 1.4 core name. Push
  descriptors were chosen deliberately over pool-allocated sets for captured command buffers: a
  pool-allocated `VkDescriptorSet` can be invalidated by a later `vkResetDescriptorPool` (`FlipPool`) while a
  replayed buffer still references it (undefined behavior — this is the same descriptor-ring hazard the
  earlier "Deliberately NOT built" call above was worried about), whereas a push-descriptor-bound set has no
  backing object for a ring to invalidate. This needed two independent descriptor-set/pipeline-layout caches
  (pool-allocated vs. `VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR`-flagged — a layout created
  with the flag cannot be pool-allocated and vice versa), selected via a `forPush`/`forCapture` bool threaded
  through `VulkanDescriptorManager` → `VulkanKernelRegistry` → `VulkanBackend.GetKernel`, not a rearchitecture
  of the ~40 existing `Dispatch()` call sites.

  All fixes landed with real GPU-resident overrides (not workarounds) and dedicated regression tests; full
  suite 130/130 green throughout. Final verification: the CLI (production path) was also run WITHOUT
  `HARTSY_DIT_GRAPH` to confirm the three new op overrides (now always-on, not gated by the env var) don't
  regress the default path every user actually gets today.
- **Krea2 VAE decode ~2300× vs CUDA — root-caused to a missing `VulkanBackend` override, NOT a kernel-
  throughput ceiling; fixed** (2026-07-31, RTX 4090). Following on from the Phase 7 work above, a
  `HARTSYINFERENCE_VK_PROFILE=1` breakdown of the ~33s VAE-decode phase showed `Conv2D` (the decoder's
  visible GPU op) totaling only ~32ms — nowhere near 33s. The other VAE ops (`Add`, `Clamp`, `Fill`, `Scale`,
  `Silu`) don't call `EnterOp()` at all (see the profiler-blind-spot entry below), so they didn't even show
  up to explain the gap; the actual culprit, `WanRmsNormChannel`, had **no `VulkanBackend` override at all**
  — every call fell through to `IBackend`'s CPU-loop default, which reads/writes `.DataPointer` directly: a
  full D2H sync, a single-threaded scalar reduction over C with a cache-hostile stride-`spatial` access
  pattern (each channel step is a full `spatial`-sized row apart — at `QwenImageVaeDecoder`'s one call site,
  the decoder's FINAL head norm, that's `[1,96,1024,1024]`, ~402 MB, the worst possible shape for this
  fallthrough), then an H2D re-upload for whatever consumes the result. `CudaBackend` has always had a real
  kernel for this op (`wan_vae_rms_norm_channel.cu`) — Vulkan never did. Fixed with a straight GLSL port
  (`wan_rms_norm_channel.comp.glsl`, one invocation per `(b, spatial)` position, F32 only, matching both
  references) — **accumulates in `float`, not `double`**: the CUDA/CPU references use double for the sumSq
  reduction, but `shaderFloat64` isn't guaranteed device-wide (this backend targets AMD/Intel too) and C is
  small (≤ a few hundred), so the float-vs-double delta is negligible; don't reach for GLSL `double` on this
  codebase's Vulkan backend without checking `shaderFloat64` is actually requested as a device feature first
  (it isn't, anywhere in this codebase). Result: **VAE decode dropped from ~33.0s to ~0.5–0.65s (~50–65×,
  run-to-run jitter)** — now inside the normal hand-written-GLSL-vs-cuDNN band (~30–50×, consistent with the
  GEMM scoreboard), not a pathological outlier. Full generation: 331.1s → 296.2s (~10.6%). Regression-gated by
  `Backend_WanRmsNormChannel_MatchesCpu` (both the gamma and no-gamma paths), 3060 + llvmpipe, 132/132 suite
  green. Lesson for future op-gap hunts: grep every `backend.*` call site in the model against
  `VulkanBackend`'s actual method list BEFORE assuming a profiled-but-small op is actually the bottleneck —
  a call that never dispatches never shows up as a large number anywhere.
- **`HARTSYINFERENCE_VK_PROFILE=1` has a blind spot: `Add`/`Clamp`/`Fill`/`Scale`/`Silu`/`Gelu`/`Sigmoid`/
  `Tanh`/`Elu` never appear in its output, however expensive, because they call `DispatchElementwise`
  directly without ever opening an `EnterOp()` scope** (found 2026-07-31 while chasing the VAE-decode entry
  above — the first profile read as "`Conv2D` = 32ms" against a 33s phase, which is the profiler silently
  omitting the real cost, not measuring a fast op). `VulkanProfiler.Record` is only called from
  `OpScope.Dispose()`; any op whose public method dispatches without wrapping `using OpScope _ =
  EnterOp();` is invisible to the profile table regardless of how much host-wall or GPU time it actually
  costs. Not fixed this session (wrapping these would also change their auto-flush timing — `Dispatch()`'s
  own inline `_dispatchesSinceSubmit >= FlushThreshold && _opNestingDepth == 0` check currently fires
  differently for un-scoped ops than `OpScope.Dispose()`'s equivalent check would, a behavior change that
  needs its own verification pass, not a same-session bolt-on). When profiling a Vulkan phase and a
  suspiciously small op total doesn't add up to the phase's wall-clock, check for un-scoped elementwise ops
  before concluding the gap is unexplained.
- **Coopmat F32-output gate: real dtype bug found and fixed, then reverted the same session — the actual
  blocker is shape, not dtype.** (2026-07-31, RTX 4090/3060). `ResolveGemmDtype` resolved the GEMM compute
  dtype directly from `output.DType` with no F32→F16 promotion, so `TryDispatchCoopmat`'s own `gemmDtype ==
  F16` gate silently failed for every F32-output Linear — even though `TryDispatchCoopmat` has supported an
  F32-output accumulate/store via its `OUTPUT_F32` spec-const since the "coopmat inert on Flux" fix earlier
  in this file. Since `Krea2Transformer` allocates every Linear's output as F32 (matching
  `CudaBackend.LinearImpl`'s own convention), this meant Krea2 could never reach coopmat regardless of input
  dtype. New engagement counters (`VulkanBackend`'s `_coopmatGemmCount`/`_tiledGemmCount`, printed alongside
  `HARTSYINFERENCE_VK_PROFILE=1`'s dump) confirmed it directly: **0.8% (16/2112) of a real Krea2 run's GEMMs
  reached coopmat**, the rest fell to `matmul_tiled`. Promoting F32 outputs to F16 compute when
  `M/N/K % 16 == 0` (mirroring `CudaBackend`'s own TF32-by-default F32-output GEMMs) raised that to... still
  16/2112 — **the fix was real but bought ~0 wall-clock**, because a SEPARATE, deeper constraint blocks
  almost every expensive GEMM in this model regardless of dtype: coopmat also hard-requires M/N/K all exact
  multiples of 16 (no buffer padding support — see `matmul_coopmat.comp.glsl`'s own comment: "spec handles
  partial fragments... IF the host pads the buffer... We require multiples of 16 here"). Every per-block
  QKV/FFN/out-proj Linear operates on the joint `[txtSeq+imgSeq]` sequence, and `txtSeq` comes straight from
  the tokenized+encoded prompt length (`Krea2Transformer.cs`: `txtSeq = (int)encoderHidden.Shape[1]`) with
  no padding to a fixed length. Measured directly on this exact prompt: `imgSeq=4096` (a multiple of 16, as
  expected for a square patchified 1024×1024 latent), but `txtSeq=13` → `jointSeq=4109`, 3 short of the next
  multiple of 16 — and since `txtSeq` is prompt-length-dependent, essentially no real prompt will land on a
  multiple of 16 by chance. **Reverted** the dtype-promotion change (kept only the engagement counters,
  which have no precision cost and were the actual diagnostic payoff): the precision trade-off it introduced
  (full F16 compute, 5-bit exponent, vs. F32/TF32's 8-bit range) was real and never actually verified under
  the condition it would matter most — FP8-dequantized-weight GEMMs, exactly where activation magnitudes are
  least predictable — and wasn't worth carrying for a measured zero-throughput win. If the shape blocker
  below gets fixed and this becomes worth revisiting, re-derive it from a real e2e SSIM/pixel-diff gate, not
  from "CUDA already runs reduced-precision GEMMs by default" alone (necessary, not sufficient — TF32's
  wider exponent range isn't the same guarantee as F16's). **The actual next lever**: pad M (and the
  corresponding buffers) up to the next multiple of 16 with safe scratch rows when it's not already one,
  matching the shader's own documented (but unbuilt) support for a padded host buffer. Real, separate,
  memory-safety-sensitive engineering — the input buffer's read past the true M rows must stay in-bounds,
  and the output buffer's padding rows must never be treated as real data by a downstream consumer — not
  attempted this session. See `benchmarks/scoreboards/VULKAN.md` and `docs/Checklists/ROADMAP.md` §3.
- **Coopmat M-padding: built, passed every test thrown at it, then caused a real `ErrorDeviceLost` on a
  full 8-step Krea2 run — reverted.** (2026-07-31, RTX 4090, follow-on to the entry above.) Built
  `TryDispatchCoopmatMPadded`: when M isn't a multiple of 16 (N/K already are, matching every real shape
  measured), allocate a scratch A buffer padded to the next multiple of 16, device-to-device copy the real
  M rows in via `RecordCopyAndBarrier` (the same primitive `CopyInto`/`CopyTo` already use — a real pipeline
  barrier, not a bare `vkCmdCopyBuffer`), dispatch the existing `TryDispatchCoopmat` unchanged against the
  padded shape, and let the caller's tensor size naturally ignore the padding tail on readback (confirmed
  `VulkanGpuTransferHelper.CacheActivation` sizes its readback from the TENSOR's `ElementCount`, never the
  `VulkanBuffer`'s physical `Size`, before writing this). Verified extensively before calling it done: (1) a
  from-scratch CPU-reference correctness test at Krea2's exact real non-alignment (M=13, the model's actual
  measured `txtSeq`) — passed, and confirmed via reflection on the engagement counters that the padded path
  actually fired rather than silently falling back. (2) A 200-iteration no-`Sync()` stress test at Krea2's
  exact real M/K/N scale (4109×6144×6144) — flat memory the whole way, in 17 seconds. (3) Two full real CLI
  runs (2-step: 84s, 4-step: 163s, GPU pinned at 100% utilization throughout, VRAM oscillating 21.6–24.1GB
  but never stuck) — both completed with coopmat engagement climbing to 60.7% then 74.9% and plausible-
  looking output.
  **Then a full 8-step run failed**: first with what looked like a 40-minute hang (GPU at 3% utilization,
  VRAM frozen at 23.97/24.56GB) — investigated and found to be confounded by an entirely unrelated external
  process (a ComfyUI backend under this box's SwarmUI install) that had started sometime mid-session and was
  independently holding ~12.7GB of the same 4090, undetected because nothing about this session's own work
  had touched it. After killing that process and re-confirming ~22GB of clean headroom (`nvidia-smi
  --query-compute-apps` before AND after — SwarmUI auto-restarted the ComfyUI backend but it stayed idle,
  no model loaded), the SAME 8-step run failed again, this time cleanly and immediately obvious as a real
  bug: `Vulkan error -4 (ErrorDeviceLost): vkQueueSubmit2`, partway through step 1, with
  `RepeatKvHeads` (an unrelated op) showing an anomalous 58ms avg / 4050ms max right before the crash — the
  same anomalous-timing signature that had appeared (and been dismissed as noise) in the earlier successful
  2-step run too. `ErrorDeviceLost` is a GPU driver-level fault/reset, categorically different from
  `ErrorOutOfDeviceMemory` (which the retry-then-throw allocator path already handles cleanly and would have
  produced instead, had this simply been a memory-pressure problem) — this is real evidence of a memory-
  safety or synchronization bug that the isolated stress test's simple repeated-Linear-call pattern didn't
  exercise. **Reverted entirely** (the method, its call site, both new regression tests) rather than debug
  further via more blind full-CLI-run attempts — the GPU is a shared resource on this box (other real
  processes, like the ComfyUI backend above, depend on it), and a device-loss bug is too severe to leave
  half-diagnosed in a shipped state. Two leading suspects for whoever picks this up, neither confirmed:
  (a) `VulkanCommandStream.AcquireRecording()` splitting the padding copy and the coopmat dispatch across
  two separately-submitted command buffers — same-queue submission order does NOT by itself guarantee
  memory-visibility ordering across separate `vkQueueSubmit2` calls, only an explicit barrier WITHIN one
  command buffer does, and `RecordCopyAndBarrier`'s barrier only covers the buffer it was recorded into;
  (b) a genuine out-of-bounds access from a size/offset miscalculation that only manifests under the real
  model's specific op-interleaving (SDPA's per-head shared-buffer reuse, `Concat`, `CfgEulerStep`'s address-
  preserving in-place update) rather than a simple repeated-Linear loop. Root-causing this properly needs
  Vulkan validation layers or `compute-sanitizer`-class tooling (already used elsewhere in this repo for
  exactly this bug class — see the CUDA im2col/CausalConv3d OOB entries above), not more full-model runs.
  **Recommended next attempt, not this same design**: the ggml/llama.cpp Vulkan backend (`mul_mm.comp`)
  handles exactly this non-tile-aligned-GEMM problem via a specialization-constant `ALIGNED` flag selecting
  between an unconditional fast path and a per-element-bounds-checked, shared-memory-staged load/store — no
  separate scratch buffer, no device-to-device copy, no risk of a cross-command-buffer barrier gap, because
  everything happens inside one dispatch's own shared memory. That's a shader change (real work, and its own
  new risk surface), but it sidesteps this entire class of host-side buffer-lifetime/barrier bug. See
  `benchmarks/scoreboards/VULKAN.md` and `docs/Checklists/ROADMAP.md` §3 for the full writeup.
- **Coopmat M-padding, THIRD attempt (the ggml shared-memory design) — built, a real bug found and fixed
  by testing (not in production), then verified stable through a full 8-step Krea2 run — but delivers ~0
  real-world speedup.** (2026-07-31, RTX 4090/3060, follow-on to both entries above.) Built
  `matmul_coopmat_partial_m.comp.glsl` — a SEPARATE shader file (not a branch inside
  `matmul_coopmat.comp.glsl`, so the existing, already-proven aligned fast path stays byte-identical and
  carries zero new risk) that mirrors `matmul_tiled.comp.glsl`'s own proven idiom: cooperatively stage the
  boundary row-tile into workgroup `shared` memory with a plain scalar `if (row < pc.M)` bounds check
  (zero-fill out-of-range), `barrier()`, `coopMatLoad` from the guaranteed-in-bounds shared copy, and drain
  the result back to the real output buffer the same way. No scratch VulkanBuffer, no device-to-device copy,
  no cross-command-buffer barrier — everything lives inside one dispatch, structurally avoiding the risk
  class that produced the SECOND attempt's `ErrorDeviceLost`. Host wiring: `TryDispatchCoopmat` now selects
  `matmul_coopmat` (M aligned) vs. `matmul_coopmat_partial_m` (M not aligned, `transposeA=false` only —
  Linear's only real usage), and the BM/BN tile-shrink loop now ALSO accounts for the new kernel's shared-
  memory footprint (`BM*16*2 + BM*BN*4` bytes), not just invocation-count limits, matching
  `PickMatmulTile`'s existing device-aware shrink pattern.
  **A real, separate bug was caught by testing, not shipped**: a 9-case parameterized correctness test
  (M ∈ {1,13,17,33,63,65,100} × bias on/off, N=48 chosen deliberately NOT a multiple of BN=32) initially
  FAILED 6/9 cases with exactly-100% relative error — not a crash, silent wrong output. Root cause:
  `matmul_coopmat.comp.glsl`'s single early `return` for an out-of-N-bounds subgroup is harmless there
  (that kernel never calls `barrier()`), but this new kernel DOES call `barrier()` after the K-loop — an
  early return for SOME subgroups in a workgroup while OTHERS continue to that barrier is a textbook
  divergent-barrier bug (undefined behavior; this GPU's manifestation was silent corruption, not a hang or
  crash). N being a multiple of 16 (the host's existing, required gate) does NOT imply N is a multiple of
  BN (the workgroup tile width, 64/32/16) — a coarser granularity mismatch that can leave a "wasted" fully-
  or partially-out-of-bounds subgroup even when the fragment-level alignment gate is satisfied. **Fixed** by
  removing every early return after the K-loop: every invocation in a workgroup now always reaches the same
  `coopMatStore`+`barrier()`+drain sequence unconditionally (gating only the *bias load*, which has no
  barrier near it, behind a `colInBounds` check to avoid reading past the real N-sized bias buffer), and the
  final scalar drain loop bounds-checks BOTH row and column before writing to the real output — the only
  point where out-of-range work is ever discarded. After the fix: full 9/9 correctness cases pass (verified
  both numerically against a CPU reference AND that the partial-M kernel actually engaged, not a silent
  matmul_tiled fallback), the pre-existing large-scale `Backend_Linear_FP8Weight_NonUnitScaleFactor_
  MatchesCpu_RealKrea2FfnShape` test (M=4108, real Krea2 FFN shape, now automatically exercises this kernel
  since 4108 isn't a multiple of 16) passes unmodified, a 200-iteration no-`Sync()` stress test at Krea2's
  exact scale (M=4109, K=N=6144) shows flat memory in 17s, and — the actual gate — three consecutive real
  Krea2 CLI runs (2-step: 80s, 4-step: 155s, **8-step: 296.7s, coopmat engagement 84.8% (1792/2112)**) all
  completed cleanly with a correct, verified-by-eye output image, GPU pinned at 100% utilization throughout
  every run (not idle/stuck), VRAM oscillating in a bounded 20.6–24.1GB range (never frozen). Full suite:
  143/143 on the 3060 and 4090; llvmpipe shows only the same two pre-existing, already-documented gaps
  (llvmpipe has no coopmat hardware at all, so these tests skip themselves there rather than exercising
  anything new).
  **The honest result: this is correct and safe, but it does not close Krea2's speed gap.** Coopmat
  engagement went from 0% → 84.8%, yet the 8-step wall-clock (296.7s) is statistically indistinguishable
  from a same-shape run with coopmat forced fully OFF via `HARTSYINFERENCE_VK_DISABLE_COOPMAT=1` (a 2-step
  A/B done earlier this session: 78.9s coopmat-off vs. 80.5s coopmat-on at 60.7% engagement — within normal
  run-to-run noise, not a real difference). This means the bottleneck was never "coopmat isn't reached" — it
  was, and remains, coopmat's own KERNEL THROUGHPUT, matching what `benchmarks/scoreboards/VULKAN.md`'s GEMM
  table already found independently: the F16 coopmat path was ALREADY measured 30–157× slower than CUDA's
  cuBLAS even where it WAS engaging. **Kept** (unlike the second attempt) because it's proven correct and
  stable, closes a real capability gap (coopmat is no longer silently inert for the ~85% of GEMMs whose
  sequence length happens to be unaligned), and any FUTURE work that speeds up the coopmat kernel itself
  (larger tiles, coopmat2, better register blocking) now automatically reaches those GEMMs too. **The actual
  next lever** is coopmat kernel throughput itself, not engagement — see `benchmarks/scoreboards/VULKAN.md`
  and `docs/Checklists/ROADMAP.md` §3.
- **Register blocking (ggml's core coopmat optimization), tested directly — a net LOSS on this RTX 4090;
  plain shared-memory staging is a real but small win; the 30–157× gap is NOT primarily an
  arithmetic-intensity problem.** (2026-07-31, follow-on to the entry above.) Read ggml/llama.cpp's actual
  `mul_mm.comp`/`mul_mm_funcs.glsl` source directly (not just PR summaries): its core technique is that each
  subgroup computes a GRID of 16×16 accumulators (register blocking) from ONE shared-memory-staged tile,
  instead of `matmul_coopmat.comp.glsl`'s one-accumulator-per-subgroup direct-global-`coopMatLoad` design —
  trading subgroup count (occupancy) for arithmetic intensity (data reuse per byte loaded). Built
  `matmul_coopmat_blocked.comp.glsl` as a standalone diagnostic kernel (an `internal
  TryDispatchCoopmatBlockedDiagnostic` entry point, NOT wired into `DispatchMatmul` — no production risk)
  to test this directly, parameterized by WM/WN (per-subgroup tile size) so the SAME kernel can isolate
  "shared-memory staging alone" (WM=WN=16, matching naive's 16-subgroups/workgroup occupancy exactly) from
  "staging + register blocking" (WM=WN=32, 4 accumulators/subgroup, only 4 subgroups/workgroup). Verified
  correct first (4/4 CPU-reference cases including a non-16-aligned M boundary), THEN benchmarked — a
  controlled 3-way A/B, same shapes as the GEMM table, same methodology for all three legs:
  **register blocking (WM=WN=32): 0.79–1.05× vs. naive — neutral to WORSE.** Reducing subgroup count from
  16 to 4 per workgroup costs more in lost occupancy than the reduced global-memory traffic gains — this
  GPU apparently has enough bandwidth that trading parallelism for reuse is a bad trade at these shapes.
  **Staging alone (WM=WN=16, occupancy unchanged): 1.00–1.41× — a real, modest win, correlated with K-depth**
  (the two K=3072 shapes benefited most; K=1280/1536 barely moved). **Neither result gets remotely close to
  closing 30–157×** — even the best case (1.41×) leaves the overwhelming majority of the gap unexplained,
  meaning the true bottleneck is NOT primarily memory access pattern/arithmetic intensity (what both
  techniques target). Likely candidates instead: raw `coopMatMulAdd` instruction throughput on this NVIDIA
  driver's `VK_KHR_cooperative_matrix` (coopmat1) implementation vs. CUDA's hand-tuned WMMA/MMA PTX, or fixed
  per-dispatch/launch overhead no in-kernel tiling change can amortize. Root-causing further needs real GPU
  profiling (Nsight Compute or `VK_LAYER_KHRONOS_validation`) — neither available on this box and no
  passwordless `sudo` to install them (`apt-get install vulkan-validationlayers` exists as a package but
  needs a password this session doesn't have). See `benchmarks/scoreboards/VULKAN.md` for the full numbers
  and the methodology caveat (this benchmark's batched-then-single-`Sync()` timing isn't directly comparable
  in absolute terms to the GEMM table's presumed per-call methodology — the staged/blocked/naive RATIOS are
  the reliable result, not the absolute microsecond figures). `VK_NV_cooperative_matrix2` (NVIDIA-only,
  already flagged in ROADMAP.md §3) is the next concrete kernel-level lever, since it changes the underlying
  instruction/memory path rather than just the tiling strategy around the same coopmat1 instructions.
- **Got real GPU profiling working after all (root access), and it changes the whole picture: the true
  kernel-level gap is 10–22×, not 30–157×, and coopmat is NOT meaningfully faster than the plain scalar
  fallback on this hardware.** (2026-07-31, follow-on to the entry above.) With `sudo`: confirmed Nsight
  Compute (already installed from prior CUDA work) is CUDA-only — its `--help` has zero mentions of
  "vulkan"/"graphics"/"opengl"/"d3d", and pointing it at the Vulkan test process produces `No kernels were
  profiled`. `vulkan-validationlayers` installs fine via apt (now installed — catches API misuse, not
  throughput). Nsight Graphics (the tool that WOULD show SM occupancy / tensor-core utilization) isn't
  apt-installable — needs an authenticated NVIDIA developer-portal download this environment can't complete.
  **Built real profiling instead**: `VulkanGpuTimer` (`src/HartsyInference.Vulkan/VulkanGpuTimer.cs`) + a new
  `VulkanBackend.MeasureGpuTimeMs` diagnostic entry point, using `VkQueryPool` GPU timestamps
  (`vkCmdWriteTimestamp2`) — a core, always-available Vulkan feature, the standard technique for exactly this
  situation (vendor profiler unavailable/uncooperative). This measures PURE GPU execution time, with no host
  submission/dispatch/wait overhead mixed in (unlike every `Stopwatch`-around-`Sync()` number in this file so
  far) — required adding `VkQueryPoolCreateInfo` + `vkCreateQueryPool`/`vkDestroyQueryPool`/
  `vkCmdResetQueryPool`/`vkCmdWriteTimestamp2`/`vkGetQueryPoolResults` P/Invoke bindings and exposing
  `VkPhysicalDeviceLimits.timestampPeriod` as `VulkanCapabilities.TimestampPeriod` (the struct field already
  existed, unused until now). Re-measured the same GEMM-table shapes: **10.6–22.4× vs. CUDA** (not 30–157×) —
  the earlier figure was mostly an artifact of the isolated per-call benchmark methodology's host/sync
  overhead, not kernel throughput. **Then, re-running the SAME shape with `HARTSYINFERENCE_VK_DISABLE_
  COOPMAT=1`** (verified via the engagement counters: `coopmat=0%, tiled-fallback=100%`) **produced a
  statistically identical number** (22.7× vs. 21.3× — within run-to-run noise) — coopmat and the plain scalar
  `matmul_tiled` fallback have the SAME real GPU throughput at these shapes. This is the finding that makes
  every other result this session consistent: raising Krea2's coopmat engagement 0%→84.8% bought ~0 real
  speedup because coopmat isn't actually faster than the scalar path here; register blocking and shared-
  memory staging only moved the needle 0.8–1.4× because both techniques assume coopmat's tensor-core path
  has meaningfully more headroom to exploit than a well-written scalar kernel — if it doesn't, tuning its
  tiling strategy can't create headroom that isn't there. Most likely explanation: `VK_KHR_cooperative_matrix`
  (coopmat1) on this NVIDIA driver either isn't compiling `coopMatMulAdd` down to real tensor-core (HMMA)
  instructions, or is doing so with far less scheduling/register-allocation sophistication than cuBLAS's
  WMMA/MMA PTX — a driver/compiler-level characteristic, not something fixable by further GLSL restructuring.
  Confirming which needs SASS/ISA-level inspection (Nsight Graphics), still not available. **Next step, in
  priority order**: (1) `VK_NV_cooperative_matrix2` — a genuinely different instruction/memory path, not just
  different tiling of the same instructions; a real win there would confirm coopmat1's tensor-core engagement
  specifically is the problem. (2) Get Nsight Graphics installed for a direct answer. See
  `benchmarks/scoreboards/VULKAN.md` for the full numbers and methodology.

- **`VK_NV_cooperative_matrix2` confirmed the hypothesis: it's a real, substantial speedup over coopmat1/
  scalar — the first one found all session — though it does NOT close the CUDA gap.** (2026-07-31,
  follow-on to the entry above.) Built `matmul_coopmat2.comp.glsl`: WORKGROUP-scope coopmat (not
  subgroup-scope like coopmat1), addressed via `tensorLayoutNV` + `coopMatLoadTensorNV`/
  `coopMatStoreTensorNV` directly against global memory — no manual shared-memory staging, and built-in
  bounds CLAMPING (`gl_CooperativeMatrixClampModeConstantNV`) so M/N/K need not be multiples of the tile
  size, unlike every coopmat1 kernel here. Extension confirmed supported (revision 1) on both the RTX 4090
  and RTX 3060 via `vulkaninfo`; wired up device-extension detection, capability querying (enumerated
  `vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV` — the RTX 4090 reports a family of
  workgroup-scope configs at 32/64/128/256 invocations; host auto-picks the largest matching FP16-in/
  FP32-out one: 32×32 M/N granularity, K granularity 16, 256 invocations), and the `VkPhysicalDeviceCoop
  erativeMatrix2FeaturesNV` feature-chain/extension-enable plumbing in `VulkanDevice.cs`
  (`HasCooperativeMatrix2` + `CoopMat2{M,N,K}Granularity`/`CoopMat2WorkgroupInvocations` on
  `VulkanCapabilities`). Correctness verified against a CPU reference including shapes where NONE of
  M/K/N are multiples of the tile granularity (129×130×144) and shapes smaller than one tile in every
  dimension (17×33×5) — all pass with zero manual bounds-checking in the shader, proving the clamp-mode
  addressing genuinely works. Swept K-block size (`BK`) {16,32,64,128,256}: the bare-minimum BK=16 (the
  device's reported K-granularity) was measurably WORSE than larger values on a K=3072 shape (14983 μs vs.
  ~9260–9925 μs) — each `coopMatLoadTensorNV` is itself an expensive workgroup-cooperative op, so smaller/
  more-frequent K-steps pay that overhead more often; BK=64 is now the default. **GPU-only-time result
  (same 5 GEMM-scoreboard shapes, same `VkQueryPool` methodology): coopmat2 beats the naive coopmat1/scalar
  baseline by 1.2–2.5× (best on the smallest-K shape), landing the CUDA gap at 8.3–15.6× — roughly HALF the
  previous 10.5–25.1× naive gap, but still far from parity.** This confirms coopmat1's tensor-core
  engagement specifically was the bottleneck (as hypothesized in the entry above), not GEMM tiling strategy
  in the abstract — a genuinely different instruction/memory path bought a real win where retiling the same
  instructions (register blocking, shared-memory staging) topped out at 0.8–1.4×. Diagnostic-only, NOT wired
  into `DispatchMatmul` — see `VulkanBackend.TryDispatchCoopMat2Diagnostic`,
  `VulkanBackendSmokeTests.Backend_CoopMat2_Diagnostic_MatchesCpu`,
  `VulkanCoopmatBlockedBenchmark.Compare_Naive_Vs_CoopMat2_GpuOnlyTime`. **Next step**: apply ggml PR
  #10942's diffusion-shaped coopmat2 tile-size tuning (they combined larger tiles + im2col tuning for
  3.68→5.07 it/s), or get Nsight Graphics installed to see exactly what's still eating the remaining
  8–16× (occupancy limits? memory-bandwidth-bound at these shapes? still-partial HMMA utilization even via
  coopmat2?). See `benchmarks/scoreboards/VULKAN.md`'s new coopmat2 section for the full numbers.

- **Wired coopmat2 into `Linear`/`DispatchMatmul` (opt-in) and ran it against real Krea2 weights: a real
  REGRESSION end-to-end, not the win the isolated benchmark predicted.** (2026-07-31, follow-on to the entry
  above.) Added a follow-up `BroadcastAdd` bias dispatch (not fused into the shader, matching ggml's own
  `mul_mm_cm2.comp`), wired `TryDispatchCoopMat2` into `DispatchMatmul` before `TryDispatchCoopmat`, gated by
  a new opt-in `VulkanBackend.EnableCoopMat2` property (`HARTSYINFERENCE_VK_COOPMAT2=1`, off by default, same
  pattern as `EnableInt8Linear`). Ran a real 2-step Krea2 CLI generation on the RTX 4090 (`HARTSY_DIT_
  GRAPH=1`, seed 42) five times: baseline, coopmat2 on twice, coopmat2 with graph capture off, coopmat2 with
  a warm on-disk pipeline cache. **Correctness: all 5 runs produced byte-identical PNG output** (same MD5) —
  the wiring, bias epilogue, and fallback gating are all correct. **Performance: worse, not better** —
  baseline 58.6s (coopmat1 engagement already 60.7%, thanks to `matmul_coopmat_partial_m` from an earlier
  pass this session) vs. 65.7-65.9s with coopmat2 opted in, reproducible across all 4 coopmat2 runs. Three
  causes found: (1) the isolated coopmat2-vs-coopmat1 benchmark compared against `matmul_coopmat` (the
  aligned-only kernel) — coopmat1's REAL fallback for most of these shapes is `matmul_coopmat_partial_m`
  (shared-memory-staged, already a real if modest win itself), which coopmat2 was never benchmarked against;
  (2) the un-fused bias costs a real extra dispatch+barrier per call the isolated benchmark (always `bias:
  null`) never paid; (3) an unexplained ~4-second one-time stall per run, reproducible regardless of
  graph-capture mode or pipeline-cache warmth, landing on whatever op happens to be executing when it
  resolves — leading hypothesis is a genuine first-use driver/hardware latency for the coopmat2 instruction
  path that the isolated benchmark's warmup loop was structurally blind to. **`EnableCoopMat2` stays off by
  default** — safe to keep in the codebase (zero effect on default behavior, verified by
  `VulkanCoopMat2LinearTests.Backend_Linear_CoopMat2OptIn_DefaultsOff`), but not currently recommendable.
  Next steps: benchmark against `matmul_coopmat_partial_m` specifically, fuse bias into the shader, isolate
  the stall's root cause. See `benchmarks/scoreboards/VULKAN.md` for full numbers and
  `tests/HartsyInference.Vulkan.Tests/VulkanCoopMat2LinearTests.cs` for the wiring tests.

- **All three coopmat2 open items chased down: the correct-competitor theory didn't hold, bias fusion barely
  moved the real number, and the "~4s stall" turned out to be a pre-existing memory-pressure bug on this
  shared box that has NOTHING to do with coopmat2.** (2026-07-31, same day, follow-on to the entry above.)
  (a) Re-benchmarked against `matmul_coopmat_partial_m` (the actual coopmat1 competitor for non-aligned M,
  not `matmul_coopmat`) through the real `Linear` path — coopmat2 still wins, 0.74-0.98x of coopmat1's time
  across 20 shape/alignment/bias combinations. (b) Fused bias directly into `matmul_coopmat2.comp.glsl` via
  a broadcast `tensorLayoutNV` (stride 0 on the M dimension — every row's `coopMatLoadTensorNV` reads the
  same `bias[n]`) loaded straight into an Accumulator-typed coopmat, eliminating the follow-up
  `BroadcastAdd` dispatch entirely. Real wins: correctness tolerance improved 1.5% -> 0.05% max relative
  error (bias no longer round-trips through F16 via a second dispatch), and isolated GPU-only-time for
  biased cases dropped sharply. But real Krea2 e2e time barely changed (65.7s before and after) — bias
  overhead wasn't the dominant real-world cost after all. (c) The "~4s stall": a 4-step real Krea2 run
  logged `[WRN] [Krea2 graph] capture invalidated — falling back to eager: ... ErrorOutOfDeviceMemory:
  vkAllocateMemory(size=134643712, typeIdx=1)` inside `TryDispatchCoopMat2 -> AllocateDedicated`, during a
  `Krea2Block.SwiGlu` Linear call on step 2. **Ran the identical 4-step generation with coopmat2 OFF as a
  control — hit the EXACT SAME OOM, same size, same step, same call site pattern.** This is a real,
  pre-existing VRAM-pressure issue on this shared box (SwarmUI + a ComfyUI Python process + rustdesk all
  hold GPU memory even at idle — confirmed via `nvidia-smi --query-compute-apps`), completely unrelated to
  coopmat2; it doesn't reproduce at 2 steps (less accumulated pressure) but reliably does at 4, in BOTH
  configs. Output correctness held perfectly through the OOM+recovery: all three 4-step runs (baseline,
  coopmat2 x2) produced byte-identical PNG output. **Corrected picture once this shared, unrelated cost is
  factored out**: at 2 steps (no OOM in any run), coopmat2+fused-bias is still ~5% slower in aggregate
  Linear time (60,952ms vs. 58,085ms baseline) — smaller than first measured, but still real. At 4 steps
  (both configs hit the same OOM once), coopmat2's Linear time was at-or-below baseline in both runs
  (121,335-131,534ms vs. baseline's 132,404ms) — genuinely competitive, possibly ahead, though run-to-run
  variance on this box (~10s spread between the two coopmat2 runs) means this isn't confidently a win yet.
  **`EnableCoopMat2` stays off by default** — the investigation converted "unexplained mystery regression"
  into "small, understood, shrinking gap," which is real progress even without a clean win yet. See
  `benchmarks/scoreboards/VULKAN.md`'s follow-up section for full numbers.

- **Ran 6 more 4-step Krea2 runs (4 ON, 2 OFF) to check the variance — 9 total samples confirm "roughly on
  par, not a confident win," and the OOM is deterministic (fires in ALL 9 runs, both configs, same exact
  allocation).** (2026-07-31, same day, follow-on to the entry above.) coopmat2 ON: n=6, mean 121,486ms,
  stdev 6,271ms (range 113,653-131,534ms). coopmat2 OFF: n=3, mean 126,456ms, stdev 8,199ms (range
  117,104-132,404ms). Mean difference: coopmat2 3.9% faster — but Welch's t ~ 0.92, well under the ~2
  threshold for significance, and the OFF group's minimum sits below 4 of the 6 ON runs, meaning the
  distributions substantially overlap. **Honest read: at 4 steps coopmat2 is roughly on par with baseline,
  plausibly a small win, but not confidently distinguishable from run-to-run noise on this shared, contended
  GPU with only 9 samples** — resolving this further would need many more runs than is a reasonable use of
  shared GPU time right now. All 9 runs (6 ON, 3 OFF) hit the identical OOM at the identical allocation/step
  and produced byte-identical PNG output — the OOM is a fully deterministic property of this box at 4+
  steps, not intermittent, confirming it's valid to treat as common-mode noise between configs. Combined
  with the clean, OOM-free 2-step result (still ~5% slower, no confound), 2-step remains the more reliable
  signal: **`EnableCoopMat2` stays off by default.** See `benchmarks/scoreboards/VULKAN.md`'s variance-sweep
  table for the full per-run numbers.

- **Root-caused the shared-box 4-step OOM: NOT a leak — a genuine one-time VRAM peak from step-graph
  capture's retain-everything design, colliding with Krea2's real footprint on this box's real budget.**
  (2026-07-31, follow-on to the entries above, investigated separately from the coopmat2 work since it
  turned out to be unrelated.) Built real diagnostics instead of guessing: a new
  `VulkanMemoryAllocator.OnOutOfMemoryDiagnostic` hook + `VulkanGpuTransferHelper.DiagnosticsSummary()`
  dump a full weight-cache / activation-cache / weight-cast-cache / step-graph-retained breakdown (each
  computed directly from its own dictionary, not the older `_cachedBytes` running counter, which turned out
  to only track weight/cast bytes — never activation-cache bytes) right before a genuine (post-retry) OOM is
  thrown. Re-ran the 4-step repro with this wired in: `allocator: used=20865.6 MB, reserved=21705.9 MB,
  blocks=264` / `weight cache: 433 tensors, 12539.9 MB` / `activation cache: 13 tensors, 146.8 MB` (small —
  not accumulating) / `weight-cast cache: 0 casts, 0.0 MB` / **`step-graph-retained: 163 buffers, 7987.0
  MB`**. The mechanism: a captured Vulkan command buffer bakes buffer ADDRESSES into its descriptor
  bindings at record time, so every buffer that gets disposed DURING the one-time capture window is
  redirected to a "retain for the graph's whole lifetime" list instead of actually being freed (see
  `VulkanGpuTransferHelper.TryRetainForCapture`) — capture must hold every intermediate activation from the
  ENTIRE forward pass alive simultaneously, unlike eager execution's free-and-reuse-per-block pattern. Krea2
  is 28 DiT blocks deep with several ~25 MB (hidden-sized) and ~135 MB (FFN-inner-sized, `Krea2Block.SwiGlu`)
  intermediates per block; by ~10 of 28 blocks into capture, retained buffers already totaled ~8 GB — on top
  of ~12.5 GB of permanently-cached weights, against a real driver-reported budget of only ~21 GB on this
  box (24 GB card minus ~1.8 GB held by other resident processes minus driver reserve, confirmed via
  `nvidia-smi --query-compute-apps` and `vulkaninfo`'s `VkPhysicalDeviceMemoryProperties` budget field). A
  full 28-block capture needs well over 15 GB just for retained activations — this was never going to fit
  here regardless of what else got freed first. **Explains the "2-step never OOMs, 4-step always does"
  pattern precisely**: capture only triggers on the 3rd call to the transformer at a given signature (2
  eager warmup passes, then capture — see `Krea2Transformer.ForwardPatched`'s `GraphCaptureCall = 3`), and a
  2-step, no-CFG generation never reaches call 3 at all — it's not step-count-driven accumulation, capture
  is simply never attempted. **Practical impact is smaller than the alarming log suggested**: this fires
  ONCE per model load (not per-step, not per-generation) — after failing once, `_graphDead` is set and the
  transformer permanently, cleanly falls back to eager for that model instance; a long-running server
  generating many images pays this cost once, not repeatedly. Correctness held perfectly through every
  occurrence (9/9 byte-identical outputs across this whole investigation, from the coopmat2 variance sweep).
  Not fixed (the underlying capacity math is real, not a bug — Krea2 + full step-graph capture genuinely
  needs more VRAM than this box's ~21 GB budget provides on top of its own weights), but now instantly
  diagnosable instead of needing a fresh multi-hour repro-and-instrument cycle next time. See
  `VulkanGpuTransferHelper.DiagnosticsSummary()` / `VulkanMemory.cs`'s `OnOutOfMemoryDiagnostic`.

- **coopmat2 flipped to default-ON: the earlier "~5% regression" was a cross-session variance artifact, not
  a real property of coopmat2 — corrected with a controlled comparison, then validated to 8 steps.**
  (2026-07-31, continuing the "keep tuning coopmat2" pass.) Added per-GEMM-path host-wall-clock timing to
  the profiler (`coopmat2/coopmat/tiled avg host-wall`) to chase down why isolated GPU-only benchmarks kept
  showing coopmat2 winning per-op while the original real-run comparison showed a net loss. First finding:
  the new timing revealed a MEASUREMENT bug of its own, not a real one — `TryDispatchCoopMat2` is
  self-contained (does its own buffer fetch/cast), while coopmat1's per-path timing only wrapped its final
  dispatch call (buffer fetch/cast happens in `DispatchMatmul`'s shared prep, outside that window) — so the
  two "avg host-wall" numbers weren't measuring the same scope. Comparing the full, apples-to-apples "Linear
  Avg(ms)" column instead (which captures the whole call for both paths regardless of internal structure),
  a back-to-back same-session pair showed coopmat2 already AHEAD (82.89 vs. 86.43 ms/call) — the opposite of
  the original finding. That prompted a proper controlled comparison: 4 runs each config at 2 steps,
  alternating ON/OFF back-to-back in the same session (canceling this box's real GPU-contention time-drift,
  the same confound already flagged for the 4-step variance sweep). Result: coopmat2 mean 59,793.5 ms vs.
  baseline 62,296.8 ms — a statistically significant ~4.0% real win (Welch's t≈2.88, n=4 each). The original
  "~5% slower" number came from comparing runs across DIFFERENT sessions with different external GPU
  contention (SwarmUI/ComfyUI/rustdesk residency varies over time on this shared box) — not a real coopmat2
  property. **Extended validation to 8 steps** — the exact step count where an EARLIER, unrelated coopmat1
  M-padding fix passed every synthetic test and then hit a real `ErrorDeviceLost` (the precedent that
  originally justified keeping this opt-in): coopmat2 completed cleanly, no device-loss, hit the same
  well-understood step-graph-capture OOM-and-recover as the baseline (not worse), and was ~10.2% faster
  (244,683 ms vs. 272,603 ms Linear time), byte-identical output. Correctness held byte-identical across
  every configuration tested this whole investigation — 2-step, 4-step, 8-step, aligned/non-aligned M,
  with/without bias, F16/F32 output — dozens of real generations, zero divergence ever. **`VulkanBackend.
  EnableCoopMat2` now defaults to `true`** (override with `HARTSYINFERENCE_VK_COOPMAT2=0`). Several existing
  tests that specifically exercised coopmat1's own engagement (`Backend_Linear_CoopmatPartialM_
  NonMultipleOf16_MatchesCpu`, `Backend_Linear_CoopmatAlignedM_StillUsesOriginalKernel`) needed an explicit
  `backend.EnableCoopMat2 = false` since coopmat2 now engages first by default. See
  `benchmarks/scoreboards/VULKAN.md` for the full numbers.

- **Investigated the beyond-GEMM part of the real Vulkan-vs-CUDA gap: real full-process wall-clock is
  ~53x slower than CUDA per marginal denoise step, far beyond the ~8-16x isolated GEMM gap — three leading
  hypotheses tested and ruled out, one real (separate) allocator bug found along the way, root cause still
  open.** (2026-08-01.) First got an honest, apples-to-apples number: full CLI process wall-clock (not just
  backend-op time) for real Krea2 generations on the same 4090 — CUDA 2-step 15.4s / 8-step 19.5s (marginal
  ~0.67s/step); Vulkan (coopmat2 default-on) 2-step 87.4s / 8-step 302.7s (marginal ~35.9s/step) — a **53.5x
  marginal-per-step gap**, well beyond the 8.3-15.6x isolated coopmat2-vs-CUDA GEMM gap. `HARTSY_LOG_LEVEL=
  Verbose` phase logging + `HARTSYINFERENCE_VK_PROFILE=1` on the same clean (no-OOM) 2-step run showed Linear
  at 99.1% of backend-op time (60,349ms of 60,910ms) — SDPA and other ops are NOT the story (their OOM-run
  "~4s" numbers were purely the already-root-caused step-graph-capture stall, not real per-op cost); the gap
  is still fundamentally inside Linear/GEMM, just far larger at full-model scale than the 5-shape isolated
  benchmark predicted (real avg ~82-135 ms/Linear-call vs. ~1.5-10 ms in isolated GPU-only timing).
  **Hypothesis 1, weight-cast caching (`CacheWeightCasts=false` — Krea2 runs with it OFF on BOTH backends,
  a deliberate ~13GB-fp8-checkpoint VRAM tradeoff, not Vulkan-specific): RULED OUT.** A controlled benchmark
  with many distinct FP8 weights at Krea2's real FFN shape (M=4109, K=6144, N=16384 — K corrected from an
  earlier wrong guess of 3072; Krea2Config.HiddenSize = 48 heads x 128 headDim = 6144) showed transient-vs-
  cached-cast ratio of 0.99-1.01x — negligible. **Hypothesis 2, dependency-chain serialization from
  `Dispatch()`'s unconditional per-dispatch `VkMemoryBarrier2` (ROADMAP.md's "per-dispatch barrier scoping"
  entry): RULED OUT.** A benchmark chaining real SwiGlu ops (gate/up Linear -> Silu -> Mul -> down Linear,
  genuine data dependencies, matching `Krea2Block.SwiGlu` exactly) measured 32.7-33.1 ms/Linear-call — same
  ballpark as independent back-to-back Linear calls at the same shape (35.6-36.2 ms/call), not worse.
  **Hypothesis 3, allocator block-count scaling: also doesn't explain per-call latency (32.7-32.8 ms/call
  flat across 4 vs. 8 distinct simulated blocks) — but surfaced a real, separate bug along the way.** An
  earlier version of the SwiGlu-chain benchmark (before it was corrected to match `Krea2Block`'s real
  per-call-fresh-tensor pattern) reused the same `g`/`u`/`silu`/`gated`/`down` `Tensor` objects across every
  block/step — this doesn't happen in real Krea2Block code (which always allocates fresh tensors and
  disposes them, verified by reading `Krea2Block.cs`), but it exposed that
  `VulkanGpuTransferHelper.CacheActivation` overwrites a tensor's cached-buffer dictionary entry WITHOUT
  freeing the buffer being replaced when the same tensor object is reused as a non-in-place op's output more
  than once (deliberately-in-place ops like `CfgEulerStep`/`CopyInto` correctly avoid this by re-caching the
  SAME buffer handle — this is a gap in the general `CacheOutput` path, not those). Real impact confirmed
  even with the CORRECTED (fresh-tensor, real-pattern) benchmark: allocator block count and `reserved` bytes
  keep growing across repeated passes (16->73->90 blocks at 4 simulated blocks; 9.6GB->17GB reserved) even
  though `used` bytes stays flat — meaning `VulkanMemoryAllocator.Allocate()`'s pooling isn't reclaiming/
  reusing already-freed dedicated blocks promptly within a tight dispatch loop (deferred-free reclamation
  lags the allocation rate when no `Sync()` happens mid-loop), so empty blocks pile up as pure `reserved`
  overhead rather than being found and reused, which independently reproduced the SAME `ErrorOutOfDeviceMemory`
  (crashed a test host outright at 14/24 simulated blocks with no step-graph capture involved at all — a
  DIFFERENT OOM trigger than the already-root-caused step-graph one). **Not fixed this pass** — a real
  reclaim-cadence fix needs care to avoid trading this away for more `Sync()`-induced host stalls, which is
  exactly the kind of overhead this whole investigation is trying to eliminate. **The core ~53x marginal
  per-step / ~82-135ms-vs-~33ms Linear-call mystery remains OPEN** — the isolated SwiGlu-only benchmark still
  doesn't reproduce it even at the real shape with real dependencies; the next things to try are a full
  `Krea2Block`-scale synthetic benchmark (attention + SDPA + RmsNorm + Modulate + GatedResidual, not just
  SwiGlu, at the real 28-block count) or genuine GPU-level profiling (Nsight Graphics, still not available
  on this box) to see what's actually different about full-scale real-model execution. See
  `VulkanFp8WeightCastOverheadBenchmark.cs` for the benchmarks backing all three ruled-out hypotheses.

---

## LLM / GGUF loading gotchas

- See also GGUF sign/rescale bakes under [Quant](#quant--fp8--w8a8-numerical-findings) (Mamba `ssm_a`,
  RWKV-6 rescale) and the SigLIP mmproj swap under [Weight layout](#weight-layout--transpose--converter-parity).
- **Qwen3.5 Gated DeltaNet:** missing `q *= 1/√head_dim` before the recurrence produces fluent-looking
  word salad (not a crash).
- **nn.Linear GGUF weights need a shape relabel to `[out,in]`, NOT a data transpose** (bytes already
  row-major); only raw params like `mm.input_projection` need an actual transpose. Recurs across every VLM.

---

## TTS / STT lessons

- **When a model degenerates but the forward looks faithful, check the checkpoint version/source FIRST**
  — don't deep-layer-diff until reference and engine load the *same* weights. Dia-1.6B looped garbage
  purely because the old `nari-labs/Dia-1.6B` was downloaded instead of `Dia-1.6B-0626` (identical 343
  keys/shapes, only values differ). Orpheus "silent output" traced to weights/sourcing (unsloth mirror),
  not engine code.
- **Always verify TTS with whisper `medium.en`, never `base.en`** — `base.en` was actively misleading in
  multiple cases.
- **F5-TTS:** ConvNeXt filler-tail masking; ×1000 timestep sinusoid scale; erf-GELU stem vs tanh-GELU FFN.
- **Fish-Speech:** the fast depth-LM must take the **PRE-norm** slow hidden (`norm_fastlayer_input=False`).
- **espeak `MatchRule` RULE_PRE OOB:** indexed past the per-word buffer start on words like "Americans" —
  guard the OOB read as a space boundary; fixes all espeak TTS.
- **CosyVoice 2 load recipe (outlier worth keeping):** the LM is **not** a single extended-vocab softmax
  (keeps Qwen text embed + separate `speech_embedding`/`llm_decoder`/`llm_embedding`); the frozen S3
  tokenizer + CAM++ must load from **chatterbox `s3gen.safetensors`** because CosyVoice's ONNX export fuses
  Conv+BN (CAM++ `head.bn*` vanish); the AR loop needs F32 (`HighPrecisionGemm`); the flow decoder must
  trim the prompt-mel region or it replays the reference clip.
- **`AudioRuntime._genLock` is a global single-slot semaphore** — one slow request makes every queued
  request look "hung." Dia's real "hang" was 1152 unbatched `cublasGemmEx` launches/decode-step (16 heads
  × 2 CFG × 18 layers × 2), each a GEMV-as-GEMM at Sq=1; fixed via `cublasGemmStridedBatchedEx` (~32× fewer
  launches): 350–800s → 44–66s.

---

## Video-specific bugs

- **LTX-2 `rope_type = "split"` vs interleaved:** the 22B config declares `split` (rotates two halves
  within each head, compact `dim/2` front-padded freqs) but only interleaved was implemented → persistent
  32-px lattice identical at 8 and 30 steps (colors survive because text cross-attn carries no RoPE). Key
  diagnostic: **a grid identical across step counts rules out under-denoising** and points at RoPE/VAE.
- **Caption projection rank-3 bug:** read the batch dim (`Shape[0]=1`) as token count, collapsing the
  caption to one token + GPU OOB write. Derive length from `ElementCount / lastDim`. Also truncate T5
  padding (unmasked ~120 PAD tokens dilute the caption).
- See CausalConv3d OOB and Concat-graph bloat under [CUDA kernel pitfalls](#cuda--ptx-kernel-pitfalls--toolchain)
  and [GPU residency](#gpu-residency--throughput-cuda) — both first surfaced on video VAEs.

---

## 3D / mesh bugs

- **TRELLIS `SparseDownsample`** = `scatter_reduce(reduce='mean', include_self=True)` → divides by
  **count+1** (zero self is in the mean), not count. A first-principles numpy port made the same mistake so
  it passed — only the real model exposed it.
- **TripoSR:** DINO pos-embed `+0.1`; a CUDA activation-cache reshape-identity bug; the triplane density
  decode host-loop (see GPU residency).
- **Hunyuan3D:** `Concat` per-slice memcpy graph bloat (see GPU residency); timestep `max_period=1000`.

---

## Vision / detection porting traps

- **Don't trust the class name — grep `class XYZ(` for the actual parent.** YOLO11 `C3k` inherits `C3`
  (3 convs, parallel cv1/cv2 both taking input), not `C2f` (expand→split→chain→concat). Wrong parent →
  wrong structure and weight keys.
- **Don't trust a default parameter because "every other call uses it."** Ultralytics `C2f`/`C3k` override
  Bottleneck `e=1.0`, but `C3k2(c3k=False)` uses default `e=0.5` (hidden compresses by half). The loader
  used actual tensor shapes so it didn't error at load — the mismatch surfaced as silent wrong output.
  Dump actual checkpoint tensor shapes per block type and verify against assumed channel widths first.
- **YOLO11 class branch uses depthwise-separable convs** (`DWConv 3×3 + Conv 1×1`) where YOLOv8 used plain
  convs — different key nesting, needs `Conv2dDepthwise` (weight `[C,1,kH,kW]`). Reusable for
  MobileNet/DINOv2 patch embed.
- **Verify what the transformer's `Forward` actually consumes — don't infer from the diffusers pipeline
  ctor signature.** Hunyuan Image 2.1 took a `ClipTextEncoder` param never used (no pooled conditioning);
  the model uses per-token T5-XXL + optional ByT5.

---

## Serving / production

- **`AccessViolationException`/corrupted-state exceptions from native code are uncatchable in .NET Core**
  — the process dies before any handler (incl. exception middleware) runs. Requires external process
  supervision (`Restart=always`), not a code fix. Verified live: a CPU-backend MoE kernel bug killed the
  whole process.
- **Decode-round fault isolation:** an exception escaping `DynamicBatchScheduler`'s decode round previously
  killed the model's background loop silently — no crash, no log, every future request to that model hung
  forever. Wrap the round to fail only that round's sequences.
- **`/ready` must check real state** (`ModelManager.UnhealthyChatModels`/`IsLoopAlive`) — an unconditional
  200 made a dead scheduler loop look healthy. Keep `/health` cheap/dependency-free (k8s split).
- **`PagedKvCache.Gather` scratch realloc bug:** the size check was "changed at all" instead of "too
  small," reallocating a full GPU tensor every decode round for every sequence → grow-only, page-rounded
  fix.
- **CUDA-graph cold-capture bug:** the first-ever capture failed `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`
  due to weight auto-promotion during capture — hidden because every prior test warmed the backend with an
  eager call first.
- **MoE D2H-sync bug (open, high-sev):** olmoe/granite-moe do per-token host routing readback (2193–3225
  syncs/rep), invalidating the tok/s number and blocking graph decode. On-device MoE routing is the fix
  (tracked in ROADMAP).
