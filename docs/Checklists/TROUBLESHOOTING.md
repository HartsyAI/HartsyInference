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

**Conditioning-inert output (well-formed but off-prompt): bisect the DIFFERENCE, not the state.**
When output is sharp and coherent but ignores the prompt, comparing absolute per-layer states against a
reference will not find it — on a quantized checkpoint those drift a few percent from GEMM ordering alone, which
swamps the signal. Instead run **two different prompts through each engine** and compare the per-layer
`Δ(A,B)`. A block where the reference's `Δ` is large and yours is small is the defect, and the ratio is readable
even when absolute states differ. (LTX-2.5, 2026-08-12: absolute states, gate, `q_shift`, `q_scale` and
`|attn2|` all matched the reference at block 0, while `Δ(A,B)` was 8–13× too weak at every block — the bug was a
timestep fed to the cross-attention's K/V modulation at the wrong scale.)

Partition it first, cheaply, before touching the model:
- Zero the **whole** conditioning. If the output barely changes, the conditioning is inert — the fault is in how
  it is consumed, not in the text encoder.
- Ablate halves of the conditioning (real tokens vs learned registers/padding) to see which half carries.
- Sweep guidance to 1 and to 10. Identical output at both rules out CFG wiring.
- Change the seed. A completely different scene confirms the model is sampling from its prior.

**Before believing ANY single-draw symptom on a generative model: sweep seeds, then check what your metric
actually measures.** Three "bugs" on LTX-2.5 were diagnosed and closed on 2026-08-13, all of them a single
observation with a plausible mechanism attached:

1. *"Audio is 20 dB too quiet."* One seed. The same prompt and settings measure **−54.1 / −25.2 / −37.9 /
   −38.5 / −21.7 dBFS across seeds 1–5**, and the ComfyUI draw it was compared against (−33.7) sits
   mid-distribution. The symptom was an outlier draw. Worse, everyone who "confirmed" it by listening had
   heard clips from that same seed — corroboration that was really one observation counted twice.
2. *"Conditioning is inert (~1% difference between prompts)."* The metric was mean pixel |Δ|, which measures
   **luminance, not semantics**: two dark night scenes score close to each other and far from a bright
   sunset one. It reproduced faithfully across seeds while being uninformative about the question asked.
   A metric can be stable, reproducible, and measuring the wrong quantity — reproducibility is not validity.
3. *"The connector dilutes conditioning (99% learnable registers vs 7% on the 2.3 path)."* Refuted by reading
   the reference: ComfyUI pads to `max(1024, L)` with `registers[p % 128]` and forces an all-attend mask,
   identical to `ReplacePaddingWithRegisters` (`comfy/ldm/lightricks/embeddings_connector.py:282-290`).
   High substitution is what the reference does too, and the reference adheres.

Two rules fall out. **Calibrate the metric against a known-good control before trusting it** — a signature can
be real and reproducible without being pathological (the LTX-2.5 audio latent's "temporal collapse", per-feature
std 0.32 vs a healthy ~1.0, was real, reproducible, and not the defect). And **a retraction needs the same
evidentiary bar as the claim it retracts**: "conditioning is inert" died at n=1, but the follow-up "there is no
conditioning bug at all" was also asserted from two seeds and the third contradicted it — 8 of 9 prompt×seed
combinations adhere, one reproducibly does not. Occasional failure to escape a strong prior is not the same
finding as a routing defect, and neither is the same as "no bug".

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

Organised by theme. The GEMM-acceleration investigation is summarised at the end — its full
run-by-run history is in git and in `ROADMAP.md` §3.

### Dtype & output-buffer selection

- **Kernel-level dtype selection must be a function of the OUTPUT tensor, not the inputs.** FP8→F16 GEMM
  writing into a caller's F32 buffer (2 bytes written, 4 read) → every other element garbage → NaN within
  1–2 ops → all-black Flux. Cast FP8 inputs up to F16 for the multiply but land the result in an
  output-dtype buffer.
- **Tensor-core/coopmat paths must store to whatever dtype the pipeline allocated.** An F16-only output
  guard made the coopmat path inert on Flux (Linears allocate F32 output) — 1.4% "improvement" = never
  launched. Relax to F16-or-F32 with an `OUTPUT_F32` spec-const. Gate coopmat on actual shape enumeration
  (`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`), not extension presence.
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
- **matmul C buffer can't be `writeonly` when `beta≠0`** (residual-add fusion reads C).

### Subgroups, reductions & index math

- **Cross-subgroup reductions drop partials when `gl_NumSubgroups > subgroupSize`.** With `local_size=256`
  on small-subgroup devices (LLVMpipe/older Intel, subgroup 4–8 → 32–64 subgroups), the single-subgroup
  final combine reads only `subgroupSize` partials → wrong GroupNorm/LayerNorm/RmsNorm/Softmax. NVIDIA (32)
  and AMD (32/64) unaffected. Fix: strided fold `for k=SubgroupInvocationID; k<NumSubgroups; k+=SubgroupSize`;
  size `warp_*` shared arrays `[64]`. Alternative: pin `requiredSubgroupSize`≥16.
- **GeGlu flat-midpoint bug (same as the CUDA one):** decompose `outerIdx=i/D; d=i%D` then
  `inputX=outerIdx*2*D+d`; a multi-row `[2,2,2*D]` test is the regression guard.
- **`Span<T>(ptr,(int)(Size/sizeof(T)))` truncates 64-bit element counts** above int.MaxValue (Vulkan
  analogue of the CUDA im2col overflow) — throw instead of returning a partial view.

### Memory, synchronization & buffer lifetime

- **"Every-N-dispatches" auto-flush must respect op atomicity — drain only at op boundaries.** Mid-op
  flush freed shared Q/K/V upload buffers still in use by later SDPA heads → garbage from head 2 onward.
  Fix: an `OpScope`/`EnterOp()` RAII nesting counter; wrap public ops.
- **Deferred-free timeline tag must be `_value + 1`** (the NEXT tick the recording op will signal), not
  `_lastSubmitted`. Tagging with the already-submitted tick lets the next reclaim destroy a still-pending
  buffer → latent UAF.
- **Slab granularity matters more than allocation count near VRAM limit.** 256MB slabs OOM'd mid-load on a
  12GB 3060 (Flux ~12GB); 64MB/8MB slabs fit into post-deferred-free gaps. Add an `OnOutOfMemory`
  drain+retry.
- **Cache-miss upload buffers must be tracked and drained** or each step leaks one `VkBuffer` per miss
  (Mesa validation catches it, the NVIDIA blob doesn't).
- **`VkAccessFlags2` sync2 constants are easy to get wrong** and NVIDIA over-flushes so tests pass anyway:
  `UniformRead=0x08`, `ShaderStorageRead=0x200000000`, `ShaderStorageWrite=0x400000000`,
  `ShaderSampledRead=0x100000000`. A wrong compute-to-compute barrier silently drops the write→read hazard.
  Pin with CPU-only constant regression tests. Same for coopmat sType values
  (`COOPERATIVE_MATRIX_PROPERTIES_KHR=1000506000`, `..._FEATURES_KHR=1000506001` — reverse of intuitive
  order).
- **`nonCoherentAtomSize` flush/invalidate must align both ends:** `start=AlignDown(off,atom);
  end=AlignUp(off+size,atom)` — computing size independently can miss the buffer tail (silent stale data on
  non-coherent/ReBAR drivers).
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

### Missing `VulkanBackend` overrides — the recurring bug class

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
- **`ConvTranspose1d`'s shipped shader has no `groups` field at all** — its inner loop sums over the full
  `cIn` range assuming dense `[cIn, cOut, K]` weights, unlike `conv1d.comp.glsl` which does support
  `groups`. Dispatching a depthwise (`groups>1`) call through it would silently read the wrong weight
  slice. `VulkanBackend.ConvTranspose1d` now throws `NotSupportedException` for `groups != 1` instead of
  producing wrong output; extending the shader for grouped transposed conv (needed for BigVGAN-style
  anti-aliased upsampling) is open work.
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

### Shader toolchain & `build.sh`

- **Older `glslangValidator` (Ubuntu 22.04) rejects `-O`** — ship unoptimized SPIR-V, let the driver
  optimize (~100ms colder first-pipeline build).
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

### Driver & device quirks

- **NVIDIA Linux blob under-reports promoted core features.** 535-series advertises `apiVersion 1.3.x` but
  returns zeros for `shaderFloat16`/`timelineSemaphore`/`bufferDeviceAddress`/`synchronization2`/
  `subgroupSizeControl`/`shaderIntegerDotProduct` through the **consolidated** v1.2/v1.3 feature structs
  (original extension structs return the truth). Fix: trust `apiVersion` for guaranteed-promoted features,
  gated by vendorID (NVIDIA 0x10DE / AMD 0x1002 / Intel 0x8086). Re-check on driver upgrades.
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
- **Push descriptors regressed 7% on NVIDIA** (pool-ring is highly tuned there) — not every "best
  practice" generalizes; kept default-off, opt-in `HARTSYINFERENCE_VK_PUSH_DESCRIPTORS=1` for AMD/Intel.
- **INT8 `dotPacked4x8` overload is selected by operand type:** `uint[]` → unsigned, `int[]` → signed.
  Same bytes, different interpretation.
- **llvmpipe gaps surfaced by a full-suite sweep (2026-07-28), not yet root-caused — tracked, not fixed
  here:** `Backend_Linear_FP8Weight_CachedCast_Matches_Cpu` fails on llvmpipe with large numeric errors
  (maxAbsErr ~7128) rather than gracefully throwing/skipping — `cast_f8e4m3_f16.spv` is unchanged by the
  2026-07-28 rebuild (byte-identical), so this is pre-existing llvmpipe FP8-cast behavior, not a
  regression. Separately, `PipelineCache_PersistsToDisk_AcrossBackends` fails on llvmpipe — the reloaded
  on-disk pipeline cache is 32 bytes (header-only) instead of >1KB, suggesting llvmpipe's
  `vkGetPipelineCacheData` doesn't retain pipeline blobs the same way the NVIDIA driver does. Both pass on
  the 3060 and 4090. Neither blocks NVIDIA-target work; flag for a dedicated llvmpipe investigation before
  treating llvmpipe as a full correctness oracle for FP8/pipeline-cache-dependent paths.
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

### Numerics

- **GLSL has no built-in `erf`** — use Abramowitz & Stegun 7.1.26 (max err 1.5e-7) for exact GELU.

### Testing & profiling

- **Env-var opt-in flags read into `static readonly` fields aren't unit-testable within one test process**
  (xUnit runs many tests in the same process, so the field is already frozen from whatever the env var was
  — or wasn't — set to at first touch of the type). `TryDispatchInt8Linear`'s gate started this way; fixed
  by moving to a settable *instance* property (`EnableInt8Linear`, defaulted from the env var in the
  constructor) matching `CudaBackend.EnableW8A8`'s existing pattern — tests flip it directly
  (`backend.EnableInt8Linear = true`) with no env var or fresh process needed. Prefer this pattern for any
  new opt-in switch that correctness tests need to exercise both on and off.
- **The Wan-video-scale smoke test (`Backend_FlashAttention_WanVideoScale_CompletesWithoutOom`) takes
  hours on llvmpipe** (measured: killed after 2.85 CPU-hours, still running) — software-rasterized
  scalar execution of 16384×24 = 393,216 workgroups × ~512 serial KV-tile iterations each has no
  business running on a CPU emulator. The test now skips itself when `Vk.DeviceName` contains
  "llvmpipe"; it proves no-OOM/finite-output on real hardware, which is not something llvmpipe's
  correctness-only role needs to cover — don't remove the guard to "get more coverage."
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

### Correctness & verification discipline

- **"No NaN" is not "correct".** Forcing F32 activations on Vulkan removed a NaN and was recorded as a
  fix; the image it produced was pure noise. Every end-to-end claim needs the output actually looked at,
  not just checked for finiteness or for not-all-black.
- **Static call-site analysis cannot rule out runtime gaps.** Five real backend gaps were found only by
  re-running the same real-weight generation after fixing the previous one. Budget for iteration.
- **Graph/step capture is model-dependent.** A Vulkan-native capture mechanism (persistent command pool +
  reusable command buffer + fence) was proven correct in isolation but did not fit the model's memory
  profile. Prove the mechanism separately from proving it helps.

### GEMM acceleration: what was tried, and what holds

A long coopmat/performance investigation (2026-07-30 → 08-02, Krea2 on RTX 4090 + 3060 + llvmpipe) is
condensed here to its transferable conclusions. The blow-by-blow — per-run timings, reverted branches,
variance tables — lived in this file and was removed on 2026-08-06; it is in git history, and the standing
writeup is `docs/Checklists/ROADMAP.md` §3 plus `benchmarks/scoreboards/VULKAN.md`.

- **Coopmat1 requires M, N and K to all be exact multiples of 16** — it has no partial-fragment support
  without host-side padding. A DiT's joint sequence is `imgSeq + txtSeq`, and `txtSeq` is the encoded
  prompt length, so it essentially never lands on a multiple of 16 (measured: `imgSeq=4096`, `txtSeq=13`
  → 4109). Result: **0.8% (16/2112) of a real generation's GEMMs reached coopmat**; the rest fell to
  `matmul_tiled`. An F32→F16 compute-dtype gate looked like the blocker and was not — fixing it moved
  engagement by zero. **Shape is the blocker, not dtype.**
- **Do not pad M with a scratch buffer + a separate device-to-device copy.** That design passed a
  CPU-reference correctness test at the real non-aligned M, a 200-iteration no-`Sync()` stress test at full
  scale, and two full CLI runs — then produced `ErrorDeviceLost` on a longer run. Same-queue submission
  order does **not** guarantee memory-visibility ordering across separate `vkQueueSubmit2` calls; only a
  barrier *within* one command buffer does. The safe design is ggml's (`mul_mm.comp`): an `ALIGNED`
  specialization constant selecting between an unconditional fast path and a bounds-checked
  shared-memory-staged load/store, so everything stays inside one dispatch.
- **`VK_NV_cooperative_matrix2` is the one real win.** Workgroup-scope (not subgroup-scope), addressed via
  `tensorLayoutNV` + `coopMatLoadTensorNV`/`coopMatStoreTensorNV` against global memory, with built-in
  bounds clamping (`gl_CooperativeMatrixClampModeConstantNV`) — so M/N/K need not be tile multiples at all,
  which is exactly the coopmat1 blocker above. It is now default-on.
- **Register blocking (ggml's core coopmat optimization) was a net loss here.** The Vulkan-vs-CUDA gap is
  not primarily an arithmetic-intensity problem, so the usual data-reuse lever does not pay.
- **Nsight Compute cannot profile Vulkan** — it is CUDA-only, and pointing it at a Vulkan process reports
  `No kernels were profiled`. `vulkan-validationlayers` (apt) catches API misuse but not throughput.
  Budget for this before planning a Vulkan profiling pass.
- **Never compare benchmark runs across sessions on a shared box.** A reported "~5% coopmat2 regression"
  was cross-session variance, not a property of the code; a controlled same-session comparison reversed the
  sign. Report n, stdev and a Welch's t — a 3.9% mean difference at t≈0.92 is not a result. Also check for
  *other* processes on the GPU (an idle-looking ComfyUI backend held 12.7 GB mid-investigation and
  confounded a whole day of runs).
- **Step-graph capture's retain-everything design causes a one-time VRAM peak, not a leak** — it is
  deterministic and fires identically with the feature on or off. Diagnose from a per-cache breakdown
  (weight / activation / weight-cast / graph-retained), not from a running byte counter.
- **Still open:** full-process wall-clock is ~53× slower than CUDA per marginal denoise step, far beyond
  the ~8–16× isolated GEMM gap. An allocator pooling bug is real and worth fixing but was measured *not* to
  explain the per-GEMM-call cost. Root cause unresolved — see ROADMAP §3.

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
