# SeedVR2 Architecture — One-Step Diffusion Video Restoration

> **STATUS: IMPLEMENTED & VERIFIED (2026-08-01).** All 7 port phases parity-gated (window partition exact;
> preprocess 2.3e-6; VAE ≤2.9e-6 real-weight; DiT trace ≤8.9e-4; E2E vs Python SSIM 0.99950 / 56.6 dB).
> 7-clip real-footage matrix green on the 4090 (~14 s/frame @ 960×540-area, 17.1 GB peak). Surfaces:
> `hartsy restore`, `hartsy video --restore`, `/v1/native/restore[/stream]`, SwarmUI "Video Restore" group.
> Evidence: `MODEL_STATUS_VIDEO.md` (SeedVR2 rows) + `PARITY_VERIFICATION.md` (§VIDEO + bugs ledger).
> Open follow-ups: fp32 720p-area activation ceiling (bf16/tiled VAE), host-math DiT perf pass, publish
> converted weights to the catalog repo. §2.5 below records the decoded semantics the port depends on.

Research notes for porting ByteDance-Seed **SeedVR2** (ICLR 2026) into HartsyInference.
Everything below was read from the reference implementation at
`github.com/ByteDance-Seed/SeedVR` (Apache 2.0), not from the paper prose — the paper
and the shipped inference code disagree in two places that matter (see *Corrections*).

| | |
|---|---|
| Task | Video/image restoration — upscale, deartifact, denoise |
| Variants | 3B (`seedvr2_ema_3b.pth`, 13.6 GB fp32) and 7B |
| License | Apache 2.0 |
| Steps | **1** (`sample_steps=1`, `cfg_scale=1.0`) |
| Text encoder | **None at inference** — frozen `pos_emb.pt` / `neg_emb.pt` |
| Target package | `HartsyInference.Video` (pipeline) + `HartsyInference.Diffusion` (DiT/VAE) |

---

## 2. Verified architecture

### 2.1 Top-level flow

```
LR video (t,h,w,3), uint8 → /255 → [0,1], TCHW
  → AreaResize(max_area = res_h*res_w, BICUBIC, downsample_only=False)   ← SEE §2.2
  → clamp [0,1]
  → DivisibleCrop(16,16)          center-crop to multiples of 16
  → Normalize(0.5,0.5)            → [-1,1]
  → rearrange t c h w -> c t h w
  → cut_videos()                  pad w/ last frame until (t-1) % 4 == 0
  → VAE encode ────────────────────► latent_blur (t', h/8, w/8, 16)     t' = t/4
  → cond   = concat(latent_blur, ones(...,1))        → (t', h', w', 17)
  → noise  = randn_like(latent_blur)                 → (t', h', w', 16)
  → DiT input = concat(noise, cond)                  → (t', h', w', 33)  == vid_in_channels
  → NaDiT(vid, txt=pos_emb, timestep=T)              → (t', h', w', 16)  == vid_out_channels
  → euler step (v_lerp, 1 step)
  → VAE decode ────────────────────► HR video
  → trim to ori_length             undo cut_videos padding
  → optional wavelet_reconstruction(sample, resized_input)  ← low-freq/color transplant
```

`vid_in_channels: 33` decomposes as **16 noisy latent + 16 conditioning latent + 1 mask
channel**. Confirmed in `projects/video_diffusion_sr/infer.py::get_condition`:

```python
cond = torch.zeros([t, h, w, c + 1], ...)
if task == "sr":
    cond[:, ..., :-1] = latent_blur[:]
    cond[:, ..., -1:] = 1.0
```

For the `sr` task the mask channel is **constant 1.0** — it exists to share weights with
the i2v/v2v tasks and carries no information here. Do not skip it; the projection weights
expect the channel.

VAE `scaling_factor: 0.9152`. VAE runs in bf16.

The reference also moves the DiT to CPU (`runner.dit.to("cpu")`) across the VAE encode
and decode calls — peak DiT and peak VAE memory never coincide. Worth mirroring on the 3060.

### 2.2 Preprocessing — the resize is load-bearing

**SeedVR2 is not an upsampler. It is a detail restorer that runs at the target
resolution.** The upscaling is done by a plain bicubic resize *before* the model sees
anything; the DiT's job is to repair what bicubic can't invent.

```python
NaResize(resolution=(res_h * res_w) ** 0.5, mode="area", downsample_only=False)
#   → AreaResize(max_area = res_h * res_w)
#     scale = sqrt(max_area / (h * w));  bicubic
```

Defaults are `res_h=1280, res_w=720` → `max_area = 921600`. Aspect ratio is preserved;
the *area* is normalized. `downsample_only=False` is what makes it upsample small inputs —
the inline comment says it outright: *"Upsample image, model only trained for high res."*

This is the same normalization the window partition applies (`sqrt((45*80)/(h*w))` in
`window.py`). Skip or mis-scale this step and the token counts land outside the training
regime, window sizing goes wrong, and output is a near-no-op or garbage. It is not
optional preprocessing — it is part of the model contract.

Then `DivisibleCrop((16, 16))` center-crops to multiples of 16 (8× VAE × 2× patch), and
`Normalize(0.5, 0.5)` maps to [-1, 1].

**Frame-count constraint.** `cut_videos` pads by repeating the last frame until
`(t - 1) % 4 == 0` — i.e. valid frame counts are **1, 5, 9, 13, 17, 21, 25, …**, the
causal-VAE 4× temporal compression requirement. `t == 1` (image input) short-circuits.
`ori_lengths` is captured before padding and the output is trimmed back:
`if ori_length < sample.shape[0]: sample = sample[:ori_length]`.

Note `wavelet_reconstruction` transplants from the **resized** input, not the original.

### 2.3 NaDiT (3B) — `configs_3b/main.yaml`

| Param | Value |
|---|---|
| `num_layers` | 32 |
| `vid_dim` | 2560 |
| `heads` × `head_dim` | 20 × 128 |
| `emb_dim` | 15360 (= 6 × vid_dim) |
| `expand_ratio` | 4 |
| `patch_size` | `[1, 2, 2]` (spatial-only patchify) |
| `mlp_type` | `swiglu` |
| `norm` / `qk_norm` / `vid_out_norm` | `fusedrms` (RMSNorm) |
| `ada` | `single` |
| `qk_bias` | `False` |
| `mm_layers` | 10 |
| `rope_type` | `mmrope3d`, `rope_dim` 128 |
| `window` | `(4, 3, 3)` on all 32 layers |
| `window_method` | alternating `720pwin_by_size_bysize` / `720pswin_by_size_bysize` |
| `txt_in_dim` | 5120 → linear → 2560 |

`mm_layers: 10` means `shared_weights = not (i < 10)`: **layers 0–9 have separate
video/text weight sets; layers 10–31 share one set across both streams.** Layer 31
(`is_last_layer`) drops the text branch of mlp/ada entirely (`vid_only=True`). Getting
this wrong is the most likely silent weight-loading bug in the port.

### 2.4 Windowed attention — the only novel piece

`models/dit_v2/window.py` is **83 lines of pure index arithmetic**. No CUDA, no Apex. It
emits a list of `(slice_t, slice_h, slice_w)` triples:

```python
scale = math.sqrt((45 * 80) / (h * w))          # normalize to a 720p token budget
resized_h, resized_w = round(h * scale), round(w * scale)
wh, ww = ceil(resized_h / 3), ceil(resized_w / 3)
wt     = ceil(min(t, 30) / 4)
```

Window *count* is fixed at `(4,3,3)`; window *size* scales with input resolution so that
token count per window stays near the 720p training regime. Boundary windows are ragged
— they fall out of `min((i+1)*w, W)` clamping, not a special case.

The shifted variant offsets by half a window (`st,sh,sw = 0.5`) and adds one window per
axis. Layers alternate regular/shifted, giving cross-window information flow — standard
Swin, in 3D, on NaViT-flattened tokens.

**Per-window sequence composition** (`NaSwinAttention.forward`): each window's attention
sequence is `[window's video tokens ; ALL text tokens]`. The text tokens are *replicated
into every window*. So it's joint attention within (window ∪ text), not cross-attention.

**RoPE is applied after windowing, against `window_shape`** — position indices restart at
zero inside each window. This is the detail that makes the port tractable: no global
position bookkeeping, no offset table. Text q/k get 1D `lang` RoPE (θ=10000) replicated
3× across axes; video temporal axis is offset by text length (`vid_freqs[l:l+f, :h, :w]`).

Sizing at 720p (1280×720): latent 160×90 → patchified 80×45. Windows are ~15×27 spatial ×
`wt` temporal. A 100-frame clip → 25 latent frames → `wt = ceil(25/4) = 7`, so
**~2.8k tokens per window, ~36 windows per layer**.

### 2.5 Decoded implementation semantics (verified against reference code + probes, 2026-08-01)

Recorded here so the port never re-derives them. All empirically verified.

**AdaSingle modulation** (`modulation.py`): shared emb (b,15360) is laid out `(d l g)` — for channel d,
layer l∈{attn=0,mlp=1}, gate-component g∈{shift=0,scale=1,gate=2}: element index = d·6 + l·3 + g.
Per-block learned vectors combine additively: mode "in" → `hid·(scaleA+scaleB) + (shiftA+shiftB)`;
mode "out" → `hid·(gateA+gateB)`. No "1+scale" at runtime — the +1 is baked into the learned scale init.

**Tail `vid_out_ada` cache-collision (LANDMINE, verified by probe).** With `layers=["out"]` the naive
reshape is dimensionally impossible (crashes standalone). In the real forward it works ONLY because the
`emb_repeat_0_vid` cache entry — stored by every block's attn-layer ada — is hit by the tail's idx=0
lookup. Effective trained semantics: `vid_out = vid·(emb_attn_scale + out_scale) + (emb_attn_shift +
out_shift)` — the tail reuses the ATTN slice of the emb, not an "out" slice. Port this rule directly.

**SwiGLU** (`mlp.py`): `proj_out(silu(proj_in_gate(x)) · proj_in(x))`, no biases;
hidden = round_up(2·dim·4/3, 256) = 6912 for dim 2560.

**TimeEmbedding** (`embedding.py`): sinusoidal dim 256, `flip_sin_to_cos=False`, `downscale_freq_shift=0`
(sin half first, then cos) → Linear(256→2560) → SiLU → Linear(2560→2560) → SiLU → Linear(2560→15360).

**Patchify** (`patch_v1.py`): tokens are channels-LAST. `b c (T t)(H h)(W w) → b T H W (t h w c)` with
patch order t-outer, h, w, c-inner; for 1×2×2: token vec = [c@(h0,w0), c@(h0,w1), c@(h1,w0), c@(h1,w1)],
then Linear(132→2560). PatchOut mirrors (Linear 2560→64, inverse rearrange). Latents already arrive
channels-last from `vae_encode` (`rearrange b c ... -> b ... c`).

**Attention block flow** (`mmattn.py` NaSwinAttention): per-branch fused QKV Linear (no bias) → window
gather of vid tokens → per-branch RMS qk-norm (head_dim 128, affine) → window-local mmrope3d (video: axial
freqs over (1024,128,128) table sliced `[l:l+f,:h,:w]` — temporal OFFSET BY TEXT LENGTH l; text: 1D lang
RoPE θ=10000 dim 42, repeated ×3 → 126 of 128 dims... freqs (21,) per checkpoint) → per-window sequence
[vid_window ; full text] → varlen attention → unconcat → scatter back → per-branch proj_out (with bias).
Text tokens are REPLICATED into every window. RoPE positions restart per window (window_shape, not global).

**VAE decoded semantics** (all parity-verified at ≤3e-6 relL2 vs real weights):
per-frame GroupNorm stats (`(b t) c h w`, NOT clip-wide); zero spatial conv padding (not replicate);
temporal causal pad = 2× frame-0 replicate; downsamplers pre-pad (0,1,0,1) zeros then stride-2 pad-0 conv;
upsampler = 1×1×1 upscale_conv → channel-to-space `(x y z c)` order → drop output frame INDEX 1 when
temporal (T→2T−1; LTX drops index 0 — different) → k3 causal conv; encoder emits mean‖logvar (32ch, no
quant convs), runner SAMPLES posterior (`use_sample: True`); latent → DiT is channels-last ×0.9152.

**v1 NaDiT (7B, `models/dit`) — decoded diff vs v2 (verified against reference code + the real
checkpoint, 2026-08-02; ported, tiny-config parity 8.96e-4).** Everything above transfers EXCEPT:
(1) **RoPE is a different function**: `RotaryEmbedding(freqs_for="pixel", max_freq=256)` →
freqs = `linspace(1,128,10)·π` (matches `blocks.*.attn.rope.rope.freqs`, shape [10]); per-axis rot
width 2·10 = 20 → **60 of 128 head dims rotated** (68 pass through); positions are
`torch.linspace(−1,1,steps=extent)` over each WINDOW axis (per-window normalization — ragged boundary
windows get different angles for the same local index; `steps=1` → `[-1]`, NOT 0); applied to
**video q/k only** — text is never rotated; no text-length temporal offset. Same interleaved-pair
convention and fp32 rotation as v2. (2) Plain GELU-**tanh** MLP with biases (hidden 4·dim = 12288),
not SwiGLU. (3) All 36 blocks fully split — no `mm_layers`, zero `.all.` keys (express as
`MmLayers == NumLayers`). (4) No `vid_out_norm`/`vid_out_ada` (bare `vid_out`) and NO last-layer
text asymmetry — block 35 computes the text branch in full (the v2 `vid_only` quirks must not fire).
Discriminator: plain-MLP + no-tail signature in `SeedVr2Config.Detect`. 7B dims: 3072 wide, 24 heads,
36 layers, emb 18432, txt_in 5120→3072 bare Linear; window/patch/VAE/scheduler identical to 3B.

**Bugs the parity harness caught so far** (for the PARITY_VERIFICATION ledger):
1. torchvision AA bicubic kernel is a=−0.5 (PIL-compatible), not torch's non-AA −0.75 → 0.18 maxAbs.
2. ATen computes resize weights in float32 (scalar_t) — double-precision weights drift linearly with
   output index (3.4e-5 at i≈1000).

### 2.6 Dependency audit

| Reference dep | Needed? | Replacement |
|---|---|---|
| `flash_attn_varlen_func` | No | Group windows by identical shape → batched SDPA. Ragged only at boundaries, so expect 2–4 distinct shapes per layer. |
| `apex` `FusedLayerNorm`/`FusedRMSNorm` | No | Pure optimization; code already has a `diffusers.RMSNorm` fallback. Engine has RMSNorm. |
| `rotary_embedding_torch` | No | ~40 lines; axial freqs + `apply_rotary_emb`. |
| `einops` | No | Rearranges are all reshape/permute. |
| Sequence-parallel ops (`gather_seq_scatter_heads_*`) | No | Identity at `sp_size=1`. Delete. |

**Nothing in the inference path requires a custom kernel.**

---

## 3. Corrections to the published description

Two things the paper/README say that the shipped code contradicts. Both change the port cost.

1. **"Three frozen pretrained text encoders."** True at training time. At inference,
   `inference_seedvr2_3b.py::_extract_text_embeds` builds the prompt strings and then
   never encodes them — it `torch.load`s `pos_emb.pt` / `neg_emb.pt`. The prompt is not
   user-controllable and no encoder ships in the HF repo.
2. **"Minimum 1× H100-80GB."** That's full-clip, no VAE tiling, no offload. It is not a
   floor on the method.

---

## 4. VRAM budget

3B weights: 13.6 GB fp32 → **~6.8 GB bf16**. VAE ~0.5 GB bf16.

| GPU | Strategy |
|---|---|
| **4090 (24 GB)** — primary target | bf16 resident, no offload, no swapping. ~16 GB headroom for activations. Handles 720p short clips directly. |
| **3060 (12 GB)** — secondary | 7.3 GB resident leaves ~4 GB. Viable for short clips with VAE tiling; W8A8 would halve DiT residency. |
| **7B** | ~16 GB bf16 weights. Off the table for the 3060; tight on the 4090 once activations are counted. Register it in the catalog, but 3B is the default and the only variant to target initially. |

On the block-swap approach the ComfyUI port uses: because this is a **one-step** model,
swapping N blocks to CPU means a full weight round-trip over PCIe amortized across
*one* forward pass. The transfer becomes the runtime. Prefer tiling and shorter clips
over block swap; treat block swap as a last resort on the 3060.

**Quantization caution.** The engine's W8A8/SmoothQuant path applies here in principle,
but restoration fidelity *is* the product — quant error surfaces directly as visible
artifacts, unlike generation where it hides in sampling variance. Given the unresolved
SmoothQuant e2e regression (SSIM 0.9210 → 0.9144), land bf16 first and treat
quantization as a separate, independently-validated follow-up.

---

## Implementation and use

The port and consumer surfaces exist. Use current CLI help and [video status](../Checklists/MODEL_STATUS_VIDEO.md) for supported requests and verification; the completed phased plan and proposed CLI examples are in git.

## 7. Reference file map

| Concern | Reference path |
|---|---|
| Inference entry | `projects/inference_seedvr2_3b.py` |
| Conditioning / runner | `projects/video_diffusion_sr/infer.py` |
| DiT top level | `models/dit_v2/nadit.py` |
| Transformer block | `models/dit_v2/nablocks/mmsr_block.py` |
| Windowed attention | `models/dit_v2/nablocks/attention/mmattn.py` |
| Window partition | `models/dit_v2/window.py` |
| RoPE | `models/dit_v2/rope.py` |
| NaViT flatten/window helpers | `models/dit_v2/na.py` |
| Patchify | `models/dit_v2/patch/patch_v1.py` |
| VAE | `models/video_vae_v3/` + `s8_c16_t4_inflation_sd3.yaml` |
| Config | `configs_3b/main.yaml`, `configs_7b/main.yaml` |
| Preprocessing transforms | `data/image/transforms/{na_resize,area_resize,divisible_crop}.py` |
| Frame padding (`cut_videos`) | `projects/inference_seedvr2_3b.py` L178–197 |
