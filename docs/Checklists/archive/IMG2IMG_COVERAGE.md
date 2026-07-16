# img2img / Inpaint / Edit — per-pipeline coverage

Production-readiness matrix for image-to-image, masked inpaint, and reference-image edit across every image
diffusion pipeline. Patterns: img2img = `Img2ImgSetup.Prepare` → VAE-encode source → noise at `plan.StartStep`
→ denoise (strength=0 pass-through); inpaint = "blend-on-vanilla" (`MaskBlendUtilities.DownsampleMaskAreaAverage`
→ per-step re-noised-source blend on the unmasked region → pixel-space `BlendChannelsInPlace` recomposite;
packed-latent pipelines route the mask through `PackLatentMask2x2`).

Updated 2026-07-02 (round 1 complete: CPU wiring tests green; GPU real-weight quality runs of the new paths pending).

| Pipeline | img2img | Inpaint | Edit | Notes |
|---|---|---|---|---|
| StableDiffusion15 | ✅ | ✅ (new) | — | Real-weight CPU e2e test passes (fresh `v1-5-pruned-emaonly` download — the old symlink was dangling). |
| SDXL | ✅ | ✅ | — | The reference blend-on-vanilla implementation. |
| SDXL Refiner | ✅ (cross-model) | — | — | Pixel-space polish over any base output. |
| SD3 | ✅ | ✅ | — | No dedicated img2img test yet. |
| Flux | ✅ | ✅ | ✅ Kontext | Reference-image tokens appended per step. |
| Flux2 | ✅ | ✅ (new) | — | Custom 16-rounded-dims validation; packed mask at featDim=128; fixed a noise-tensor leak. |
| Chroma | ✅ (new) | ✅ (new) | — | New VaeEncoder ctor overload (Flux 16-ch VAE). |
| ChromaRadiance | ✅ (new) | ✅ (new) | — | Pixel-space: source noised directly, no VAE; padded border choice needs real-checkpoint eyeball. |
| ZetaChroma | ✅ (new) | ✅ (new) | — | Inherits the checkpoint's VALIDATION-PENDING sampling recipe. |
| Z-Image | ✅ | ✅ (new) | — | |
| Anima | ✅ | ✅ (new) | — | |
| Qwen-Image | ✅ (new) | ✅ (new) | ✅ Qwen-Image-Edit (r2) | `editRefImage`: ref latent 2×2-packed, appended tokens w/ frame-1 RoPE, dropped after proj_out (ComfyUI `index` method). **VL vision-token half deferred** — needs a Qwen2.5-VL vision encoder (the existing Qwen3-VL one is a different arch); callers pass text-only edit-template tokens meanwhile. Single-ref only. |
| Lumina2 | ✅ (new) | ✅ (new) | — | Flux VAE; velocity negation orthogonal to noising math. |
| Krea2 | ✅ (r2) | ✅ (r2) | — | QwenImageVaeEncoder, unpacked 16-ch, flow-match. |
| ERNIE | ✅ (r2) | ✅ (r2) | — | Flux2-config VaeEncoder → patchify → BN-normalize (symmetric with decode). |
| AuraFlow | ✅ (r2) | ✅ (r2) | — | AuraFlow-config VaeEncoder, 4-ch. |
| F-Lite | ✅ (r2) | ✅ (r2) | — | Custom dynamic-shift integrator; mixes via shared `Img2ImgSetup.MixAtSigma`; inpaint blends the accumulator. |
| Kandinsky5 | ✅ (r2) | ✅ (r2) | — | Flux-config VaeEncoder (normalization inside the encoder — differs from its decode wiring, documented on ctor). |
| LanceImage | ✅ (r2) | ❌ by design | — | 32× effective downscale → mask cells are 32×32 px; inpaint throws NotSupportedException. |
| Ideogram4 | ✅ (r2) | ✅ (r2) | — | Token patchify + inverse fixed-constant latent norm; reversed-loop integration (t: 0→1); packed-channel-inner mask blend. |
| Boogu | ❌ | ❌ | ✅ EditFromEmbeddings | Triple-guidance edit path; no plain img2img. |
| SdxlInpaint (9-ch UNet) | — | ❌ stub | — | Superseded by SDXL blend-on-vanilla; dedicated 9-ch path unimplemented. |
| HunyuanImage | ❌ | ❌ | — | Skipped: no encoder for its 32×/32-ch VAE + model itself blocked e2e (incompatible checkpoint keys). |
| HiDream, OmniGen2 | ❌ | ❌ | ❌ | HiDream numerically broken; OmniGen2 edit out of scope per its docs. |

## GPU quality validation (`Img2ImgGpuQualityTests`, 2026-07-02, real weights on the 4090)

| Test | Verdict |
|---|---|
| SDXL img2img + inpaint | ✅ strength-ordering (diff 38→51); inpaint (production `RecompositeAtEnd=true`) border **exact 0.00**, center 41.1 changed. The latent-blend-only mode's ~40/255 border drift is **root-caused as pure SDXL VAE roundtrip loss on textured content** — the per-step blend is exact at the latent level (0.000000/step, `HARTSY_SDXL_DEBUG=1` instrumentation) and `SdxlVaeRoundtrip_Gpu` measures decode(encode(generated image)) = 36.79/255 standalone (smooth gradient = 8.34). Not a blend bug; recomposite default exists precisely for this. |
| Flux img2img + inpaint | ✅ strength-ordering (17→37); inpaint with `RecompositeAtEnd=false` (latent blend under test) border 14.35 (< 20 tuned for fp8 VAE round-trip), center 35.4. |
| Qwen-Image-Edit | ✅ ref-conditioning causally proven (same seed with/without editRefImage → clearly divergent, both coherent). Unlocked by two real-checkpoint fixes to `QwenImageVaeEncoder`: the probed downsamples schedule (channel widening lives in shortcut RESIDUAL blocks — indices 2/5/8 are resamples) and 2-D `resample.1.weight` handling (native rank-4, no temporal slice). |
| Chroma img2img | ✅ strength-ordering clean (diff 8.3 → 21.5 vs source), coherent at both strengths (fp8 checkpoint + SD3.5-bundle T5). |

## Open

1. Remaining round-1/2 paths without real-weight GPU quality runs (SD3, Z-Image, Anima, Lumina2, Krea2, ERNIE,
   AuraFlow, F-Lite, Kandinsky5, Lance, Ideogram4) — same harness pattern extends per checkpoint availability.
   Lance/Ideogram4 base T2I numerics are themselves first-run pending.
2. Qwen2.5-VL vision encoder → complete the Qwen-Image-Edit VL prompt half; multi-ref (Edit-Plus).
3. ERNIE ctor doc says BN stats are `[1,32,1,1]` but decode applies them to the 128-ch patchified latent —
   stale doc, encode side follows the decode order.
