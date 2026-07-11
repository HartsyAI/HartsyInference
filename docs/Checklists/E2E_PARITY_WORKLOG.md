# Image-model e2e + parity grind — session worklog

## ▶▶ NEXT AGENT — HunyuanImage 2.1 bring-up (staged + scoped 2026-07-10, start here)

**All weights staged, no downloads needed:** transformer `Models/Stable-Diffusion/HunyuanImage/HunyuanImage2.1-Q4_K_M.gguf`
(QuantStack, 856 tensors: 20 double + 40 single blocks, Tencent-style keys, no VAE/TE inside);
VAE `Models/VAE/HunyuanImage/hunyuan_image_2.1_vae_fp16.safetensors` (NEW download);
ByT5 `Models/clip/byt5_small_glyphxl_fp16.safetensors` (NEW download);
Qwen2.5-VL-7B TE `Models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` (already present, + full-fp16 sibling).

**PROGRESS 2026-07-10 pt2 — converter DONE + VALIDATED on the real GGUF:**
`HunyuanImageCheckpointConverter.ConvertTencentToDiffusers` implemented (mirrors diffusers
`convert_hunyuan_image_to_diffusers.py`, fetched + condensed this session): full byt5_in/txt_in-refiner/
double/single/final mapping, fused qkv + single-linear1 [3h|4h] splits quant-block-aligned, final-layer
[shift,scale]→[scale,shift] swap, img_in 1x1-conv weight flattened to a [3584,64] Linear (GGUF rank-4 dims
are ggml-reversed — the rank-2 relabel skips it). `HunyuanImageKeyMapper.MatchesByKeys` now also matches
Tencent GGUFs (img_attn_qkv + byt5_in; note the file declares arch 'hyvid' and previously fell through to
FluxKeyMapper identity — worked, but the explicit match is safer). MRE validation: 856 GGUF tensors →
1264 diffusers keys → `HunyuanImageTransformer(V21).LoadWeights` **OK** (scratchpad `hymre/`).

**PROGRESS 2026-07-10 pt3 — loader WIRED + deployed (`44.50-local`); blocked on the pixel-shuffle VAE:**
`HunyuanImageLoader` written + all 6 dispatch sites wired (compiles, routes; `ModelSupport.cs` refusal gate
lifted — the old "T5-XXL stand-in" note was stale, the Qwen2.5-VL path is real). In Swarm: GGUF transformer
converts + loads ✓, Qwen2.5-VL-7B fp8 TE loads ✓ (auto-download entries `SideModels.Qwen25Vl7BHunyuan` /
`HunyuanImageVae` added with hashes; note EnsureSideModel re-downloaded the VAE to `Models/VAE/` root —
the pre-staged `Models/VAE/HunyuanImage/` copy is a deletable duplicate). `VaeConfig.HunyuanImage` fixed
(64-ch latent, [128,256,512,512,1024,1024], scale 0.75289 per ComfyUI `HunyuanImage21`); `ConvertVaeKey`
gained `reverseUpIndices:false` (this file's `up.0` = deepest, opposite of SD-LDM).
**BLOCKER: the decoder is a PIXEL-SHUFFLE VAE, not standard LDM** — per-level upsamplers are
`conv([4·C_next, C, 3, 3]) → PixelShuffle(2)` carrying the channel transitions (up.0 [4096,1024] →
up.1 1024; up.1 [2048,1024] → 512; …; up.4 [512,256] → 128; up.5 has none), resnets are all
channel-preserving (zero shortcuts). The engine's `VaeDecoder`/UpDecoderBlock2D (resnet transitions +
interpolate-conv upsample) cannot represent this → `KeyNotFoundException conv_shortcut`. NEXT: write a
dedicated HunyuanImage VAE decoder (reference: ComfyUI's hunyuan_image VAE / diffusers
`AutoencoderKLHunyuanImage`), swap into `HunyuanImageLoader`, rerun — everything upstream of the VAE is
proven loading.

**The remaining work (in order):**
1. ~~Converter~~ ✅ DONE (see PROGRESS above).
2. ~~GGUF key-mapper match~~ ✅ DONE (byt5_in + img_attn_qkv heuristic; file declares arch 'hyvid').
3. **Extension loader**: new `HunyuanImageLoader` per the ErnieImageLoader/Lumina2Loader anatomy (6 wiring sites),
   GGUF branch per QwenImageLoader/Flux2Loader (native quant + `RelabelRank2ToPyTorchOrder`), Qwen2.5-VL-7B encode
   via the existing `HunyuanImageQwenTextEncoder` (template already matches ComfyUI), ByT5 second stream
   (`TextEmbedDim2` + `encoder_hidden_states_2` — transformer support exists), VAE is 32×/64-ch (own config).
   Arch detection: check what Swarm classes the GGUF as (`DescribeModel`) — may need `EditModelMetadata`.
   **Findings this session:** `HunyuanImageQwenTextEncoder` is COMPLETE (encode template, hidden_states[-3]
   tap, 34-token prefix drop — needs `LlamaStyleEncoderConfig.Qwen2_5_VL_7B` + `Qwen2Tokenizer.EncodeChat`);
   the PIPELINE has NO byt5 wiring (transformer's `encoder_hidden_states_2` is optional — skip ByT5 for the
   first run, wire the glyph branch later); `VaeConfig.HunyuanImage` preset EXISTS but is marked
   "TODO: confirm exact architecture" — the 32×/64-ch VAE (`Models/VAE/HunyuanImage/hunyuan_image_2.1_vae_fp16.safetensors`)
   has never been loaded and is the likeliest debugging front. Do NOT copy `HunyuanImageGenerationTests`
   wiring — it uses the legacy CLIP-L/T5 ctor and `VaeConfig.Flux` (wrong VAE). Scratchpad `hymre/` has the
   validated GGUF→transformer load sequence to lift into the loader verbatim.
4. Deploy → 1024² gen → visual verify → warm bench + scoreboard row. GPU etiquette + flagship gate as always.

**Traps from the Flux.2 Dev bring-up (2026-07-10, memory `flux2-dev-gguf-bringup`):** verify WHICH file the Swarm
model resolver picks when a canonical name exists in multiple roots (THE Dev noise bug was an fp4 file shadowing
the fp8 TE); `HARTSY_TE_PROBE=1` dumps per-layer TE absmax for conditioning debug; model-switch away from a
GGUF pipeline currently NREs in `GpuTransferHelper.OnPromotedHostAccess` (restart Swarm between model switches);
BF16 GGUF tensors must be host-cast to F16 in the loader (transient path skips BF16 weights → blank).

---


## ▶▶ NEXT AGENT — finish HiDream i1 Dev + OmniGen-2 (start here)

This session verified **8 image models e2e** (Qwen-Image, Anima, Lumina-2, Chroma, Krea-2, ERNIE, Boogu, Kandinsky-5) and landed reusable engine fixes (committed): the GPU-residency DiT-block rewrite pattern, fp8 `.weight_scale`-companion remap, SDPA `[B,1,Sq,Skv]` mask broadcast, GroupNormSilu **and** LayerNorm F32-path F16/BF16-affine casts, BF16→F16 transformer cast for blank-image avoidance, and `ClipTextEncoder` CLIP-L raw-EOS pooled. Two models remain; both download + load but need correctness/perf work.

> ### ⏩ UPDATE (next session, 2026-06-30 pt2) — FFN inner-dim bug fixed in BOTH models
> **Environment gotchas found (all real):** (a) no system CUDA 13 — copied the runtime out of Trash to `~/.local/lib/cuda13`; every GPU run needs `LD_LIBRARY_PATH=~/.local/lib/cuda13:$LD_LIBRARY_PATH`. (b) **The device mapping below is BACKWARDS** — CUDA orders fastest-first, so `CUDA_VISIBLE_DEVICES=0` = **4090** (HiDream), `=1` = **3060** (OmniGen2). (c) tests multi-target `net8.0;net10.0` and only net10.0 has a runtime → **pin `--framework net10.0`**. (d) view BMPs via an existing ComfyUI venv's PIL.
> **OmniGen-2 — FIXED (512-nocfg verified: coherent astronaut-on-horse, no artifacts).** ROOT CAUSE of BOTH the blocky-bottom-third AND the wrong-subject: `OmniGen2Transformer.ComputeFfnInnerDim` used Llama's `8/3·dim` (=6912) but the checkpoint's `feed_forward.linear_1` is `[10240,2520]` (= `round_up(4·dim,256)`). SwiGLU buffers were sized 6912 while `backend.Linear` writes N=10240 from the weight → out-of-bounds GEMM writes → tail image tokens corrupted (bottom third) + adjacent-memory corruption (wrong subject). Fix: base `= 4·HiddenSize`. The embeds were fine. **This also likely fixes the 1024-CFG illegal-address** (same overflow at 4096 tokens) — verifying.
> **Found via the proven numerical-diff loop:** dump per-stage (`OMNIGEN2_DEBUG_DIR`) → localize to noise-refiner block 1 → isolated PyTorch reference from cloned `VectorSpaceLab/OmniGen2` (existing ComfyUI venv torch) fed our exact dumped input → sub-component relL2 showed attn matched (0.009) but MLP-out bottom rows = 1.0 → FFN buffer sizing.
> **HiDream i1 Dev — ✅ FIXED (coherent astronaut-on-horse @1024/8-step).** Two bugs, both found via the numerical-diff harness (dump transformer inputs+stages with `HIDREAM_DEBUG_DIR`, PyTorch reference from naked-fp8 weights `.float()`, first-divergence relL2). **Bug 1 (FFN):** `SwiGluForward` sized SwiGLU buffers from computed `ffDim=4·hidden=10240` vs the 6912/3584 weights → overflow; now derives inner dim from `w1.Shape[0]`. **Bug 2 (DOMINANT — the brown cloud):** `HiDreamTransformer.LoadWeights` set `numCaptionProjections = CaptionChannels.Length` (=2), loading only 2 of the **49** caption projections → every Llama layer projected through `caption_projection[0]`, T5 through `[1]`. Diffusers uses 49 (= num_layers+num_single_layers+1): Llama layer `i` → `caption_projection[i]`, T5 → `[-1]`=[48]. Fix: load all 49, project per-block. (`t5_proj` relL2 was 17.86; matched proj[1] not [48].) **HiDream FULLY VERIFIED:** Dev nocfg coherent at full **25-step** (~49s/step, sharp astronaut-on-horse) AND the **1024-CFG path is functional** (4-step smoke test, 2 sequential B=1 forwards, no illegal-address/OOM — the handoff's "1024-CFG illegal address" was a symptom of the same FFN/caption bugs). **GPU-residency perf rewrite KEPT (52s→~29s/step, 44% faster):** block CPU glue → backend ops (SliceLastDim / AffineBroadcastLastDim / GatedResidualLastDim / LayerNormNoAffine). An earlier run OOM'd and was WRONGLY blamed on the rewrite — the real cause was ~550MB of leftover Python CUDA contexts (my own reference-diff scripts) + rustdesk on the 4090 eating the 47.5MB-margin headroom; on a clean 4090 it fits fine. (Rewritten image differs 9/255 from the CPU version — diffusion sensitivity to GPU-vs-CPU LayerNorm float ordering, not a bug.) **Reliability caveat:** intermittent hard-fault crash (~5-10%/forward, no OOM warning) during the fp8 denoise loop under memory pressure — retry-able (25-step crashed once then completed on retry); `pkill -9 -f testhost` + kill dotnet holding GPU VRAM between runs. Memory: `hidream-caption-projection-per-block`.
> **(superseded note) earlier HiDream FFN-only attempt:** Code used `ffDim = 4·hidden = 10240` but HiDream's checkpoint FFN is `6912` (routed experts / text FFN) and `3584` (shared expert) — the Llama `8/3` width. `SwiGluForward` allocated with the computed dim → overflow. Fix: `SwiGluForward` now derives the inner dim from `w1.Shape[0]` (the weight = the GEMM's N), robust for all three FFN paths. **General lesson: never size a SwiGLU/proj buffer from a computed dim — read it from the weight, since `backend.Linear` takes N from the weight.** But 8-step output is STILL a brown cloud — necessary, not sufficient. QK-norm verified correct vs diffusers (RMSNorm over inner_dim, before head reshape). NEXT: HiDream needs the same per-stage-dump numerical diff (vs diffusers `HiDreamImageTransformer2DModel` / `comfy/ldm/hidream`). Top suspects: final `norm_out` scale/shift order, image-only RoPE, 12/6-way AdaLN chunk order, fp8 dequant. See memory `hidream-status-still-incoherent`.

**HiDream i1 Dev (run on the 4090, `CUDA_VISIBLE_DEVICES=1`):** the encoder crash is FIXED (CLIP-L pooled, SD3/SDXL-safe) and `HiDreamBlock.cs` is reverted in the working tree to its original correct-but-CPU-slow version (cpu=27; the half-rewrite committed at `591e6ad` was BROKEN — crashed the forward — do not use it). START by running `HiDream_I1_Dev_Gpu_1024_NoCfg` (drop to ~8 steps for a fast read) to confirm the original block makes a COHERENT image. The load occasionally HANGS during the 17GB-fp8 transformer load (retry; `pkill -9 -f testhost` first to free VRAM). The OOM fix is in (naked fp8 + `CacheWeightCasts=false`). If coherent → redo the `HiDreamBlock` GPU-residency rewrite PROPERLY (cpu→0, copy `BooguImageDoubleBlock.cs`, verify pixels unchanged); if not → numerical-diff vs ComfyUI `comfy/ldm/hidream`. NOTE 1024-CFG may hit the same illegal-address as OmniGen2.

**OmniGen-2 (run on the 3060, `CUDA_VISIBLE_DEVICES=0` — its 8GB F16 fits):** RUNS at 512-nocfg (68s) but has 4 bugs — fix in order: (1) **conditioning first** — it renders the WRONG subject, so verify the precomputed embeds (`OmniGen2/TestEmbeddings/{prompt,negative}.bin`, generated via ComfyUI `clip.encode_from_tokens`) actually match OmniGen2's instruction template + hidden-layer tap vs diffusers `pipeline_omnigen2.py`; (2) the **blocky bottom-third artifact** (full Decode, so it's in the LATENT → transformer corrupts bottom image tokens — check RoPE positions for later tokens, the unverified `OmniGen2Block` rewrite, and whether the ref-image/edit token path is wrongly active in t2i); (3) **1024-CFG `CUDA_ERROR_ILLEGAL_ADDRESS`** (B=2 out-of-bounds); (4) host-bound at 1024 (once-per-forward transformer glue scales with tokens). Use the numerical-diff loop (fixed input → per-component dump ours vs reference → first divergence). **DO NOT run silent poll-loops or `CUDA_LAUNCH_BLOCKING=1` timing runs — that's how the agents stalled this session; bound every step and report findings even if partial.**

---

## ▶ RESUME HERE — state as of 2026-06-30 (fresh-session handoff)

**Nothing is committed** (user does all git manually). All changes are in the working tree. Run recipe: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`(4090)`/=0`(3060) + cuBLAS-12 `LD_LIBRARY_PATH` (cu12 libs, see below) + `-f net10.0 --no-build` + `-- xUnit.ParallelizeTestCollections=false`.

**Verified ✅ this session:** Qwen-Image, Anima, Lumina-2, **Chroma**, **Krea-2 Turbo**, **ERNIE-Image**, **Boogu-Image Base**, **Kandinsky-5 Lite** (8 new). Chroma + ERNIE transformers also numerically verified vs diffusers.
**General engine fix (Kandinsky):** `CudaBackend.LayerNorm` F32-input path now casts F16/BF16 affine weight/bias UP to F32 (was unguarded → near-zero output → noise; same class as GroupNormSilu). Audit every norm op's F32-input affine read when casting a model to F16/BF16.
**OmniGen-2 — 4 distinct bugs (NOT close):** (1) 512-nocfg runs (3060, 68s) but renders WRONG subject (man, not the astronaut-horse the embeds were made for) → conditioning/embeds likely wrong (embeds generated via ComfyUI `clip.encode_from_tokens` for Qwen2.5-VL — may not match OmniGen2's instruction template/tap). (2) BLOCKY bottom-third artifact (full `Decode`, not tiled → it's in the latent → transformer corrupts bottom image tokens; suspect RoPE-positions / unverified `OmniGen2Block` rewrite / ref-token path active in t2i). (3) host-bound at 1024 (9s/step, GPU 4% — once-per-forward transformer glue scales with tokens). (4) **1024-CFG crashes `CUDA_ERROR_ILLEGAL_ADDRESS`** at ~step 5 (B=2 bounds bug). Needs a dedicated multi-bug debug campaign. 3B TE + embeds staged.
**HiDream — encoder FIXED, now crashes in the block:** the `ConcatPooled` NRE is FIXED — `ClipTextEncoder.ExtractPooledOutput`/`EncodePenultimate` now return the raw EOS pooled when there's no `_textProjectionWeight` (CLIP-L), guarded so SD3/SDXL (which load projections) are unchanged. HiDream now gets past encoding + into the denoising loop ("HiDream t2i: 1024x1024" prints). BUT it then CRASHES in the transformer forward (exit 1 @155s, no Step output) — the half-rewritten `HiDreamBlock` (cpu=16, broken intermediate) dies on the first forward. **NEXT: finish the HiDreamBlock GPU-residency rewrite (cpu 16→0) OR revert it to the committed original, then re-run.** Also host-bound until the block is done.

**Reusable ENGINE fixes landed (working tree, uncommitted):**
- `Tensor.SetKeepAlive` + `SafeTensorsLoader`/`GgufLoader` root the mmap handle (fixes GC-unmap AccessViolation). `InternalsVisibleTo ModelHandler` added.
- `GgufModelLoader.RelabelRank2ToPyTorchOrder` (GGUF ggml [in,out]→[out,in]); applied in the Qwen GGUF path.
- `CudaBackend.DequantizeToF32` (test/tooling GGUF dequant) + GGUF dequant verified vs gguf-python (4.6e-4).
- **Qwen final-layer scale/shift swap fix** (`QwenImageTransformer.ApplyFinalLayer`: AdaLayerNormContinuous is `(scale,shift)` scale-first) — was THE Qwen grid bug.
- **Qwen conditioning fix** (test: `Qwen2Tokenizer.EncodeChat` template + `promptDropIndex:34`, not raw-512).
- **`QwenImageBlock` GPU-residency rewrite** (30min→13min; mirrors ChromaDoubleStreamBlock + fused `QwenImageRope.ApplyJoint`).
- **ERNIE BF16-BN-cast fix** (`ErnieImagePipeline` ctor casts bn stats to F32).
- **General SDPA mask-broadcast fix** (`AttentionKernels` already handles full-Sq; `ErnieImageTransformer.BuildAttentionMask` now emits `[B,1,Sq,Skv]`; new CUDA `B·Sq·Skv` broadcast-over-heads branch in `CudaBackend.ScaledDotProductAttention`).
- **Krea-2 fp8 scale-companion fix** (`Krea2CheckpointConverter.RemapTransformerKey`: when renaming fp8 keys, the `.weight_scale`/`.scale_weight` companion MUST be remapped through the same logic as its `.weight` or `ApplyFp8ScaledDequant` can't pair them → scale dropped → weights ~250-900× too big → noise). ⚠ **Check every fp8-key-renaming converter for this** (BooguImageCheckpointConverter next).
- **ERNIE VAE banding fix** (`ErnieImagePipeline.cs:188` DecodeTiled→full `Decode`; tiled per-tile GroupNorm + bad overlap blend caused horizontal bands).
- **GroupNormSilu F32-path BF16-affine cast fix** (`CudaBackend.GroupNormSilu` ~line 1465: the F32-activation path passed BF16/F16 weight+bias straight to the kernel WITHOUT casting → wrong affine. Was THE ERNIE washout. Affects any F32-activations + BF16-affine GroupNorm; F16/BF16 paths already cast). + ERNIE pipeline frees TE weights right after encode (3060 OOM fix).
- Worktree (UNMERGED, pending validation): `agent-a9925eeaa59218004` — Flux2/HiDream/Krea2/Lumina2 block GPU-residency rewrites.

**Downloaded + key-verified (in /tmp, symlinked into Models/ — /tmp won't survive reboot!):**
- Qwen-Image-Edit Q5_K_M GGUF (`QwenImageEdit/`), Krea-2 Base+Turbo fp8 + Qwen3-VL-4B TE (`Krea2/{Base,Turbo}/`), Boogu Base/Turbo/Edit fp8 + Qwen3-VL-8B TE (`Boogu/{Base,Turbo,Edit}/`). All keys verified loadable. TestPaths blocks added (`Krea2`, `Boogu`, `QwenImageEdit`).
- **Kandinsky 5.0 Lite** (BF16 12GB, `Kandinsky5/...Diffusers/{transformer,vae}/`) — ✅ MATCH, precomputed embeds present (`TestEmbeddings/prompt_qwen.bin`+`prompt_clip.bin`) → **READY to test, no fix needed**.
- **HiDream i1 Dev fp8** (17GB naked-fp8, `HiDream/{hidream_i1,vae,clip_l,clip_g,t5xxl,llama_3_1_8b}.safetensors`) — ✅ MATCH, quad-encoder all staged, no scale companions → **READY to test**.
- **OmniGen-2** (F16 8GB, `OmniGen2/{omnigen2,vae}.safetensors`) — staged but BLOCKED: (a) engine key-remap — file ships `time_caption_embed.timestep_embedder.linear_1/2`, `caption_embedder.0/1`, `norm_out.linear_1/linear_2` but `OmniGen2Transformer.LoadWeights` wants `time_proj.0/2`/top-level `caption_embedder`/`norm_out.linear`+`norm` (silent-null via TryGetValue → wrong forward); fix `OmniGen2CheckpointConverter`/`LoadWeights`. (b) needs Qwen2.5-VL-**3B** TE (`qwen_2.5_vl_fp16` 7.5GB, NOT the local 7B) + missing precomputed `TestEmbeddings/prompt.bin`.

**IN FLIGHT / NEXT:**
1. **Krea-2 Turbo: ✅ DONE** (verified `krea2_turbo_1024_20260630_095817.bmp`, std 66.5/grid 0.042, fp8 scale-companion fix above). Remaining: Base/CFG path (shares converter+transformer; CFG anchoring `Krea2Pipeline.cs:100` validation-pending) + GPU-residency (28-step CFG host-bound, item 7).
2. **Boogu Base: ✅ DONE** (`boogu_base_1024_20260630_122132.bmp`, std 97.8/grid 0.038, ~6min @1024-28cfg). Fixes: VAE bare-ldm key remap (`ConvertVaeKey`) + double-block GPU-residency rewrite (`BooguImageDoubleBlock`). Test `BooguImageGenerationTests.cs` (+`512_Fast`). **Turbo** = same path, needs a run; **Edit** needs the Qwen3-VL vision tower. ⚠ Boogu converter is pass-through so NO fp8 scale-companion bug.
3b. **STATUS UPDATE (agents stalled — driving directly):** Kandinsky-5: OOM+perf FIXED, runs 512/25-step in 52s on 4090, but the BF16+CacheWeightCasts=false path gave a BLANK image → switched to **F16 cast** (std 5.6→70.2, weights now applied) → but now NOISE = a real transformer CORRECTNESS bug (needs numerical diff vs ComfyUI; same class as Qwen final-layer/Chroma). HiDream: block HALF-rewritten (27→16 sites, broken intermediate — finish or revert before running); OOM handled (fp8+CacheWeightCasts=false, naked fp8). OmniGen2: block rewritten (32→0) but needs the 3B TE + embeds. **Lesson: the remaining models each need a correctness diff, not just download+run.**
3. **Kandinsky-5 / HiDream / OmniGen-2 — ALL IN PROGRESS.** Kandinsky-5 (4090): fixing the OOM (12GB BF16 transformer cast to 24GB F32 → keep BF16 + CacheWeightCasts=false) + perf/correctness. HiDream Dev (4090-when-free): same fp8-F32-cast OOM fix (17GB→34GB) + verify. OmniGen-2 (3060, 8GB fits): key-remap fix (file ships `time_caption_embed.timestep_embedder`/`caption_embedder.0/1`/`norm_out.linear_1/2`; LoadWeights wants old names) + Qwen2.5-VL-**3B** TE download + embeds. **RECURRING PERF/VRAM PATTERN for fp8/BF16 models: tests do `CastWeightsToF32` on the big transformer → 2× VRAM → OOM; fix = pass quantized weights to LoadWeights + `CacheWeightCasts=false` (transient per-GEMM cast).** Operational: `pkill -9 -f testhost` between runs (orphans hold VRAM → false OOM).
4b. **Qwen-Image-Edit / Boogu-Edit** — TODO: build the EDIT pipeline (VAE-encode ref image + Qwen-VL multimodal conditioning); transformers load but no edit pipeline/TestPaths exist.
4. **ERNIE — ✅ DONE** (verified `ernie_image_v1_512_cfg_20260630_103703.bmp`, std 60.9/grid 0.069). 4 bugs fixed (BN-cast, SDPA mask, VAE banding, GroupNormSilu F32-path BF16-affine cast — the washout). Remaining: GPU-residency (CPU-bound blocks, item 7).
5. **Chroma checkpoint**: only `/tmp/chroma_dl/Chroma1-HD-fp8mixed-final.safetensors` remains (the `do_not_use/…exp` variants + alternates were deleted to reclaim 137GB; the symlink points to the good one). **Disk policy (user-approved): verified-✅ models' /tmp downloads may be deleted when low on space** — keep only the file the Models/ symlink uses.
6. **HunyuanImage 2.1: BLOCKED** on 24GB (35GB bf16; fp8/GGUF repacks use incompatible original-Tencent keys). Deleted from /tmp.
7. **GPU-residency block rewrite (PERF — ~27s/forward + GPU ~7% util = host-bound CPU glue).** Done: Qwen, Chroma, **Boogu** (double-block rewrite → 7%→72% util, ~6min/1024; NOTE the per-block bottleneck was only the DOUBLE blocks — Boogu's single blocks were already GPU-resident, and the transformer's remaining ~27 CPU sites are once-per-forward glue, not per-block). **STILL PENDING: Krea-2, ERNIE, Flux2, HiDream, Lumina2** (apply the same — check whether the CPU glue is in the double/single block class or once-per-forward before rewriting).
**GLUE SCAN (run this FIRST per model — `grep -cE "for \(int|float\*|DataPointer" <file>` vs Backend ops; BLOCK files = per-block hot path that MUST be GPU-resident, TRANSFORMER files = mostly once-per-forward [minor], ROPE files = intentionally CPU):**
  - NEED block rewrite: `ErnieImageBlock` cpu=36, `Lumina2Block` cpu=38, `Flux2DoubleBlock` cpu=33 / `Flux2SingleBlock` cpu=29, `OmniGen2Block` cpu=32, `HiDreamBlock` cpu=27, `Krea2Block` cpu=15.
  - ALREADY GPU-resident (skip): `Kandinsky5Block` cpu=6/gpu=50, `BooguImageSingleBlock` (done), `QwenImageBlock`/`ChromaDoubleStreamBlock` (done).
  - Transformer once-per-forward glue (don't bother unless still slow after block fix): Lumina2Transformer 54, ErnieImageTransformer 46, Kandinsky5Transformer 40, HiDreamTransformer 38, OmniGen2Transformer 36, Krea2Transformer 25. Pattern: CPU for-loops over batch×seq×hidden reading DataPointer → `IBackend` GPU ops (RmsNorm/LayerNormNoAffine/Concat/Permute0213/SliceRows/AffineBroadcastLastDim/GatedResidualLastDim); copy `QwenImageBlock.cs`/`ChromaDoubleStreamBlock.cs`/`BooguImageDoubleBlock.cs`. **Verify the rewrite preserves pixels (perf-only).**

---

Goal: systematically take every **unverified** image model (🔬/🔧) to visual e2e + Python parity on this
box (RTX 4090 24 GB = CUDA index 1, RTX 3060 12 GB = CUDA index 0). Per-model loop: download official
weights (ungated Comfy-Org/community mirror if gated) → dump safetensors keys → check C# logic vs the
official reference → run the gated generation test → **inspect the output image myself** → if wrong, dump
a Python layer reference and diff until the first `avg_err>1e-3` layer → fix C# → re-run → document the
bug. Final bug entries go to `PARITY_VERIFICATION.md` §Bugs; status flips in `MODEL_STATUS_IMAGE.md`.

GPU selection: tests hard-code `deviceOrdinal:0`, so pick the card with `CUDA_VISIBLE_DEVICES`
(`=1` → 4090, `=0` → 3060). Big/dual-transformer models → 4090; ≤8 GB models → 3060 (frees 4090 for a
parallel run).

## Environment ready
- dotnet 10.0.109; diffusion test project builds clean (Release). Solution multi-targets `net8.0;net10.0`
  but only net10 SDK is installed → **always run tests with `-f net10.0`** (net8 testhost errors out).
- **CUDA cuBLAS version is the dominant env issue.** Driver is 13.0, but `CudaLibraryResolver` resolves
  `cublas` by trying `.so.13` → **`.so.12`** → `.so.11` (engine's validated target is **cuBLAS 12**, per the
  `CublasApi` doc). Forcing cuBLAS 13 (cu13 libs on the path) loads but throws **`CUBLAS_STATUS_NOT_SUPPORTED`**
  on some GEMM configs (hit on Flux Dev) and is the suspected cause of the bad AuraFlow/Chroma images.
  **Use cuBLAS 12** by putting ONLY the cu12 libs on the path (so `.so.13` isn't found → resolver falls back to 12):
  `export LD_LIBRARY_PATH="/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI/venv/lib/python3.12/site-packages/nvidia/cublas/lib:/home/hartsy/Desktop/Swarm/SwarmUI.not too old/dlbackend/ComfyUI/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/lib/x86_64-linux-gnu"`
  (cuBLAS 12 + cudart 12 run fine on the CUDA-13 driver via forward-compat.) `libcuda.so` (driver API) is system.
  **NOTE: all earlier results in this doc used cuBLAS 13 and are suspect — re-run on cuBLAS 12.**
- GPU select: **must set `CUDA_DEVICE_ORDER=PCI_BUS_ID`** (CUDA's default order ≠ nvidia-smi; without it
  `CUDA_VISIBLE_DEVICES=1` wrongly picked the 3060). With it: `CUDA_VISIBLE_DEVICES=1` → 4090, `=0` → 3060.
  Tests hard-code `deviceOrdinal:0`. AuraFlow fp8 (+T5) OOMs the 3060's 12 GB → use the 4090.
- HF CLI venv: `/home/hartsy/hfvenv/bin/hf`. No HF token (gated → Comfy-Org/community mirror).
- `Models/` wired via symlinks to local SwarmUI weights (see below) + Qwen3 tokenizer downloaded.

## Per-model status (this session)

| # | Model | Tier | Weights | e2e | Parity | Notes |
|---|---|---|---|---|---|---|
| 1 | AuraFlow v0.3 | ✅ | ✅ `calcuis/aura` fp8 (self-contained) | **✅ on-prompt** | ✅ visual | **DONE @1024 → clean photoreal horse + mounted rider, on-prompt ("an astronaut riding a horse").** Fixed by TWO changes: (a) T5 attention scale 1.0 (was 1/√64), (b) Pile-T5-XL tokenizer `pile_t5xl_spiece.model` (was wrongly fed `t5_xxl_spiece.model`, different vocab). Denoiser/VAE/scheduler were always correct. Image: `Output/auraflow_v03_1024_cfg_20260629_105400.bmp`. NOTE: transient path @1024 took 26 min — slow; optimize later. Rider reads soldier-not-astronaut (AuraFlow v0.3 weakness + crop), composition correct. |
| 2 | Chroma | 🔧 | ✅ fp8 (`silveroxides/Chroma1-HD-fp8-scaled`, downloaded mxfp8mixed + fp8_scaled variants) | pending | — | Logic-checked: 2 fixes needed before run — (a) CFG should be **uncond-anchored** (`ChromaPipeline.cs:272` uses cond-anchored), (b) T5-XXL `AttentionScale=1.0` (`Xxl` preset unset → 0.125). **Trap: `Xxl` is shared with ✅ Flux/SD3 — give Chroma its OWN XXL config, don't edit the shared one.** mod-table/approximator wiring verified correct. |
| - | Anima (Cosmos-Predict2 2B) | ✅ | local fp32 | **✅ on-prompt @512 on 3060** | visual | Clean coherent "anime girl with blue hair in a garden" (`anima_t2i_512_nocfg_20260630_000534.bmp`), 93.9s/25-step on the 3060. Precomputed Qwen3-0.6B embeds. Verified e2e this session. |
| - | Qwen-Image | ✅ | ✅ Q4_K_M GGUF (`QuantStack/Qwen-Image-GGUF`, 13 GB) | **✅ clean photoreal on-prompt astronaut-on-horse @1024** (`qwen_image_v1_1024_gguf_20260630_020418.bmp`) | ✅ visual | **GGUF downloaded + pre-flight verified** (arch `qwen_image`, bare diffusers keys, no `model.diffusion_model.` prefix, no fused QKV; Q4_K/Q5_K/Q6_K all have GPU dequant kernels). 4 earlier fixes (scheduler maxSeqLen 8192/maxShift 0.9, `shift_terminal=0.02`, RoPE `scale_rope` centering, lazy GGUF load). **THREE NEW BLOCKER BUGS FIXED THIS SESSION (each crashed the run in sequence):** (1) **mmap dangle** — `Tensor` borrowing an mmap pointer didn't root the `MmapHandle`, so a GC during the 1933-tensor GGUF convert finalized it and unmapped the fp8 text-encoder mid-load → AccessViolation in `ApplyFp8ScaledDequant`. Fixed engine-wide: `Tensor.SetKeepAlive` + `SafeTensorsLoader`/`GgufLoader` root the handle. (2) **GGUF shape transposed** — `GgufLoader` emits ggml `[in,out]` dims; the image GGUF path never relabeled to PyTorch `[out,in]` (the LLM path does), so the first Linear (timestep embedder) got K=3072>input256 → M=0 → zero-grid kernel launch `CUDA_ERROR_INVALID_VALUE`. Fixed: shared `GgufModelLoader.RelabelRank2ToPyTorchOrder` applied in the image GGUF branch. (3) **VRAM OOM** — test never set `CacheWeightCasts=false`, so it cached an F16 dequant of every weight of a ~20B model (~40 GB). Fixed: transient per-GEMM dequant on the GGUF path → stable 15.3 GB resident on the 4090. **PERF BUG FOUND + FIXED:** `QwenImageBlock.Forward` ran its entire glue (LayerNorm/AdaLN-modulation/QK-norm/reshape/concat/split/gated-residual) on the CPU via `DiTUtils`/`QkNorm`/`AdaLNModulation` — each forced a GPU→host→GPU round-trip of every ~50 MB intermediate (~16 ops × 60 blocks × CFG = thousands of PCIe syncs/step → 10+ min, GPU util ~15%). Rewrote it GPU-resident mirroring the verified `ChromaDoubleStreamBlock` (interleaved RoPE fused into one joint CPU pass `QwenImageRope.ApplyJoint`). Result: **30+ min host-stalled → ~13 min denoise + 37s VAE, GPU util 15%→72%**. **BUT the image is garbage** (periodic woven grid, not the prompt) — a *pre-existing* Qwen correctness bug (first-ever completed run; rope-positions/unpatchify/VAE/fp8-conditioning never validated — block rewrite is faithful per-op to the old CPU path, so not the cause). **Next: Qwen parity loop** (diffusers reference, per-component diff) + further perf (per-GEMM Q4_K dequant repeated every step is the remaining ~13 min; needs fused dequant-GEMM or bounded F16 cache).

**UPDATE (2026-06-30) — perf FIXED, correctness still open, bug localized to the transformer forward:**
- **Perf root cause = `QwenImageBlock.Forward` ran all glue on CPU** (DiTUtils/QkNorm/AdaLNModulation reading `DataPointer` → D2H per ~50MB intermediate, ~16×60blocks×CFG). Rewrote GPU-resident mirroring verified `ChromaDoubleStreamBlock` + fused interleaved RoPE into one joint CPU pass (`QwenImageRope.ApplyJoint`, **bit-exact** via `QwenImageRopeFusionTests`). 30+min host-stalled → **~13min, GPU util 15%→72%** (remaining cost = per-GEMM Q4_K dequant, HBM-bound).
- **Conditioning bug FOUND + FIXED (real, needed):** test fed the RAW prompt **padded to 512** through `Qwen3Tokenizer`, no template, no prefix-drop. diffusers `_get_qwen_prompt_embeds` requires the ChatML encode-template + real length + drop 34. Fixed: `Qwen2Tokenizer.EncodeChat(prompt, qwenImageSystem, addGenerationPrompt:true)` + `promptDropIndex:34` → condHidden seqLen **512→12**. (The codebase already had this pattern in `HunyuanImageQwenTextEncoder`.)
- **Image STILL a woven grid at 28 steps with correct conditioning** → conditioning was NOT the grid cause. **Systematically ruled out:** Q4_K GPU dequant (new `DequantizeToF32` + `QwenGgufDequantTests`: matches gguf-python to 4.6e-4), block math (faithful to Chroma), VAE (`QwenImageVaeSmokeTests`: constant→smooth, gradient→smooth gradient, only faint banding), scheduler (standard `x+=v·dt`), pack/unpack (exact inverse, diffusers `(c,py,px)` order), rope GPU/host coherence (activation-cache evict-on-read works). **Remaining: the transformer produces a too-weak/wrong velocity (step-0 std 0.57) → latent stays noise → grid.** Needs a per-component reference dump of the DiT forward (ComfyUI same-Q4_K-weights or diffusers) to localize (rope position VALUES vs diffusers QwenEmbedRope, img_in/txt_in/time_embed, or final AdaLN-continuous layer are the un-diffed suspects).
- VAE has a minor faint-banding artifact (follow-up, not the dominant bug).
- **ROOT CAUSE FOUND + FIXED (2026-06-30, via ComfyUI `comfy/ldm/qwen_image/model.py` component diff):** the final layer `ApplyFinalLayer` (norm_out / AdaLayerNormContinuous) read `shift=firstHalf, scale=secondHalf` — but diffusers/ComfyUI `LastLayer` does `scale, shift = chunk(emb, 2)` (**scale first**). The per-block AdaLayerNormZero IS [shift,scale,gate] (ours was right there), but the final continuous norm is [scale,shift] — they differ. Swapped → velocity std collapsed to 0.57, latent stayed noise → woven grid. **Fix → noisePred std 0.57→1.29, finalLatent std 1.6→0.65, grid→coherent image** (6-step shows clear subject+sky). ComfyUI diff also CONFIRMED matching: rope position ids (img `row/col - len//2`, txt `max(h//2,w//2)+s`), patchify `(c,py,px)`, attention `[txt,img]` concat, block modulation order, SiLU+Linear mod, timestep sinusoid (cos-first, scale 1000). 28-step clean-image confirmation running. See [[adaln-continuous-scale-shift-order]]. |
| - | Flux.2 Dev (32B) | 🔧 | local Q4 gguf | blocked | — | no gen test + 32B > 24 GB |
| - | Ideogram 4 | 🔬 | local nvfp4 single-file | blocked | 🔬 1e-7 | test wants diffusers folder layout + ≥22 GB (4090 only); local copy is single-file nvfp4 — download Comfy-Org folder layout |
| - | Kandinsky 5.0 Lite | 🔧 | ⬇ transformer 12 GB + VAE (`kandinskylab/...T2I-Lite-sft-Diffusers`) | **embeds ready, dl-ing** | — | Embeds generated from on-disk Qwen2.5-VL-7B fp8 (dim 3584, `hidden_states[-1][:,41:]`) + CLIP-L pooled (768) — `dump_kandinsky5_embeddings.py`. Transformer+VAE downloading (text encoders skipped — embeds precomputed). Denoiser-only run after dl. |
| - | OmniGen2 | 🔧 | local fp16 (7.9 GB) | **10-fix plan ready** | — | `NotImplementedException` was STALE — Forward is structurally wired. Real blockers (ref `comfy/ldm/omnigen/omnigen2.py`): **(1) 4 wrong weight-key names** in `LoadWeights` (`caption_embedder`→`time_caption_embed.caption_embedder.1`, time_proj→timestep_embedder.linear_1/2, norm_out.linear→linear_1, proj_out→norm_out.linear_2, delete fabricated norm_out.norm) → null→NRE; (2) missing caption pre-RMSNorm (caption_embedder.0); (3) FFN inner-dim 10240 not 6912; (4) final norm = affine-free LayerNorm eps 1e-6 not RMSNorm; (5) timestep sinusoid width 256 not 2520; (6) timestep double-scaled (pass t=1-sigma); (7) verify velocity sign; (8) CFG needs neg embeds (test passes null→ArgumentException). Mostly S/M. Defer to focused session. |
| - | Anima | 🔧 | local (Cosmos-Predict2 2B) | **RUNNING on 3060 @1024** | — | Unblocked: precomputed Qwen3-0.6B embeds + T5 token-id `.bin` files generated (`tests/python-reference/encode_anima_prompt.py`, last_hidden_state dim 1024). Fits 3060 (MinReq 8 GB). Inspect on completion. |
| - | ERNIE-Image | 🔧 | ⬇ FP8 (`rootlocalghost/ERNIE-Image-FP8` transformer + `Comfy-Org/ERNIE-Image` ministral-3-3b TE, ~15.75 GB) | **downloading, fixes applied** | — | Real e2e (bundled Ministral-3B encoder). Runnable on 4090 w/ FP8. **2 test fixes applied:** (a) `CacheWeightCasts=false` (was `PreloadWeights` → OOM), (b) wired VAE BatchNorm un-norm (`bn.running_mean/var`, eps 1e-4 — was silently skipped → wrong latent scale). MUST use `model.`-prefixed Comfy TE (baidu TE has `language_model.model.` prefix → KeyNotFound). Compiles. Run when dl done + GPU free. |
| - | HiDream i1 | 🔧 | partial (CLIP-L+VAE+llama-tok staged) | **DEFERRED — needs A100/H100** | — | Encoders unconditionally upcast to F32 → ~54 GB peak (T5 19 + Llama 32 + CLIP 3) + 30 GB test gate. Won't fit 24 GB without keeping encoders fp8-on-GPU (larger code change). Arch verified correct (real top-2 MoE, not single-expert fallback). 30 GB transformer dl NOT pulled. Latent bugs noted: T5/CLIP `LoadWeights` skip `TextEncoderQuantNormalizer.Normalize` (fp8_scaled → wrong); worked around w/ non-scaled T5 + fp16 CLIP. |
| - | Lumina2 | 🔧 | local transformer + Flux.1 VAE (symlinked) | **ready to run** | — | Gemma-2 2B embeds generated (`hidden_states[-2]`, dim 2304, `encode_lumina2_prompt.py`). VAE unblocked: symlinked the Flux.1 16-ch `ae.safetensors` (bare keys verified, `VaeConfig.Flux`). 2B → fits 3060. Run after Anima frees the 3060. |

## ⚠️ BASELINE VALIDATED — resolution was the red herring
- **SDXL @ 1024×1024 produces a clean, on-prompt lion-in-sunflowers on the 4090.** Engine, both GPUs,
  cuBLAS 12 AND 13 are all sound. (cuBLAS 12-vs-13 made NO difference for SDXL — identical output; the
  Flux `NOT_SUPPORTED` is an fp8-transient-path-specific issue, separate.)
- **Root cause of the earlier "everything is broken" panic: testing 1024-native models at 256²/512².**
  SDXL@256 = blocky primary-color garbage; SDXL@1024 = flawless. These DiT/UNet models are 1024-trained and
  degenerate badly at low res. **RULE: always test image models at 1024×1024** (512 only for quick smoke,
  never 256). AuraFlow@512 (green banding) and Chroma@512 (noise) must be re-judged at 1024.
- **Validated run recipe:** `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (4090) +
  `LD_LIBRARY_PATH=<cu13>/lib:/usr/lib/x86_64-linux-gnu` + `-f net10.0`, gen at 1024².

## Engine change this session (broadly useful)
- **`CudaBackend.CacheWeightCasts`** (new, default true) — set `false` for low-VRAM: forces transient per-GEMM
  fp8/quant→F16 weight cast instead of a resident cache. Cuts large-fp8-DiT weight VRAM ~3×. This is the
  lever that lets AuraFlow (and other big fp8 models) fit the 4090. `CudaBackend.cs` ~line 57 + the
  `LinearImpl` cache branch (~line 332).

## Local weight symlinks created under Models/
SD15, SDXL (sd_xl_base_1.0), SD3.5-Large fp8, Flux-dev fp8, Z-Image-Turbo fp8, Qwen-Image fp8,
Ideogram4 nvfp4, OmniGen2 fp16, Anima, Lumina2, Flux2-dev Q4 gguf; text encoders (qwen3-4b, qwen3-0.6b,
qwen2.5-vl-7b, mistral-fp8, umt5); VAEs (flux2-vae, qwen_image_vae).

## ⚡ DiT PERF: GPU-residency rewrite (root cause of ~40-min runs)
**Definitive diagnosis** (profiler `HARTSY_PROFILE=1` in `NvtxRange` + `CUDA_LAUNCH_BLOCKING=1`): a 2-step Chroma@1024 run = 277s, of which **GPU kernels are only ~35s** (Linear 19.5s, SDPA 12s). The other **~240s is host-side**: the DiT blocks do reshape/concat/split/QK-norm/RoPE on the **CPU** via `DiTUtils.*` (all read `(float*)tensor.DataPointer` → D2H sync of the full Q/K/V tensor + nested CPU loops), `QkNorm.Forward` (CPU), and `FluxRope.Forward` (CPU). Per double-block that's ~12 full-tensor `DiTUtils` ops + 4 QK-norms + rope, ×19 double+38 single blocks × forwards.
**Red herrings ruled out:** RoPE alone (`HARTSY_SKIP_ROPE=1` saved only 5s); weight-cast cache (F16 cache size = param-count, not GGUF size → OOMs Chroma on 24GB regardless of quant); tensor alloc (lazy — never zeroes host mem for GPU-resident tensors). See [[dit-inference-host-overhead]].
**Template:** Ideogram4 (the one fast DiT) does everything via GPU ops — `backend.RmsNorm`/`Permute0213`/`SliceLastDim`/`AffineBroadcastLastDim`/`GatedResidualLastDim`/`ApplyRope`, never touches `DataPointer`.
**Rewrite spec (per DiT block, existing GPU ops unless noted):**
1. QK-norm `_normQ.Forward` → `backend.RmsNorm(out, q_viewed_[*,headDim], _normQ.Weight, _normQ.Eps)`.
2. `DiTUtils.ReshapeToMultiHead([B,S,H,D]→[B,H,S,D])` → `backend.Permute0213(out, in, S, H, D)`.
3. Joint concat/split: concat txt+img in **[S, H·D] layout BEFORE the permute** (contiguous row-concat via `ScatterRowsAfter`/copy-with-offset), permute after; split = `SliceRows`. Avoids interleaved-by-head concat.
4. `FluxRope.Forward` (CPU) → `backend.ApplyRope` (GPU): needs cos/sin as GPU tensors + apply on `[B,L,H,D]` pre-permute (like Ideogram4).
5. Modulation/residual already could use `AffineBroadcastLastDim`/`GatedResidualLastDim`.
**Risk:** `DiTUtils`/`FluxRope` are shared by verified Flux + Flux2/HiDream/Krea2/Lumina2 — but the rewrite only edits `ChromaDoubleStreamBlock`/`ChromaSingleStreamBlock` (used ONLY by `ChromaTransformer`), so it's **Chroma-only** and can't regress other models. Keep changes bit-exact; replicate per-model afterward with a CPU-vs-GPU parity test. Profiler kept (`NvtxRange`), RmsNorm F16 GPU-path fix landed. Temp `HARTSY_SKIP_ROPE` gate in FluxRope to remove.
**→ Full actionable TODO with op-by-op conversion table + gotchas + validation steps: [`TODO_CHROMA_GPU_RESIDENCY.md`](TODO_CHROMA_GPU_RESIDENCY.md).**

## Bugs found (promote to PARITY_VERIFICATION.md once confirmed)
- **AuraFlow / Pile-T5-XL — attention scale `1/√64` instead of `1.0` (FIXED).** `T5TextEncoderConfig.PileT5Xl`
  didn't set `AttentionScale`, so `T5Block` fell back to `1/√head_dim`. All T5/UMT5/Pile-T5 use scale=1.0
  (sibling `T5Base` already does). Wrong scale across all 24 encoder layers → corrupted conditioning. Caught
  by official-source review (diffusers/ComfyUI). Same latent bug confirmed in the shared `Xxl` preset (affects
  Chroma; must be fixed via a Chroma-specific config to avoid touching verified Flux/SD3).
- **AuraFlow — final `norm_out` "missing LayerNorm": FALSE POSITIVE.** A research pass flagged it, but the
  official `AuraFlowPreFinalBlock` genuinely applies no LayerNorm before `x*(1+scale)+shift`. C# was correct.
  (Lesson: verify agent findings against the real upstream source before editing.)
- **AuraFlow — QK-norm uses RMSNorm; official is `qk_norm="fp32_layer_norm"` (non-affine LayerNorm). NOT the
  bug — image came out clean+on-prompt with the existing RMSNorm path once tokenizer+scale were fixed.** The
  "horizontal banding" that triggered this suspicion was a low-res artifact (512 on a 1024-native model).
  Leave as-is unless a future layer-diff shows drift; checkpoint ships no `norm_q/k` weights either way.
- **AuraFlow / Pile-T5-XL — WRONG TOKENIZER (FIXED, was the conditioning bug).** Test fed `t5_xxl_spiece.model`
  (the UMT5/T5-XXL SentencePiece) to a Pile-T5-XL encoder, which uses a **LLaMA-derived 32000-vocab SP model
  with different token IDs** (`pile_t5xl_spiece.model`). Same prompt tokenized to entirely different ids
  (astronaut/horse → `[385,29132,20546,364,4821,263,10435]` pile vs `[46,30059,7494,3,9,4952]` xxl), so the
  model rendered a clean-but-off-prompt image. Fix: `TestPaths.PileT5XlSpiece` now defaults to the real pile
  model file. Confirmed byte-identical (md5 `eeec4125…`) to EleutherAI/pile-t5-xl's `spiece.model`. **Combined
  with the T5-scale-1.0 fix, AuraFlow v0.3 now produces correct on-prompt output → promoted to ✅.**
