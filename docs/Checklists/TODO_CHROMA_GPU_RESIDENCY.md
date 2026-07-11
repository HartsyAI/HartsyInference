# TODO: Chroma DiT GPU-residency rewrite (perf)

**Status:** scoped & de-risked, not yet implemented. **Scope: Chroma-only.**
**Owner:** unassigned. **Created this session.**

## Why
Chroma (and the other DiT image models) is host-CPU-bound, not GPU-bound. A 2-step Chroma@1024
run was **277s wall, of which only ~35s is GPU kernels** (Linear 19.5s, SDPA 12s, measured with
`CUDA_LAUNCH_BLOCKING=1` + the `HARTSY_PROFILE=1` per-op profiler). The other **~240s is host-side**:
`ChromaDoubleStreamBlock`/`ChromaSingleStreamBlock` do reshape / concat / split / QK-norm / RoPE on the
**CPU** via `DiTUtils.*`, `QkNorm.Forward`, `AdaLNModulation`, and `FluxRope.Forward` — every one reads
`(float*)tensor.DataPointer` (forces a GPU→host sync of the FULL Q/K/V tensor) then runs a nested CPU loop.
Per double-block: ~2 LayerNorm + 4 modulate + 4 QK-norm + 6 reshape + 3 concat + 1 split + 2 reshape-back +
1 rope ≈ 20 full-tensor CPU ops, ×19 double + 38 single blocks × forwards → ~56s/forward.

**Ruled out (don't re-chase):** RoPE alone (`HARTSY_SKIP_ROPE=1` saved only 5s); the fp8→F16 weight-cast
cache (its size = param count, so a smaller GGUF still OOMs Chroma on 24GB); tensor allocation (lazy — never
zeroes host mem for GPU-resident tensors). See memory `[[dit-inference-host-overhead]]`.

## Scope — Chroma-only
- Edit ONLY `src/HartsyInference.Diffusion/Models/Denoisers/DiTBlocks/ChromaDoubleStreamBlock.cs` and
  `ChromaSingleStreamBlock.cs`. Both are instantiated only by `ChromaTransformer.cs` (verified) → no other
  model is affected.
- Do NOT edit the shared helpers (`DiTUtils`, `AdaLNModulation`, `FluxRope`, `QkNorm`) — verified Flux and
  Flux2/HiDream/Krea2/Lumina2 use them via their own block classes and must stay byte-identical.
- (Once Chroma is proven, replicate the same pattern into each other DiT model's block class, one at a time,
  each with its own parity check.)

## Template
`Ideogram4Block` is the one already-GPU-resident DiT block — copy its idiom. It uses, end to end:
`backend.RmsNorm`, `backend.Permute0213`, `backend.SliceLastDim`, `backend.AffineBroadcastLastDim`,
`backend.GatedResidualLastDim`, `backend.Silu`, `backend.Mul`, `backend.ApplyRope`, `backend.ScaledDotProductAttention`
— and never touches `DataPointer`.

## Op-by-op conversion (CPU → existing GPU op)
| CPU op in Chroma block | GPU replacement | Notes |
|---|---|---|
| `DiTUtils.LayerNormNoAffine(out,in,B,S,H)` | `backend.LayerNormNoAffine(out,in,1e-6f)` | full tensors, norms last dim — drop-in |
| `AdaLNModulation.ApplyModulation(in,shift,scale,…)` (x*(1+scale)+shift) | `backend.AffineBroadcastLastDim(out,in,scale,shift)` | **verify formula**: Affine = x*scale+shift vs modulation x*(1+scale)+shift → may need scale+1 or a 1+scale variant |
| `_normQ.Forward(out,in,vectors)` (per-head RMSNorm over headDim) | `backend.RmsNorm(out2,in2,_normQ.Weight,_normQ.Eps)` where `in2=in.Reshape([vectors,headDim])` | QkNorm exposes `.Weight`/`.Eps`. See **view-identity caveat** below |
| `DiTUtils.ReshapeToMultiHead([B,S,H·D]→[B,H,S,D])` | `backend.Permute0213(out,in.Reshape([B,S,H,D]),S,H,D)` | Permute0213 = swap dims 1,2 |
| `DiTUtils.ReshapeFromMultiHead([B,H,S,D]→[B,S,H·D])` | `backend.Permute0213(...)` reverse then Reshape | |
| `DiTUtils.ConcatAlongSeqDimMultiHead(txt,img)` | **concat in `[S,H·D]` layout BEFORE the permute** (contiguous row-concat via `backend.ScatterRowsAfter` / copy-with-offset), then `Permute0213` once | concatting in `[B,H,S,D]` is interleaved-by-head — avoid by reordering: norm → concat([S,H·D]) → permute → rope → SDPA → permute → split |
| `DiTUtils.SplitAlongSeqDimMultiHead` | `backend.SliceRows` (row ranges) | reverse of the concat |
| `rope.Forward(q,k,…)` (FluxRope CPU) | `backend.ApplyRope(q,k,cos,sin)` | needs cos/sin as **GPU F32 tensors** and applied on `[B,L,H,D]` PRE-permute (like Ideogram4). The CPU `FluxRope.Precompute` builds `_cosCache/_sinCache` host arrays — upload them to GPU tensors once per resolution |
| final gated residual add | `backend.GatedResidualLastDim` | |

## Hard parts / gotchas
1. **Reorder to make concat cheap**: apply QK-norm + concat in `[S, H·D]` (concat = contiguous row concat),
   THEN `Permute0213` to `[B,H,S,D]`, THEN rope+SDPA, THEN permute back + `SliceRows` split. This avoids a
   per-head interleaved concat that the existing GPU ops don't do directly.
2. **RoPE layout/convention**: `backend.ApplyRope` expects `[B,L,H,D]` and is applied pre-permute; FluxRope
   currently runs post-permute on `[B,H,L,D]`. The rotation is interleaved pairs `(x[2i],x[2i+1])` — confirm the
   GPU `LaunchRope` kernel matches (Ideogram4 is the proof it does for `[B,L,H,D]`). Upload cos/sin to GPU once.
3. **Activation-cache view identity**: `Tensor.Reshape` returns a *new* Tensor object sharing memory. The GPU
   activation cache keys on the Tensor object → passing a `Reshape` view to a backend op may miss the cache or
   leave the original object's cached pointer stale. **Verify**: either reshape the underlying tensor in place,
   or confirm the cache follows the shared buffer. This is the #1 correctness risk.
4. Keep everything **bit-exact** vs the current CPU path.

## Validation procedure (per increment)
1. Build: `dotnet build tests/HartsyInference.Diffusion.Tests/... -c Release -f net10.0`.
2. Correctness: run `ChromaGenerationTests.Chroma_V1_Gpu_512_Cfg5` (currently 30 steps, fp8, 4090) and
   **visually inspect** the output BMP — must be on-prompt "astronaut riding a horse", not noise/garbage.
   (A 6-step run is NOT enough to judge — flow-matching needs ~20-30 steps to converge.)
3. Perf: re-run 2-step with `HARTSY_PROFILE=1 CUDA_LAUNCH_BLOCKING=1`; the wall time should collapse toward the
   ~35s GPU-kernel floor as host ops move to GPU. Compare `/tmp/hartsy_profile.txt`.
4. Regression: once Chroma is converted, run a verified model that shares the helpers as-is (e.g. a Flux test)
   to confirm the shared helpers/verified path is untouched.

## Run recipe (env)
`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (4090) +
`LD_LIBRARY_PATH=<cu13>/lib:/usr/lib/x86_64-linux-gnu` + `-f net10.0` +
`CHROMA_T5XXL_SOURCE=.../SD3/sd3.5_large_fp8_scaled.safetensors`. Profiler: `HARTSY_PROFILE=1` (+ optional
`CUDA_LAUNCH_BLOCKING=1` for true per-op GPU time). Dumps to `/tmp/hartsy_profile.txt`.

## Already landed this session (do NOT redo)
- `HARTSY_PROFILE` per-op profiler in `src/HartsyInference.Cuda/Profiling/NvtxRange.cs` (env-gated, dumps on `CudaBackend.Dispose`).
- `CudaBackend.RmsNorm` F16/BF16 inputs now cast→F32 on-GPU + F32 kernel + cast back (was a CPU loop). Shared backend, correctness win.
- `FluxRope.Forward/ForwardSingle` parallelized with `Parallel.For` (shared, bit-exact). **Temp `HARTSY_SKIP_ROPE=1` diagnostic gate in `FluxRope.Forward` — REMOVE when done.**

## Open follow-ups (separate, not this rewrite)
- GGUF Q4_K transient path crashes `CUDA_ERROR_ILLEGAL_ADDRESS` in Chroma — separate bug.
- Lumina2 converter key-map: `time_caption_embed.timestep_embedder.linear_1.weight` not found.
- Z-Image Turbo fp8mix load: "Unsupported dtype conversion: U8 → F32" (comfy_quant metadata).

---

## CLOSED 2026-07-09 (rounds 1-3 executed)

The block residency rewrite shipped (round 1), cuDNN masked SDPA (round 2), and round 3 landed the
F16 block loop (exact 1/32 residual damp — see `ChromaF16.cs`), the persistent cross-generation
CFG-pair CUDA graph (default-on via `DitStepGraph.EnabledDefaultOn`), the T5 context TRIM (replaces
the SDPA mask entirely), the approximator device RmsNorm, and the two-slot rope cache.
**550s → 28.3s** (Comfy 16.6s). Remaining levers tracked in
`benchmarks/results/image_comfy-vs-hartsy_2026-07-05.md` (2026-07-09 round-3 entry): batched CFG,
Lt algo tuning, VAE F16 tail. The `HARTSY_SKIP_ROPE` diagnostic gate in FluxRope remains (shared file,
harmless, still useful).
