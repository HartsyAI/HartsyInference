namespace HartsyInference.Core.Configuration;

public static partial class KnobProfiles
{
    /// <summary>The <c>reference</c> profile: every approximation and fast path pinned to its most faithful setting.</summary>
    /// <remarks>Derived from a per-knob audit of what each setting does to the arithmetic, not from its name. Of the
    /// numerics surface, 33 knobs were judged bit-exact (graph capture, scheduling, pure memory-traffic chunking) and
    /// are not pinned, and 18 change output but have no more-faithful direction — they select which variant loads, or
    /// a model requires them — so forcing those would break a model rather than make it faithful.
    /// <para>Two entries read backwards from their names and are correct as written. <c>bf16Gemv</c> is pinned ON
    /// because the fused GEMV keeps activations F32 while the cuBLAS fallback casts them to BF16, so ON is the more
    /// accurate arm. <c>vkCoopmat2</c> and <c>vkDisableCoopmat</c> are BOTH required: the coopmat2 path is dispatched
    /// first under its own flag and <c>vkDisableCoopmat</c> does not gate it, so pinning only one achieves nothing.</para>
    /// <para>Knobs are pinned even when the value equals the declared default. That is the point: a profile has to beat
    /// whatever the machine has exported, so stating the faithful value explicitly is what makes it authoritative.</para>
    /// <para>Several pins cost VRAM or time rather than saving it — <c>audioLmQuant=off</c>, <c>vaeF32</c>,
    /// <c>seedvr2VaeF32</c>, <c>vaeFullres</c>, and a monolithic HeartCodec decode. That is the intended trade.</para>
    /// <para>Not a promise of bit-exactness against any particular reference implementation, and two things stay
    /// outside its reach: SDPA still auto-tiles under memory pressure regardless of <c>sdpaForceTiled</c>, and
    /// <c>vram.h3ChunkRows</c> decides from free VRAM, so H3 parity runs must pin it per run.</para></remarks>
    private static readonly KnobProfile ReferenceProfile = KnobProfile.Create("reference")
        // Bypass skips the checkpoint's LlmAdapter and feeds raw Qwen-3 hiddens to DiT cross-attn; a diagnostic, not the model path.
        .With(EngineKnobs.AnimaBypassLlmAdapter, false)
        // cuDNN 1D-as-2D audio convs compute F32 output via TF32 tensor cores, corr 0.9999 vs the direct kernel it replaces.
        .With(EngineKnobs.AudioConvCudnn, false)
        // AudioLmQuantPolicy maps "off" to AudioLmQuant.Off (no weight quant); unset applies Q4_K on a single device.
        .With(EngineKnobs.AudioLmQuant, "off")
        // The fused GEMV keeps activations F32 with F32 accumulate; the cuBLAS fallback casts them to BF16, so ON is the accurate arm.
        .With(EngineKnobs.Bf16Gemv, true)
        // Unset yields GuidanceInterval.Always, byte-identical to pre-feature CFG; a spec skips the uncond forward on some steps.
        .With(EngineKnobs.CfgInterval, null)
        // Off keeps the proven 3-GEMM + 2-RMSNorm split; ON swaps in one qkv GEMM plus a fused QkvSplitNorm kernel.
        .With(EngineKnobs.ChromaFusedQkv, false)
        // Routes 2D convs back to im2col+cuBLAS, which honors the engine's TF32/precision policy; cuDNN picks its own algo and math.
        .With(EngineKnobs.ConvCudnn, false)
        // Forces F32 activations in every opted-in DiT block/attention hot path instead of the default F16 (DitDtype.Act).
        .With(EngineKnobs.DitF16, false)
        // The dp4a decode GEMV quantizes activations to Q8_1; the code calls it lossy within the Q8_1 rounding bound.
        .With(EngineKnobs.Dp4aOn, false)
        // Forcing split-K flash (engage floor dropped to 8) adds a cross-split online-softmax merge to every attention call.
        .With(EngineKnobs.FlashSplitForce, false)
        // Keeps attention on the monolithic single-accumulator reduction; the split merge is only claimed exact, never bit-identical.
        .With(EngineKnobs.FlashSplitOff, true)
        // Enables an FP8 GEMM with F16 accumulate, narrowing both the operands and the accumulator.
        .With(EngineKnobs.Fp8F16, false)
        // Enables the FP8 GEMM path; operands are still quantized to e4m3 even though the accumulator stays F32.
        .With(EngineKnobs.Fp8F32, false)
        // Quantizes with the checkpoint's own .input_scale, which is what ComfyUI fp8_scaled references use; absmax is engine-invented.
        .With(EngineKnobs.Fp8StaticInputScale, true)
        // Fusing w1/w3 into w13 REQUANTIZES the fp8 pair to a common scale (per-block fallback above 1/16 error), altering weights.
        .With(EngineKnobs.FusedFfn, false)
        // Selects CUBLAS_COMPUTE_32F_FAST_16F, i.e. F16-mantissa tensor-core math for F32-operand GEMMs.
        .With(EngineKnobs.GemmF16, false)
        // 0 pins one warp per row, removing the shape-dependent 4-way split-K whose shared-memory combine reorders float accumulation.
        .With(EngineKnobs.GemvKsplit, 0)
        // noiseSeed -1 disables NSF additive noise; the code calls this the deterministic source for parity validation.
        .With(EngineKnobs.HiftDeterministic, true)
        // Compute32F then returns plain CUBLAS_COMPUTE_32F, overriding both TF32 and FAST_16F for every F32-operand GEMM.
        .With(EngineKnobs.HighPrecisionGemm, true)
        // Backend doc says it changes numerics: the tiled kernel is an online softmax over the tile's union window, not one dense pass.
        .With(EngineKnobs.Ltx25Na3dTiled, false)
        // F16 rope tables cost a measured real output change (SSIM 0.9956 across the clip); F32 tables are the faithful arm.
        .With(EngineKnobs.Ltx2Ropef16, false)
        // Detours a token-major SDPA call through a permute pair into the INT8 SageAttention kernel.
        .With(EngineKnobs.Ltx2SageTokenmajor, false)
        // Off leaves the MiniMax-Music3 depth decoder at checkpoint precision instead of quantizing it at construction.
        .With(EngineKnobs.Mm3DepthQuant, false)
        // Adds an e4m3 re-quantization of modulated activations; disabling it measured cross-device SSIM 0.9795 against 0.9597 with it on.
        .With(EngineKnobs.ModulateEmitFp8, false)
        // Kill-switch: true clears _allowTf32 so F32 GEMMs keep the full 23-bit mantissa instead of TF32's 10.
        .With(EngineKnobs.NoTf32, true)
        // Off keeps Chroma Radiance's DiT blocks at wide BF16 instead of requantizing them to fp8 during checkpoint conversion.
        .With(EngineKnobs.RadianceFp8, false)
        // =1 feeds the COND audio to the uncond branch as an interim workaround; the reference zeroes it, which silence-audio approximates.
        .With(EngineKnobs.S2vTextCfg, false)
        // SageAttention quantizes Q and K to INT8 (K-smoothed per-row) for QK^T; off keeps the cuDNN/F32 attention routes.
        .With(EngineKnobs.SageAttn, false)
        // The explicit =1 sense; it is the first of two opt-ins that unlock Sage's unsafe F32-to-F16 V narrowing.
        .With(EngineKnobs.SageAttnExplicit, false)
        // Anything other than the literal f16acc keeps SageAttention's F32-accumulate PV; f16acc accumulates PV in F16.
        .With(EngineKnobs.SagePv, null)
        // Accepts materializing F32 V as F16, so any V outside the finite F16 range becomes infinity.
        .With(EngineKnobs.SageUnsafeF32VNarrow, false)
        // Falls back to the materialized F32 score-matrix paths instead of cuDNN's fused flash engine, which picks its own internal math.
        .With(EngineKnobs.SdpaCudnn, false)
        // Would force F16 SDPA on for callers the per-call allowF16 gate deliberately excludes (unbounded-score architectures).
        .With(EngineKnobs.SdpaF16, false)
        // Forces the online-softmax FlashAttention kernel for every SDPA call instead of the materialized F32 route.
        .With(EngineKnobs.SdpaForceFlash, false)
        // Kill-switch: true disables the F16 SDPA path everywhere, including callers that opt in via allowF16.
        .With(EngineKnobs.SdpaNoF16, true)
        // The experimental fused FlashAttention-2 kernel runs its QK^T on TF32 tensor cores.
        .With(EngineKnobs.SdpaV2, false)
        // Builds the SeedVR2 VAE with F32 activations instead of the default CUDA BF16.
        .With(EngineKnobs.Seedvr2VaeF32, true)
        // The speculative verify pass recomputes logits with multi-row GEMMs, so an accepted token's argmax can differ from m=1 decode.
        .With(EngineKnobs.SpecDecode, false)
        // Hand-written HGEMM; TensorCoreGemmTests only bounds avg err < 0.05 vs cuBLAS, so the "bit-exact" doc claim is stale.
        .With(EngineKnobs.TensorcoreGemm, false)
        // Casts VAE weights to F32 at recipe load instead of the default BF16 on BF16-capable backends.
        .With(EngineKnobs.VaeF32, true)
        // Byte-identical to coopmat1 but still an F16 coop-matrix GEMM; it is tried FIRST and vkDisableCoopmat does not gate it.
        .With(EngineKnobs.VkCoopmat2, false)
        // Kills the coopmat1 F16 path so F16 GEMMs fall to the scalar matmul_tiled shaders; only covers coopmat1, so pin vkCoopmat2 too.
        .With(EngineKnobs.VkDisableCoopmat, true)
        // Opts Vulkan Linear into an INT8 dot-product GEMM path.
        .With(EngineKnobs.VkInt8, false)
        // Runs large-M float Linears as per-channel-int8 weight x per-row-int8 activation IMMA; the declaration itself calls it lossy.
        .With(EngineKnobs.W8a8, false)
        // 2 is the FlowUniPC solver order Wan's own scheduler ships; the coercion already sends non-positive values back to 2.
        .With(EngineKnobs.WanSolverOrder, 2)
        // 0 restores the monolithic whole-latent HeartCodec scalar decode, removing every chunk boundary.
        .With(EngineKnobs.HeartcodecScalarChunk, (int?)0)
        // Keeps the single-sequence decode KV cache at F32; F16 storage rounds every cached key and value.
        .With(EngineKnobs.KvF16, false)
        // false never forces the online-softmax tiled path; note the engine still auto-tiles when scores exceed half of free VRAM.
        .With(EngineKnobs.SdpaForceTiled, false)
        // Keeps the full-resolution direct VAE decode; =0 forces always-tiled decoding with tent-blended overlaps.
        .With(EngineKnobs.VaeFullres, true);
}
