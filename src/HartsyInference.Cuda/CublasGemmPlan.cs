using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>The dtype a batched cuBLAS product runs in (operands are cast to it) and the dtype cuBLAS writes its output in.</summary>
/// <remarks>
/// <para>cuBLAS accumulates in F32 (<c>COMPUTE_32F</c>). Supported (GEMM dtype → output dtype), with operands F32, F16,
/// BF16, FP8 or quantized as <see cref="ResolveGemmDtype"/> maps them:</para>
/// <list type="bullet">
/// <item>F32 GEMM → F32 output.</item>
/// <item>F16 GEMM → F16 or F32 output.</item>
/// <item>BF16 GEMM → BF16 or F32 output.</item>
/// </list>
/// <para>Anything else (an F32 GEMM into F16 or BF16, an F16 GEMM into BF16, a BF16 GEMM into F16, a non-float GEMM or output) has no
/// cuBLAS kernel here and is refused before any device work.</para>
/// </remarks>
internal readonly record struct CublasGemmPlan(DType GemmDtype, DType OutputDtype)
{
    /// <summary>Plans a batched GEMM for operands <paramref name="a"/> and <paramref name="b"/> writing <paramref name="output"/>.</summary>
    /// <exception cref="NotSupportedException">The operand/output combination has no cuBLAS kernel.</exception>
    public static CublasGemmPlan Resolve(DType a, DType b, DType output, bool highPrecisionGemm, bool fp8F32Gemm, bool fp8F16Gemm)
    {
        DType gemm = ResolveGemmDtype(a, b, highPrecisionGemm, fp8F32Gemm, fp8F16Gemm);
        bool gemmSupported = gemm == DType.F32 || gemm == DType.F16 || gemm == DType.BF16;
        if (!gemmSupported || (output != gemm && output != DType.F32))
        {
            throw new NotSupportedException(
                $"cuBLAS batched GEMM does not support operands {a.Name} x {b.Name} with output {output.Name} (GEMM dtype {gemm.Name}).");
        }
        return new CublasGemmPlan(gemm, output);
    }

    /// <summary>Resolves the GEMM dtype for two operands, preferring F16 over F32 when either is F16 (or fp8, which casts to F16).</summary>
    /// <remarks>cublasGemmEx COMPUTE_32F does not support F32×F32→F16, so when an F16 activation feeds an F16 output
    /// and the weight happens to be F32, the F32 weight must cast down to F16 — F16 also gets Tensor Core
    /// acceleration on Ampere+.</remarks>
    internal static DType ResolveGemmDtype(DType a, DType b, bool highPrecisionGemm, bool fp8F32Gemm, bool fp8F16Gemm)
    {
        // FP8 forces a 16-bit GEMM (Ampere has fast Tensor Cores in F16/BF16, not F32). Pick:
        //  - BF16 when the other operand is F32 — BF16 has F32's full dynamic range, so the
        //    F32→16-bit activation cast cannot produce ±Inf even when SwiGLU's `gated`
        //    intermediate momentarily exceeds 65504 in F32. F16 *would* overflow there
        //    (Z-Image L0 ffnOut had INF=9024 at step 1 of CUDA bring-up before this fix).
        //  - F16 otherwise — keeps the existing F16 fast path that Flux/SDXL FP8 paths rely on
        //    when their activations are already F16 (and therefore in-range).
        if (a.IsFp8 || b.IsFp8)
        {
            // F16 (10-bit mantissa) is more accurate than BF16 (7-bit) for the activation cast; BF16 is the default only
            // because SwiGLU MLPs can momentarily exceed F16's 65504. For GELU-FFN models (Wan) F16 is safe AND needed:
            // over a deep DiT (40 layers) + CFG, BF16's coarser mantissa lets a small per-step velocity bias compound
            // into a diverging trajectory. numerics.fp8F16 opts the fp8 path into F16.
            if (fp8F32Gemm) return DType.F32;
            if (fp8F16Gemm) return DType.F16;
            return (a == DType.F32 || b == DType.F32) ? DType.BF16 : DType.F16;
        }
        // GGUF quants always dequantize to F16 (or BF16 if the other operand is F32). The
        // dequant kernels emit F16 directly; routing through F32 would force an extra F16→F32
        // cast pass for no benefit. Same precedence rule as FP8 above.
        if (a.IsQuantized || b.IsQuantized)
        {
            return (a == DType.F32 || b == DType.F32) ? DType.BF16 : DType.F16;
        }
        if (a == DType.F16 || b == DType.F16) return DType.F16;
        if (highPrecisionGemm && (a == DType.BF16 || b == DType.BF16) && (a == DType.F32 || b == DType.F32))
            return DType.F32;
        if (a == DType.BF16 || b == DType.BF16) return DType.BF16;
        return a == DType.F32 || b == DType.F32 ? DType.F32 : a;
    }
}
