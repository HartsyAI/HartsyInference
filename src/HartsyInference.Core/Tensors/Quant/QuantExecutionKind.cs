namespace HartsyInference.Core.Tensors.Quant;

/// <summary>How a backend runs a recipe-carrying weight.</summary>
public enum QuantExecutionKind
{
    /// <summary>A GEMM that consumes the packed weight and its block scales directly.</summary>
    NativeBlockScaled,

    /// <summary>A per-tensor-scaled FP8 GEMM.</summary>
    NativePerTensorFp8,

    /// <summary>A fused dequant GEMV for single-token decode.</summary>
    FusedGemv,

    /// <summary>Dequantize to BF16 into a bounded workspace per call, then a normal GEMM.</summary>
    DequantBf16,

    /// <summary>Dequantize to F32 on the host (reference and CPU backend).</summary>
    DequantF32Host,
}
