namespace HartsyInference.Core.Backends.Norm;

/// <summary>How a <see cref="NormModulateCall"/> normalizes each row before its affine and modulation.</summary>
public enum NormKind
{
    /// <summary>No normalization: the row is modulated as it is.</summary>
    None,

    /// <summary><c>(x − mean(x)) · rsqrt(var(x) + eps)</c> with the biased variance.</summary>
    LayerNorm,

    /// <summary><c>x · rsqrt(mean(x²) + eps)</c>.</summary>
    RmsNorm,

    /// <summary><c>x · sqrt(n) / max(‖x‖₂, eps)</c>: an RMS norm whose epsilon floors the L2 norm (Wan 2.2's <c>RMS_norm</c>, <c>F.normalize</c> scaled by <c>sqrt(C)</c>).</summary>
    L2Norm,
}
