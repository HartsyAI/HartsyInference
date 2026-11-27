using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Normalization: plain, affine, channel and fused residual/norm forms, with host reference defaults where a backend declares none.
public partial interface IBackend
{
    /// <summary>Splits channels into <paramref name="groups"/> groups, normalizes each over channel+spatial, then applies the per-channel affine weight/bias.</summary>
    void GroupNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps);

    /// <summary>Normalizes over the last dim (zero mean, unit variance) then applies the affine weight/bias.</summary>
    void LayerNorm(Tensor output, Tensor input, Tensor weight, Tensor bias, float eps);

    /// <summary>RMS-normalizes over the last dim (no mean subtraction) then scales by weight: <c>out = x / sqrt(mean(x²)+eps) · weight</c>.</summary>
    void RmsNorm(Tensor output, Tensor input, Tensor weight, float eps);

    /// <summary>AdaIN 1D: per-(batch,channel) normalize <c>[B,C,T]</c> over T, scale by <c>(1+gamma)</c>, shift by beta.</summary>
    /// <param name="gamma"><c>[B, C]</c> (or <c>[C]</c>, broadcast across batch), typically a Linear projection of a style/speaker embedding.</param>
    /// <param name="beta"><c>[B, C]</c> (or <c>[C]</c>, broadcast across batch), typically a Linear projection of a style/speaker embedding.</param>
    void AdaInstanceNorm1d(Tensor output, Tensor input, Tensor gamma, Tensor beta, float eps);

    /// <summary>Fused residual-add + RMSNorm: <c>residOut = a + b; normOut = rmsnorm(residOut) · weight</c>, one pass instead of two.</summary>
    void AddRmsNorm(Tensor residOut, Tensor normOut, Tensor a, Tensor b, Tensor weight, float eps)
    {
        Add(residOut, a, b);
        RmsNorm(normOut, residOut, weight, eps);
    }

    /// <summary>RMSNorm that may also emit a Q8_1 sidecar, so the dp4a GEMV skips its own quantize; semantically = <see cref="RmsNorm"/>.</summary>
    void RmsNormEmitQ8(Tensor output, Tensor input, Tensor weight, float eps)
        => RmsNorm(output, input, weight, eps);

    /// <summary>Residual-add + RMSNorm that may emit a Q8_1 sidecar (<see cref="RmsNormEmitQ8"/>); semantically = <see cref="AddRmsNorm"/>.</summary>
    void AddRmsNormEmitQ8(Tensor residOut, Tensor normOut, Tensor a, Tensor b, Tensor weight, float eps)
        => AddRmsNorm(residOut, normOut, a, b, weight, eps);

    /// <summary>Sandwich-norm fusion: <c>residOut = a + rmsnorm(b)·w1; normOut = rmsnorm(residOut)·w2</c>, one launch.</summary>
    /// <remarks>Replaces the post-attention norm, residual add, and pre-FFN norm; may emit a Q8_1 sidecar for
    /// <paramref name="normOut"/> (see <see cref="RmsNormEmitQ8"/>). Backends without a fused kernel fall back to the exact composition.</remarks>
    void NormAddRmsNormEmitQ8(Tensor residOut, Tensor normOut, Tensor a, Tensor b, Tensor w1, Tensor w2, float eps)
    {
        Tensor n1 = new(b.Shape, DType.F32);
        try
        {
            RmsNorm(n1, b, w1, eps);
            AddRmsNormEmitQ8(residOut, normOut, a, n1, w2, eps);
        }
        finally { n1.Dispose(); }
    }

    /// <summary>Sandwich-norm fusion: <c>output = a + rmsnorm(b)·w</c>, one launch replacing the post-FFN norm and residual add.</summary>
    /// <remarks>Backends without a fused kernel fall back to the exact composition.</remarks>
    void RmsNormAdd(Tensor output, Tensor a, Tensor b, Tensor weight, float eps)
    {
        Tensor n = new(b.Shape, DType.F32);
        try
        {
            RmsNorm(n, b, weight, eps);
            Add(output, a, n);
        }
        finally { n.Dispose(); }
    }

    /// <summary>One row-wise normalization with its affine and adaLN modulation: <c>y = (norm(x) · w + b) · scale + shift</c>
    /// as the <see cref="NormModulateCall"/> says, along the last axis or the channel axis, F32, F16 or BF16 in and out.
    /// LayerNorm, RMSNorm and Wan's L2-floored RMS norm, the broadcast and row-indexed affines and every modulated norm the
    /// DiT blocks run are this one op. Contract in <c>docs/BACKEND_OPS.md</c>; <see cref="NormModulateReference.Apply"/> is
    /// its executable form.</summary>
    void NormModulate(in NormModulateCall call) => NormModulateReference.Apply(call);

    /// <summary>Non-affine LayerNorm over the last dim: per row, zero mean and unit variance (biased var), no learned scale or bias;
    /// <see cref="NormModulate"/> with <see cref="NormKind.LayerNorm"/> and nothing else.</summary>
    void LayerNormNoAffine(Tensor output, Tensor input, float eps)
        => NormModulate(NormModulateCall.Rows(output, input, NormKind.LayerNorm, eps));

    /// <summary>Fused GroupNorm + SiLU: normalize, apply affine, then SiLU in one pass, eliminating an intermediate allocation.</summary>
    void GroupNormSilu(Tensor output, Tensor input, Tensor weight, Tensor bias, int groups, float eps)
    {
        GroupNorm(output, input, weight, bias, groups, eps);
        Silu(output, output);
    }
}
