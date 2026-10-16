using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Fused DiT block primitives with a host reference default; GPU backends override them.
public partial interface IBackend
{
    /// <summary>SiTU-GLU over the two halves of the last dimension: with <c>u</c> the first half and <c>g</c> the second,
    /// <c>out = (4·tanh(g/4)·sigmoid(g)) · (25·tanh(u/25))</c> (MOSS-Image's soft-capped gated FFN activation).
    /// <paramref name="input"/> is <c>[.., 2·F]</c> and <paramref name="output"/> <c>[.., F]</c>, both F32 or both F16.</summary>
    void SituGlu(Tensor output, Tensor input) => SituGluReference.Apply(output, input);

    /// <summary>Weighted RMSNorm followed by an adaLN modulation in one pass:
    /// <c>out[r,d] = x[r,d]·rsqrt(mean(x[r,:]²) + eps)·weight[d]·(1 + scale[d]) + shift[d]</c>.</summary>
    /// <remarks>Input and output may differ in dtype (F32 or F16 each), so an F32 residual stream can feed an F16 block
    /// branch without a separate cast. <paramref name="weight"/> is F32 <c>[D]</c>; <paramref name="scale"/> and
    /// <paramref name="shift"/> are F32 <c>[D]</c> or <c>[B, D]</c> broadcast over each batch's rows, with the raw scale
    /// (the <c>1 +</c> is applied here).</remarks>
    void RmsNormModulate(Tensor output, Tensor input, Tensor weight, Tensor scale, Tensor? shift, float eps) =>
        RmsNormModulateReference.Apply(output, input, weight, scale, shift, eps);
}
