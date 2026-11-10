using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// RMSNorm, ReLU and a scale in one pass; the default composes the three ops.
public partial interface IBackend
{
    /// <summary><c>output = ReLU(RMSNorm(input)·weight)·scale</c> per row of <c>weight.ElementCount</c> values: the q/k
    /// normalization of linear-attention layers. The default runs <see cref="RmsNorm"/>, <see cref="LeakyRelu"/> and
    /// <see cref="Scale"/>; a backend may fuse them into one read and one write.</summary>
    /// <param name="output">F32, shaped like <paramref name="input"/>.</param>
    /// <param name="input">F32 rows of <c>weight.ElementCount</c> values.</param>
    /// <param name="weight">F32 per-channel norm weight.</param>
    void RmsNormRelu(Tensor output, Tensor input, Tensor weight, float eps, float scale)
    {
        RmsNorm(output, input, weight, eps);
        LeakyRelu(output, output, 0f);
        if (scale != 1f)
            Scale(output, output, scale);
    }
}
