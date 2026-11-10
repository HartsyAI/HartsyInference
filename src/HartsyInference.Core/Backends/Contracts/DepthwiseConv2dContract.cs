using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>Validated NCHW geometry for <see cref="IBackend.Conv2dDepthwise"/>.</summary>
internal readonly record struct DepthwiseConv2dGeometry(int Batch, int Channels, int InH, int InW, int OutH, int OutW, int KernelH, int KernelW);

/// <summary>The depthwise Conv2D shape contract, shared by the host reference and the device backends so they cannot disagree on geometry.</summary>
internal static class DepthwiseConv2dContract
{
    /// <summary>Checks NCHW input and output, the <c>[C, 1, kH, kW]</c> weight and the optional <c>[C]</c> bias agree on channels.</summary>
    /// <exception cref="ArgumentException">A tensor has the wrong rank, weight layout, or channel or bias count.</exception>
    internal static DepthwiseConv2dGeometry Validate(Tensor input, Tensor weight, Tensor? bias, Tensor output)
    {
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4)
            throw new ArgumentException($"Conv2dDepthwise requires [N, C, H, W] tensors; got input {input.Shape} / output {output.Shape}.");
        if (weight.Shape.Rank != 4 || weight.Shape[1] != 1)
            throw new ArgumentException($"Conv2dDepthwise weight must be [C, 1, kH, kW]; got {weight.Shape}.");
        if (input.Shape[1] != weight.Shape[0] || output.Shape[1] != weight.Shape[0])
            throw new ArgumentException("Conv2dDepthwise requires input/output channel count to equal weight channel count.");
        if (bias is not null && bias.ElementCount != weight.Shape[0])
            throw new ArgumentException($"Conv2dDepthwise bias must hold one value per channel ({weight.Shape[0]}); got {bias.Shape}.");

        return new DepthwiseConv2dGeometry(
            (int)input.Shape[0], (int)input.Shape[1], (int)input.Shape[2], (int)input.Shape[3],
            (int)output.Shape[2], (int)output.Shape[3], (int)weight.Shape[2], (int)weight.Shape[3]);
    }
}
