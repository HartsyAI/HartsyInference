using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>The output contract of <see cref="IBackend.UpsampleNearest2DToSize"/>, shared by every backend's implementation.</summary>
public static class UpsampleNearestExtent
{
    /// <summary>Returns the output's height and width, or throws unless both tensors are NCHW with the same batch,
    /// channels and dtype and each output extent is twice the input's or one less.</summary>
    public static (int OutH, int OutW) Validate(Tensor output, Tensor input)
    {
        const int scale = 2;
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4 || input.Shape[0] != output.Shape[0]
            || input.Shape[1] != output.Shape[1] || input.DType != output.DType)
            throw new ArgumentException($"UpsampleNearest2DToSize needs matching NCHW tensors; got input {input.Shape} {input.DType}, output {output.Shape} {output.DType}.");
        long inH = input.Shape[2], inW = input.Shape[3], outH = output.Shape[2], outW = output.Shape[3];
        if (outH > inH * scale || outH < inH * scale - 1 || outW > inW * scale || outW < inW * scale - 1)
            throw new ArgumentException($"UpsampleNearest2DToSize output {outH}x{outW} must be {inH * scale}x{inW * scale} or one short per axis (input {inH}x{inW}).");
        return ((int)outH, (int)outW);
    }
}
