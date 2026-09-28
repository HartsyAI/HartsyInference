using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>The output contract of <see cref="IBackend.UpsampleNearest2DToSize"/>, shared by every backend's implementation.</summary>
public static class UpsampleNearestExtent
{
    /// <summary>Returns the output's height and width, or throws unless both tensors are NCHW with the same batch,
    /// channels and dtype and each output extent lies in <c>((in − 1)·scale, in·scale]</c> — the range where every input
    /// row still lands in the output and <c>oh / scale</c> is the nearest-neighbour source.</summary>
    public static (int OutH, int OutW) Validate(Tensor output, Tensor input, int scale)
    {
        if (scale < 1)
            throw new ArgumentOutOfRangeException(nameof(scale), $"Upsample scale must be positive; got {scale}.");
        if (input.Shape.Rank != 4 || output.Shape.Rank != 4 || input.Shape[0] != output.Shape[0]
            || input.Shape[1] != output.Shape[1] || input.DType != output.DType)
            throw new ArgumentException($"UpsampleNearest2DToSize needs matching NCHW tensors; got input {input.Shape} {input.DType}, output {output.Shape} {output.DType}.");
        long inH = input.Shape[2], inW = input.Shape[3], outH = output.Shape[2], outW = output.Shape[3];
        if (outH > inH * scale || outH <= (inH - 1) * scale || outW > inW * scale || outW <= (inW - 1) * scale)
            throw new ArgumentException($"UpsampleNearest2DToSize output {outH}x{outW} is not within scale-1 of {inH * scale}x{inW * scale} (input {inH}x{inW}, scale {scale}).");
        return ((int)outH, (int)outW);
    }
}
