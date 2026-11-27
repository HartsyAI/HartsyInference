using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.TopKGuidedLogits"/>: three identically shaped F32 tensors of
/// <c>[rows, width]</c> logits, a top-K within the width, a finite guidance, and an output sharing no storage with either
/// input. Returns the row count and width.</summary>
public static class TopKGuidedContract
{
    public static (long Rows, int Width) Validate(Tensor output, Tensor cond, Tensor uncond, int topK, float guidance)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(cond);
        ArgumentNullException.ThrowIfNull(uncond);
        if (output.DType != DType.F32 || cond.DType != DType.F32 || uncond.DType != DType.F32)
            throw new NotSupportedException($"TopKGuidedLogits takes F32 tensors; got output {output.DType}, cond {cond.DType}, uncond {uncond.DType}.");
        if (!output.Shape.Equals(cond.Shape) || !output.Shape.Equals(uncond.Shape))
            throw new ArgumentException($"TopKGuidedLogits requires identical shapes; got output {output.Shape}, cond {cond.Shape}, uncond {uncond.Shape}.", nameof(cond));
        if (output.Shape.Rank < 1 || output.ElementCount <= 0)
            throw new ArgumentException("TopKGuidedLogits requires nonempty tensors.", nameof(output));
        long width = output.Shape[output.Shape.Rank - 1];
        if (width <= 0 || width > int.MaxValue)
            throw new ArgumentException($"TopKGuidedLogits requires a positive row width within int range; got {output.Shape}.", nameof(output));
        if (topK < 1 || topK > width)
            throw new ArgumentOutOfRangeException(nameof(topK), topK, $"TopKGuidedLogits needs 1 <= topK <= {width}.");
        if (!float.IsFinite(guidance))
            throw new ArgumentOutOfRangeException(nameof(guidance), guidance, "Guidance must be finite.");
        if (output.HasOverlappingHostStorageWithoutSync(cond) || output.HasOverlappingHostStorageWithoutSync(uncond))
            throw new ArgumentException("TopKGuidedLogits output storage must not overlap either input.", nameof(output));
        return (output.ElementCount / width, (int)width);
    }
}
