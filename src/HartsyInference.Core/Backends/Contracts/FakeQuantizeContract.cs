using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.FakeQuantizeUniform"/>: F32 input and output of one shape sharing no
/// storage, a finite range with <c>low &lt; high</c>, and at least two levels. Returns the element count.</summary>
public static class FakeQuantizeContract
{
    public static long Validate(Tensor output, Tensor input, float low, float high, int levels)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        if (output.DType != DType.F32 || input.DType != DType.F32)
            throw new NotSupportedException($"FakeQuantizeUniform takes F32 tensors; got output {output.DType}, input {input.DType}.");
        if (!output.Shape.Equals(input.Shape))
            throw new ArgumentException($"FakeQuantizeUniform requires identical shapes; got output {output.Shape}, input {input.Shape}.", nameof(output));
        if (!float.IsFinite(low) || !float.IsFinite(high) || low >= high)
            throw new ArgumentOutOfRangeException(nameof(high), (low, high), "FakeQuantizeUniform needs a finite range with low < high.");
        if (levels < 2)
            throw new ArgumentOutOfRangeException(nameof(levels), levels, "FakeQuantizeUniform needs at least two levels.");
        if (output.HasOverlappingHostStorageWithoutSync(input))
            throw new ArgumentException("FakeQuantizeUniform output storage must not overlap the input.", nameof(output));
        return input.ElementCount;
    }
}
