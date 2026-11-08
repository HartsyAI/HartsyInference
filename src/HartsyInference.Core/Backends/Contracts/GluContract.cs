using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.Glu"/>: input and output of one dtype (F32, F16 or BF16), the output's last
/// extent half the input's, the same row count, no shared storage. Returns the inner width and the row count.</summary>
public static class GluContract
{
    public static (int Inner, long Rows) Validate(Tensor output, Tensor input, in GluSpec spec)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        if (input.DType != output.DType || (input.DType != DType.F32 && input.DType != DType.F16 && input.DType != DType.BF16))
            throw new NotSupportedException($"Glu takes an F32, F16 or BF16 input and an output of the same dtype; got input {input.DType}, output {output.DType}.");
        if (!Enum.IsDefined(spec.Activation) || !Enum.IsDefined(spec.Gate))
            throw new ArgumentOutOfRangeException(nameof(spec), spec, "Unknown GLU activation or gate half.");
        int inner = (int)output.Shape[output.Shape.Rank - 1];
        if (inner <= 0 || input.Shape[input.Shape.Rank - 1] != 2L * inner)
            throw new ArgumentException($"Glu input {input.Shape} must be twice the output's last extent {output.Shape}.", nameof(input));
        long rows = output.ElementCount / inner;
        if (input.ElementCount != rows * 2 * inner)
            throw new ArgumentException($"Glu input {input.Shape} and output {output.Shape} disagree on the row count.", nameof(input));
        if (output.HasOverlappingHostStorageWithoutSync(input))
            throw new ArgumentException("Glu output storage must not overlap the input.", nameof(output));
        return (inner, rows);
    }
}
