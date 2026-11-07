using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.GroupScale"/>: an activation and output of one dtype (F32, F16 or BF16) and equal
/// element count, the output either the input itself (in place) or storage apart from it, gates F32 or the activation's dtype
/// that cover every row's groups, a group width that divides the row, and a function the gates pass through. Returns the geometry.</summary>
public static class GroupScaleContract
{
    public static GroupScaleGeometry Validate(Tensor output, Tensor input, Tensor gates, in GroupScaleSpec spec)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(gates);
        if (input.DType != output.DType || (input.DType != DType.F32 && input.DType != DType.F16 && input.DType != DType.BF16))
            throw new NotSupportedException($"GroupScale takes an F32, F16 or BF16 input and an output of the same dtype; got input {input.DType}, output {output.DType}.");
        if (gates.DType != DType.F32 && gates.DType != input.DType)
            throw new NotSupportedException($"GroupScale gates are F32 or the activation's dtype; got {gates.DType} with {input.DType}.");
        if (output.ElementCount != input.ElementCount)
            throw new ArgumentException($"GroupScale output {output.Shape} and input {input.Shape} differ in element count.", nameof(output));
        if (!Enum.IsDefined(spec.Function))
            throw new ArgumentOutOfRangeException(nameof(spec), spec.Function, "Unknown gate function.");
        int width = (int)input.Shape[input.Shape.Rank - 1];
        long rows = input.ElementCount / width;
        int groupWidth = spec.GroupWidth == 0 ? width : spec.GroupWidth;
        if (groupWidth <= 0 || width % groupWidth != 0)
            throw new ArgumentException($"A group width of {spec.GroupWidth} does not divide rows of {width}.", nameof(spec));
        int groups = width / groupWidth;
        if (spec.GateStride < groups || spec.GateOffset < 0)
            throw new ArgumentException($"Gates are at least {groups} apart per row with a non-negative offset; got stride {spec.GateStride}, offset {spec.GateOffset}.", nameof(spec));
        if (rows > 0 && spec.GateOffset + (rows - 1) * spec.GateStride + groups > gates.ElementCount)
            throw new ArgumentException($"The gates {gates.Shape} do not cover {rows} rows of {groups} groups at stride {spec.GateStride} from {spec.GateOffset}.", nameof(gates));
        if (!ReferenceEquals(output, input) && output.HasOverlappingHostStorageWithoutSync(input))
            throw new ArgumentException("GroupScale output storage is the input itself or apart from it.", nameof(output));
        if (output.HasOverlappingHostStorageWithoutSync(gates))
            throw new ArgumentException("GroupScale output storage must not overlap the gates.", nameof(output));
        return new GroupScaleGeometry(rows, width, groupWidth, groups);
    }
}
