using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.Pad"/>: the output extent of every axis is the input extent plus
/// the padding, reflection never reaches past the far edge, and the tensors share a byte-addressable dtype and no
/// storage.</summary>
public static class PadContract
{
    public static PadGeometry Validate(Tensor output, Tensor input, in PadSpec spec)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        if (input.DType != output.DType)
            throw new ArgumentException($"Pad fills tensors of one dtype; got output {output.DType}, input {input.DType}.", nameof(input));
        int size = input.DType.SizeInBytes;
        if (input.DType.IsQuantized || (size != 1 && size != 2 && size != 4 && size != 8))
            throw new NotSupportedException($"Pad moves 1, 2, 4 or 8-byte elements; {input.DType} is not addressable that way.");
        if (input.Shape.Rank != spec.Rank || output.Shape.Rank != spec.Rank)
            throw new ArgumentException($"Pad spec covers {spec.Rank} axes; got input {input.Shape} and output {output.Shape}.", nameof(spec));
        Span<long> outputDims = stackalloc long[spec.Rank];
        Span<long> inputDims = stackalloc long[spec.Rank];
        Span<long> before = stackalloc long[spec.Rank];
        for (int axis = 0; axis < spec.Rank; axis++)
        {
            long extent = input.Shape[axis];
            if (extent <= 0)
                throw new ArgumentException($"Pad input axis {axis} must have a positive extent; got {input.Shape}.", nameof(input));
            long expected = checked(extent + spec.Before(axis) + spec.After(axis));
            if (output.Shape[axis] != expected)
                throw new ArgumentException($"Pad output axis {axis} must be {expected} ({extent} + {spec.Before(axis)} + {spec.After(axis)}); got {output.Shape}.", nameof(output));
            if (spec.Mode == PadMode.Reflect && (spec.Before(axis) >= extent || spec.After(axis) >= extent))
                throw new ArgumentException($"Pad reflect on axis {axis} pads by ({spec.Before(axis)}, {spec.After(axis)}), which must stay below the extent {extent}.", nameof(spec));
            outputDims[axis] = output.Shape[axis];
            inputDims[axis] = extent;
            before[axis] = spec.Before(axis);
        }
        if (output.HasOverlappingHostStorageWithoutSync(input))
            throw new ArgumentException("Pad output storage must not overlap its input storage.", nameof(output));
        return new PadGeometry(outputDims, inputDims, before, spec.Mode, size);
    }
}
