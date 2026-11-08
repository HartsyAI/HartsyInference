using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.CopyStrided"/> and <see cref="IBackend.AccumulateStrided"/>:
/// element widths, bounds of both views against their tensors, a destination that addresses every element once,
/// and no storage aliasing. Every implementation validates through here, so host and accelerator agree on what is
/// refused; the result is the collapsed geometry a kernel runs.</summary>
public static class StridedCopyContract
{
    /// <summary>Validates a copy of 1, 2, 4 or 8-byte elements between tensors of one dtype.</summary>
    public static StridedCopyGeometry ValidateCopy(Tensor output, Tensor input, in StridedCopySpec spec)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        if (input.DType != output.DType)
            throw new ArgumentException($"CopyStrided moves bits between tensors of one dtype; got output {output.DType}, input {input.DType}.", nameof(input));
        int size = input.DType.SizeInBytes;
        if (input.DType.IsQuantized || (size != 1 && size != 2 && size != 4 && size != 8))
            throw new NotSupportedException($"CopyStrided moves 1, 2, 4 or 8-byte elements; {input.DType} is not addressable that way.");
        return Validate(output, input, spec, "CopyStrided");
    }

    /// <summary>Validates an accumulation of F32, F16 or BF16 elements between tensors of one dtype, or of an F32 input
    /// into a BF16 output (the sum is formed in F32 and rounded once, so an F32 bias keeps its precision).</summary>
    public static StridedCopyGeometry ValidateAccumulate(Tensor output, Tensor input, in StridedCopySpec spec)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        bool mixed = input.DType == DType.F32 && output.DType == DType.BF16;
        if (input.DType != output.DType && !mixed)
            throw new ArgumentException($"AccumulateStrided adds tensors of one dtype, or an F32 input into a BF16 output; got output {output.DType}, input {input.DType}.", nameof(input));
        if (input.DType != DType.F32 && input.DType != DType.F16 && input.DType != DType.BF16)
            throw new NotSupportedException($"AccumulateStrided adds F32, F16 or BF16 elements; got {input.DType}.");
        return Validate(output, input, spec, "AccumulateStrided");
    }

    private static StridedCopyGeometry Validate(Tensor output, Tensor input, in StridedCopySpec spec, string operation)
    {
        StridedView source = spec.Source;
        StridedView destination = spec.Destination;
        if (source.MaxElementIndex >= input.ElementCount)
            throw new ArgumentException($"{operation} source view {source} reaches element {source.MaxElementIndex} of an input with {input.ElementCount} elements.", nameof(spec));
        if (destination.MaxElementIndex >= output.ElementCount)
            throw new ArgumentException($"{operation} destination view {destination} reaches element {destination.MaxElementIndex} of an output with {output.ElementCount} elements.", nameof(spec));
        RequireInjective(destination, operation);
        // Shape, dtype and bounds are checked above; the aliasing query never materializes lazy host storage, so a
        // device-resident copy is not pulled to the host just to be validated.
        if (output.HasOverlappingHostStorageWithoutSync(input))
            throw new ArgumentException($"{operation} output storage must not overlap its input storage.", nameof(output));
        // Injective, in bounds and as many elements as the output: the destination is a permutation of the whole output.
        bool covers = destination.ElementCount == output.ElementCount;
        return Collapse(spec, input.DType.SizeInBytes, covers);
    }

    /// <summary>Every destination element must be written once: axes of extent above 1 need a positive stride, and
    /// each axis's stride must exceed the span of the axes with smaller strides.</summary>
    private static void RequireInjective(in StridedView destination, string operation)
    {
        int rank = destination.Rank;
        Span<int> order = stackalloc int[rank];
        int n = 0;
        for (int axis = 0; axis < rank; axis++)
        {
            if (destination[axis] == 1) continue;
            if (destination.Stride(axis) == 0)
                throw new ArgumentException($"{operation} destination view {destination} broadcasts axis {axis}, so it would write one element {destination[axis]} times.", nameof(destination));
            order[n++] = axis;
        }
        // Insertion sort by stride, descending: ranks are at most six.
        for (int i = 1; i < n; i++)
        {
            int axis = order[i];
            int j = i - 1;
            while (j >= 0 && destination.Stride(order[j]) < destination.Stride(axis))
            {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = axis;
        }
        long innerSpan = 0;
        for (int k = n - 1; k >= 0; k--)
        {
            int axis = order[k];
            if (destination.Stride(axis) <= innerSpan)
                throw new ArgumentException($"{operation} destination view {destination} overlaps itself: axis {axis} steps by {destination.Stride(axis)} over inner axes spanning {innerSpan + 1} elements.", nameof(destination));
            innerSpan = checked(innerSpan + (destination[axis] - 1) * destination.Stride(axis));
        }
    }

    /// <summary>Drops axes of extent 1 and merges adjacent axes both views walk as one run.</summary>
    private static StridedCopyGeometry Collapse(in StridedCopySpec spec, int elementSize, bool covers)
    {
        int rank = spec.Source.Rank;
        Span<long> d = stackalloc long[rank];
        Span<long> s = stackalloc long[rank];
        Span<long> t = stackalloc long[rank];
        int n = 0;
        for (int axis = 0; axis < rank; axis++)
        {
            long dim = spec.Source[axis];
            long src = spec.Source.Stride(axis);
            long dst = spec.Destination.Stride(axis);
            if (dim == 1) continue;
            if (n > 0 && s[n - 1] == src * dim && t[n - 1] == dst * dim)
            {
                // The previous (outer) axis steps exactly one inner run in both views: fold it into this axis.
                d[n - 1] *= dim;
                s[n - 1] = src;
                t[n - 1] = dst;
                continue;
            }
            d[n] = dim;
            s[n] = src;
            t[n] = dst;
            n++;
        }
        if (n == 0)
        {
            d[0] = 1;
            s[0] = 0;
            t[0] = 0;
            n = 1;
        }
        return new StridedCopyGeometry(d[..n], s[..n], t[..n], spec.Source.Offset, spec.Destination.Offset, elementSize, covers);
    }
}
