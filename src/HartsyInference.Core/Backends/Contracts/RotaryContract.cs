using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.Rotary"/>: dtypes, equal extents across the three views, an
/// activation view that addresses every element once (a pair never shares an element with another pair), bounds of
/// every view against its tensor, and a device position only with a row stride. Every implementation validates
/// through here.</summary>
public static class RotaryContract
{
    public static RotaryGeometry Validate(in RotaryCall call)
    {
        ArgumentNullException.ThrowIfNull(call.X);
        ArgumentNullException.ThrowIfNull(call.Cos);
        ArgumentNullException.ThrowIfNull(call.Sin);
        if (call.X.DType != DType.F32 && call.X.DType != DType.F16 && call.X.DType != DType.BF16)
            throw new NotSupportedException($"Rotary rotates F32, F16 or BF16 activations; got {call.X.DType}.");
        if (call.Cos.DType != DType.F32 || call.Sin.DType != DType.F32)
            throw new NotSupportedException($"Rotary reads F32 tables; got cos {call.Cos.DType}, sin {call.Sin.DType}.");
        RotarySpec spec = call.Spec;
        if (spec.X.Rank != RotarySpec.Rank || spec.Cos.Rank != RotarySpec.Rank || spec.Sin.Rank != RotarySpec.Rank)
            throw new ArgumentException($"Rotary views have {RotarySpec.Rank} axes [B, H, S1, S2, S3, P, 2]; got x {spec.X}, cos {spec.Cos}, sin {spec.Sin}.", nameof(call));
        if (!spec.X.HasSameDims(spec.Cos) || !spec.X.HasSameDims(spec.Sin))
            throw new ArgumentException($"Rotary views must share their extents; got x {spec.X}, cos {spec.Cos}, sin {spec.Sin}.", nameof(call));
        if (spec.X[RotaryGeometry.IndexAxes] != 2)
            throw new ArgumentException($"The last axis of a rotary view holds the two elements of a pair; got {spec.X}.", nameof(call));
        if (spec.X.Stride(RotaryGeometry.IndexAxes) == 0)
            throw new ArgumentException($"A pair's two elements must be distinct; the activation view {spec.X} broadcasts them.", nameof(call));
        RequireInjective(spec.X);
        if (spec.X.MaxElementIndex >= call.X.ElementCount)
            throw new ArgumentException($"Rotary activation view {spec.X} reaches element {spec.X.MaxElementIndex} of a tensor with {call.X.ElementCount} elements.", nameof(call));
        long positions = spec.DevicePosition == 0 ? 0 : spec.PositionRowStride;
        if (spec.DevicePosition != 0 && spec.PositionRowStride <= 0)
            throw new ArgumentException("A rotary call with a device position needs a positive table row stride.", nameof(call));
        if (spec.DevicePosition == 0 && (spec.PositionRowStride != 0 || spec.PositionSlot != 0))
            throw new ArgumentException("Position row stride and slot apply only with a device position buffer.", nameof(call));
        // With a device position the row offset is unknown here; the static part of the view must still fit.
        if (spec.Cos.MaxElementIndex >= call.Cos.ElementCount)
            throw new ArgumentException($"Rotary cosine view {spec.Cos} reaches element {spec.Cos.MaxElementIndex} of a table with {call.Cos.ElementCount} elements.", nameof(call));
        if (spec.Sin.MaxElementIndex >= call.Sin.ElementCount)
            throw new ArgumentException($"Rotary sine view {spec.Sin} reaches element {spec.Sin.MaxElementIndex} of a table with {call.Sin.ElementCount} elements.", nameof(call));
        if (positions != 0 && (call.Cos.ElementCount % positions != 0 || call.Sin.ElementCount % positions != 0))
            throw new ArgumentException($"The tables hold {call.Cos.ElementCount} and {call.Sin.ElementCount} elements, not whole rows of {positions}.", nameof(call));
        if (call.X.HasOverlappingHostStorageWithoutSync(call.Cos) || call.X.HasOverlappingHostStorageWithoutSync(call.Sin))
            throw new ArgumentException("Rotary activation storage must not overlap its tables.", nameof(call));
        return new RotaryGeometry(spec, call.X.DType.SizeInBytes);
    }

    /// <summary>Every rotated element is written once: the same rule as a strided copy's destination.</summary>
    private static void RequireInjective(in StridedView x)
    {
        int rank = x.Rank;
        Span<int> order = stackalloc int[rank];
        int n = 0;
        for (int axis = 0; axis < rank; axis++)
        {
            if (x[axis] == 1) continue;
            if (x.Stride(axis) == 0)
                throw new ArgumentException($"Rotary activation view {x} broadcasts axis {axis}, so pairs would share elements.", nameof(x));
            order[n++] = axis;
        }
        for (int i = 1; i < n; i++)
        {
            int axis = order[i];
            int j = i - 1;
            while (j >= 0 && x.Stride(order[j]) < x.Stride(axis))
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
            if (x.Stride(axis) <= innerSpan)
                throw new ArgumentException($"Rotary activation view {x} overlaps itself on axis {axis}.", nameof(x));
            innerSpan = checked(innerSpan + (x[axis] - 1) * x.Stride(axis));
        }
    }
}
