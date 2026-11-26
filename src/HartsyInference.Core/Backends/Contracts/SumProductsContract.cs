using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Reduction;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.SumProducts"/>: F32 operands that may be one tensor, views inside them, an F32 or F64 output of
/// exactly groups times terms elements that shares no storage with either operand. Returns the geometry.</summary>
public static class SumProductsContract
{
    public static SumProductsGeometry Validate(Tensor output, Tensor a, Tensor b, in SumProductsSpec spec)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (spec.Terms == ProductTerms.None)
            throw new ArgumentException("SumProducts needs a spec built with its constructor or factories.", nameof(spec));
        if (a.DType != DType.F32 || b.DType != DType.F32)
            throw new NotSupportedException($"SumProducts takes F32 operands; got {a.DType} and {b.DType}.");
        if (output.DType != DType.F32 && output.DType != DType.F64)
            throw new NotSupportedException($"SumProducts writes F32 or F64 sums; got an output of {output.DType}.");
        if (spec.A.MaxElementIndex >= a.ElementCount)
            throw new ArgumentException($"SumProducts view {spec.A} reaches element {spec.A.MaxElementIndex} of {a.Shape}.", nameof(a));
        if (spec.B.MaxElementIndex >= b.ElementCount)
            throw new ArgumentException($"SumProducts view {spec.B} reaches element {spec.B.MaxElementIndex} of {b.Shape}.", nameof(b));
        long groups = 1, perGroup = 1;
        for (int axis = 0; axis < spec.A.Rank; axis++)
        {
            if (axis < spec.KeptAxes) groups = checked(groups * spec.A[axis]);
            else perGroup = checked(perGroup * spec.A[axis]);
        }
        int terms = spec.TermCount;
        if (output.ElementCount != checked(groups * terms))
            throw new ArgumentException($"SumProducts writes {groups} groups of {terms} sums; the output {output.Shape} has {output.ElementCount} elements.", nameof(output));
        if (output.HasOverlappingHostStorageWithoutSync(a) || output.HasOverlappingHostStorageWithoutSync(b))
            throw new ArgumentException("SumProducts output storage must not overlap an operand.", nameof(output));
        return new SumProductsGeometry(groups, perGroup, terms,
            StridedCopyContract.Collapse(new StridedCopySpec(spec.B, spec.A), DType.F32.SizeInBytes, covers: false));
    }
}
