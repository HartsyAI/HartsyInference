using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The contract of <see cref="IBackend.Exp"/> and <see cref="IBackend.Reciprocal"/>: input and output each F32 or BF16,
/// of equal element count (shapes may differ), sharing no storage. Returns the element count.</summary>
public static class ElementwiseMathContract
{
    /// <exception cref="NotSupportedException">A tensor is neither F32 nor BF16.</exception>
    /// <exception cref="ArgumentException">The element counts differ or the output overlaps the input.</exception>
    public static long Validate(Tensor output, Tensor input, ElementwiseMathFunction function)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        if (!IsSupported(output.DType) || !IsSupported(input.DType))
            throw new NotSupportedException($"{function} takes F32 or BF16 tensors; got output {output.DType}, input {input.DType}.");
        if (output.ElementCount != input.ElementCount)
            throw new ArgumentException($"{function} needs equal element counts; got output {output.Shape}, input {input.Shape}.", nameof(output));
        if (output.HasOverlappingHostStorageWithoutSync(input))
            throw new ArgumentException($"{function} output storage must not overlap the input.", nameof(output));
        return input.ElementCount;
    }

    /// <summary>Whether <paramref name="dtype"/> is an input or output dtype the op takes.</summary>
    public static bool IsSupported(DType dtype) => dtype == DType.F32 || dtype == DType.BF16;
}
