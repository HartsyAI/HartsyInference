using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Validated contiguous geometry for <see cref="IBackend.Split"/>.</summary>
internal readonly record struct SplitGeometry(
    long Outer,
    long InputDimension,
    long Inner,
    int ElementSize);

/// <summary>
/// Centralizes the exact split contract so host and accelerator implementations cannot disagree about
/// dtype width, output geometry, or storage aliasing.
/// </summary>
internal static class SplitContract
{
    internal static SplitGeometry Validate(ReadOnlySpan<Tensor> outputs, Tensor input, int dim)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (outputs.IsEmpty)
            throw new ArgumentException("Split requires at least one output tensor.", nameof(outputs));
        if (input.DType != DType.F32 && input.DType != DType.F16 && input.DType != DType.BF16)
            throw new NotSupportedException(
                $"Split supports the byte-addressable F32, F16, and BF16 dtypes; got {input.DType}.");

        int elementSize = input.DType.SizeInBytes;
        ValidateTensorGeometry(input, elementSize, nameof(input));
        if ((uint)dim >= (uint)input.Shape.Rank)
            throw new ArgumentOutOfRangeException(
                nameof(dim), dim, $"Split dimension must be in [0,{input.Shape.Rank}); got {dim}.");

        long splitDimensionSum = 0;
        for (int outputIndex = 0; outputIndex < outputs.Length; outputIndex++)
        {
            Tensor? output = outputs[outputIndex];
            if (output is null)
                throw new ArgumentNullException(nameof(outputs), $"Split output {outputIndex} is null.");
            if (output.DType != input.DType)
                throw new ArgumentException(
                    $"Split output {outputIndex} dtype {output.DType} must match input dtype {input.DType}.",
                    nameof(outputs));
            ValidateTensorGeometry(output, elementSize, nameof(outputs));
            if (output.Shape.Rank != input.Shape.Rank)
                throw new ArgumentException(
                    $"Split output {outputIndex} rank {output.Shape.Rank} must match input rank {input.Shape.Rank}.",
                    nameof(outputs));

            for (int axis = 0; axis < input.Shape.Rank; axis++)
            {
                if (axis != dim && output.Shape[axis] != input.Shape[axis])
                    throw new ArgumentException(
                        $"Split output {outputIndex} axis {axis} has size {output.Shape[axis]}, "
                        + $"but input axis {axis} has size {input.Shape[axis]}.",
                        nameof(outputs));
            }

            try
            {
                splitDimensionSum = checked(splitDimensionSum + output.Shape[dim]);
            }
            catch (OverflowException)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(outputs), "Split output dimension sum exceeds signed 64-bit indexing.");
            }
        }

        if (splitDimensionSum != input.Shape[dim])
            throw new ArgumentException(
                $"Split output sizes along dimension {dim} sum to {splitDimensionSum}; "
                + $"the input dimension has size {input.Shape[dim]}.",
                nameof(outputs));

        // Validate every byte span before asking Tensor to compare its materialized storage range.
        // The helper never materializes lazy host storage and therefore never triggers a GPU readback.
        for (int outputIndex = 0; outputIndex < outputs.Length; outputIndex++)
        {
            Tensor output = outputs[outputIndex];
            if (output.HasOverlappingHostStorageWithoutSync(input))
                throw new ArgumentException(
                    $"Split output {outputIndex} storage must not overlap input storage.", nameof(outputs));
            for (int previous = 0; previous < outputIndex; previous++)
            {
                if (output.HasOverlappingHostStorageWithoutSync(outputs[previous]))
                    throw new ArgumentException(
                        $"Split outputs {previous} and {outputIndex} must not have overlapping storage.",
                        nameof(outputs));
            }
        }

        long outer = 1;
        long inner = 1;
        try
        {
            for (int axis = 0; axis < dim; axis++)
                outer = checked(outer * input.Shape[axis]);
            for (int axis = dim + 1; axis < input.Shape.Rank; axis++)
                inner = checked(inner * input.Shape[axis]);
        }
        catch (OverflowException)
        {
            // ValidateTensorGeometry already establishes that this is unreachable for a valid tensor, but keep
            // this explicit in case TensorShape's representation changes independently of this contract.
            throw new ArgumentOutOfRangeException(nameof(input), "Split geometry exceeds signed 64-bit indexing.");
        }

        return new SplitGeometry(outer, input.Shape[dim], inner, elementSize);
    }

    private static void ValidateTensorGeometry(Tensor tensor, int elementSize, string parameterName)
    {
        if (tensor.Shape.Rank <= 0)
            throw new ArgumentException("Split tensors must have rank at least one.", parameterName);

        long elementCount = 1;
        try
        {
            for (int axis = 0; axis < tensor.Shape.Rank; axis++)
            {
                long dimension = tensor.Shape[axis];
                if (dimension <= 0)
                    throw new ArgumentOutOfRangeException(
                        parameterName, dimension, $"Split tensor axis {axis} must be positive.");
                elementCount = checked(elementCount * dimension);
            }
        }
        catch (OverflowException)
        {
            throw new ArgumentOutOfRangeException(
                parameterName, "Split tensor geometry exceeds signed 64-bit indexing.");
        }

        if (tensor.ElementCount != elementCount)
            throw new ArgumentException(
                $"Split requires valid contiguous tensor geometry; shape {tensor.Shape} reports "
                + $"{tensor.ElementCount} elements instead of {elementCount}.",
                parameterName);
        if (elementCount > long.MaxValue / elementSize)
            throw new ArgumentOutOfRangeException(
                parameterName, "Split tensor byte span exceeds signed 64-bit addressable storage.");
    }
}
