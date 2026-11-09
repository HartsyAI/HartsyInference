using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Streaming;

/// <summary>Slices and joins of activations along one axis, each a new tensor the caller owns. The streaming caches build
/// their carried tensors with these, so a device backend keeps every piece resident.</summary>
public static class StreamingTensors
{
    /// <summary><c>source[..., start : start + length, ...]</c> along <paramref name="axis"/> as a new tensor.</summary>
    public static Tensor Slice(IBackend backend, Tensor source, int axis, long start, long length)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(source);
        if (axis < 0 || axis >= source.Shape.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis), axis, $"{source.Shape} has no axis {axis}.");
        if (start < 0 || length < 1 || start + length > source.Shape[axis])
            throw new ArgumentOutOfRangeException(nameof(start), $"[{start}, {start + length}) is outside axis {axis} of {source.Shape}.");
        Tensor slice = new(WithAxis(source.Shape, axis, length), source.DType);
        try
        {
            StridedCopies.SliceAxis(backend, slice, source, axis, start, length);
            return slice;
        }
        catch
        {
            slice.Dispose();
            throw;
        }
    }

    /// <summary>The tensors of <paramref name="parts"/> joined along <paramref name="axis"/> as a new tensor. Every part has the
    /// same dtype and the same extents off the axis.</summary>
    public static Tensor Concat(IBackend backend, ReadOnlySpan<Tensor> parts, int axis)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (parts.IsEmpty)
            throw new ArgumentException("Nothing to join.", nameof(parts));
        Tensor first = parts[0];
        if (axis < 0 || axis >= first.Shape.Rank)
            throw new ArgumentOutOfRangeException(nameof(axis), axis, $"{first.Shape} has no axis {axis}.");
        long total = 0;
        foreach (Tensor part in parts)
        {
            if (part.DType != first.DType || part.Shape.Rank != first.Shape.Rank)
                throw new ArgumentException($"Cannot join {part.DType} {part.Shape} with {first.DType} {first.Shape}.", nameof(parts));
            for (int d = 0; d < first.Shape.Rank; d++)
            {
                if (d != axis && part.Shape[d] != first.Shape[d])
                    throw new ArgumentException($"{part.Shape} and {first.Shape} differ off axis {axis}.", nameof(parts));
            }

            total += part.Shape[axis];
        }

        Tensor joined = new(WithAxis(first.Shape, axis, total), first.DType);
        try
        {
            backend.Concat(joined, parts, axis);
            return joined;
        }
        catch
        {
            joined.Dispose();
            throw;
        }
    }

    /// <summary>A copy of <paramref name="source"/> as a new tensor of the same shape.</summary>
    public static Tensor Clone(IBackend backend, Tensor source) => Slice(backend, source, 0, 0, source.Shape[0]);

    private static TensorShape WithAxis(TensorShape shape, int axis, long extent)
    {
        long[] dims = new long[shape.Rank];
        for (int d = 0; d < dims.Length; d++)
            dims[d] = d == axis ? extent : shape[d];
        return new TensorShape(dims);
    }
}
