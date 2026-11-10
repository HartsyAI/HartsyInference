using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Numerics;
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
    /// same dtype and the same extents off the axis. A device backend joins any dtype with its <c>Concat</c> (device-to-device
    /// copies sized by the dtype); the CPU backend's <c>Concat</c> copies four bytes an element, so there narrower dtypes are
    /// scattered into place.</summary>
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
            // The scatter is one strided copy per part, about five times slower than Concat on CUDA for the BF16 FFN tail
            // that every streamed SANA-WM pass joins (+40 ms a pass on a 4090), so only the CPU backend takes it.
            if (first.DType.SizeInBytes == sizeof(float) || !backend.Device.IsCpu)
            {
                backend.Concat(joined, parts, axis);
            }
            else
            {
                // The CPU backend's Concat copies four bytes per element whatever the dtype, which overruns a 16-bit tensor; a
                // strided scatter is a pure move of bits for any 1, 2, 4 or 8-byte dtype.
                long start = 0;
                foreach (Tensor part in parts)
                {
                    StridedCopies.ScatterAxis(backend, joined, part, axis, start);
                    start += part.Shape[axis];
                }
            }

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

    /// <summary>A new tensor holding <paramref name="source"/> converted to <paramref name="target"/>, a copy when the dtype
    /// already matches. F32 and BF16 convert in both directions; narrowing rounds to nearest even. A device backend casts on
    /// the device; the CPU backend casts on the host, because the <see cref="IBackend"/> cast defaults read their input as F16
    /// and narrow by truncation.</summary>
    /// <exception cref="NotSupportedException">The pair of dtypes is not F32 and BF16.</exception>
    public static Tensor CastTo(IBackend backend, Tensor source, DType target)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(source);
        if (source.DType == target)
            return Clone(backend, source);
        bool narrowing = source.DType == DType.F32 && target == DType.BF16;
        bool widening = source.DType == DType.BF16 && target == DType.F32;
        if (!narrowing && !widening)
            throw new NotSupportedException($"Streaming state converts between F32 and BF16 only; asked for {source.DType} to {target}.");
        Tensor cast = new(source.Shape, target);
        try
        {
            if (backend.Device.IsCpu)
            {
                if (narrowing)
                    NarrowOnHost(source, cast);
                else
                    WidenOnHost(source, cast);
            }
            else if (narrowing)
            {
                backend.CastToBf16(cast, source);
            }
            else
            {
                backend.CastToF32(cast, source);
            }

            return cast;
        }
        catch
        {
            cast.Dispose();
            throw;
        }
    }

    private static void NarrowOnHost(Tensor source, Tensor destination)
    {
        ReadOnlySpan<float> values = source.AsReadOnlySpan<float>();
        Span<ushort> bits = destination.AsSpan<ushort>();
        for (int i = 0; i < values.Length; i++)
            bits[i] = BFloat16Bits.FromFloat(values[i]);
    }

    private static void WidenOnHost(Tensor source, Tensor destination)
    {
        ReadOnlySpan<ushort> bits = source.AsReadOnlySpan<ushort>();
        Span<float> values = destination.AsSpan<float>();
        for (int i = 0; i < bits.Length; i++)
            values[i] = BFloat16Bits.ToFloat(bits[i]);
    }

    private static TensorShape WithAxis(TensorShape shape, int axis, long extent)
    {
        long[] dims = new long[shape.Rank];
        for (int d = 0; d < dims.Length; d++)
            dims[d] = d == axis ? extent : shape[d];
        return new TensorShape(dims);
    }
}
