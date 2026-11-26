using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>The stream windows the Duo video and audio decoders share: a window is parts concatenated along one axis, and a crop is a slice.
/// Both run as backend copies on the tensors the caller holds, and nothing is read back to the host here. A window is filled once and then
/// scattered into, so a device backend never uploads a partly written tensor.</summary>
public static class DuoDeviceWindow
{
    /// <summary>Steps <c>[Start, Start + Count)</c> of a source tensor along the window's axis. The source is borrowed.</summary>
    public readonly record struct Part(Tensor Source, int Start, int Count);

    /// <summary>A new tensor holding the parts in order along <paramref name="axis"/>; empty parts are skipped. The caller owns it.</summary>
    /// <exception cref="ArgumentException">No part has steps, or the parts do not agree on the other axes.</exception>
    public static Tensor Concat(IBackend backend, int axis, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(parts);
        List<Part> kept = [];
        foreach (Part part in parts)
        {
            if (part.Count > 0)
            {
                kept.Add(part);
            }
        }
        if (kept.Count == 0)
        {
            throw new ArgumentException("A window needs at least one part with steps.", nameof(parts));
        }
        long[] dims = new long[kept[0].Source.Shape.Rank];
        kept[0].Source.Shape.CopyDimsTo(dims);
        dims[axis] = 0;
        foreach (Part part in kept)
        {
            if (part.Source.Shape.Rank != dims.Length)
            {
                throw new ArgumentException($"A part is {part.Source.Shape}; the window's parts must have the same rank.", nameof(parts));
            }
            dims[axis] += part.Count;
        }
        Tensor window = new(new TensorShape(dims), DType.F32);
        try
        {
            StridedCopies.FillFrom(backend, window, kept[0].Source);
            long at = 0;
            foreach (Part part in kept)
            {
                Tensor piece = Slice(backend, part.Source, axis, part.Start, part.Count);
                try
                {
                    StridedCopies.ScatterAxis(backend, window, piece, axis, at);
                }
                finally
                {
                    piece.Dispose();
                }
                at += part.Count;
            }
        }
        catch
        {
            window.Dispose();
            throw;
        }
        return window;
    }

    /// <summary>Steps <c>[start, start + count)</c> of <paramref name="source"/> along <paramref name="axis"/>, as a new tensor the caller owns.</summary>
    public static Tensor Slice(IBackend backend, Tensor source, int axis, int start, int count)
    {
        long[] dims = new long[source.Shape.Rank];
        source.Shape.CopyDimsTo(dims);
        dims[axis] = count;
        Tensor piece = new(new TensorShape(dims), DType.F32);
        try
        {
            StridedCopies.SliceAxis(backend, piece, source, axis, start, count);
        }
        catch
        {
            piece.Dispose();
            throw;
        }
        return piece;
    }

    /// <summary>A new F32 tensor of <paramref name="shape"/> holding <paramref name="values"/> in row-major order: the host write a caller
    /// makes before it hands a tensor to the backend. Every Duo host upload goes through here.</summary>
    /// <exception cref="ArgumentException">The value count is not the shape's element count.</exception>
    public static Tensor Upload(TensorShape shape, float[] values)
    {
        Tensor tensor = new(shape, DType.F32);
        if (values.Length != tensor.ElementCount)
        {
            tensor.Dispose();
            throw new ArgumentException($"A {shape} tensor holds {tensor.ElementCount} values; got {values.Length}.", nameof(values));
        }
        values.AsSpan().CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    /// <summary>The <see cref="Upload(TensorShape, float[])"/> of a shape given as its dimensions.</summary>
    public static Tensor Upload(long[] dims, float[] values) => Upload(new TensorShape(dims), values);
}
