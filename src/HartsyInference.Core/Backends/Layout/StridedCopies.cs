using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Layout;

/// <summary>The common shapes of <see cref="IBackend.CopyStrided"/>, named: slicing and scattering along one axis
/// and repeating an axis in place (<c>repeat_interleave</c>). A model that needs another layout builds its own
/// <see cref="StridedView"/> and calls the backend directly.</summary>
public static class StridedCopies
{
    /// <summary><c>output = input[..., start : start + length, ...]</c> along <paramref name="axis"/>; the output is the slice's row-major shape.</summary>
    public static void SliceAxis(IBackend backend, Tensor output, Tensor input, int axis, long start, long length)
        => backend.CopyStrided(output, input, StridedCopySpec.Gather(StridedView.Of(input.Shape).Slice(axis, start, length)));

    /// <summary><c>output[..., start : start + input.Shape[axis], ...] = input</c> along <paramref name="axis"/>, in place; the rest of the output keeps its contents.</summary>
    public static void ScatterAxis(IBackend backend, Tensor output, Tensor input, int axis, long start)
        => backend.CopyStrided(output, input, StridedCopySpec.Scatter(StridedView.Of(output.Shape).Slice(axis, start, input.Shape[axis])));

    /// <summary>Repeats every position of <paramref name="axis"/> <paramref name="repeats"/> times in place, so output
    /// position <c>p</c> reads input position <c>p / repeats</c>: the GQA key/value head expansion
    /// (<c>[B,Hkv,L,D] → [B,Hkv·g,L,D]</c>, HF <c>repeat_kv</c>) and any other <c>repeat_interleave</c>.</summary>
    public static void RepeatInterleave(IBackend backend, Tensor output, Tensor input, int axis, long repeats)
    {
        if (repeats <= 0)
            throw new ArgumentOutOfRangeException(nameof(repeats), repeats, "The repeat count must be positive.");
        StridedView source = StridedView.Of(input.Shape).Broadcast(axis + 1, repeats);
        if (output.ElementCount != source.ElementCount)
            throw new ArgumentException($"RepeatInterleave of {input.Shape} by {repeats} on axis {axis} needs {source.ElementCount} output elements; got {output.Shape}.", nameof(output));
        backend.CopyStrided(output, input, StridedCopySpec.Gather(source));
    }
}
