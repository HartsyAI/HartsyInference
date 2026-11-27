using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.NeighborhoodAttention"/>: F32 <c>[batch, T, H, W, heads, headDim]</c> q, k
/// and v of one shape; an output of that shape or its contiguous <c>[batch · T · H · W, heads · headDim]</c> flattening (the
/// same bytes, so a caller whose next op is row-wise allocates it flat and skips a reshape); a positive window, clamped here to
/// the axes; a finite scale; an output sharing no storage with any input, since it is accumulated while neighbouring queries
/// still read the originals.</summary>
public static class NeighborhoodAttentionContract
{
    public static NeighborhoodGeometry Validate(Tensor output, Tensor q, Tensor k, Tensor v, in NeighborhoodWindow window, float scale)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(q);
        ArgumentNullException.ThrowIfNull(k);
        ArgumentNullException.ThrowIfNull(v);
        if (output.DType != DType.F32 || q.DType != DType.F32 || k.DType != DType.F32 || v.DType != DType.F32)
            throw new NotSupportedException($"NeighborhoodAttention takes F32 tensors; got output {output.DType}, q {q.DType}, k {k.DType}, v {v.DType}.");
        if (q.Shape.Rank != 6)
            throw new ArgumentException($"NeighborhoodAttention expects [batch, T, H, W, heads, headDim]; got {q.Shape}.", nameof(q));
        if (!q.Shape.Equals(k.Shape) || !q.Shape.Equals(v.Shape))
            throw new ArgumentException($"NeighborhoodAttention requires q, k and v of one shape; got q {q.Shape}, k {k.Shape}, v {v.Shape}.", nameof(k));
        if (!OutputShapeMatches(output.Shape, q.Shape))
            throw new ArgumentException($"NeighborhoodAttention output must be the q grid {q.Shape} or its flat [tokens, heads · headDim] form; got {output.Shape}.", nameof(output));
        if (window.Depth <= 0 || window.Height <= 0 || window.Width <= 0)
            throw new ArgumentOutOfRangeException(nameof(window), window, "The neighborhood window must be positive on every axis.");
        if (!float.IsFinite(scale))
            throw new ArgumentOutOfRangeException(nameof(scale), scale, "The score scale must be finite.");
        if (ReferenceEquals(output, q) || ReferenceEquals(output, k) || ReferenceEquals(output, v)
            || output.HasOverlappingHostStorageWithoutSync(q) || output.HasOverlappingHostStorageWithoutSync(k) || output.HasOverlappingHostStorageWithoutSync(v))
            throw new ArgumentException("NeighborhoodAttention output must not share storage with q, k or v.", nameof(output));
        for (int axis = 0; axis < 6; axis++)
        {
            if (q.Shape[axis] <= 0 || q.Shape[axis] > int.MaxValue)
                throw new ArgumentException($"NeighborhoodAttention extents must be positive and within int range; got {q.Shape}.", nameof(q));
        }
        int depth = (int)q.Shape[1], height = (int)q.Shape[2], width = (int)q.Shape[3];
        return new NeighborhoodGeometry((int)q.Shape[0], depth, height, width, (int)q.Shape[4], (int)q.Shape[5],
            Math.Min(window.Depth, depth), Math.Min(window.Height, height), Math.Min(window.Width, width));
    }

    /// <summary>Start of the attended window along one axis: the window is always exactly <paramref name="kernel"/> wide and
    /// slides inward at the borders rather than being truncated, so every query attends the same number of keys.
    /// <paramref name="kernel"/> must already be clamped to <paramref name="length"/>.</summary>
    public static int WindowStart(int index, int length, int kernel)
    {
        int start = index - kernel / 2;
        int last = length - kernel;
        if (start < 0) start = 0;
        if (start > last) start = last;
        return start;
    }

    /// <summary>True when <paramref name="output"/> is the rank-6 q grid or its contiguous <c>[batch · T · H · W, heads · headDim]</c> flattening.</summary>
    public static bool OutputShapeMatches(TensorShape output, TensorShape q)
        => output.Equals(q) || (output.Rank == 2 && output[0] == q[0] * q[1] * q[2] * q[3] && output[1] == q[4] * q[5]);
}
