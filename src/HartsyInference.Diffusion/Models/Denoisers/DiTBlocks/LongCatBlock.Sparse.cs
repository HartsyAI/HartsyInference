using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

// The refinement's self-attention: every segment of the plan through IBackend.BlockSparseAttention, a group of heads at a time.
public sealed partial class LongCatBlock
{
    /// <summary>The most one head-major operand of a block-sparse call holds; the heads are split into groups that stay under it.</summary>
    private const long SparseOperandBytes = 512L << 20;

    private int _sparseHeadGroup;

    /// <summary>Sets the heads one block-sparse call takes, 0 for as many as <see cref="SparseOperandBytes"/> holds; tests use small
    /// groups to exercise the seams.</summary>
    internal void SetSparseHeadGroup(int heads)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(heads);
        _sparseHeadGroup = heads;
    }

    /// <summary>Attends rotated token-major <paramref name="q"/>, <paramref name="k"/> and <paramref name="v"/>
    /// (<c>[S, heads, headDim]</c> F32): segment <c>i</c>'s queries against its key rows with <paramref name="options"/><c>[i]</c>.
    /// Heads do not mix in the operation, so the split into groups does not change the result. Returns <c>[S, heads · headDim]</c>,
    /// which the caller owns.</summary>
    private Tensor BlockSparseSelfAttention(IBackend backend, Tensor q, Tensor k, Tensor v, AttentionSegment[] segments,
        BlockSparseAttentionOptions[] options)
    {
        int heads = _config.NumHeads, headDim = _config.HeadDim;
        int tokens = (int)(q.ElementCount / _config.HiddenSize);
        long[] dims = [tokens, heads, headDim];
        Tensor output = new(new TensorShape(tokens, _config.HiddenSize), DType.F32);
        try
        {
            // A scatter into part of a tensor first uploads it. A fill that covers the whole output places it on the device
            // instead; every element is then overwritten.
            using (Tensor zero = new(new TensorShape(1), DType.F32))
            {
                zero.AsSpan<float>()[0] = 0f;
                StridedCopies.FillFrom(backend, output, zero);
            }
            for (int i = 0; i < segments.Length; i++)
            {
                AttentionSegment segment = segments[i];
                long keyBytes = (long)segment.KeyCount * headDim * sizeof(float);
                int group = _sparseHeadGroup > 0 ? _sparseHeadGroup : (int)Math.Clamp(SparseOperandBytes / keyBytes, 1, heads);
                for (int head = 0; head < heads; head += group)
                {
                    int count = Math.Min(group, heads - head);
                    using Tensor qHeads = HeadGroup(backend, q, dims, segment.QueryStart, segment.QueryCount, head, count);
                    using Tensor kHeads = HeadGroup(backend, k, dims, segment.KeyStart, segment.KeyCount, head, count);
                    using Tensor vHeads = HeadGroup(backend, v, dims, segment.KeyStart, segment.KeyCount, head, count);
                    using Tensor attended = new(qHeads.Shape, DType.F32);
                    backend.BlockSparseAttention(attended, qHeads, kHeads, vHeads, options[i]);
                    StridedView place = HeadGroupView(dims, segment.QueryStart, segment.QueryCount, head, count);
                    backend.CopyStrided(output, attended, StridedCopySpec.Scatter(place));
                }
            }
            return output;
        }
        catch
        {
            output.Dispose();
            throw;
        }
    }

    /// <summary>Rows <c>[start, start + rows)</c> and heads <c>[head, head + count)</c> of a token-major <c>[S, heads, headDim]</c>
    /// operand, seen head-major <c>[count, rows, headDim]</c>.</summary>
    private static StridedView HeadGroupView(long[] dims, int start, int rows, int head, int count) =>
        StridedView.Of(dims).Slice(0, start, rows).Slice(1, head, count).Permute([1, 0, 2]);

    /// <summary>That view copied into a new head-major <c>[1, count, rows, headDim]</c> tensor of the source's dtype.</summary>
    private static Tensor HeadGroup(IBackend backend, Tensor source, long[] dims, int start, int rows, int head, int count)
    {
        Tensor group = new(new TensorShape(1, count, rows, dims[2]), source.DType);
        try
        {
            backend.CopyStrided(group, source, StridedCopySpec.Gather(HeadGroupView(dims, start, rows, head, count)));
            return group;
        }
        catch
        {
            group.Dispose();
            throw;
        }
    }
}
