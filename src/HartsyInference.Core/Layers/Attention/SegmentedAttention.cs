using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Layers.Attention;

/// <summary>Runs self-attention as one call per <see cref="AttentionSegment"/> over operands that are already normed and rotated, and joins the
/// outputs in query order. Each call is the layer's ordinary head-major attention, so a backend's fused kernels serve it unchanged.</summary>
public static class SegmentedAttention
{
    /// <summary>Attends token-major <paramref name="q"/>, <paramref name="k"/> and <paramref name="v"/> (<c>[S, heads, headDim]</c>, F32 or F16)
    /// per segment. The operands stay the caller's. Returns <c>[S, heads · headDim]</c>, which the caller owns.</summary>
    public static Tensor Run(IBackend backend, AttentionLayer layer, Tensor q, Tensor k, Tensor v, ReadOnlySpan<AttentionSegment> segments)
    {
        AttentionLayerConfig config = layer.Config;
        int heads = config.Heads, headDim = config.HeadDim, dim = config.InnerDim;
        if (config.KvHeads != heads)
            throw new NotSupportedException("Segmented attention runs equal query and key/value heads.");
        long elements = q.ElementCount;
        if (elements % dim != 0 || k.ElementCount != elements || v.ElementCount != elements)
            throw new ArgumentException($"Segmented attention takes equal [S, {heads}, {headDim}] operands; got {q.Shape}, {k.Shape}, {v.Shape}.");
        int tokens = (int)(elements / dim);
        AttentionSegment.Validate(segments, tokens);
        DType dtype = q.DType;

        Tensor? sharedKeys = null, sharedValues = null;
        List<Tensor> parts = new(segments.Length);
        try
        {
            foreach (AttentionSegment s in segments)
            {
                bool fullKeys = s.KeyStart == 0 && s.KeyCount == tokens;
                Tensor qMh = HeadMajorRows(backend, q, s.QueryStart, s.QueryCount, tokens, heads, headDim, dtype);
                Tensor kMh, vMh;
                OperandOwnership consume = OperandOwnership.Query;
                if (fullKeys)
                {
                    sharedKeys ??= HeadMajorRows(backend, k, 0, tokens, tokens, heads, headDim, dtype);
                    sharedValues ??= HeadMajorRows(backend, v, 0, tokens, tokens, heads, headDim, dtype);
                    (kMh, vMh) = (sharedKeys, sharedValues);
                }
                else
                {
                    kMh = HeadMajorRows(backend, k, s.KeyStart, s.KeyCount, tokens, heads, headDim, dtype);
                    vMh = HeadMajorRows(backend, v, s.KeyStart, s.KeyCount, tokens, heads, headDim, dtype);
                    consume = OperandOwnership.All;
                }
                Tensor attn = layer.AttendHeadMajorOperands(backend, qMh, kMh, vMh, default, consume);
                parts.Add(layer.MergeHeads(backend, attn, new TensorShape(s.QueryCount, dim)));
            }

            if (parts.Count == 1)
            {
                Tensor only = parts[0];
                parts.Clear();
                return only;
            }
            Tensor joined = new(new TensorShape(tokens, dim), dtype);
            backend.Concat(joined, parts.ToArray(), 0);
            return joined;
        }
        finally
        {
            sharedKeys?.Dispose();
            sharedValues?.Dispose();
            foreach (Tensor part in parts)
                part.Dispose();
        }
    }

    /// <summary>Rows <c>[start, start + count)</c> of a token-major operand as head-major <c>[1, heads, count, headDim]</c>.</summary>
    private static Tensor HeadMajorRows(IBackend backend, Tensor source, int start, int count, int tokens, int heads, int headDim, DType dtype)
    {
        int dim = heads * headDim;
        Tensor? rows = null;
        try
        {
            if (start != 0 || count != tokens)
            {
                rows = new Tensor(new TensorShape(count, dim), dtype);
                backend.SliceRows(rows, source, start);
            }
            Tensor headMajor = new(new TensorShape(1, heads, count, headDim), dtype);
            backend.Permute0213(headMajor, rows ?? source, count, heads, headDim);
            return headMajor;
        }
        finally
        {
            rows?.Dispose();
        }
    }
}
