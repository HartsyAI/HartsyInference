using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
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

    /// <summary>Attends head-major operands with key and value rows that may outnumber the queries: <paramref name="q"/>
    /// <c>[B, heads, Sq, headDim]</c>, <paramref name="k"/> and <paramref name="v"/> <c>[B, heads, Skv, headDim]</c>. Each segment's queries read
    /// the key ranges it names. Nothing is masked: a key outside a segment's ranges is not in its call, so there is no <c>[Sq, Skv]</c> bias and every
    /// call is the layer's ordinary unmasked attention. The operands stay the caller's. Returns <c>[B, heads, Sq, headDim]</c>, which the caller owns.</summary>
    /// <remarks>Equal to one attention with a bias of 0 on the named pairs and negative infinity elsewhere. Segments that name the same ranges share
    /// one gathered copy of the keys and values, and a segment that reads every key reads the operands themselves.</remarks>
    public static Tensor RunHeadMajor(IBackend backend, AttentionLayer layer, Tensor q, Tensor k, Tensor v, ReadOnlySpan<KeyRangeSegment> segments)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(layer);
        if (q.Shape.Rank != 4 || k.Shape.Rank != 4 || !v.Shape.Equals(k.Shape))
            throw new ArgumentException($"Segmented attention takes head-major [B, heads, S, headDim] operands; got {q.Shape}, {k.Shape}, {v.Shape}.");
        int queryTokens = (int)q.Shape[2], keyTokens = (int)k.Shape[2];
        KeyRangeSegment.Validate(segments, queryTokens, keyTokens);
        if (segments.Length == 1 && ReadsEveryKey(segments[0], keyTokens))
            return layer.AttendHeadMajorOperands(backend, q, k, v, default, OperandOwnership.None);

        List<Tensor> parts = new(segments.Length);
        Tensor? keys = null, values = null;
        try
        {
            for (int i = 0; i < segments.Length; i++)
            {
                KeyRangeSegment s = segments[i];
                bool every = ReadsEveryKey(s, keyTokens);
                if (!every && (keys is null || !SameKeys(s, segments[i - 1])))
                {
                    keys?.Dispose();
                    values?.Dispose();
                    keys = values = null;
                    keys = GatherTokens(backend, k, s);
                    values = GatherTokens(backend, v, s);
                }
                using Tensor queries = SliceTokens(backend, q, s.QueryStart, s.QueryCount);
                parts.Add(layer.AttendHeadMajorOperands(backend, queries, every ? k : keys!, every ? v : values!, default, OperandOwnership.None));
                if (every || i + 1 == segments.Length || !SameKeys(segments[i + 1], s))
                {
                    keys?.Dispose();
                    values?.Dispose();
                    keys = values = null;
                }
            }

            if (parts.Count == 1)
            {
                Tensor only = parts[0];
                parts.Clear();
                return only;
            }
            Tensor joined = new(new TensorShape(q.Shape[0], q.Shape[1], queryTokens, q.Shape[3]), parts[0].DType);
            backend.Concat(joined, parts.ToArray(), 2);
            return joined;
        }
        finally
        {
            keys?.Dispose();
            values?.Dispose();
            foreach (Tensor part in parts)
                part.Dispose();
        }
    }

    private static bool ReadsEveryKey(in KeyRangeSegment segment, int keyTokens) => segment.KeyCount == keyTokens;

    private static bool SameKeys(in KeyRangeSegment a, in KeyRangeSegment b)
    {
        if (ReferenceEquals(a.Keys, b.Keys))
            return true;
        if (a.Keys.Count != b.Keys.Count)
            return false;
        for (int i = 0; i < a.Keys.Count; i++)
        {
            if (a.Keys[i] != b.Keys[i])
                return false;
        }
        return true;
    }

    /// <summary>The key or value rows a segment names, in order, as one head-major tensor.</summary>
    private static Tensor GatherTokens(IBackend backend, Tensor source, in KeyRangeSegment segment)
    {
        if (segment.Keys.Count == 1)
            return SliceTokens(backend, source, segment.Keys[0].Start, segment.Keys[0].Count);
        Tensor[] slices = new Tensor[segment.Keys.Count];
        try
        {
            for (int i = 0; i < slices.Length; i++)
                slices[i] = SliceTokens(backend, source, segment.Keys[i].Start, segment.Keys[i].Count);
            Tensor gathered = new(new TensorShape(source.Shape[0], source.Shape[1], segment.KeyCount, source.Shape[3]), source.DType);
            backend.Concat(gathered, slices, 2);
            return gathered;
        }
        finally
        {
            foreach (Tensor? slice in slices)
                slice?.Dispose();
        }
    }

    private static Tensor SliceTokens(IBackend backend, Tensor source, int start, int count)
    {
        Tensor slice = new(new TensorShape(source.Shape[0], source.Shape[1], count, source.Shape[3]), source.DType);
        StridedCopies.SliceAxis(backend, slice, source, 2, start, count);
        return slice;
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
