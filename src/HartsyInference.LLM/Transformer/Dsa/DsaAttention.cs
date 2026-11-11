using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer.Sparse;

namespace HartsyInference.LLM.Transformer.Dsa;

/// <summary>Host reference for one DSA attention layer (vLLM-ERNIE <c>Ernie5_5DSAAttention</c>, <c>compress_ratio</c> -2): low-rank
/// queries and a shared key/value latent (MLA), RoPE on the trailing query/key slice, a per-head sink, a sparse key set of an indexer
/// top-k plus a sliding window, an optional head mix, an optional sigmoid output gate, and the output projection.</summary>
/// <remarks>
/// <para>The attention runs in the absorbed MQA form the reference's sparse kernel uses: each head's no-position query is multiplied
/// into the latent space by its <c>kv_b_proj</c> key rows, every head attends the same <c>[latent | rotated key]</c> rows, the value is
/// the latent prefix of each row, and each head's latent output is expanded by its value rows. Every dot product equals the
/// expanded form's, so the scale stays <c>QkHeadDim^-0.5</c>. The per-query attention goes through <see cref="IBackend.Attention"/>
/// over the gathered rows of that query's sparse set.</para>
/// <para>Latent rows are cached in F32. The reference may store them in an FP8 MLA cache, depending on the serving configuration.</para>
/// </remarks>
public sealed class DsaAttention : IDisposable
{
    private readonly IBackend _backend;
    private readonly DsaAttentionSettings _s;
    private readonly DsaAttentionWeights _w;
    private readonly IRopeRowSource _rope;
    private readonly DsaIndexer _indexer;
    private Tensor? _keyAbsorb;
    private Tensor? _valueExpand;

    /// <param name="backend">Runs every projection, norm, rotation and the attention itself.</param>
    /// <param name="settings">Shape of this layer.</param>
    /// <param name="weights">This layer's weights; <c>KvB</c> is folded into absorption matrices here and not read afterwards.</param>
    /// <param name="rope">Rotary rows for the main query/key slice, <c>QkRopeDim / 2</c> wide, covering every position that will be seen.</param>
    /// <param name="indexerRope">Rotary rows for the indexer slice, <c>Indexer.RopeDim / 2</c> wide; ERNIE 5.5 uses the same plain table.</param>
    public DsaAttention(IBackend backend, DsaAttentionSettings settings, DsaAttentionWeights weights, IRopeRowSource rope, IRopeRowSource indexerRope)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(rope);
        ArgumentNullException.ThrowIfNull(indexerRope);
        settings.Validate();
        if (rope.HalfDim * 2 != settings.QkRopeDim) throw new ArgumentException("The rope table width must equal QkRopeDim.", nameof(rope));
        int h = settings.Heads, qkv = settings.QkNopeDim + settings.VDim;
        DsaIndexer.RequireShape(weights.QA, settings.QLoraRank, settings.Dim, "q_a_proj");
        DsaIndexer.RequireShape(weights.KvA, settings.LatentDim, settings.Dim, "kv_a_proj_with_mqa");
        DsaIndexer.RequireShape(weights.QB, h * settings.QkHeadDim, settings.QLoraRank, "q_b_proj");
        DsaIndexer.RequireShape(weights.KvB, h * qkv, settings.KvLoraRank, "kv_b_proj");
        DsaIndexer.RequireShape(weights.WoB, settings.Dim, h * settings.VDim, "wo_b");
        if (weights.OGate is { } gate) DsaIndexer.RequireShape(gate, h * settings.VDim, settings.QLoraRank, "o_gate");
        if (weights.QANorm.ElementCount != settings.QLoraRank || weights.KvANorm.ElementCount != settings.KvLoraRank)
            throw new ArgumentException("The latent norms must match QLoraRank and KvLoraRank.", nameof(weights));
        if (weights.Sink is { } sink && (sink.DType != DType.F32 || sink.ElementCount != h))
            throw new ArgumentException("The sink must be F32 with one logit per head.", nameof(weights));
        if (weights.HeadMix is { } mix && (mix.DType != DType.F32 || mix.Shape.Rank != 2 || mix.Shape[0] != h || mix.Shape[1] != h))
            throw new ArgumentException("The head mix must be F32 [Heads, Heads].", nameof(weights));
        _backend = backend;
        _s = settings;
        _w = weights;
        _rope = rope;
        _indexer = new DsaIndexer(backend, settings.Indexer, weights.Indexer, indexerRope, settings.Dim, settings.QLoraRank);
        (_keyAbsorb, _valueExpand) = FoldKvB(weights.KvB, settings);
    }

    /// <summary>The shape this layer was built with, used to size its per-sequence state.</summary>
    public DsaAttentionSettings Settings => _s;

    /// <summary>Runs the layer over <paramref name="tokens"/> rows at positions <c>startPos ..</c> and writes <c>[tokens, Dim]</c> to
    /// <paramref name="y"/>. Any chunk length works; a call at position 0 starts a new sequence, any other must continue the state.</summary>
    /// <remarks>The indexer scores each chunk as one F32 <c>[tokens, startPos + tokens]</c> buffer (<see cref="DsaIndexer.Select"/>), so a
    /// long prompt should be prefilled in bounded chunks.</remarks>
    /// <param name="x">Normalized hidden states, <c>[tokens, Dim]</c>.</param>
    public void Forward(ReadOnlySpan<float> x, int tokens, int startPos, DsaAttentionState state, Span<float> y)
    {
        ArgumentNullException.ThrowIfNull(state);
        DsaAttentionSettings s = _s;
        if (tokens < 1) throw new ArgumentOutOfRangeException(nameof(tokens));
        if (x.Length != (long)tokens * s.Dim || y.Length != x.Length) throw new ArgumentException("x and y must each hold tokens x Dim values.");
        if (startPos != 0 && startPos != state.Length) throw new InvalidOperationException($"The state holds {state.Length} positions; this call starts at {startPos}.");
        if (startPos + tokens > state.Capacity) throw new InvalidOperationException("The DSA cache is full; allocate state for a longer sequence.");
        if (startPos + tokens > _rope.Length) throw new ArgumentOutOfRangeException(nameof(startPos), "The rope table does not reach this position.");
        Tensor keyAbsorb = _keyAbsorb ?? throw new ObjectDisposedException(nameof(DsaAttention));
        Tensor valueExpand = _valueExpand!;
        if (startPos == 0) state.Reset();
        int h = s.Heads, kvr = s.KvLoraRank, rd = s.QkRopeDim, latent = s.LatentDim, endPos = startPos + tokens;
        int[] positions = HostOps.Range(startPos, tokens);

        float[] qLatent = HostOps.Project(_backend, x, tokens, s.Dim, _w.QA, s.QLoraRank);
        HostOps.RmsNorm(_backend, qLatent, s.QLoraRank, _w.QANorm, s.NormEps);
        float[] q = HostOps.Project(_backend, qLatent, tokens, s.QLoraRank, _w.QB, h * s.QkHeadDim);
        PartialRotary.Rotate(_backend, _rope, rd, q, [1, tokens, h, s.QkHeadDim], positions, inverse: false, s.RopePairing);

        // the latent row is [RMS-normed latent | rotated shared key], cached at its position
        float[] kv = HostOps.Project(_backend, x, tokens, s.Dim, _w.KvA, latent);
        float[] kvLatent = new float[tokens * kvr], keyPe = new float[tokens * rd];
        for (int t = 0; t < tokens; t++)
        {
            kv.AsSpan(t * latent, kvr).CopyTo(kvLatent.AsSpan(t * kvr, kvr));
            kv.AsSpan(t * latent + kvr, rd).CopyTo(keyPe.AsSpan(t * rd, rd));
        }
        HostOps.RmsNorm(_backend, kvLatent, kvr, _w.KvANorm, s.NormEps);
        PartialRotary.Rotate(_backend, _rope, rd, keyPe, [1, tokens, rd], positions, inverse: false, s.RopePairing);
        Span<float> cache = state.Latent.AsSpan<float>();
        for (int t = 0; t < tokens; t++)
        {
            kvLatent.AsSpan(t * kvr, kvr).CopyTo(cache.Slice((startPos + t) * latent, kvr));
            keyPe.AsSpan(t * rd, rd).CopyTo(cache.Slice((startPos + t) * latent + kvr, rd));
        }

        _indexer.AppendKeys(x, tokens, startPos, state.IndexKeys);
        state.Length = endPos;
        int[] limits = new int[tokens];
        for (int t = 0; t < tokens; t++) limits[t] = SparseWindowMerge.IndexerLimit(positions[t], s.Window);
        int[] topk = _indexer.Select(x, qLatent, positions, limits, state.IndexKeys, out int topkCols);
        int[] rows = MergeWindow(topk, topkCols, positions, out int rowCols, out int rowWidth);

        float[] query = AbsorbQueries(q, tokens, keyAbsorb);
        float[] latentOut = Attend(query, tokens, positions, rows, rowCols, rowWidth, state);
        float[] o = ExpandValues(latentOut, tokens, valueExpand);

        if (_w.OGate is { } gateWeight)
            AttentionOutputGate.ApplySigmoid(o, HostOps.Project(_backend, qLatent, tokens, s.QLoraRank, gateWeight, h * s.VDim));
        HostOps.Project(_backend, o, tokens, h * s.VDim, _w.WoB, s.Dim).CopyTo(y);
    }

    // [tokens, heads, QkHeadDim] queries to [tokens, heads, LatentDim]: the no-position part times each head's key rows, then the rotated part.
    private float[] AbsorbQueries(float[] q, int tokens, Tensor keyAbsorb)
    {
        int h = _s.Heads, nope = _s.QkNopeDim, rd = _s.QkRopeDim, qk = _s.QkHeadDim, kvr = _s.KvLoraRank, latent = _s.LatentDim;
        using Tensor nopeT = new(new TensorShape(h, tokens, nope), DType.F32);
        Span<float> nopeHm = nopeT.AsSpan<float>();
        for (int t = 0; t < tokens; t++)
            for (int head = 0; head < h; head++)
                q.AsSpan((t * h + head) * qk, nope).CopyTo(nopeHm.Slice((head * tokens + t) * nope, nope));
        using Tensor absorbed = new(new TensorShape(h, tokens, kvr), DType.F32);
        _backend.BatchedMatMul(absorbed, nopeT, keyAbsorb);
        ReadOnlySpan<float> a = absorbed.AsReadOnlySpan<float>();
        float[] query = new float[tokens * h * latent];
        for (int t = 0; t < tokens; t++)
            for (int head = 0; head < h; head++)
            {
                int dst = (t * h + head) * latent;
                a.Slice((head * tokens + t) * kvr, kvr).CopyTo(query.AsSpan(dst, kvr));
                q.AsSpan((t * h + head) * qk + nope, rd).CopyTo(query.AsSpan(dst + kvr, rd));
            }
        return query;
    }

    // Every query's [topk | window | -1 ...] row through IBackend.MergeWindowTopK. With no indexer columns yet (no query is past its
    // window) a one-column placeholder stands in; no query takes from it, since each takes min(max(seq - window, 0), cols) = 0.
    private int[] MergeWindow(int[] topk, int topkCols, int[] positions, out int cols, out int width)
    {
        int tokens = positions.Length;
        cols = Math.Max(topkCols, 1);
        width = cols + _s.Window;
        int[] source = topk;
        if (topkCols == 0)
        {
            source = new int[tokens];
            Array.Fill(source, -1);
        }
        using Tensor topkT = HostMath.Tensor(source, tokens, cols);
        using Tensor positionT = HostMath.Tensor(positions, tokens);
        using Tensor rows = new(new TensorShape(tokens, width), DType.I32);
        _backend.MergeWindowTopK(rows, topkT, positionT, _s.Window);
        return rows.AsReadOnlySpan<int>().ToArray();
    }

    // Each query attends its own [topk | window] rows; K is the whole latent row, V its latent prefix (MQA, one KV head).
    private float[] Attend(float[] query, int tokens, int[] positions, int[] rows, int rowCols, int rowWidth, DsaAttentionState state)
    {
        int h = _s.Heads, kvr = _s.KvLoraRank, latent = _s.LatentDim, win = _s.Window;
        float[] latentOut = new float[tokens * h * kvr];
        for (int t = 0; t < tokens; t++)
        {
            int seq = positions[t] + 1, picked = Math.Min(Math.Max(seq - win, 0), rowCols);
            int count = picked + Math.Min(seq, win);
            // A NaN indexer score ranks below the masked tail and can leave a -1 among the picks; name it here, not in GatherRows.
            for (int i = 0; i < picked; i++)
            {
                if ((uint)rows[t * rowWidth + i] >= (uint)(seq - win))
                    throw new InvalidOperationException($"DSA top-k entry {rows[t * rowWidth + i]} for position {positions[t]} is "
                        + $"outside [0, {seq - win}); the indexer scores held a NaN.");
            }
            using Tensor keys = new(new TensorShape(count, latent), DType.F32);
            _backend.GatherRows(keys, state.Latent, rows.AsSpan(t * rowWidth, count));
            using Tensor qT = HostMath.Tensor(query.AsSpan(t * h * latent, h * latent), 1, h * latent);
            using Tensor oT = new(new TensorShape(1, h * kvr), DType.F32);
            AttentionCall call = new()
            {
                Query = AttentionOperand.TokenMajor(qT, h, latent),
                Key = AttentionOperand.Packed(keys, count, latent, 0, 1, latent),
                Value = AttentionOperand.Packed(keys, count, latent, 0, 1, kvr),
                Output = AttentionOperand.TokenMajor(oT, h, kvr),
                Scale = _s.Scale,
                Logits = new AttentionLogitTerms(0f, _w.Sink, null),
            };
            _backend.Attention(in call);
            oT.AsReadOnlySpan<float>().CopyTo(latentOut.AsSpan(t * h * kvr, h * kvr));
        }
        return latentOut;
    }

    // [tokens, heads, KvLoraRank] latent outputs to [tokens, heads * VDim] values, through the optional head mix.
    private float[] ExpandValues(float[] latentOut, int tokens, Tensor valueExpand)
    {
        int h = _s.Heads, kvr = _s.KvLoraRank, vd = _s.VDim;
        using Tensor latentHm = new(new TensorShape(h, tokens, kvr), DType.F32);
        Span<float> l = latentHm.AsSpan<float>();
        for (int t = 0; t < tokens; t++)
            for (int head = 0; head < h; head++)
                latentOut.AsSpan((t * h + head) * kvr, kvr).CopyTo(l.Slice((head * tokens + t) * kvr, kvr));
        using Tensor values = new(new TensorShape(h, tokens, vd), DType.F32);
        _backend.BatchedMatMul(values, latentHm, valueExpand);
        Tensor result = values;
        using Tensor? mixed = _w.HeadMix is null ? null : new Tensor(new TensorShape(h, tokens * vd), DType.F32);
        if (_w.HeadMix is { } mix)
        {
            using Tensor flat = values.Reshape(new TensorShape(h, tokens * vd));
            HeadMix.Apply(_backend, mix, flat, mixed!);
            result = mixed!;
        }
        ReadOnlySpan<float> r = result.AsReadOnlySpan<float>();
        float[] o = new float[tokens * h * vd];
        for (int head = 0; head < h; head++)
            for (int t = 0; t < tokens; t++)
                r.Slice((head * tokens + t) * vd, vd).CopyTo(o.AsSpan((t * h + head) * vd, vd));
        return o;
    }

    // Splits kv_b_proj into per-head key-absorption [heads, nope, kvr] and value-expansion [heads, kvr, vd] matrices, once at load.
    private static (Tensor KeyAbsorb, Tensor ValueExpand) FoldKvB(Tensor kvB, DsaAttentionSettings s)
    {
        int h = s.Heads, nope = s.QkNopeDim, vd = s.VDim, kvr = s.KvLoraRank, per = nope + vd;
        Tensor f32 = ProjectionOps.EnsureF32(kvB);
        try
        {
            ReadOnlySpan<float> w = f32.AsReadOnlySpan<float>();
            Tensor keyAbsorb = new(new TensorShape(h, nope, kvr), DType.F32);
            Tensor valueExpand = new(new TensorShape(h, kvr, vd), DType.F32);
            Span<float> k = keyAbsorb.AsSpan<float>(), v = valueExpand.AsSpan<float>();
            for (int head = 0; head < h; head++)
            {
                w.Slice(head * per * kvr, nope * kvr).CopyTo(k.Slice(head * nope * kvr, nope * kvr));
                for (int j = 0; j < vd; j++)
                {
                    ReadOnlySpan<float> src = w.Slice((head * per + nope + j) * kvr, kvr);
                    for (int c = 0; c < kvr; c++) v[(head * kvr + c) * vd + j] = src[c];
                }
            }
            return (keyAbsorb, valueExpand);
        }
        finally
        {
            if (!ReferenceEquals(f32, kvB)) f32.Dispose();
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Interlocked.Exchange(ref _keyAbsorb, null)?.Dispose();
        Interlocked.Exchange(ref _valueExpand, null)?.Dispose();
        _indexer.Dispose();
    }
}
