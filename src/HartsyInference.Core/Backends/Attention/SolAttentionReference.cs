using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>What a Sol-Attn call routes and adds, beyond its operands.</summary>
public sealed record SolAttentionOptions
{
    /// <summary>The routing threshold, in standard deviations of a query block's scores across key blocks. Ignored when
    /// <see cref="TopKRatio"/> is set. A block is routed when its score is above the threshold, so a threshold far
    /// below zero routes every block.</summary>
    public float Tau { get; init; } = 1f;

    /// <summary>The score multiplier; null is <c>headDim^-1/2</c>.</summary>
    public float? Scale { get; init; }

    /// <summary>Key blocks <c>[Start, End)</c> every query block reads exactly. They take no part in top-k ranking and
    /// do not count against its budget.</summary>
    public (int Start, int End) SinkBlocks { get; init; }

    /// <summary>Query blocks <c>[Start, End)</c> that read every key block exactly.</summary>
    public (int Start, int End) SinkQueries { get; init; }

    /// <summary>An additive natural-log bias per key, F32 <c>[T]</c> or <c>[B, T]</c>; negative infinity masks a key. It
    /// applies only where the key's block is read exactly: a block that is pooled ignores it, so biased keys belong
    /// under <see cref="SinkBlocks"/>.</summary>
    public Tensor? KeyBias { get; init; }

    /// <summary>When above zero, the fraction of non-sink key blocks each query block routes by rank, in place of the
    /// <see cref="Tau"/> threshold; below one.</summary>
    public float TopKRatio { get; init; }

    /// <summary>Whether key blocks that are not read exactly still take part, each as one pooled term. Off, the softmax
    /// runs over the exactly read keys only.</summary>
    public bool Tail { get; init; } = true;

    /// <summary>Live rows per block, one entry per block, each clamped to 1 to 64; null means every row that exists is
    /// live. A row past its block's length is dead: it is no key, and it is left out of every block mean and sum.</summary>
    public IReadOnlyList<int>? BlockLengths { get; init; }

    /// <summary>A per-token, per-channel gate, F32 in the operands' shape. With it, a second, dense attention over block
    /// means is added to the output, multiplied by the gate.</summary>
    public Tensor? CoarseGate { get; init; }
}

/// <summary>The definition of general Sol-Attn block-sparse self-attention, written from Comfy Kitchen's eager
/// definition at the pinned commit (benchmarks/attention/protocol.json) and checked against its outputs.</summary>
/// <remarks>
/// <para>Queries and keys are the same sequence, cut into blocks of <see cref="Block"/> tokens. For each (batch, head),
/// every query block picks the key blocks it reads token by token: its neighbours, the sink blocks, and those whose
/// pooled score against the block's mean query passes a threshold (<see cref="SolAttentionOptions.Tau"/>) or ranks high
/// enough (<see cref="SolAttentionOptions.TopKRatio"/>). Each remaining key block enters the same softmax once, as a
/// pooled term scored against the block's mean query and weighted by its live length: the tail.</para>
/// <para>Operands are F32 <c>[B, T, H, D]</c>, token-major, with one head count for Q, K and V. It computes in F32, in
/// plain loops, and holds no approximation: a GPU port is compared with it, not the other way round. This is a
/// different operation from the checkpoint-specific video sparse attention behind
/// <see cref="IVideoSparseAttentionSession"/>, which it does not replace.</para>
/// </remarks>
public static class SolAttentionReference
{
    /// <summary>Tokens per block.</summary>
    public const int Block = 64;

    private const float Log2E = 1.4426950408889634f;

    /// <summary>Blocks of a sequence of <paramref name="tokens"/> tokens.</summary>
    public static int Blocks(int tokens) => (tokens + Block - 1) / Block;

    /// <summary>Key blocks a query block routes by rank among <paramref name="blocks"/> blocks of which
    /// <paramref name="sinkBlocks"/> are sinks: <c>round(ratio × non-sink blocks)</c> to nearest even, at least one, and
    /// always leaving one non-sink block unrouted; zero when at most one block is not a sink.</summary>
    public static int TopKBudget(float ratio, int blocks, int sinkBlocks)
    {
        int eligible = blocks - sinkBlocks;
        int wanted = Math.Max(1, (int)Math.Round((double)ratio * eligible, MidpointRounding.ToEven));
        return Math.Max(0, Math.Min(eligible - 1, wanted));
    }

    /// <summary>Attends <paramref name="query"/> over <paramref name="key"/> and <paramref name="value"/> into
    /// <paramref name="output"/>, all F32 <c>[B, T, H, D]</c>. With <paramref name="routed"/>, also reports which key
    /// blocks each query block read exactly, as <c>[B, H, queryBlock, keyBlock]</c>.</summary>
    /// <exception cref="ArgumentException">An operand or option does not fit the call.</exception>
    public static void Run(Tensor output, Tensor query, Tensor key, Tensor value, SolAttentionOptions options, Span<bool> routed = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        Check(output, query, key, value, options);
        int batch = (int)query.Shape[0], tokens = (int)query.Shape[1], heads = (int)query.Shape[2], dim = (int)query.Shape[3];
        int blocks = Blocks(tokens);
        if (!routed.IsEmpty && routed.Length != (long)batch * heads * blocks * blocks)
            throw new ArgumentException($"The routing report holds {routed.Length} entries; [B, H, blocks, blocks] needs {(long)batch * heads * blocks * blocks}.", nameof(routed));
        float scale = options.Scale ?? 1f / MathF.Sqrt(dim);
        float log2s = scale * Log2E;
        int[] lengths = LiveLengths(options.BlockLengths, tokens, blocks);
        bool deadRows = options.BlockLengths is not null;
        int sinkStart = Math.Min(options.SinkBlocks.Start, blocks), sinkEnd = Math.Min(options.SinkBlocks.End, blocks);
        int sinkQueryStart = Math.Min(options.SinkQueries.Start, blocks), sinkQueryEnd = Math.Min(options.SinkQueries.End, blocks);
        int budget = options.TopKRatio > 0f ? TopKBudget(options.TopKRatio, blocks, Math.Max(0, sinkEnd - sinkStart)) : -1;

        ReadOnlySpan<float> q = query.AsReadOnlySpan<float>(), k = key.AsReadOnlySpan<float>(), v = value.AsReadOnlySpan<float>();
        ReadOnlySpan<float> bias = options.KeyBias is null ? default : options.KeyBias.AsReadOnlySpan<float>();
        bool biasPerBatch = options.KeyBias is { } b && b.Shape.Rank == 2 && b.Shape[0] == batch && batch > 1;
        ReadOnlySpan<float> gate = options.CoarseGate is null ? default : options.CoarseGate.AsReadOnlySpan<float>();
        Span<float> o = output.AsSpan<float>();

        float[] keyMeans = new float[blocks * dim], valueSums = new float[blocks * dim], centroids = new float[blocks * dim];
        float[] keyMean = new float[dim], centered = new float[blocks * dim], variance = new float[dim];
        float[] pooled = new float[blocks * blocks], tokenLogits = new float[tokens], row = new float[dim], ranked = new float[blocks];
        bool[] exact = new bool[blocks * blocks];
        float[] coarse = options.CoarseGate is null ? [] : new float[blocks * dim];

        for (int bi = 0; bi < batch; bi++)
            for (int h = 0; h < heads; h++)
            {
                // Block summaries over live rows: mean K, summed V, mean Q.
                Array.Clear(keyMeans);
                Array.Clear(valueSums);
                Array.Clear(centroids);
                for (int n = 0; n < blocks; n++)
                {
                    for (int t = n * Block; t < n * Block + lengths[n]; t++)
                    {
                        int at = Row(bi, t, h, tokens, heads, dim);
                        for (int d = 0; d < dim; d++)
                        {
                            keyMeans[n * dim + d] += k[at + d];
                            valueSums[n * dim + d] += v[at + d];
                            centroids[n * dim + d] += q[at + d];
                        }
                    }
                    for (int d = 0; d < dim; d++)
                    {
                        keyMeans[n * dim + d] /= lengths[n];
                        centroids[n * dim + d] /= lengths[n];
                    }
                }
                // Keys are centered on the mean of the block means, and scores are taken against the centered keys.
                for (int d = 0; d < dim; d++)
                {
                    float sum = 0f;
                    for (int n = 0; n < blocks; n++) sum += keyMeans[n * dim + d];
                    keyMean[d] = sum / blocks;
                    float squares = 0f;
                    for (int n = 0; n < blocks; n++)
                    {
                        float c = keyMeans[n * dim + d] - keyMean[d];
                        centered[n * dim + d] = c;
                        squares += c * c;
                    }
                    variance[d] = squares / blocks;
                }
                for (int qb = 0; qb < blocks; qb++)
                    for (int n = 0; n < blocks; n++)
                        pooled[qb * blocks + n] = Dot(centroids.AsSpan(qb * dim, dim), centered.AsSpan(n * dim, dim)) * log2s;

                // Routing: which key blocks each query block reads token by token.
                for (int qb = 0; qb < blocks; qb++)
                {
                    Span<bool> reads = exact.AsSpan(qb * blocks, blocks);
                    ReadOnlySpan<float> scores = pooled.AsSpan(qb * blocks, blocks);
                    if (budget >= 0)
                    {
                        float threshold = float.PositiveInfinity;
                        if (budget > 0)
                        {
                            for (int n = 0; n < blocks; n++) ranked[n] = n >= sinkStart && n < sinkEnd ? float.NegativeInfinity : scores[n];
                            Array.Sort(ranked);
                            threshold = ranked[blocks - budget];
                        }
                        // A group tied at the boundary is kept whole.
                        for (int n = 0; n < blocks; n++)
                            reads[n] = budget > 0 && !(n >= sinkStart && n < sinkEnd) && scores[n] >= threshold;
                    }
                    else
                    {
                        float spread = 0f;
                        for (int d = 0; d < dim; d++) spread += centroids[qb * dim + d] * centroids[qb * dim + d] * variance[d];
                        float threshold = options.Tau * MathF.Sqrt(spread * log2s * log2s + 1e-6f);
                        for (int n = 0; n < blocks; n++) reads[n] = scores[n] > threshold;
                    }
                    bool readsAll = qb >= sinkQueryStart && qb < sinkQueryEnd;
                    for (int n = 0; n < blocks; n++)
                        reads[n] |= readsAll || Math.Abs(qb - n) <= 1 || (n >= sinkStart && n < sinkEnd);
                    if (!routed.IsEmpty) reads.CopyTo(routed.Slice(((bi * heads + h) * blocks + qb) * blocks, blocks));
                }

                if (coarse.Length > 0) CoarseAttention(coarse, centroids, keyMeans, valueSums, lengths, blocks, dim, scale);

                // One base-2 softmax per query row over the exactly read keys and the pooled blocks.
                for (int i = 0; i < tokens; i++)
                {
                    int qb = i / Block, qAt = Row(bi, i, h, tokens, heads, dim);
                    ReadOnlySpan<bool> reads = exact.AsSpan(qb * blocks, blocks);
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < tokens; j++)
                    {
                        int n = j / Block;
                        if (!reads[n] || (deadRows && j % Block >= lengths[n]))
                        {
                            tokenLogits[j] = float.NegativeInfinity;
                            continue;
                        }
                        int kAt = Row(bi, j, h, tokens, heads, dim);
                        float dot = 0f;
                        for (int d = 0; d < dim; d++) dot += q[qAt + d] * (k[kAt + d] - keyMean[d]);
                        float logit = dot * log2s;
                        if (!bias.IsEmpty) logit += bias[biasPerBatch ? bi * tokens + j : j] * Log2E;
                        tokenLogits[j] = logit;
                        max = MathF.Max(max, logit);
                    }
                    if (options.Tail)
                        for (int n = 0; n < blocks; n++)
                            if (!reads[n]) max = MathF.Max(max, pooled[qb * blocks + n]);

                    Array.Clear(row);
                    float denominator = 0f;
                    for (int j = 0; j < tokens; j++)
                    {
                        if (float.IsNegativeInfinity(tokenLogits[j])) continue;
                        float p = Exp2(tokenLogits[j] - max);
                        denominator += p;
                        int vAt = Row(bi, j, h, tokens, heads, dim);
                        for (int d = 0; d < dim; d++) row[d] += p * v[vAt + d];
                    }
                    if (options.Tail)
                        for (int n = 0; n < blocks; n++)
                        {
                            if (reads[n]) continue;
                            // The pooled term: the block's summed V under one probability, counted once per live row.
                            float p = Exp2(pooled[qb * blocks + n] - max);
                            denominator += p * lengths[n];
                            for (int d = 0; d < dim; d++) row[d] += p * valueSums[n * dim + d];
                        }
                    float inverse = 1f / MathF.Max(denominator, 1e-30f);
                    for (int d = 0; d < dim; d++)
                    {
                        float result = row[d] * inverse;
                        if (coarse.Length > 0) result += gate[qAt + d] * coarse[qb * dim + d];
                        o[qAt + d] = result;
                    }
                }
            }
    }

    /// <summary>The coarse branch: each query block's mean query attends every block's mean key (uncentered, natural
    /// softmax, no length weighting) over the blocks' mean values.</summary>
    private static void CoarseAttention(float[] coarse, float[] centroids, float[] keyMeans, float[] valueSums, int[] lengths, int blocks,
        int dim, float scale)
    {
        float[] logits = new float[blocks];
        Array.Clear(coarse);
        for (int qb = 0; qb < blocks; qb++)
        {
            float max = float.NegativeInfinity;
            for (int n = 0; n < blocks; n++)
            {
                logits[n] = Dot(centroids.AsSpan(qb * dim, dim), keyMeans.AsSpan(n * dim, dim)) * scale;
                max = MathF.Max(max, logits[n]);
            }
            float total = 0f;
            for (int n = 0; n < blocks; n++)
            {
                logits[n] = MathF.Exp(logits[n] - max);
                total += logits[n];
            }
            for (int n = 0; n < blocks; n++)
            {
                float p = logits[n] / total / lengths[n];
                for (int d = 0; d < dim; d++) coarse[qb * dim + d] += p * valueSums[n * dim + d];
            }
        }
    }

    /// <summary>Live rows per block: 64, or what is left in the last block; with lengths given, each clamped to 1 to 64
    /// and the last also to the rows that exist.</summary>
    private static int[] LiveLengths(IReadOnlyList<int>? given, int tokens, int blocks)
    {
        int[] lengths = new int[blocks];
        for (int n = 0; n < blocks; n++)
        {
            int exists = Math.Min(Block, tokens - n * Block);
            lengths[n] = given is null ? exists : Math.Min(exists, Math.Clamp(given[n], 1, Block));
        }
        return lengths;
    }

    private static int Row(int batch, int token, int head, int tokens, int heads, int dim) => ((batch * tokens + token) * heads + head) * dim;

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float sum = 0f;
        for (int i = 0; i < a.Length; i++) sum += a[i] * b[i];
        return sum;
    }

    private static float Exp2(float x) => (float)Math.Pow(2.0, x);

    private static void Check(Tensor output, Tensor query, Tensor key, Tensor value, SolAttentionOptions options)
    {
        if (query.Shape.Rank != 4 || query.Shape != key.Shape || query.Shape != value.Shape || query.Shape != output.Shape)
            throw new ArgumentException(
                $"Sol-Attn takes Q, K, V and an output of one [B, T, H, D] shape; got Q={query.Shape}, K={key.Shape}, V={value.Shape}, O={output.Shape}.");
        if (query.DType != DType.F32 || key.DType != DType.F32 || value.DType != DType.F32 || output.DType != DType.F32)
            throw new ArgumentException("The Sol-Attn reference reads and writes F32.");
        if (ReferenceEquals(output, query) || ReferenceEquals(output, key) || ReferenceEquals(output, value))
            throw new ArgumentException("The output must not alias an input.", nameof(output));
        int batch = (int)query.Shape[0], tokens = (int)query.Shape[1];
        if (query.ElementCount == 0) throw new ArgumentException($"Sol-Attn needs positive extents; got {query.Shape}.", nameof(query));
        if (options.Scale is { } scale && !float.IsFinite(scale)) throw new ArgumentException("The scale must be finite.", nameof(options));
        if (float.IsNaN(options.Tau)) throw new ArgumentException("Tau is not a number.", nameof(options));
        if (options.TopKRatio is < 0f or >= 1f || float.IsNaN(options.TopKRatio))
            throw new ArgumentException($"The top-k ratio is 0 (off) or inside (0, 1); got {options.TopKRatio}.", nameof(options));
        foreach ((int start, int end) in (ReadOnlySpan<(int, int)>)[options.SinkBlocks, options.SinkQueries])
            if (start < 0 || end < start)
                throw new ArgumentException($"A sink range is [start, end) with 0 <= start <= end; got [{start}, {end}).", nameof(options));
        if (options.BlockLengths is { } lengths && lengths.Count != Blocks(tokens))
            throw new ArgumentException($"Block lengths hold {lengths.Count} entries; {tokens} tokens make {Blocks(tokens)} blocks.", nameof(options));
        if (options.KeyBias is { } bias)
        {
            bool perKey = bias.Shape.Rank == 1 && bias.Shape[0] == tokens;
            bool perBatch = bias.Shape.Rank == 2 && (bias.Shape[0] == 1 || bias.Shape[0] == batch) && bias.Shape[1] == tokens;
            if (bias.DType != DType.F32 || !(perKey || perBatch))
                throw new ArgumentException($"A key bias is F32 [T] or [B, T]; got {bias.DType} {bias.Shape}.", nameof(options));
        }
        if (options.CoarseGate is { } gate && (gate.DType != DType.F32 || gate.Shape != query.Shape))
            throw new ArgumentException($"A coarse gate is F32 in the operands' shape; got {gate.DType} {gate.Shape}.", nameof(options));
    }
}
