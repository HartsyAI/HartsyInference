using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Maxwell;

/// <summary>One layer's block-sparse GQA attention with its indexer. The projections run on the backend; norms, RoPE, the block route and the
/// attention itself run on the host, so every backend serves any sequence length. Below <see cref="MaxwellSparseConfig.DenseCausalLimit"/> a query's
/// route holds every causal block and the indexer is not consulted.</summary>
internal sealed class MaxwellAttention : IDisposable
{
    private readonly IBackend _backend;
    private readonly MaxwellConfig _cfg;
    private readonly MaxwellRope _rope;
    private readonly Tensor _q, _k, _v, _o, _indexQ, _indexK;
    private readonly float[] _qNorm, _kNorm, _indexQNorm, _indexKNorm;
    private int _disposed;

    /// <param name="prefix">The layer's key prefix, <c>model.layers.N</c>.</param>
    public MaxwellAttention(IBackend backend, MaxwellConfig cfg, string prefix, IModelWeightSource source, MaxwellRope rope)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cfg);
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(rope);
        _backend = backend;
        _cfg = cfg;
        _rope = rope;
        List<Tensor> read = [];
        try
        {
            Tensor Read(string leaf)
            {
                Tensor t = source.Read(MaxwellWeightNames.Key(prefix, $"self_attn.{leaf}"));
                read.Add(t);
                return t;
            }
            float[] Host(string leaf) => WeightReads.HostF32(source, MaxwellWeightNames.Key(prefix, $"self_attn.{leaf}"));
            string[] a = MaxwellWeightNames.AttnLeaves, ix = MaxwellWeightNames.IndexerLeaves;
            _q = Read(a[0]);
            _k = Read(a[1]);
            _v = Read(a[2]);
            _o = Read(a[3]);
            _qNorm = Host(a[4]);
            _kNorm = Host(a[5]);
            _indexQ = Read($"indexer.{ix[0]}");
            _indexK = Read($"indexer.{ix[1]}");
            _indexQNorm = Host($"indexer.{ix[2]}");
            _indexKNorm = Host($"indexer.{ix[3]}");
        }
        catch
        {
            foreach (Tensor t in read) t.Dispose();
            throw;
        }
    }

    /// <summary>Every weight the layer's attention reads, each FP8 weight followed by its block-scale grid.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in (Tensor[])[_q, _k, _v, _o, _indexQ, _indexK])
        {
            yield return t;
            if (t.QuantInfo?.Fp8BlockScaleInv is { } scale) yield return scale;
        }
    }

    /// <summary>Attends <paramref name="tokens"/> rows of <paramref name="input"/> (<c>[tokens, hidden]</c>, already normed) starting at
    /// <paramref name="startPos"/>, appends their keys to the layer's caches, and writes <paramref name="output"/> (<c>[tokens, hidden]</c>).</summary>
    public void Forward(ReadOnlySpan<float> input, int tokens, int startPos, MaxwellSequenceState state, int layer, Span<float> output)
    {
        int hidden = _cfg.HiddenSize, heads = _cfg.NumAttentionHeads, kvHeads = _cfg.NumKeyValueHeads, dim = _cfg.HeadDim;
        MaxwellSparseConfig sparse = _cfg.Sparse;
        int groups = sparse.IndexerGroups, indexDim = sparse.IndexerHeadDim;
        float norm = (float)_cfg.RmsNormEps;

        float[] q = HostOps.Project(_backend, input, tokens, hidden, _q, heads * dim);
        float[] k = HostOps.Project(_backend, input, tokens, hidden, _k, kvHeads * dim);
        float[] v = HostOps.Project(_backend, input, tokens, hidden, _v, kvHeads * dim);
        float[] indexQ = HostOps.Project(_backend, input, tokens, hidden, _indexQ, groups * indexDim);
        float[] indexK = HostOps.Project(_backend, input, tokens, hidden, _indexK, indexDim);
        HostMath.RmsNormRows(q, _qNorm, dim, norm);
        HostMath.RmsNormRows(k, _kNorm, dim, norm);
        HostMath.RmsNormRows(indexQ, _indexQNorm, indexDim, norm);
        HostMath.RmsNormRows(indexK, _indexKNorm, indexDim, norm);
        for (int t = 0; t < tokens; t++)
        {
            int position = startPos + t;
            for (int h = 0; h < heads; h++) _rope.Apply(q.AsSpan((t * heads + h) * dim, dim), position);
            for (int h = 0; h < kvHeads; h++) _rope.Apply(k.AsSpan((t * kvHeads + h) * dim, dim), position);
            for (int g = 0; g < groups; g++) _rope.Apply(indexQ.AsSpan((t * groups + g) * indexDim, indexDim), position);
            _rope.Apply(indexK.AsSpan(t * indexDim, indexDim), position);
        }

        (float[] keys, float[] values, float[] indexKeys) = state.Reserve(layer, startPos + tokens);
        k.CopyTo(keys.AsSpan(startPos * kvHeads * dim));
        v.CopyTo(values.AsSpan(startPos * kvHeads * dim));
        indexK.CopyTo(indexKeys.AsSpan(startPos * indexDim));

        float[] attended = new float[tokens * heads * dim];
        int perGroup = heads / groups, maxBlocks = MaxwellBlockRoute.MaxBlocks(sparse);
        // Queries below the dense limit see every causal key; later ones read their route's blocks.
        int[]?[] routes = new int[]?[tokens * groups];
        Parallel.For(0, tokens * groups, slot =>
        {
            int t = slot / groups, g = slot % groups, position = startPos + t;
            if (position < sparse.DenseCausalLimit) return;
            int[] blocks = new int[maxBlocks];
            int count = MaxwellBlockRoute.Select(sparse, indexQ.AsSpan(slot * indexDim, indexDim), indexKeys, position, blocks);
            routes[slot] = blocks[..count];
        });
        float scale = 1f / MathF.Sqrt(dim);
        Parallel.For(0, tokens * heads, slot =>
        {
            int t = slot / heads, h = slot % heads, g = h / perGroup, kv = h * kvHeads / heads, position = startPos + t;
            int[]? route = routes[t * groups + g];
            int ranges = route is null ? 1 : route.Length, span = route is null ? position + 1 : route.Length * sparse.BlockSize;
            float[] scores = System.Buffers.ArrayPool<float>.Shared.Rent(span);
            try
            {
                ReadOnlySpan<float> query = q.AsSpan(slot * dim, dim);
                int n = 0;
                for (int r = 0; r < ranges; r++)
                {
                    (int first, int last) = KeyRange(route, r, sparse.BlockSize, position);
                    for (int key = first; key <= last; key++)
                        scores[n++] = HostMath.Dot(query, keys.AsSpan((key * kvHeads + kv) * dim, dim)) * scale;
                }
                Softmax(scores.AsSpan(0, n));
                Span<float> result = attended.AsSpan(slot * dim, dim);
                n = 0;
                for (int r = 0; r < ranges; r++)
                {
                    (int first, int last) = KeyRange(route, r, sparse.BlockSize, position);
                    for (int key = first; key <= last; key++)
                    {
                        float weight = scores[n++];
                        ReadOnlySpan<float> value = values.AsSpan((key * kvHeads + kv) * dim, dim);
                        for (int d = 0; d < dim; d++) result[d] += weight * value[d];
                    }
                }
            }
            finally
            {
                System.Buffers.ArrayPool<float>.Shared.Return(scores);
            }
        });
        float[] projected = HostOps.Project(_backend, attended, tokens, heads * dim, _o, hidden);
        projected.CopyTo(output);
    }

    // Range r of the keys a query reads: every causal key without a route, else block r of the route clipped at the query.
    private static (int First, int Last) KeyRange(int[]? route, int r, int blockSize, int position)
    {
        if (route is null) return (0, position);
        int block = route[r];
        return (block * blockSize, Math.Min((block + 1) * blockSize - 1, position));
    }

    private static void Softmax(Span<float> scores)
    {
        float max = float.NegativeInfinity;
        foreach (float s in scores) max = MathF.Max(max, s);
        float sum = 0f;
        for (int i = 0; i < scores.Length; i++)
        {
            scores[i] = MathF.Exp(scores[i] - max);
            sum += scores[i];
        }
        for (int i = 0; i < scores.Length; i++) scores[i] /= sum;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in (Tensor[])[_q, _k, _v, _o, _indexQ, _indexK]) t.Dispose();
    }
}
