using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Host reference of the Hy4 Flash compressed indexer (transformers <c>HYV4FlashCompressedIndexer</c>, PR 50903 at <c>eb1d5db</c>): per-token keys are
/// cached, every <c>Kpool</c> complete tokens are pooled in F32 into one group key, and each query recalls the best-scoring complete groups plus its own group;
/// attention then covers every token of those groups.</summary>
/// <remarks>
/// <para>Token key <c>RMSNorm(x Wtk^T)</c>. Group key: <c>values = Wv g</c>, <c>gates = Wg g + ape</c>, <c>pooled = sum over the slots of values * softmax(gates over the slots)</c>,
/// then RMS norm, then the trailing <c>RopeDim</c> channels rotated (interleaved pairs) at the group's first token. Query: <c>Wq q_latent</c> per head, trailing channels rotated at the
/// query position. Score: <c>sum_h w_h * relu(q_h . k_g)</c> with <c>w = Whw x * Heads^-0.5 * HeadDim^-0.5</c>. A query ignores groups at or after its own, recalls the top
/// <c>TopK / Kpool - 1</c> others (all of them when fewer) and always its own.</para>
/// <para>Up to <c>TopK</c> cached keys the layer is dense and the indexer does nothing, as in the reference. Slots equal rotary positions (text-only); image spans need positions passed separately.
/// Ties between equal scores resolve to the lower group index; the reference <c>topk</c> leaves them undefined.</para>
/// </remarks>
public sealed class Hy4FlashGroupIndexer : IDsaKeySelector
{
    private readonly IBackend _backend;
    private readonly Hy4FlashIndexerSettings _s;
    private readonly Hy4FlashIndexerWeights _w;
    private readonly IRopeRowSource _rope;
    private readonly int _dim;
    private readonly int _qLoraRank;

    /// <param name="backend">Runs the projections, norms and rotations.</param>
    /// <param name="settings">Indexer shape.</param>
    /// <param name="weights">Indexer weights.</param>
    /// <param name="rope">Rotary rows <c>RopeDim / 2</c> wide covering every position that will be seen.</param>
    /// <param name="dim">Hidden width the keys and head weights are projected from.</param>
    /// <param name="qLoraRank">Query latent width the queries are projected from.</param>
    public Hy4FlashGroupIndexer(IBackend backend, Hy4FlashIndexerSettings settings, Hy4FlashIndexerWeights weights, IRopeRowSource rope, int dim, int qLoraRank)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(rope);
        settings.Validate();
        if (rope.HalfDim * 2 != settings.RopeDim) throw new ArgumentException("The rope table width must equal RopeDim.", nameof(rope));
        int hd = settings.HeadDim;
        DsaIndexer.RequireShape(weights.QueryProjection, settings.Heads * hd, qLoraRank, "indexer query_projection");
        DsaIndexer.RequireShape(weights.HeadWeightProjection, settings.Heads, dim, "indexer head_weight_projection");
        DsaIndexer.RequireShape(weights.TokenKeyProjection, hd, dim, "indexer token_key_projection");
        DsaIndexer.RequireShape(weights.KeyValueProjection, hd, hd, "indexer key_value_projection");
        DsaIndexer.RequireShape(weights.KeyGateProjection, hd, hd, "indexer key_gate_projection");
        DsaIndexer.RequireShape(weights.IntraGroupApe, settings.Kpool, hd, "indexer intra_group_ape");
        if (weights.TokenKeyNorm.ElementCount != hd || weights.KeyNorm.ElementCount != hd)
            throw new ArgumentException("The indexer norms must hold HeadDim values.", nameof(weights));
        _backend = backend;
        _s = settings;
        _w = weights;
        _rope = rope;
        _dim = dim;
        _qLoraRank = qLoraRank;
    }

    /// <inheritdoc />
    public int KeyWidth => _s.HeadDim;

    /// <inheritdoc />
    public void Append(ReadOnlySpan<float> x, int tokens, int startPos, Tensor keyStore)
    {
        int hd = _s.HeadDim;
        if (keyStore.DType != DType.F32 || keyStore.Shape.Rank != 2 || keyStore.Shape[1] != hd || startPos + tokens > keyStore.Shape[0])
            throw new ArgumentException("keyStore must be F32 [capacity, HeadDim] with room for these positions.", nameof(keyStore));
        float[] k = HostOps.Project(_backend, x, tokens, _dim, _w.TokenKeyProjection, hd);
        HostOps.RmsNorm(_backend, k, hd, _w.TokenKeyNorm, _s.NormEps);
        k.CopyTo(keyStore.AsSpan<float>().Slice(startPos * hd, tokens * hd));
    }

    /// <inheritdoc />
    public DsaKeySelection Select(ReadOnlySpan<float> x, ReadOnlySpan<float> qLatent, ReadOnlySpan<int> positions, int length, Tensor keyStore)
    {
        int tokens = positions.Length;
        if (length <= _s.TopK) return Dense(positions);
        int kpool = _s.Kpool;
        float[] scores = Scores(x, qLatent, positions, length, keyStore, out int groups);
        int recalled = Math.Min(_s.RecalledGroups, groups);
        List<int>[] rows = new List<int>[tokens];
        int width = 0;
        for (int t = 0; t < tokens; t++)
        {
            int slot = positions[t], own = slot / kpool;
            HashSet<int> chosen = [own];
            foreach (int g in TopGroups(scores.AsSpan(t * groups, groups), recalled)) chosen.Add(g);
            List<int> list = [];
            foreach (int g in chosen.Order())
                for (int i = 0; i < kpool; i++)
                {
                    int token = g * kpool + i;
                    if (token <= slot && token < length) list.Add(token);
                }
            rows[t] = list;
            width = Math.Max(width, list.Count);
        }
        return Pack(rows, width);
    }

    /// <summary>The group scores of every query, <c>[tokens, groups]</c>, with groups at or after the query own group at -inf. Only meaningful past <c>TopK</c> keys.</summary>
    internal float[] Scores(ReadOnlySpan<float> x, ReadOnlySpan<float> qLatent, ReadOnlySpan<int> positions, int length, Tensor keyStore, out int groups)
    {
        int tokens = positions.Length;
        int hd = _s.HeadDim, nh = _s.Heads, kpool = _s.Kpool;
        groups = length / kpool;
        int lastPosition = 0;
        foreach (int p in positions) lastPosition = Math.Max(lastPosition, p);
        if (lastPosition >= _rope.Length || (groups - 1) * kpool >= _rope.Length)
            throw new ArgumentOutOfRangeException(nameof(positions), "The rope table does not reach these positions.");

        float[] groupKeys = PoolGroups(keyStore, groups);
        int[] groupPositions = new int[groups];
        for (int g = 0; g < groups; g++) groupPositions[g] = g * kpool;
        PartialRotary.Rotate(_backend, _rope, _s.RopeDim, groupKeys, [1, groups, hd], groupPositions, inverse: false, RotaryPairing.Interleaved);

        float[] q = HostOps.Project(_backend, qLatent, tokens, _qLoraRank, _w.QueryProjection, nh * hd);
        PartialRotary.Rotate(_backend, _rope, _s.RopeDim, q, [1, tokens, nh, hd], positions.ToArray(), inverse: false, RotaryPairing.Interleaved);
        float[] weights = HostOps.Project(_backend, x, tokens, _dim, _w.HeadWeightProjection, nh);
        float weightScale = MathF.Pow(nh, -0.5f) * MathF.Pow(hd, -0.5f);
        float[] scores = new float[tokens * groups];
        for (int t = 0; t < tokens; t++)
        {
            int own = positions[t] / kpool;
            for (int g = 0; g < groups; g++)
            {
                if (g >= own)
                {
                    scores[t * groups + g] = float.NegativeInfinity;
                    continue;
                }
                float sum = 0f;
                for (int h = 0; h < nh; h++)
                {
                    float dot = 0f;
                    for (int d = 0; d < hd; d++) dot += q[(t * nh + h) * hd + d] * groupKeys[g * hd + d];
                    if (dot > 0f) sum += weights[t * nh + h] * weightScale * dot;
                }
                scores[t * groups + g] = sum;
            }
        }
        return scores;
    }

    /// <inheritdoc />
    public void Dispose()
    {
    }

    // Pools the first groups * Kpool cached token keys: values * softmax(gates + ape over the slots), then the key RMS norm. F32 throughout.
    private float[] PoolGroups(Tensor keyStore, int groups)
    {
        int hd = _s.HeadDim, kpool = _s.Kpool, rows = groups * kpool;
        ReadOnlySpan<float> tokenKeys = keyStore.AsReadOnlySpan<float>().Slice(0, rows * hd);
        float[] values = HostOps.Project(_backend, tokenKeys, rows, hd, _w.KeyValueProjection, hd);
        float[] gates = HostOps.Project(_backend, tokenKeys, rows, hd, _w.KeyGateProjection, hd);
        ReadOnlySpan<float> ape = _w.IntraGroupApe.AsReadOnlySpan<float>();
        float[] pooled = new float[groups * hd];
        for (int g = 0; g < groups; g++)
            for (int d = 0; d < hd; d++)
            {
                float max = float.NegativeInfinity;
                for (int s = 0; s < kpool; s++) max = MathF.Max(max, gates[(g * kpool + s) * hd + d] + ape[s * hd + d]);
                float denominator = 0f, acc = 0f;
                for (int s = 0; s < kpool; s++)
                {
                    float e = MathF.Exp(gates[(g * kpool + s) * hd + d] + ape[s * hd + d] - max);
                    denominator += e;
                    acc += values[(g * kpool + s) * hd + d] * e;
                }
                pooled[g * hd + d] = acc / denominator;
            }
        HostOps.RmsNorm(_backend, pooled, hd, _w.KeyNorm, _s.NormEps);
        return pooled;
    }

    // The best groups by score, ties to the lower index; groups scored -inf are taken last, as the reference top-k does when fewer groups are valid.
    internal static IEnumerable<int> TopGroups(ReadOnlySpan<float> scores, int count)
    {
        float[] copy = scores.ToArray();
        int[] order = [.. Enumerable.Range(0, copy.Length)];
        Array.Sort(order, (a, b) => copy[a] != copy[b] ? copy[b].CompareTo(copy[a]) : a.CompareTo(b));
        return order.Take(count);
    }

    private static DsaKeySelection Dense(ReadOnlySpan<int> positions)
    {
        List<int>[] rows = new List<int>[positions.Length];
        int width = 0;
        for (int t = 0; t < positions.Length; t++)
        {
            rows[t] = [.. Enumerable.Range(0, positions[t] + 1)];
            width = Math.Max(width, rows[t].Count);
        }
        return Pack(rows, width);
    }

    private static DsaKeySelection Pack(List<int>[] rows, int width)
    {
        int[] packed = new int[rows.Length * width], counts = new int[rows.Length];
        Array.Fill(packed, -1);
        for (int t = 0; t < rows.Length; t++)
        {
            rows[t].CopyTo(packed, t * width);
            counts[t] = rows[t].Count;
        }
        return new DsaKeySelection(packed, width, counts);
    }
}
