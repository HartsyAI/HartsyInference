using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Latent;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer.Sparse;

namespace HartsyInference.LLM.Transformer.Dsa;

/// <summary>Host reference for a DSA lightning indexer (vLLM-ERNIE <c>Ernie5_5Indexer</c>): scores each older token for each query and
/// keeps the top-k.</summary>
/// <remarks>
/// <para>Keys: <c>k = LayerNorm(wk(x))</c>, rotated, Hadamard-rotated, then round-tripped through the key cache's encoding and stored per
/// position. Queries: <c>q = wq_b(q_latent)</c> per head, rotated and Hadamard-rotated the same way, then round-tripped. Score:
/// <c>score[t,s] = sum_h w[t,h] * ReLU(q[t,h] . k[s])</c> with <c>w = weights_proj(x) * HeadDim^-0.5 * Heads^-0.5</c>
/// (<see cref="IBackend.IndexerScores"/>). The reference folds the query's power-of-two FP8 scale into <c>w</c>; ReLU commutes with a
/// positive scale, so scoring the dequantized query is the same sum.</para>
/// <para>UNVERIFIED: the ReLU. The scoring kernel (DeepGEMM <c>fp8_mqa_logits</c>) is not in the reference sources; the ReLU is taken
/// from vLLM's comment on that kernel and from the DeepSeek V3.2 indexer it serves. No ERNIE checkpoint run has confirmed it.</para>
/// </remarks>
public sealed class DsaIndexer : IDisposable
{
    private readonly IBackend _backend;
    private readonly DsaIndexerSettings _s;
    private readonly DsaIndexerWeights _w;
    private readonly IRopeRowSource _rope;
    private readonly int _dim;
    private readonly int _qLoraRank;
    private Tensor? _ownedZeroBias;

    /// <param name="backend">Runs the projections, norms, rotations, Hadamard, round trips and scores.</param>
    /// <param name="settings">Indexer shape and numerics.</param>
    /// <param name="weights">Indexer weights.</param>
    /// <param name="rope">Rotary rows <c>RopeDim / 2</c> wide covering every position that will be seen.</param>
    /// <param name="dim">Hidden width the keys and weights are projected from.</param>
    /// <param name="qLoraRank">Query latent width the queries are projected from.</param>
    public DsaIndexer(IBackend backend, DsaIndexerSettings settings, DsaIndexerWeights weights, IRopeRowSource rope, int dim, int qLoraRank)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(rope);
        settings.Validate();
        if (rope.HalfDim * 2 != settings.RopeDim) throw new ArgumentException("The indexer rope table width must equal RopeDim.", nameof(rope));
        RequireShape(weights.WqB, settings.Heads * settings.HeadDim, qLoraRank, "indexer wq_b");
        RequireShape(weights.Wk, settings.HeadDim, dim, "indexer wk");
        RequireShape(weights.WeightsProj, settings.Heads, dim, "indexer weights_proj");
        if (weights.KeyNormWeight.ElementCount != settings.HeadDim || weights.KeyNormBias is { } b && b.ElementCount != settings.HeadDim)
            throw new ArgumentException("The indexer key norm must hold HeadDim values.", nameof(weights));
        _backend = backend;
        _s = settings;
        _w = weights;
        _rope = rope;
        _dim = dim;
        _qLoraRank = qLoraRank;
        if (weights.KeyNormBias is null)
        {
            _ownedZeroBias = new Tensor(new TensorShape(settings.HeadDim), DType.F32);
            _ownedZeroBias.AsSpan<float>().Clear();
        }
    }

    /// <summary>The settings the indexer was built with.</summary>
    public DsaIndexerSettings Settings => _s;

    /// <summary>Computes the keys of <paramref name="tokens"/> rows at positions <c>startPos ..</c> and writes them into
    /// <paramref name="keyStore"/> rows of the same positions.</summary>
    /// <param name="x">Normalized hidden states, <c>[tokens, dim]</c>.</param>
    /// <param name="keyStore">F32 <c>[capacity, HeadDim]</c>.</param>
    public void AppendKeys(ReadOnlySpan<float> x, int tokens, int startPos, Tensor keyStore)
    {
        int hd = _s.HeadDim;
        if (keyStore.DType != DType.F32 || keyStore.Shape.Rank != 2 || keyStore.Shape[1] != hd || startPos + tokens > keyStore.Shape[0])
            throw new ArgumentException("keyStore must be F32 [capacity, HeadDim] with room for these positions.", nameof(keyStore));
        RequireRope(startPos + tokens);
        Tensor bias = _w.KeyNormBias ?? _ownedZeroBias ?? throw new ObjectDisposedException(nameof(DsaIndexer));
        float[] k = HostOps.Project(_backend, x, tokens, _dim, _w.Wk, hd);
        HostOps.LayerNorm(_backend, k, hd, _w.KeyNormWeight, bias, _s.KeyNormEps);
        Rotate(k, [1, tokens, hd], HostOps.Range(startPos, tokens));
        if (_s.Hadamard) HostOps.Hadamard(_backend, k, hd);
        HostOps.RoundTrip(_backend, k, hd, _s.KeyEncoding);
        k.CopyTo(keyStore.AsSpan<float>().Slice(startPos * hd, tokens * hd));
    }

    /// <summary>Selects, per query, the ascending top-k among keys <c>[0, limits[t])</c> of <paramref name="keyStore"/>.</summary>
    /// <param name="x">Normalized hidden states, <c>[tokens, dim]</c>, for the head weights.</param>
    /// <param name="qLatent">Normalized query latents, <c>[tokens, qLoraRank]</c>.</param>
    /// <param name="positions">Each query's absolute position, for its rotation.</param>
    /// <param name="limits">Keys each query may score; positions at or past it are never picked.</param>
    /// <param name="keyStore">The keys written by <see cref="AppendKeys"/>; at least <c>max(limits)</c> rows are read.</param>
    /// <param name="width">Receives the row width, <c>min(TopK, max(limits))</c>; 0 when no query may score any key.</param>
    /// <returns><c>[tokens, width]</c> ascending key positions; each row's first <c>min(limits[t], TopK)</c> are valid, the rest -1.</returns>
    /// <remarks>The scores are one F32 <c>[tokens, max(limits)]</c> buffer, so memory grows with chunk length times context
    /// (about 4 GB for a 32k-token chunk at 32k context). Callers should prefill in bounded chunks.</remarks>
    public int[] Select(ReadOnlySpan<float> x, ReadOnlySpan<float> qLatent, ReadOnlySpan<int> positions, ReadOnlySpan<int> limits, Tensor keyStore,
        out int width)
    {
        int tokens = positions.Length, nh = _s.Heads, hd = _s.HeadDim;
        if (limits.Length != tokens) throw new ArgumentException("limits must hold one entry per query.", nameof(limits));
        int keyRows = 0, lastPos = -1;
        foreach (int l in limits) keyRows = Math.Max(keyRows, l);
        foreach (int p in positions) lastPos = Math.Max(lastPos, p);
        RequireRope(lastPos + 1);
        if (keyRows > keyStore.Shape[0]) throw new ArgumentException("A limit reaches past the key store.", nameof(limits));
        width = Math.Min(_s.TopK, keyRows);
        if (keyRows == 0) return [];

        float[] q = HostOps.Project(_backend, qLatent, tokens, _qLoraRank, _w.WqB, nh * hd);
        Rotate(q, [1, tokens, nh, hd], positions.ToArray());
        if (_s.Hadamard) HostOps.Hadamard(_backend, q, hd);
        HostOps.RoundTrip(_backend, q, hd, _s.QueryEncoding);

        float[] weights = HostOps.Project(_backend, x, tokens, _dim, _w.WeightsProj, nh);
        float weightScale = 1f / MathF.Sqrt(hd) / MathF.Sqrt(nh);
        for (int i = 0; i < weights.Length; i++) weights[i] *= weightScale;

        using Tensor scores = new(new TensorShape(tokens, keyRows), DType.F32);
        using Tensor queryT = HostMath.Tensor(q, tokens, nh, hd);
        using Tensor keys = keyStore.SliceRows(0, keyRows);
        using Tensor weightT = HostMath.Tensor(weights, tokens, nh);
        using Tensor limitT = HostMath.Tensor(limits, tokens);
        _backend.IndexerScores(scores, queryT, new LatentSource(LatentEncoding.F32, keys, null, keyRows, hd), weightT, limitT, null, 1f);

        ReadOnlySpan<float> all = scores.AsReadOnlySpan<float>();
        int[] picked = new int[tokens * width];
        for (int t = 0; t < tokens; t++)
            SparseTopK.SelectSorted(all.Slice(t * keyRows, keyRows), limits[t], _s.TopK, 0, picked.AsSpan(t * width, width));
        return picked;
    }

    private void RequireRope(int endPos)
    {
        if (endPos > _rope.Length)
            throw new ArgumentOutOfRangeException(nameof(endPos), $"The indexer rope table covers {_rope.Length} positions; position {endPos - 1} was asked for.");
    }

    private void Rotate(float[] data, long[] shape, int[] positions)
    {
        if (_s.RopeLeading) PartialRotary.RotateLeading(_backend, _rope, _s.RopeDim, data, shape, positions, inverse: false, _s.RopePairing);
        else PartialRotary.Rotate(_backend, _rope, _s.RopeDim, data, shape, positions, inverse: false, _s.RopePairing);
    }

    /// <inheritdoc />
    public void Dispose() => Interlocked.Exchange(ref _ownedZeroBias, null)?.Dispose();

    internal static void RequireShape(Tensor w, int rows, int cols, string name)
    {
        if (w.Shape.Rank != 2 || w.Shape[0] != rows || w.Shape[1] != cols)
            throw new ArgumentException($"{name} must be [{rows}, {cols}]; got {w.Shape}.");
    }
}
