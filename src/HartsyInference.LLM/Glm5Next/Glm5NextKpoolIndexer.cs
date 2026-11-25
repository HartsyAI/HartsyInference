using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;

namespace HartsyInference.LLM.Glm5Next;

/// <summary>Host reference of the glm5_next compressed-pool indexer (transformers <c>Glm5NextTextIndexer</c> at 52bac5d0): per-token keys and pool gates are
/// cached, every <c>Pool</c> complete tokens are merged into one pool key by a per-channel softmax over their gate scores (plus <c>ape</c>), each query
/// scores the pools whose last token it can see, keeps the best <c>TopK / Pool</c>, expands them to raw tokens and appends its own incomplete pool.</summary>
/// <remarks>
/// <para>Per slot the store holds <c>[k | gate]</c>: <c>k = LayerNorm(x Wk^T)</c> (weight and bias, eps 1e-6) and <c>gate = x Wgate^T</c>, both <c>HeadDim</c> wide.
/// Pool <c>g</c> covers slots <c>g·Pool .. g·Pool + Pool - 1</c>; its key is <c>sum_j softmax_j(gate_j + ape_j) · k_j</c> per channel; there is no value
/// projection, no norm after pooling and no rotary. Query: <c>q = Wq_b q_latent</c> per head; score <c>sum_h w_h · relu(q_h · pool · HeadDim^-0.5)</c> with
/// <c>w = Wweights x · Heads^-0.5</c>. A pool is selectable only when its last slot is at or before the query's, so a query on the last slot of a pool may
/// pick its own pool or not, by score. With fewer selectable pools than slots every one is kept. Tail: the <c>(position + 1) mod Pool</c> newest slots up
/// to the query, as raw tokens, when <c>AlwaysSelectTail</c>.</para>
/// <para>Equal scores resolve to the lower pool index; the reference <c>topk</c> leaves them undefined, so a parity input must keep the last kept and first
/// dropped score apart. Pools are rebuilt from the cached keys on every call, so cost grows with context; this is the host reference, not the fast path.
/// Slots equal positions (text only).</para>
/// </remarks>
public sealed class Glm5NextKpoolIndexer : IDsaKeySelector
{
    /// <summary>Epsilon of the key LayerNorm (transformers <c>k_norm</c>).</summary>
    internal const float KeyNormEps = 1e-6f;

    private readonly IBackend _backend;
    private readonly Glm5NextConfig _c;
    private readonly Tensor _wqB, _wk, _kNormW, _kNormB, _weightsProj, _gate, _ape;
    private readonly int _dim, _qLoraRank;

    /// <param name="backend">Runs the projections and the key norm.</param>
    /// <param name="config">Supplies the indexer shape (heads, head width, pool, top-k, tail).</param>
    /// <param name="wqB"><c>wq_b</c>, <c>[Heads·HeadDim, QLoraRank]</c>.</param>
    /// <param name="wk"><c>wk</c>, <c>[HeadDim, Hidden]</c>.</param>
    /// <param name="kNormWeight">Key LayerNorm gain, <c>[HeadDim]</c>.</param>
    /// <param name="kNormBias">Key LayerNorm bias, <c>[HeadDim]</c>.</param>
    /// <param name="weightsProj"><c>weights_proj</c>, <c>[Heads, Hidden]</c>.</param>
    /// <param name="compressGate"><c>index_kpool_compress_gate</c>, <c>[HeadDim, Hidden]</c>.</param>
    /// <param name="compressApe"><c>index_kpool_compress_ape</c>, <c>[Pool, HeadDim]</c>.</param>
    public Glm5NextKpoolIndexer(IBackend backend, Glm5NextConfig config, Tensor wqB, Tensor wk, Tensor kNormWeight, Tensor kNormBias, Tensor weightsProj,
        Tensor compressGate, Tensor compressApe)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        int hd = config.IndexHeadDim, nh = config.IndexHeads;
        DsaIndexer.RequireShape(wqB, nh * hd, config.QLoraRank, "indexer wq_b");
        DsaIndexer.RequireShape(wk, hd, config.Hidden, "indexer wk");
        DsaIndexer.RequireShape(weightsProj, nh, config.Hidden, "indexer weights_proj");
        DsaIndexer.RequireShape(compressGate, hd, config.Hidden, "indexer index_kpool_compress_gate");
        DsaIndexer.RequireShape(compressApe, config.IndexPool, hd, "indexer index_kpool_compress_ape");
        if (kNormWeight.ElementCount != hd || kNormBias.ElementCount != hd) throw new ArgumentException("The indexer key norm must hold HeadDim values.");
        (_backend, _c, _wqB, _wk, _kNormW, _kNormB, _weightsProj, _gate, _ape) = (backend, config, wqB, wk, kNormWeight, kNormBias, weightsProj, compressGate, compressApe);
        (_dim, _qLoraRank) = (config.Hidden, config.QLoraRank);
    }

    /// <inheritdoc />
    public int KeyWidth => 2 * _c.IndexHeadDim;

    /// <inheritdoc />
    public void Append(ReadOnlySpan<float> x, int tokens, int startPos, Tensor keyStore)
    {
        int hd = _c.IndexHeadDim, width = 2 * hd;
        if (keyStore.DType != DType.F32 || keyStore.Shape.Rank != 2 || keyStore.Shape[1] != width || startPos + tokens > keyStore.Shape[0])
            throw new ArgumentException("keyStore must be F32 [capacity, 2 * HeadDim] with room for these positions.", nameof(keyStore));
        float[] k = HostOps.Project(_backend, x, tokens, _dim, _wk, hd);
        HostOps.LayerNorm(_backend, k, hd, _kNormW, _kNormB, KeyNormEps);
        float[] gate = HostOps.Project(_backend, x, tokens, _dim, _gate, hd);
        Span<float> store = keyStore.AsSpan<float>();
        for (int t = 0; t < tokens; t++)
        {
            k.AsSpan(t * hd, hd).CopyTo(store.Slice((startPos + t) * width, hd));
            gate.AsSpan(t * hd, hd).CopyTo(store.Slice((startPos + t) * width + hd, hd));
        }
    }

    /// <inheritdoc />
    public DsaKeySelection Select(ReadOnlySpan<float> x, ReadOnlySpan<float> qLatent, ReadOnlySpan<int> positions, int length, Tensor keyStore)
    {
        int tokens = positions.Length, hd = _c.IndexHeadDim, nh = _c.IndexHeads, pool = _c.IndexPool, pools = length / pool, width = 2 * hd;
        float[] poolKeys = PoolKeys(keyStore.AsReadOnlySpan<float>(), pools, width);
        float[] q = HostOps.Project(_backend, qLatent, tokens, _qLoraRank, _wqB, nh * hd);
        float[] w = HostOps.Project(_backend, x, tokens, _dim, _weightsProj, nh);
        float scale = MathF.Pow(hd, -0.5f), headScale = MathF.Pow(nh, -0.5f);
        int keep = Math.Min(_c.IndexSelectedPools, pools);

        List<int>[] rows = new List<int>[tokens];
        float[] scores = new float[pools];
        for (int t = 0; t < tokens; t++)
        {
            int position = positions[t], visiblePools = Math.Min((position + 1) / pool, pools);
            for (int g = 0; g < visiblePools; g++)
            {
                float sum = 0f;
                for (int h = 0; h < nh; h++)
                {
                    float dot = 0f;
                    for (int d = 0; d < hd; d++) dot += q[(t * nh + h) * hd + d] * poolKeys[g * hd + d];
                    float s = dot * scale;
                    if (s > 0f) sum += w[t * nh + h] * headScale * s;
                }
                scores[g] = sum;
            }
            List<int> row = [];
            foreach (int g in DsaCompressedPool.TopK(scores.AsSpan(0, visiblePools), keep).Order())
                for (int i = 0; i < pool; i++) row.Add(g * pool + i);
            if (_c.IndexAlwaysSelectTail)
            {
                int tail = (position + 1) % pool;
                for (int i = tail; i >= 1; i--) row.Add(position + 1 - i);
            }
            rows[t] = row;
        }
        return DsaCompressedPool.Pack(rows);
    }

    /// <inheritdoc />
    public void Dispose()
    {
    }

    // The pool key of every complete pool: per channel, sum_j softmax_j(gate_j + ape_j) * k_j over the pool's slots, F32.
    private float[] PoolKeys(ReadOnlySpan<float> store, int pools, int width)
    {
        int hd = _c.IndexHeadDim, pool = _c.IndexPool;
        ReadOnlySpan<float> ape = _ape.AsReadOnlySpan<float>();
        float[] keys = new float[pools * hd];
        // Values are the first HeadDim of each slot, gates the rest of it, so the gate view starts HeadDim in.
        DsaCompressedPool.GatedSoftmaxPool(store, store.Slice(hd), width, ape, pools, pool, hd, keys);
        return keys;
    }
}
