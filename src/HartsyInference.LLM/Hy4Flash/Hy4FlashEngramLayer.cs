using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Host reference of one Hy4 Flash Engram layer (transformers <c>HYV4FlashEngramLayer</c>, PR 50903 at <c>eb1d5db</c>): a hashed n-gram lookup whose rows are
/// normalized per head, projected once into a shared value and gated per residual stream by how well that stream matches the memory key. The result is added to the
/// <c>[tokens, hc, hidden]</c> stream before the layer attention.</summary>
/// <remarks>
/// Hash: token ids go through the frozen <c>lookup_table</c> (ids at or past its length read as the pad id's entry), each lookback is multiplied by its multiplier, the first
/// <c>n</c> products are XORed and reduced modulo the prime of each head of order <c>n</c>; heads index one flat table through per-head row offsets. Every small tensor (biases, gains,
/// per-stream scales) is widened to F32 once here, whatever its stored dtype: the released checkpoint keeps them in BF16. Norm epsilon: the reference builds its Engram norms with
/// <c>nn.RMSNorm(dim)</c> and no epsilon, so PyTorch uses the machine epsilon of the activation dtype; pass 1.1920929e-7 for F32 and 0.0078125 for a BF16 run (not yet matched
/// against a BF16 reference). The layer reads one document per sequence: a packed batch needs the reference's per-document positions, which this layer does not take.
/// </remarks>
public sealed class Hy4FlashEngramLayer : IDisposable
{
    private readonly IBackend _backend;
    private readonly Hy4FlashConfig _cfg;
    private readonly Hy4FlashEngramWeights _w;
    private readonly Hy4FlashRowTable _rows;
    private readonly int[] _lookup;
    private readonly long[] _multipliers;
    private readonly long[] _moduli;
    private readonly long[] _offsets;
    private readonly Tensor[] _perHeadNorms;
    private readonly Tensor[] _keyNorms, _queryNorms;
    private readonly Tensor _ones;
    private readonly float[][] _keyBiases, _perHcScales;
    private readonly float[] _valueBias, _gamma;
    private readonly List<Tensor> _owned = [];
    private readonly float _eps;
    private readonly int _headDim, _width, _heads, _hc, _hidden;
    private readonly float _gainScale;
    private int _disposed;

    /// <param name="backend">Runs the projections and norms.</param>
    /// <param name="config">The model config; supplies the Engram shape, seed and moduli.</param>
    /// <param name="layerId">The config layer id (the decoder index plus one).</param>
    /// <param name="weights">Dense weights, borrowed; any small tensor in BF16 or F16 is widened into a copy this layer owns.</param>
    /// <param name="rows">The embedding table of this layer, borrowed.</param>
    /// <param name="lookupTable">The frozen token map stored with the layer, <c>engram_lookup_table_size</c> non-negative entries.</param>
    /// <param name="normEps">Epsilon of every Engram RMS norm.</param>
    public Hy4FlashEngramLayer(IBackend backend, Hy4FlashConfig config, int layerId, Hy4FlashEngramWeights weights, Hy4FlashRowTable rows,
        int[] lookupTable, float normEps)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(rows);
        ArgumentNullException.ThrowIfNull(lookupTable);
        if (!config.EngramHeadVocabSizes.TryGetValue(layerId, out long[][]? sizes)) throw new ArgumentException($"The config has no Engram layer id {layerId}.", nameof(layerId));
        if (lookupTable.Length != config.EngramLookupTableSize) throw new ArgumentException("The lookup table must hold engram_lookup_table_size entries.", nameof(lookupTable));
        if (config.EngramPadId >= lookupTable.Length) throw new ArgumentException("The pad id lies outside the lookup table.", nameof(lookupTable));
        if (Array.Exists(lookupTable, static v => v < 0)) throw new ArgumentException("The lookup table holds a negative id.", nameof(lookupTable));
        _headDim = config.EngramEmbedPerNgram / config.EngramHeadsPerNgram;
        if (rows.Dim != _headDim || rows.Rows != config.EngramTableRows(layerId))
            throw new ArgumentException($"The table must be [{config.EngramTableRows(layerId)}, {_headDim}]; got [{rows.Rows}, {rows.Dim}].", nameof(rows));
        _heads = config.EngramHeadsPerNgram;
        _hc = config.HcMult;
        _hidden = config.HiddenSize;
        _width = (config.EngramMaxNgramSize - 1) * config.EngramEmbedPerNgram;
        if (weights.PerHeadNorms.Count != config.EngramHashHeads || weights.KeyProjs.Count != _hc || weights.KeyBiases.Count != _hc || weights.NormKey.Count != _hc
            || weights.NormQuery.Count != _hc || weights.PerHcScales.Count != _hc)
            throw new ArgumentException("The Engram weights must carry one norm per hash head and one key, norm and scale per residual stream.", nameof(weights));
        _backend = backend;
        _cfg = config;
        _w = weights;
        _rows = rows;
        _lookup = lookupTable;
        _eps = normEps;
        _gainScale = MathF.Pow(_hidden, -0.5f);
        _multipliers = Hy4FlashEngramMultipliers.Draw(config.EngramSeed, layerId, lookupTable.Max() + 1, config.EngramMaxNgramSize);
        _moduli = [.. sizes.SelectMany(static order => order)];
        _offsets = new long[_moduli.Length];
        for (int i = 1; i < _offsets.Length; i++) _offsets[i] = _offsets[i - 1] + _moduli[i - 1];
        try
        {
            _perHeadNorms = [.. weights.PerHeadNorms.Select(Own)];
            _keyNorms = [.. weights.NormKey.Select(Own)];
            _queryNorms = [.. weights.NormQuery.Select(Own)];
            _valueBias = Host(weights.ValueBias);
            _gamma = Host(weights.UnifiedPreGamma);
            _keyBiases = [.. weights.KeyBiases.Select(Host)];
            _perHcScales = [.. weights.PerHcScales.Select(Host)];
            _ones = new Tensor(new TensorShape(_hidden), DType.F32);
            _ones.AsSpan<float>().Fill(1f);
            _owned.Add(_ones);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Stacks each position's token with the <c>MaxNgramSize - 1</c> before it, padding before the sequence start, and marks the positions that own an n-gram
    /// (not an image, video or pad token): <c>build_ngram_tokens</c> and <c>build_engram_inputs</c> of the reference for one text-only or single-document sequence.</summary>
    /// <param name="ids">The whole token history so far.</param>
    /// <param name="start">First position to build rows for.</param>
    /// <param name="count">Positions to build.</param>
    public static (int[] Ngram, bool[] Mask) BuildInputs(Hy4FlashConfig config, ReadOnlySpan<int> ids, int start, int count)
    {
        ArgumentNullException.ThrowIfNull(config);
        int n = config.EngramMaxNgramSize;
        int[] ngram = new int[count * n];
        bool[] mask = new bool[count];
        for (int i = 0; i < count; i++)
        {
            int position = start + i;
            for (int back = 0; back < n; back++) ngram[i * n + back] = position - back >= 0 ? ids[position - back] : config.EngramPadId;
            int id = ids[position];
            mask[i] = id != config.ImageTokenId && id != config.VideoTokenId && id != config.EngramPadId;
        }
        return (ngram, mask);
    }

    /// <summary>The residual to add to <paramref name="stream"/> (<c>[tokens, hc, hidden]</c>, unchanged here): zero at positions outside <paramref name="mask"/>.</summary>
    /// <param name="ngram"><c>[tokens, MaxNgramSize]</c> token ids, the current token first.</param>
    public float[] Residual(ReadOnlySpan<float> stream, int tokens, ReadOnlySpan<int> ngram, ReadOnlySpan<bool> mask)
    {
        int hc = _hc, h = _hidden, maxN = _cfg.EngramMaxNgramSize, heads = _heads, width = _width, headDim = _headDim, flatCount = (maxN - 1) * heads;
        if (stream.Length != (long)tokens * hc * h) throw new ArgumentException("stream must hold tokens x hc x hidden values.", nameof(stream));
        if (ngram.Length != tokens * maxN || mask.Length != tokens) throw new ArgumentException("ngram and mask must cover every token.");

        // Gather every head's rows for all tokens first, so each head is normalized in one backend call rather than one per token.
        float[][] perHead = new float[flatCount][];
        for (int f = 0; f < flatCount; f++) perHead[f] = new float[tokens * headDim];
        long[] mixed = new long[maxN];
        for (int t = 0; t < tokens; t++)
        {
            for (int k = 0; k < maxN; k++)
            {
                int id = ngram[t * maxN + k];
                int compressed = (uint)id < (uint)_lookup.Length ? _lookup[id] : _lookup[_cfg.EngramPadId];
                mixed[k] = compressed * _multipliers[k];
            }
            for (int order = 2; order <= maxN; order++)
            {
                long mix = mixed[0];
                for (int k = 1; k < order; k++) mix ^= mixed[k];
                for (int head = 0; head < heads; head++)
                {
                    int flat = (order - 2) * heads + head;
                    _rows.Read(_offsets[flat] + mix % _moduli[flat], perHead[flat].AsSpan(t * headDim, headDim));
                }
            }
        }
        if (_cfg.EngramFp8Embedding)
            for (int f = 0; f < flatCount; f++)
                for (int t = 0; t < tokens; t++) Hy4FlashFp8Rounding.RoundTripVector(perHead[f].AsSpan(t * headDim, headDim));

        float[] embeddings = new float[tokens * width];
        for (int f = 0; f < flatCount; f++)
        {
            HostOps.RmsNorm(_backend, perHead[f], headDim, _perHeadNorms[f], _eps);
            for (int t = 0; t < tokens; t++) perHead[f].AsSpan(t * headDim, headDim).CopyTo(embeddings.AsSpan(t * width + f * headDim, headDim));
        }

        float[][] gates = new float[hc][];
        for (int i = 0; i < hc; i++)
        {
            float[] key = HostOps.Project(_backend, embeddings, tokens, width, _w.KeyProjs[i], h);
            AddBias(key, _keyBiases[i], tokens, h);
            HostOps.RmsNorm(_backend, key, h, _keyNorms[i], _eps);
            float[] query = new float[tokens * h];
            for (int t = 0; t < tokens; t++) stream.Slice((t * hc + i) * h, h).CopyTo(query.AsSpan(t * h, h));
            HostOps.RmsNorm(_backend, query, h, _queryNorms[i], _eps);
            gates[i] = new float[tokens];
            for (int t = 0; t < tokens; t++)
            {
                float dot = 0f;
                for (int d = 0; d < h; d++) dot += key[t * h + d] * query[t * h + d];
                dot *= _gainScale;
                // A NaN in the stream stays NaN through the gate, as in the reference, instead of reading as a zero score.
                float sign = dot > 0f ? 1f : dot < 0f ? -1f : 0f;
                float shaped = float.IsNaN(dot) ? dot : MathF.Sqrt(MathF.Max(MathF.Abs(dot), 1e-6f)) * sign;
                gates[i][t] = 1f / (1f + MathF.Exp(-shaped));
            }
        }

        float[] main = HostOps.Project(_backend, embeddings, tokens, width, _w.ValueProj, h);
        AddBias(main, _valueBias, tokens, h);
        HostOps.RmsNorm(_backend, main, h, _ones, _eps);
        for (int t = 0; t < tokens; t++)
            for (int d = 0; d < h; d++) main[t * h + d] *= _gamma[d];

        float[] residual = new float[tokens * hc * h];
        for (int t = 0; t < tokens; t++)
        {
            if (!mask[t]) continue;
            for (int i = 0; i < hc; i++)
                for (int d = 0; d < h; d++)
                    residual[(t * hc + i) * h + d] = gates[i][t] * main[t * h + d] + _perHcScales[i][d] * main[t * h + d];
        }
        return residual;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }

    // A small tensor the layer reads as host floats: widened from BF16 or F16 to a copy this layer owns, read in place when already F32.
    private float[] Host(Tensor t)
    {
        Tensor f32 = Own(t);
        return f32.AsReadOnlySpan<float>().ToArray();
    }

    // A norm gain RMSNorm reads as a tensor: F32 is borrowed, anything else is widened to a copy this layer owns.
    private Tensor Own(Tensor t)
    {
        if (t.DType == DType.F32) return t;
        Tensor widened = ProjectionOps.EnsureF32(t);
        _owned.Add(widened);
        return widened;
    }

    private static void AddBias(float[] data, float[] bias, int rows, int width)
    {
        for (int t = 0; t < rows; t++)
            for (int d = 0; d < width; d++) data[t * width + d] += bias[d];
    }
}
