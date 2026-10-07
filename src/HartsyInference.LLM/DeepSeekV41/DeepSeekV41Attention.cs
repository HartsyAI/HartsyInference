using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Host reference for one V4.1 attention layer: low-rank queries, a shared key/value latent, a sliding window plus optional compressed positions, and a grouped low-rank output.</summary>
/// <remarks>Follows upstream <c>Attention.forward</c> step for step. A prefill must start at position 0 and decode takes one token at a time, as upstream does;
/// a later multi-token chunk is the caller's to split. Cached latents are kept after their quantize-dequantize round trip, which is what the quantized cache decodes to.</remarks>
public sealed class DeepSeekV41Attention
{
    private readonly IBackend _backend;
    private readonly DeepSeekV41AttentionSettings _s;
    private readonly DeepSeekV41AttentionWeights _w;
    private readonly DeepSeekV41RopeTable _rope;

    /// <param name="backend">Provides rope, quantize round trips, window indices, indexer scores and sparse attention.</param>
    /// <param name="settings">Shape and role of this layer.</param>
    /// <param name="weights">Dequantized weights; compressor and indexer must be present exactly when the settings call for them.</param>
    /// <param name="rope">This layer's rotary table (compressed layers use the YaRN table with the compress theta), covering every position that will be seen.</param>
    public DeepSeekV41Attention(IBackend backend, DeepSeekV41AttentionSettings settings, DeepSeekV41AttentionWeights weights, DeepSeekV41RopeTable rope)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(rope);
        if (settings.Heads * settings.HeadDim % settings.OGroups != 0) throw new ArgumentException("Heads x HeadDim must divide into OGroups.", nameof(settings));
        if (rope.HalfDim * 2 != settings.RopeDim) throw new ArgumentException("The rope table width must equal RopeDim.", nameof(rope));
        if (settings.IsKvSource != (weights.Compressor is not null) && settings.CompressRatio > 0)
            throw new ArgumentException("Compressor weights must be present exactly on a KV-source layer.", nameof(weights));
        if (settings.CompressRatio > 0 && settings.IsIndexSource != (weights.Indexer is not null))
            throw new ArgumentException("Indexer weights must be present exactly on an index-source layer.", nameof(weights));
        _backend = backend;
        _s = settings;
        _w = weights;
        _rope = rope;
    }

    /// <summary>Runs the layer over <paramref name="tokens"/> rows and writes <c>[tokens, Dim]</c> to <paramref name="y"/>.</summary>
    /// <param name="x">Normalized hidden states, <c>[tokens, Dim]</c>.</param>
    /// <param name="tokens">Row count; exactly 1 unless <paramref name="startPos"/> is 0.</param>
    /// <param name="startPos">Absolute position of the first row.</param>
    /// <param name="state">This layer's cache for the sequence.</param>
    /// <param name="shared">Slots handed between layers within this forward pass.</param>
    /// <param name="y">Receives the output.</param>
    public void Forward(ReadOnlySpan<float> x, int tokens, int startPos, DeepSeekV41AttentionState state, DeepSeekV41SharedAttention shared, Span<float> y)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(shared);
        DeepSeekV41AttentionSettings s = _s;
        if (tokens < 1) throw new ArgumentOutOfRangeException(nameof(tokens));
        if (startPos > 0 && tokens != 1) throw new NotSupportedException("After position 0 attention takes one token per call; split a longer chunk.");
        if (x.Length != (long)tokens * s.Dim || y.Length != x.Length) throw new ArgumentException("x and y must each hold tokens x Dim values.");
        if (startPos + tokens > _rope.Length) throw new ArgumentOutOfRangeException(nameof(startPos), "The rope table does not reach this position.");
        int hd = s.HeadDim, rd = s.RopeDim, heads = s.Heads, win = s.Window;

        float[] qr = DeepSeekV41HostMath.Linear(x, _w.WqA, tokens, s.Dim, s.QLoraRank);
        DeepSeekV41HostMath.RmsNormRows(qr, _w.QNorm, s.QLoraRank, s.NormEps);
        float[] q = DeepSeekV41HostMath.Linear(qr, _w.WqB, tokens, s.QLoraRank, heads * hd);
        int[] positions = Range(startPos, tokens);
        Rotate(q, [1, tokens, heads, hd], positions, hd - rd, inverse: false);

        float[] kv = DeepSeekV41HostMath.Linear(x, _w.Wkv, tokens, s.Dim, hd);
        DeepSeekV41HostMath.RmsNormRows(kv, _w.KvNorm, hd, s.NormEps);
        Rotate(kv, [1, tokens, hd], positions, hd - rd, inverse: false);
        RoundTrip(kv, hd, LatentEncoding.Fp8E4M3Ue8m0x32);
        for (int p = Math.Max(0, tokens - win); p < tokens; p++)
            kv.AsSpan(p * hd, hd).CopyTo(state.Window.AsSpan((startPos + p) % win * hd, hd));

        // prefill attends over its own chunk, decode over the whole ring
        bool prefill = startPos == 0;
        int windowRows = prefill ? tokens : win;
        (int idxRows, int idxCols) = WindowIndicesReference.Shape(win, tokens, startPos);
        using Tensor windowIdx = new(new TensorShape(idxRows, idxCols), DType.I32);
        _backend.BuildWindowIndices(windowIdx, win, tokens, startPos);
        using Tensor windowCodes = DeepSeekV41HostMath.Tensor(prefill ? kv : state.Window, windowRows, hd);
        LatentSource window = new(LatentEncoding.F32, windowCodes, null, windowRows, hd);

        int compressCols = 0;
        int[]? compressIdx = null;
        LatentSource main = LatentSource.Empty;
        Tensor? mainCodes = null;
        try
        {
            if (s.CompressRatio > 0)
            {
                int compressLen = (startPos + tokens) / s.CompressRatio;
                (float[]? latent, int latentRows) = s.IsKvSource ? Compress(x, tokens, startPos, state) : (null, 0);
                if (s.IsKvSource) shared.CompressKv = state.CompressKv;

                if (s.IsIndexSource)
                {
                    if (compressLen == 0) (compressIdx, compressCols) = ([], 0);
                    else (compressIdx, compressCols) = Index(x, qr, tokens, startPos, windowRows, latent, latentRows, state, shared);
                    shared.Topk = compressIdx;
                    shared.TopkWidth = compressCols;
                }
                else
                {
                    compressIdx = shared.Topk ?? throw new InvalidOperationException("A reusing layer ran before any index-source layer.");
                    compressCols = shared.TopkWidth;
                }

                if (latent is not null) StoreLatent(latent, latentRows, startPos, state);
                if (compressLen > 0)
                {
                    float[] cache = shared.CompressKv ?? throw new InvalidOperationException("No compressed KV is available to this layer.");
                    mainCodes = DeepSeekV41HostMath.Tensor(cache.AsSpan(0, compressLen * hd), compressLen, hd);
                    main = new LatentSource(LatentEncoding.F32, mainCodes, null, compressLen, hd);
                }
            }

            int[] indices = new int[tokens * (idxCols + compressCols)];
            ReadOnlySpan<int> windowSlots = windowIdx.AsReadOnlySpan<int>();
            for (int t = 0; t < tokens; t++)
            {
                int row = idxRows == 1 ? 0 : t;
                windowSlots.Slice(row * idxCols, idxCols).CopyTo(indices.AsSpan(t * (idxCols + compressCols), idxCols));
                if (compressCols > 0) compressIdx.AsSpan(t * compressCols, compressCols).CopyTo(indices.AsSpan(t * (idxCols + compressCols) + idxCols, compressCols));
            }

            using Tensor query = DeepSeekV41HostMath.Tensor(q, tokens, heads, hd);
            using Tensor output = new(new TensorShape(tokens, heads, hd), DType.F32);
            using Tensor indexT = DeepSeekV41HostMath.Tensor(indices, tokens, idxCols + compressCols);
            using Tensor sink = DeepSeekV41HostMath.Tensor(_w.Sink, heads);
            _backend.SparseLatentAttention(output, query, window, main, indexT, windowRows, sink, 1f / MathF.Sqrt(hd));

            float[] o = output.AsReadOnlySpan<float>().ToArray();
            Rotate(o, [1, tokens, heads, hd], positions, hd - rd, inverse: true);
            float[] grouped = new float[tokens * s.OGroups * s.OLoraRank];
            DeepSeekV41GroupedProjection.Apply(o, _w.WoA, tokens, s.OGroups, s.OLoraRank, s.GroupDim, grouped);
            DeepSeekV41HostMath.Linear(grouped, _w.WoB, tokens, s.OGroups * s.OLoraRank, s.Dim).CopyTo(y);
        }
        finally
        {
            mainCodes?.Dispose();
        }
    }

    private (float[]? Latent, int Rows) Compress(ReadOnlySpan<float> x, int tokens, int startPos, DeepSeekV41AttentionState state)
    {
        DeepSeekV41CompressorWeights c = _w.Compressor!;
        int hd = _s.HeadDim, ratio = _s.CompressRatio;
        float[] kv = DeepSeekV41HostMath.Linear(x, c.Wkv, tokens, _s.Dim, hd);
        if (ratio == 1)
        {
            DeepSeekV41HostMath.RmsNormRows(kv, c.Norm, hd, _s.NormEps);
            return (kv, tokens);
        }
        float[] score = DeepSeekV41HostMath.Linear(x, c.Wgate!, tokens, _s.Dim, hd);
        DeepSeekV41CompressorState pool = state.Compressor ?? throw new InvalidOperationException("A pooling layer needs compressor state.");
        float[] pooled = new float[pool.MaxRows(startPos, tokens) * hd];
        int rows = pool.Pool(kv, score, tokens, startPos, pooled);
        if (rows == 0) return (null, 0);
        float[] latent = pooled.AsSpan(0, rows * hd).ToArray();
        DeepSeekV41HostMath.RmsNormRows(latent, c.Norm, hd, _s.NormEps);
        return (latent, rows);
    }

    // The latent is stored rotated at its group's first position and through the compressed-cache quantization.
    private void StoreLatent(float[] latent, int rows, int startPos, DeepSeekV41AttentionState state)
    {
        int hd = _s.HeadDim, ratio = _s.CompressRatio, first = startPos / ratio;
        if ((first + rows) * hd > state.CompressKv!.Length) throw new InvalidOperationException("The compressed cache is full; allocate state for a longer sequence.");
        Rotate(latent, [1, rows, hd], GroupPositions(first, rows, ratio), hd - _s.RopeDim, inverse: false);
        RoundTrip(latent, hd, LatentEncoding.Fp4E2M1E4M3x16);
        latent.AsSpan(0, rows * hd).CopyTo(state.CompressKv.AsSpan(first * hd, rows * hd));
    }

    private (int[] Indices, int Width) Index(ReadOnlySpan<float> x, float[] qr, int tokens, int startPos, int offset, float[]? latent, int latentRows,
        DeepSeekV41AttentionState state, DeepSeekV41SharedAttention shared)
    {
        DeepSeekV41IndexerWeights ix = _w.Indexer!;
        int ihd = _s.IndexHeadDim, nh = _s.IndexHeads, ratio = _s.CompressRatio, rd = _s.RopeDim, endPos = startPos + tokens;

        if (_s.IsKvSource && latent is not null)
        {
            int first = startPos / ratio;
            float[] k = DeepSeekV41HostMath.Linear(latent, ix.Wk!, latentRows, _s.HeadDim, ihd);
            DeepSeekV41HostMath.RmsNormRows(k, ix.KNorm!, ihd, _s.NormEps);
            Rotate(k, [1, latentRows, ihd], GroupPositions(first, latentRows, ratio), ihd - rd, inverse: false);
            RoundTrip(k, ihd, LatentEncoding.Fp4E2M1E8M0x32);
            k.CopyTo(state.IndexKeys!.AsSpan(first * ihd, latentRows * ihd));
            shared.IndexKeys = state.IndexKeys;
        }

        float[] q = DeepSeekV41HostMath.Linear(qr, ix.WqB, tokens, _s.QLoraRank, nh * ihd);
        Rotate(q, [1, tokens, nh, ihd], Range(startPos, tokens), ihd - rd, inverse: false);
        RoundTrip(q, ihd, LatentEncoding.Fp4E2M1E8M0x32);

        int keyRows = endPos / ratio;
        float[] keys = shared.IndexKeys ?? throw new InvalidOperationException("No index keys are available to this layer.");
        float[] weights = DeepSeekV41HostMath.Linear(x, ix.WeightsProj, tokens, _s.Dim, nh);
        float weightScale = 1f / MathF.Sqrt(ihd) / MathF.Sqrt(nh);
        for (int i = 0; i < weights.Length; i++) weights[i] *= weightScale;

        // a block becomes visible once the query has passed its last token; a decode step sees everything
        int[] lens = new int[tokens];
        for (int t = 0; t < tokens; t++) lens[t] = startPos == 0 ? (t + 1) / ratio : keyRows;

        using Tensor scores = new(new TensorShape(tokens, keyRows), DType.F32);
        using Tensor queryT = DeepSeekV41HostMath.Tensor(q, tokens, nh, ihd);
        using Tensor keyCodes = DeepSeekV41HostMath.Tensor(keys.AsSpan(0, keyRows * ihd), keyRows, ihd);
        using Tensor weightT = DeepSeekV41HostMath.Tensor(weights, tokens, nh);
        using Tensor lensT = DeepSeekV41HostMath.Tensor(lens, tokens);
        using Tensor? candidateT = _s.UsesCandidates ? CandidateTensor(shared, tokens, keyRows) : null;
        _backend.IndexerScores(scores, queryT, new LatentSource(LatentEncoding.F32, keyCodes, null, keyRows, ihd), weightT, lensT, candidateT, 1f);

        ReadOnlySpan<float> all = scores.AsReadOnlySpan<float>();
        if (_s.IsCandidateSource)
        {
            byte[] candidates = new byte[tokens * keyRows];
            bool[] mask = new bool[keyRows];
            for (int t = 0; t < tokens; t++)
            {
                DeepSeekV41IndexSelection.SelectCandidateBlocks(all.Slice(t * keyRows, keyRows), lens[t], _s.CandidateTopkBlocks, _s.CandidateBlockSize, mask);
                for (int n = 0; n < keyRows; n++) candidates[t * keyRows + n] = mask[n] ? (byte)1 : (byte)0;
            }
            shared.Candidates = candidates;
            shared.CandidateWidth = keyRows;
        }

        int width = Math.Min(_s.IndexTopk, keyRows);
        int[] picked = new int[tokens * width];
        for (int t = 0; t < tokens; t++)
            DeepSeekV41IndexSelection.SelectTopK(all.Slice(t * keyRows, keyRows), lens[t], _s.IndexTopk, offset, picked.AsSpan(t * width, width));
        return (picked, width);
    }

    private static Tensor CandidateTensor(DeepSeekV41SharedAttention shared, int tokens, int keyRows)
    {
        if (shared.Candidates is null || shared.CandidateWidth != keyRows)
            throw new InvalidOperationException("A candidate-restricted layer ran before a matching candidate source.");
        Tensor t = new(new TensorShape(tokens, keyRows), DType.U8);
        shared.Candidates.AsSpan(0, tokens * keyRows).CopyTo(t.AsSpan<byte>());
        return t;
    }

    // Rotates the trailing RopeDim slice of every vector in place; the inverse negates sin, which is how the output undoes the query's rotation.
    private void Rotate(float[] data, long[] shape, int[] positions, int dimOffset, bool inverse)
    {
        int half = _rope.HalfDim;
        float[] cos = new float[positions.Length * half], sin = new float[cos.Length];
        for (int i = 0; i < positions.Length; i++)
        {
            _rope.CosRow(positions[i]).CopyTo(cos.AsSpan(i * half, half));
            for (int j = 0; j < half; j++) sin[i * half + j] = inverse ? -_rope.SinRow(positions[i])[j] : _rope.SinRow(positions[i])[j];
        }
        using Tensor x = DeepSeekV41HostMath.Tensor(data, shape);
        using Tensor c = DeepSeekV41HostMath.Tensor(cos, 1, positions.Length, half);
        using Tensor sn = DeepSeekV41HostMath.Tensor(sin, 1, positions.Length, half);
        _backend.ApplyRopeInterleaved(x, c, sn, _s.RopeDim, dimOffset);
        x.AsReadOnlySpan<float>().CopyTo(data);
    }

    private void RoundTrip(float[] data, int width, LatentEncoding encoding)
    {
        using Tensor x = DeepSeekV41HostMath.Tensor(data, data.Length / width, width);
        _backend.ActQuantDequantInPlace(x, encoding);
        x.AsReadOnlySpan<float>().CopyTo(data);
    }

    private static int[] Range(int start, int count)
    {
        int[] r = new int[count];
        for (int i = 0; i < count; i++) r[i] = start + i;
        return r;
    }

    // A latent stands for the first token of its group.
    private static int[] GroupPositions(int firstGroup, int count, int ratio)
    {
        int[] r = new int[count];
        for (int i = 0; i < count; i++) r[i] = (firstGroup + i) * ratio;
        return r;
    }
}
