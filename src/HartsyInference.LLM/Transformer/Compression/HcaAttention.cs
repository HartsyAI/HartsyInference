using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.LLM.Transformer.Sparse;

namespace HartsyInference.LLM.Transformer.Compression;

/// <summary>Host reference for one heavily compressed attention (HCA) layer (vLLM-ERNIE <c>Ernie5_5HCAAttentionMixin</c> over DeepSeek V4
/// <c>DeepseekV4Attention</c>, a <c>compress_ratios</c> entry of at least 2 except 4 with no indexer): low-rank queries, one shared key/value
/// row per token that is both key and value, a sliding window plus every completed compressed latent, a per-head sink, the head
/// mix, the inverse RoPE, a grouped low-rank output with an optional sigmoid gate, and the output projection.</summary>
/// <remarks>
/// <para>Each query at position <c>p</c> attends the positions <c>max(p - Window + 1, 0) .. p</c> and the latents
/// <c>0 .. (p + 1) / CompressRatio - 1</c>, as DeepSeek V4's <c>fill_short_context_topk_indices</c> builds them for C128 layers. The
/// attention goes through <see cref="IBackend.Attention"/> over that query's gathered rows.</para>
/// <para>Any chunk length works; a call at position 0 starts a new sequence. Rows are cached in F32; the reference stores them as
/// BF16 or FP8 depending on the serving KV-cache dtype.</para>
/// </remarks>
public sealed class HcaAttention : IDisposable
{
    private readonly IBackend _backend;
    private readonly HcaAttentionSettings _s;
    private readonly HcaAttentionWeights _w;
    private readonly IRopeRowSource _rope;
    private Tensor? _headNormOnes;

    /// <param name="backend">Runs every projection, norm, rotation and the attention itself.</param>
    /// <param name="settings">Shape of this layer.</param>
    /// <param name="weights">This layer's weights.</param>
    /// <param name="rope">Rotary rows for the trailing <c>RopeDim</c> slice, covering every position that will be seen; compressed latents
    /// rotate at their group's first position in the same table.</param>
    public HcaAttention(IBackend backend, HcaAttentionSettings settings, HcaAttentionWeights weights, IRopeRowSource rope)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(rope);
        settings.Validate();
        if (rope.HalfDim * 2 != settings.RopeDim) throw new ArgumentException("The rope table width must equal RopeDim.", nameof(rope));
        int h = settings.Heads, hd = settings.HeadDim, grouped = settings.OGroups * settings.OLoraRank;
        DsaIndexer.RequireShape(weights.WqA, settings.QLoraRank, settings.Dim, "wq_a");
        DsaIndexer.RequireShape(weights.WqB, h * hd, settings.QLoraRank, "wq_b");
        DsaIndexer.RequireShape(weights.Wkv, hd, settings.Dim, "wkv");
        DsaIndexer.RequireShape(weights.CompressorWkv, hd, settings.Dim, "compressor.wkv");
        DsaIndexer.RequireShape(weights.CompressorWgate, hd, settings.Dim, "compressor.wgate");
        DsaIndexer.RequireShape(weights.WoB, settings.Dim, grouped, "wo_b");
        if (weights.OGate is { } gate) DsaIndexer.RequireShape(gate, grouped, settings.QLoraRank, "o_gate");
        if (weights.QNorm.ElementCount != settings.QLoraRank || weights.KvNorm.ElementCount != hd || weights.CompressorNorm.Length != hd)
            throw new ArgumentException("The norms must match QLoraRank and HeadDim.", nameof(weights));
        if (weights.CompressorApe is { } ape && ape.Length != settings.CompressRatio * hd)
            throw new ArgumentException("The compressor bias must hold CompressRatio x HeadDim values.", nameof(weights));
        DsaIndexer.RequireShape(weights.WoA, grouped, settings.GroupDim, "wo_a");
        if (weights.Sink.DType != DType.F32 || weights.Sink.ElementCount != h)
            throw new ArgumentException("The sink must be F32 with one logit per head.", nameof(weights));
        if (weights.HeadMix is { } mix && (mix.DType != DType.F32 || mix.Shape.Rank != 2 || mix.Shape[0] != h || mix.Shape[1] != h))
            throw new ArgumentException("The head mix must be F32 [Heads, Heads].", nameof(weights));
        _backend = backend;
        _s = settings;
        _w = weights;
        _rope = rope;
        if (settings.QueryHeadNorm)
        {
            _headNormOnes = new Tensor(new TensorShape(hd), DType.F32);
            _headNormOnes.AsSpan<float>().Fill(1f);
        }
    }

    /// <summary>The shape this layer was built with, used to size its per-sequence state.</summary>
    public HcaAttentionSettings Settings => _s;

    /// <summary>Runs the layer over <paramref name="tokens"/> rows at positions <c>startPos ..</c> and writes <c>[tokens, Dim]</c> to
    /// <paramref name="y"/>.</summary>
    /// <param name="x">Normalized hidden states, <c>[tokens, Dim]</c>.</param>
    public void Forward(ReadOnlySpan<float> x, int tokens, int startPos, HcaAttentionState state, Span<float> y)
    {
        ArgumentNullException.ThrowIfNull(state);
        HcaAttentionSettings s = _s;
        if (tokens < 1) throw new ArgumentOutOfRangeException(nameof(tokens));
        if (x.Length != (long)tokens * s.Dim || y.Length != x.Length) throw new ArgumentException("x and y must each hold tokens x Dim values.");
        if (startPos != 0 && startPos != state.Length)
            throw new InvalidOperationException($"The state holds {state.Length} positions; this call starts at {startPos}.");
        if (startPos + tokens > state.Capacity) throw new InvalidOperationException("The HCA cache is full; allocate state for a longer sequence.");
        if (startPos + tokens > _rope.Length) throw new ArgumentOutOfRangeException(nameof(startPos), "The rope table does not reach this position.");
        if (startPos == 0) state.Reset();
        int h = s.Heads, hd = s.HeadDim, rd = s.RopeDim;
        int[] positions = HostOps.Range(startPos, tokens);

        float[] qr = HostOps.Project(_backend, x, tokens, s.Dim, _w.WqA, s.QLoraRank);
        HostOps.RmsNorm(_backend, qr, s.QLoraRank, _w.QNorm, s.NormEps);
        float[] q = HostOps.Project(_backend, qr, tokens, s.QLoraRank, _w.WqB, h * hd);
        if (_headNormOnes is not null) HostOps.RmsNorm(_backend, q, hd, _headNormOnes, s.NormEps);
        PartialRotary.Rotate(_backend, _rope, rd, q, [1, tokens, h, hd], positions, inverse: false, s.RopePairing);

        float[] kv = HostOps.Project(_backend, x, tokens, s.Dim, _w.Wkv, hd);
        HostOps.RmsNorm(_backend, kv, hd, _w.KvNorm, s.NormEps);
        PartialRotary.Rotate(_backend, _rope, rd, kv, [1, tokens, hd], positions, inverse: false, s.RopePairing);
        Compress(x, tokens, startPos, state);

        float[] o = Attend(q, kv, tokens, startPos, state);
        for (int p = Math.Max(0, tokens - s.Window); p < tokens; p++)
            kv.AsSpan(p * hd, hd).CopyTo(state.Window.AsSpan((startPos + p) % s.Window * hd, hd));
        state.Length = startPos + tokens;

        if (_w.HeadMix is { } mix) o = MixHeads(o, tokens, mix);
        PartialRotary.Rotate(_backend, _rope, rd, o, [1, tokens, h, hd], positions, inverse: true, s.RopePairing);
        float[] z = HostOps.BlockDiagonal(_backend, o, tokens, s.OGroups, _w.WoA, _w.WoAScaleInv);
        if (_w.OGate is { } gateWeight)
            AttentionOutputGate.ApplySigmoid(z, HostOps.Project(_backend, qr, tokens, s.QLoraRank, gateWeight, s.OGroups * s.OLoraRank));
        HostOps.Project(_backend, z, tokens, s.OGroups * s.OLoraRank, _w.WoB, s.Dim).CopyTo(y);
    }

    // Pools the chunk into the compressor and appends each completed latent, rotated at its group's first position.
    private void Compress(ReadOnlySpan<float> x, int tokens, int startPos, HcaAttentionState state)
    {
        HcaAttentionSettings s = _s;
        int hd = s.HeadDim, ratio = s.CompressRatio;
        state.Compressor ??= new KvCompressorState(ratio, hd, _w.CompressorApe, _w.CompressorNorm, s.NormEps);
        float[] kv = HostOps.Project(_backend, x, tokens, s.Dim, _w.CompressorWkv, hd);
        float[] score = HostOps.Project(_backend, x, tokens, s.Dim, _w.CompressorWgate, hd);
        float[] latent = new float[state.Compressor.MaxRows(startPos, tokens) * hd];
        int rows = state.Compressor.Pool(kv, score, tokens, startPos, latent);
        if (rows == 0) return;
        int first = startPos / ratio;
        int[] groupStarts = CompressedRopePositions.GroupStart(first, rows, ratio);
        PartialRotary.Rotate(_backend, _rope, s.RopeDim, latent, [1, rows, hd], groupStarts, inverse: false, s.RopePairing);
        latent.AsSpan(0, rows * hd).CopyTo(state.Compressed.AsSpan(first * hd, rows * hd));
    }

    // Each query attends its own window rows plus every visible compressed latent; the row is both key and value (MQA, one KV head).
    private float[] Attend(float[] q, float[] kv, int tokens, int startPos, HcaAttentionState state)
    {
        HcaAttentionSettings s = _s;
        int h = s.Heads, hd = s.HeadDim, win = s.Window;
        float[] o = new float[tokens * h * hd];
        for (int t = 0; t < tokens; t++)
        {
            int pos = startPos + t, first = Math.Max(pos - win + 1, 0), windowRows = pos - first + 1, compressed = (pos + 1) / s.CompressRatio;
            int count = windowRows + compressed;
            using Tensor rows = new(new TensorShape(count, hd), DType.F32);
            Span<float> r = rows.AsSpan<float>();
            state.Compressed.AsSpan(0, compressed * hd).CopyTo(r);
            for (int j = first; j <= pos; j++)
            {
                ReadOnlySpan<float> row = j >= startPos ? kv.AsSpan((j - startPos) * hd, hd) : state.Window.AsSpan(j % win * hd, hd);
                row.CopyTo(r.Slice((compressed + j - first) * hd, hd));
            }
            using Tensor qT = HostMath.Tensor(q.AsSpan(t * h * hd, h * hd), 1, h * hd);
            using Tensor oT = new(new TensorShape(1, h * hd), DType.F32);
            AttentionCall call = new()
            {
                Query = AttentionOperand.TokenMajor(qT, h, hd),
                Key = AttentionOperand.Packed(rows, count, hd, 0, 1, hd),
                Value = AttentionOperand.Packed(rows, count, hd, 0, 1, hd),
                Output = AttentionOperand.TokenMajor(oT, h, hd),
                Scale = s.Scale,
                Logits = new AttentionLogitTerms(0f, _w.Sink, null),
            };
            _backend.Attention(in call);
            oT.AsReadOnlySpan<float>().CopyTo(o.AsSpan(t * h * hd, h * hd));
        }
        return o;
    }

    // [tokens, heads, HeadDim] through the head mix, head-major for one matrix product.
    private float[] MixHeads(float[] o, int tokens, Tensor mix)
    {
        int h = _s.Heads, hd = _s.HeadDim;
        using Tensor headMajor = new(new TensorShape(h, tokens * hd), DType.F32);
        Span<float> hm = headMajor.AsSpan<float>();
        for (int t = 0; t < tokens; t++)
            for (int head = 0; head < h; head++)
                o.AsSpan((t * h + head) * hd, hd).CopyTo(hm.Slice((head * tokens + t) * hd, hd));
        using Tensor mixed = new(new TensorShape(h, tokens * hd), DType.F32);
        HeadMix.Apply(_backend, mix, headMajor, mixed);
        ReadOnlySpan<float> m = mixed.AsReadOnlySpan<float>();
        float[] result = new float[o.Length];
        for (int head = 0; head < h; head++)
            for (int t = 0; t < tokens; t++)
                m.Slice((head * tokens + t) * hd, hd).CopyTo(result.AsSpan((t * h + head) * hd, hd));
        return result;
    }

    /// <inheritdoc />
    public void Dispose() => Interlocked.Exchange(ref _headNormOnes, null)?.Dispose();
}
