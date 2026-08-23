using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.BiCodec;

/// <summary>BiCodec — Spark-TTS's dual-stream tokenizer, producing 50 Hz semantic VQ tokens plus 32 global FSQ tokens from a reference audio clip.</summary>
/// <remarks>Used by the SparkTTS pipeline: the text-conditioned Qwen2.5 LM generates new (semantic, global)
/// token streams from these, and a separate DAC HiFi-GAN wave generator converts them back to PCM.
///
/// <para>Two encoders, no shared decoder: the semantic decoder isn't used at TTS
/// inference, only the encode path. We provide a minimal decode for round-trip
/// validation but the production-relevant path is <see cref="Encode"/>.</para>
///
/// <para>State-dict roots:</para>
/// <list type="bullet">
///   <item><c>tokenizer.encoder.*</c> — w2v-BERT projection + downsample → semantic latent</item>
///   <item><c>tokenizer.quantizer.*</c> — single 8192-entry VQ codebook on semantic</item>
///   <item><c>tokenizer.speaker_encoder.*</c> — attention-pool head + FSQ on global</item>
/// </list></remarks>
public sealed unsafe class BiCodec
{
    public BiCodecConfig Config { get; }

    private readonly BiCodecSemanticEncoder _semanticEncoder;
    private readonly BiCodecGlobalEncoder _globalEncoder;

    public BiCodec(BiCodecConfig cfg)
    {
        Config = cfg;
        _semanticEncoder = new BiCodecSemanticEncoder(cfg, "tokenizer");
        _globalEncoder = new BiCodecGlobalEncoder(cfg, "tokenizer.speaker_encoder");
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _semanticEncoder.LoadWeights(w);
        _globalEncoder.LoadWeights(w);
    }

    /// <summary>Encodes a w2v-BERT feature stream into BiCodec's dual token streams; caller owns disposal of both results.</summary>
    /// <param name="features">Channels-last <c>[B, T_features, 1024]</c> (w2v-BERT 2.0 output at 50 Hz).</param>
    /// <returns>Semantic codes <c>[B, T_features]</c> and global codes <c>[B, 32]</c>.</returns>
    public (Tensor SemanticCodes, Tensor GlobalCodes) Encode(IBackend backend, Tensor features, int batch, int tFeatures)
    {
        Tensor semantic = _semanticEncoder.Forward(backend, features, batch, tFeatures);
        Tensor global = _globalEncoder.Forward(backend, features, batch, tFeatures);
        return (semantic, global);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _semanticEncoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _globalEncoder.EnumerateWeights()) yield return t;
    }
}

public sealed record BiCodecConfig
{
    public int FeatureDim { get; init; } = 1_024;
    public int SemanticHiddenDim { get; init; } = 1_024;
    public int SemanticCodebookSize { get; init; } = 8_192;
    public int SemanticCodebookDim { get; init; } = 8;
    public int GlobalTokens { get; init; } = 32;
    public IReadOnlyList<int> GlobalFsqLevels { get; init; } = [4, 4, 4, 4, 4, 4];   // reference fsq_levels [4,4,4,4,4,4]; full speaker/semantic encoder rewrite is Phase 4
    public int GlobalQueryHeads { get; init; } = 8;
    public static BiCodecConfig Default => new();
}

/// <summary>Projects w2v-BERT features to a 50 Hz semantic latent, then nearest-neighbor matches by cosine similarity against an L2-normalized 8192-entry codebook.</summary>
internal sealed unsafe class BiCodecSemanticEncoder
{
    private readonly BiCodecConfig _cfg;
    private readonly string _prefix;

    private Tensor? _projW;
    private Tensor? _projB;
    private Tensor? _vqInProjW;
    private Tensor? _vqInProjB;
    private Tensor? _codebook;
    private Tensor? _codebookNorm;

    public BiCodecSemanticEncoder(BiCodecConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _projW = WhisperOps.EnsureF32(w[$"{_prefix}.encoder.proj.weight"]);
        _projB = WhisperOps.EnsureF32(w[$"{_prefix}.encoder.proj.bias"]);
        _vqInProjW = WhisperOps.EnsureF32(w[$"{_prefix}.quantizer.in_proj.weight"]);
        _vqInProjB = WhisperOps.EnsureF32(w[$"{_prefix}.quantizer.in_proj.bias"]);
        _codebook = WhisperOps.EnsureF32(w[$"{_prefix}.quantizer.codebook.weight"]);
        _codebookNorm = VqOps.L2NormalizeRows(_codebook, _cfg.SemanticCodebookSize, _cfg.SemanticCodebookDim);
    }

    /// <summary>Forward — features <c>[B, T, 1024]</c> → semantic codes <c>[B, T]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor features, int batch, int t)
    {
        // 1) Linear projection to semantic hidden.
        Tensor projected = WhisperOps.ProjectLinear(backend, features, _projW!, _projB, batch, t, _cfg.FeatureDim, _cfg.SemanticHiddenDim);
        // 2) VQ in_proj to codebook_dim.
        Tensor vqInput = WhisperOps.ProjectLinear(backend, projected, _vqInProjW!, _vqInProjB, batch, t, _cfg.SemanticHiddenDim, _cfg.SemanticCodebookDim);
        projected.Dispose();

        // 3) Cosine-similarity nearest-neighbor.
        Tensor codes = new(new TensorShape(batch, t), DType.I32);
        int* cp = (int*)codes.DataPointer;
        float* vp = (float*)vqInput.DataPointer;
        float* cbNorm = (float*)_codebookNorm!.DataPointer;
        Span<float> normQuery = stackalloc float[_cfg.SemanticCodebookDim];

        for (int b = 0; b < batch; b++)
        {
            for (int ti = 0; ti < t; ti++)
            {
                int rowBase = (b * t + ti) * _cfg.SemanticCodebookDim;
                double sumSq = 0d;
                for (int d = 0; d < _cfg.SemanticCodebookDim; d++)
                {
                    float v = vp[rowBase + d];
                    normQuery[d] = v;
                    sumSq += (double)v * v;
                }
                float invNorm = (float)(1.0 / Math.Sqrt(sumSq + 1e-12));
                for (int d = 0; d < _cfg.SemanticCodebookDim; d++) normQuery[d] *= invNorm;

                int bestIdx = 0;
                float bestDot = float.MinValue;
                for (int k = 0; k < _cfg.SemanticCodebookSize; k++)
                {
                    float dot = 0f;
                    int cbBase = k * _cfg.SemanticCodebookDim;
                    for (int d = 0; d < _cfg.SemanticCodebookDim; d++) dot += normQuery[d] * cbNorm[cbBase + d];
                    if (dot > bestDot) { bestDot = dot; bestIdx = k; }
                }
                cp[b * t + ti] = bestIdx;
            }
        }
        vqInput.Dispose();
        return codes;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_projW, _projB, _vqInProjW, _vqInProjB, _codebook];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}

/// <summary>Pools w2v-BERT features via cross-attention against 32 learned queries, then FSQ-quantizes the result into global tokens capturing speaker/timbre identity independent of content.</summary>
internal sealed unsafe class BiCodecGlobalEncoder
{
    private readonly BiCodecConfig _cfg;
    private readonly string _prefix;
    private readonly int _globalTokens;
    private readonly int _featureDim;
    private readonly int _headDim;
    private readonly int _heads;

    private Tensor? _queryEmbed;        // [num_tokens, feature_dim]
    private Tensor? _qProjW;
    private Tensor? _qProjB;
    private Tensor? _kProjW;
    private Tensor? _kProjB;
    private Tensor? _vProjW;
    private Tensor? _vProjB;
    private Tensor? _outProjW;
    private Tensor? _outProjB;
    private Tensor? _fsqProjW;          // projects pooled feature to len(levels)
    private Tensor? _fsqProjB;

    public BiCodecGlobalEncoder(BiCodecConfig cfg, string prefix)
    {
        _cfg = cfg;
        _prefix = prefix;
        _globalTokens = cfg.GlobalTokens;
        _featureDim = cfg.FeatureDim;
        _heads = cfg.GlobalQueryHeads;
        _headDim = cfg.FeatureDim / cfg.GlobalQueryHeads;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _queryEmbed = WhisperOps.EnsureF32(w[$"{_prefix}.queries"]);
        _qProjW = WhisperOps.EnsureF32(w[$"{_prefix}.attn.q_proj.weight"]);
        _qProjB = WhisperOps.EnsureF32(w[$"{_prefix}.attn.q_proj.bias"]);
        _kProjW = WhisperOps.EnsureF32(w[$"{_prefix}.attn.k_proj.weight"]);
        _kProjB = WhisperOps.EnsureF32(w[$"{_prefix}.attn.k_proj.bias"]);
        _vProjW = WhisperOps.EnsureF32(w[$"{_prefix}.attn.v_proj.weight"]);
        _vProjB = WhisperOps.EnsureF32(w[$"{_prefix}.attn.v_proj.bias"]);
        _outProjW = WhisperOps.EnsureF32(w[$"{_prefix}.attn.out_proj.weight"]);
        _outProjB = WhisperOps.EnsureF32(w[$"{_prefix}.attn.out_proj.bias"]);
        _fsqProjW = WhisperOps.EnsureF32(w[$"{_prefix}.fsq_proj.weight"]);
        _fsqProjB = WhisperOps.EnsureF32(w[$"{_prefix}.fsq_proj.bias"]);
    }

    /// <summary>Forward — features <c>[B, T_features, 1024]</c> → global codes <c>[B, 32]</c> Int32.</summary>
    public Tensor Forward(IBackend backend, Tensor features, int batch, int tFeatures)
    {
        // Q: tile learned queries across batch → [B, num_tokens, feature_dim].
        Tensor queries = new(new TensorShape(batch, _globalTokens, _featureDim), DType.F32);
        float* qp = (float*)queries.DataPointer;
        float* qep = (float*)_queryEmbed!.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int j = 0; j < _globalTokens; j++)
            {
                int srcBase = j * _featureDim;
                int dstBase = (b * _globalTokens + j) * _featureDim;
                for (int d = 0; d < _featureDim; d++) qp[dstBase + d] = qep[srcBase + d];
            }
        }

        // Cross-attention: queries attend to features.
        Tensor qProj = WhisperOps.ProjectLinear(backend, queries, _qProjW!, _qProjB, batch, _globalTokens, _featureDim, _featureDim);
        queries.Dispose();
        Tensor kProj = WhisperOps.ProjectLinear(backend, features, _kProjW!, _kProjB, batch, tFeatures, _featureDim, _featureDim);
        Tensor vProj = WhisperOps.ProjectLinear(backend, features, _vProjW!, _vProjB, batch, tFeatures, _featureDim, _featureDim);

        // Reshape to [B, H, S, D] for SDPA.
        Tensor qMh = new(new TensorShape(batch, _heads, _globalTokens, _headDim), DType.F32);
        Tensor kMh = new(new TensorShape(batch, _heads, tFeatures, _headDim), DType.F32);
        Tensor vMh = new(new TensorShape(batch, _heads, tFeatures, _headDim), DType.F32);
        WhisperOps.ReshapeToMultiHead4D(qMh, qProj, batch, _globalTokens, _heads, _headDim);
        WhisperOps.ReshapeToMultiHead4D(kMh, kProj, batch, tFeatures, _heads, _headDim);
        WhisperOps.ReshapeToMultiHead4D(vMh, vProj, batch, tFeatures, _heads, _headDim);
        qProj.Dispose(); kProj.Dispose(); vProj.Dispose();

        Tensor attnOut = new(new TensorShape(batch, _heads, _globalTokens, _headDim), DType.F32);
        backend.ScaledDotProductAttention(attnOut, qMh, kMh, vMh, null, 1f / MathF.Sqrt(_headDim));
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        // Reshape back to [B, num_tokens, feature_dim].
        Tensor attnFlat = new(new TensorShape(batch, _globalTokens, _featureDim), DType.F32);
        float* ap = (float*)attnOut.DataPointer;
        float* fp = (float*)attnFlat.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int j = 0; j < _globalTokens; j++)
                for (int h = 0; h < _heads; h++)
                    for (int d = 0; d < _headDim; d++)
                    {
                        int srcIdx = ((b * _heads + h) * _globalTokens + j) * _headDim + d;
                        int dstIdx = (b * _globalTokens + j) * _featureDim + h * _headDim + d;
                        fp[dstIdx] = ap[srcIdx];
                    }
        attnOut.Dispose();

        Tensor pooled = WhisperOps.ProjectLinear(backend, attnFlat, _outProjW!, _outProjB, batch, _globalTokens, _featureDim, _featureDim);
        attnFlat.Dispose();

        // FSQ projection: pooled features → len(levels) scalars per token, then FSQ quantize.
        int nLevels = _cfg.GlobalFsqLevels.Count;
        Tensor fsqInput = WhisperOps.ProjectLinear(backend, pooled, _fsqProjW!, _fsqProjB, batch, _globalTokens, _featureDim, nLevels);
        pooled.Dispose();

        // FSQ pack to int codes via existing Fsq helper.
        Tensor codes = new(new TensorShape(batch, _globalTokens), DType.I32);
        int[] levelsArr = [.. _cfg.GlobalFsqLevels];
        Fsq.Quantize(codes, fsqInput, levelsArr);
        fsqInput.Dispose();
        return codes;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_queryEmbed, _qProjW, _qProjB, _kProjW, _kProjB, _vProjW, _vProjB,
            _outProjW, _outProjB, _fsqProjW, _fsqProjB];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
