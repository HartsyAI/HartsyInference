using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Compression;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.LLM.Transformer.Mhc;
using HartsyInference.LLM.Transformer.Sparse;

namespace HartsyInference.LLM.Ernie5;

/// <summary>One ERNIE-5.5 decoder layer (vLLM-ERNIE <c>Ernie5_5DecoderLayer</c>), or the decoder block of an MTP layer: an HCA or DSA
/// attention and a dense or latent-MoE SiTU FFN, each wrapped in its own manifold-constrained hyper-connection over the
/// <c>[tokens, hc, hidden]</c> residual stream.</summary>
/// <remarks>Each sublayer derives its mixes from the stream, collapses it, RMS-normalizes the result, runs, and expands its output back.
/// The FFN's mixes are derived from the stream after the attention's expand, which is what the reference's fused post-pre computes.</remarks>
internal sealed class Ernie5Layer : IDisposable
{
    private const int VhaRank = 16;
    private readonly IBackend _backend;
    private readonly Ernie5Config _cfg;
    private readonly List<Tensor> _owned = [];
    private readonly HyperConnection _attnHc;
    private readonly HyperConnection _ffnHc;
    private readonly Tensor _attnNorm;
    private readonly Tensor _ffnNorm;
    private readonly HcaAttention? _hca;
    private readonly DsaAttention? _dsa;
    private readonly MoeFeedForward? _moe;
    private readonly Tensor? _denseGate;
    private readonly Tensor? _denseUp;
    private readonly Tensor? _denseDown;
    private int _disposed;

    /// <summary>Reads layer <paramref name="index"/>'s weights from <paramref name="source"/> and builds its modules; a failure disposes
    /// what it read.</summary>
    public Ernie5Layer(IBackend backend, Ernie5Config cfg, int index, IErnie5WeightSource source, IRopeRowSource hcaRope, IRopeRowSource dsaRope)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cfg);
        ArgumentNullException.ThrowIfNull(source);
        _backend = backend;
        _cfg = cfg;
        Index = index;
        try
        {
            _attnHc = BuildHc(backend, cfg, index, Ernie5WeightNames.AttnHc, source);
            _ffnHc = BuildHc(backend, cfg, index, Ernie5WeightNames.FfnHc, source);
            string prefix = Ernie5WeightNames.Layer(index);
            _attnNorm = Own(ReadF32(source, $"{prefix}.{Ernie5WeightNames.AttnNorm}"));
            _ffnNorm = Own(ReadF32(source, $"{prefix}.{Ernie5WeightNames.FfnNorm}"));
            if (cfg.IsDsaLayer(index)) _dsa = new DsaAttention(backend, DsaSettings(cfg), DsaWeights(index, source), dsaRope, dsaRope);
            else _hca = new HcaAttention(backend, HcaSettings(cfg, index), HcaWeights(index, source), hcaRope);
            if (index < cfg.FirstKDenseReplace)
            {
                _denseGate = Own(source.Read(Ernie5WeightNames.DenseFfn(index, Ernie5WeightNames.DenseFfnLeaves[0])));
                _denseUp = Own(source.Read(Ernie5WeightNames.DenseFfn(index, Ernie5WeightNames.DenseFfnLeaves[1])));
                _denseDown = Own(source.Read(Ernie5WeightNames.DenseFfn(index, Ernie5WeightNames.DenseFfnLeaves[2])));
            }
            else
            {
                Dictionary<string, Tensor> moe = new(StringComparer.Ordinal);
                foreach ((string checkpoint, string engine) in Ernie5WeightNames.MoeKeys(cfg, index)) moe[engine] = Own(source.Read(checkpoint));
                _moe = new MoeFeedForward(MoeSettings(cfg), cfg.HiddenSize, lowVram: false);
                _moe.LoadWeights(moe, prefix);
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Layer index; MTP layers follow the decoder layers.</summary>
    public int Index { get; }

    /// <summary>Runs the layer on <paramref name="stream"/> (<c>[tokens, hc, hidden]</c>) in place.</summary>
    public void Forward(float[] stream, int tokens, int startPos, Ernie5SequenceState state)
    {
        int d = _cfg.HiddenSize;
        float[] x = new float[tokens * d], sub = new float[tokens * d];
        RunSublayer(_attnHc, _attnNorm, stream, tokens, x, sub, (input, output) =>
        {
            if (_hca is not null) _hca.Forward(input, tokens, startPos, state.Hca(Index), output);
            else _dsa!.Forward(input, tokens, startPos, state.Dsa(Index), output);
        });
        RunSublayer(_ffnHc, _ffnNorm, stream, tokens, x, sub, (input, output) => Ffn(input, tokens, output));
    }

    /// <summary>Shape of layer <paramref name="index"/>'s HCA attention.</summary>
    public static HcaAttentionSettings HcaSettings(Ernie5Config cfg, int index) => new()
    {
        Dim = cfg.HiddenSize,
        Heads = cfg.NumAttentionHeads,
        HeadDim = cfg.HeadDim,
        RopeDim = cfg.QkRopeHeadDim,
        QLoraRank = cfg.QLoraRank,
        OGroups = cfg.OGroups,
        OLoraRank = cfg.OLoraRank,
        Window = cfg.SlidingWindow,
        CompressRatio = cfg.CompressRatios[index],
        NormEps = (float)cfg.RmsNormEps,
        QueryHeadNorm = string.Equals(cfg.QkNormType, "per_head", StringComparison.Ordinal),
    };

    /// <summary>Shape of a DSA attention; the indexer rotates <c>hybrid_mla_qk_rope_head_dim</c> values like the main attention.</summary>
    public static DsaAttentionSettings DsaSettings(Ernie5Config cfg) => new()
    {
        Dim = cfg.HiddenSize,
        Heads = cfg.MlaHeads,
        QLoraRank = cfg.MlaQLoraRank,
        KvLoraRank = cfg.MlaKvLoraRank,
        QkNopeDim = cfg.MlaQkNopeHeadDim,
        QkRopeDim = cfg.MlaQkRopeHeadDim,
        VDim = cfg.MlaVHeadDim,
        Window = cfg.SlidingWindow,
        NormEps = (float)cfg.RmsNormEps,
        Indexer = new DsaIndexerSettings
        {
            Heads = cfg.IndexNHeads, HeadDim = cfg.IndexHeadDim, RopeDim = cfg.MlaQkRopeHeadDim, TopK = cfg.IndexTopk,
        },
    };

    /// <summary>The latent MoE block of every layer from <c>first_k_dense_replace</c> on (PR #695's options).</summary>
    public static MoeConfig MoeSettings(Ernie5Config cfg) => new()
    {
        NumExperts = cfg.NRoutedExperts,
        NumExpertsPerTok = cfg.NumExpertsPerTok,
        MoeIntermediateSize = cfg.MoeIntermediateSize,
        NormTopKProb = cfg.NormTopkProb,
        Scoring = MoeScoring.Sigmoid,
        SharedExpertIntermediateSize = cfg.NSharedExperts * cfg.MoeIntermediateSize,
        Activation = ActivationKind.Situ,
        ActivationSituBeta = cfg.ActivationSituBeta,
        ActivationSituLinearBeta = cfg.ActivationSituLinearBeta,
        FirstDenseLayers = cfg.FirstKDenseReplace,
        RoutedScalingFactor = (float)cfg.RoutedScalingFactor,
        SplitFeatureRouting = cfg.MoeSplitFeatureRouting,
        LearnedRoutedScale = cfg.RoutedScalingFactorLearnable,
        LatentSize = cfg.MoeLatentSize,
        LatentNorm = cfg.LatentMoeUseNorm ? MoeLatentNorm.AfterCombine : MoeLatentNorm.None,
        LatentNormEps = (float)cfg.RmsNormEps,
    };

    private void RunSublayer(HyperConnection hc, Tensor norm, float[] stream, int tokens, float[] x, float[] sub, Action<float[], float[]> body)
    {
        int h = hc.Hc;
        float[] pre = new float[tokens * h], post = new float[tokens * h], comb = new float[tokens * h * h];
        hc.Mixes(stream, tokens, pre, post, comb);
        hc.Collapse(stream, pre, tokens, x);
        HostOps.RmsNorm(_backend, x, _cfg.HiddenSize, norm, (float)_cfg.RmsNormEps);
        body(x, sub);
        hc.Expand(sub, stream, post, comb, tokens, stream);
    }

    private void Ffn(float[] x, int tokens, float[] y)
    {
        int d = _cfg.HiddenSize;
        using Tensor input = HostMath.Tensor(x, 1, tokens, d);
        if (_moe is not null)
        {
            using Tensor routed = _moe.Forward(_backend, input, tokens);
            routed.AsReadOnlySpan<float>().CopyTo(y);
            return;
        }
        int inner = _cfg.IntermediateSize;
        Tensor gate = new(new TensorShape(1, tokens, inner), DType.F32);
        Tensor up = new(new TensorShape(1, tokens, inner), DType.F32);
        ProjectionOps.Project(_backend, gate, input, _denseGate!, null, lowVram: false);
        ProjectionOps.Project(_backend, up, input, _denseUp!, null, lowVram: false);
        using Tensor act = GatedFfn.SituGlu(_backend, gate, up, _cfg.ActivationSituBeta, _cfg.ActivationSituLinearBeta);
        using Tensor output = new(new TensorShape(1, tokens, d), DType.F32);
        ProjectionOps.Project(_backend, output, act, _denseDown!, null, lowVram: false);
        output.AsReadOnlySpan<float>().CopyTo(y);
    }

    private HcaAttentionWeights HcaWeights(int index, IErnie5WeightSource source)
    {
        Tensor Read(string leaf) => Own(source.Read(Ernie5WeightNames.Attn(index, leaf)));
        float[] Host(string leaf) => HostF32(source, Ernie5WeightNames.Attn(index, leaf));
        string[] l = Ernie5WeightNames.HcaLeaves;
        return new HcaAttentionWeights
        {
            WqA = Read(l[0]),
            QNorm = Own(ReadF32(source, Ernie5WeightNames.Attn(index, l[1]))),
            WqB = Read(l[2]),
            Wkv = Read(l[3]),
            KvNorm = Own(ReadF32(source, Ernie5WeightNames.Attn(index, l[4]))),
            WoA = Own(ReadF32(source, Ernie5WeightNames.Attn(index, l[5]))),
            WoB = Read(l[6]),
            Sink = Own(ReadF32(source, Ernie5WeightNames.Attn(index, l[7]))),
            CompressorWkv = Read(l[8]),
            CompressorWgate = Read(l[9]),
            CompressorApe = Host(l[10]),
            CompressorNorm = Host(l[11]),
            OGate = _cfg.UseGatedAttn ? Read(Ernie5WeightNames.AttnGate) : null,
            HeadMix = _cfg.UseVhaAttention ? FoldVha(index, source, _cfg.NumAttentionHeads) : null,
        };
    }

    private DsaAttentionWeights DsaWeights(int index, IErnie5WeightSource source)
    {
        Tensor Read(string leaf) => Own(source.Read(Ernie5WeightNames.Attn(index, leaf)));
        Tensor ReadVector(string leaf) => Own(ReadF32(source, Ernie5WeightNames.Attn(index, leaf)));
        string[] l = Ernie5WeightNames.DsaLeaves;
        return new DsaAttentionWeights
        {
            QA = Read(l[0]),
            QANorm = ReadVector(l[1]),
            QB = Read(l[2]),
            KvA = Read(l[3]),
            KvANorm = ReadVector(l[4]),
            KvB = Read(l[5]),
            WoB = Read(l[6]),
            Sink = ReadVector(l[7]),
            Indexer = new DsaIndexerWeights
            {
                WqB = Read(l[8]),
                Wk = Read(l[9]),
                KeyNormWeight = ReadVector(l[10]),
                KeyNormBias = ReadVector(l[11]),
                WeightsProj = Read(l[12]),
            },
            OGate = _cfg.UseGatedAttn ? Read(Ernie5WeightNames.AttnGate) : null,
            HeadMix = _cfg.UseVhaAttention ? FoldVha(index, source, _cfg.MlaHeads) : null,
        };
    }

    // m = I + V U^T, rounded to BF16 as the reference's fold in the BF16 weights' dtype is (vLLM fold_vha).
    private Tensor FoldVha(int index, IErnie5WeightSource source, int heads)
    {
        float[] u = HostF32(source, Ernie5WeightNames.Attn(index, Ernie5WeightNames.VhaLeaves[0]));
        float[] v = HostF32(source, Ernie5WeightNames.Attn(index, Ernie5WeightNames.VhaLeaves[1]));
        float[] mix = HeadMix.FoldLowRank(u, v, heads, VhaRank, roundToBf16: true);
        return Own(HostMath.Tensor(mix, heads, heads));
    }

    // The checkpoint stores the mapper input-major, [hc * hidden, mixes]; the module wants [mixes, hc * hidden].
    private static HyperConnection BuildHc(IBackend backend, Ernie5Config cfg, int index, string module, IErnie5WeightSource source)
    {
        int hc = cfg.HcMult, mixes = (2 + hc) * hc, flat = hc * cfg.HiddenSize;
        float[] stored = HostF32(source, Ernie5WeightNames.Hc(cfg, index, module, Ernie5WeightNames.HcLeaves[0]));
        if (stored.Length != (long)mixes * flat)
            throw new InvalidDataException($"Layer {index} {module} mapper holds {stored.Length} values, not {flat} x {mixes}.");
        float[] fn = Transpose(stored, flat, mixes);
        float[] bias = HostF32(source, Ernie5WeightNames.Hc(cfg, index, module, Ernie5WeightNames.HcLeaves[1]));
        float[] scale = new float[3];
        for (int i = 0; i < 3; i++)
        {
            float[] alpha = HostF32(source, Ernie5WeightNames.Hc(cfg, index, module, Ernie5WeightNames.HcLeaves[2 + i]));
            if (alpha.Length != 1) throw new InvalidDataException($"Layer {index} {module} {Ernie5WeightNames.HcLeaves[2 + i]} must hold one value.");
            scale[i] = alpha[0];
        }
        return new HyperConnection(backend, HcConfig(cfg), fn, scale, bias);
    }

    /// <summary>The ERNIE hyper-connection formula for <paramref name="cfg"/>. The shipped config has only <c>hc_eps</c>, so it is both the
    /// Sinkhorn and the pre-gate epsilon, and the post scale is the reference model's 2.0.</summary>
    public static HyperConnectionConfig HcConfig(Ernie5Config cfg) =>
        HyperConnectionConfig.Ernie55(cfg.HcMult, cfg.HiddenSize, cfg.HcSinkhornIters, (float)cfg.HcEps, (float)cfg.HcEps, 2f, (float)cfg.RmsNormEps);

    /// <summary>Row-major transpose of a <c>[rows, cols]</c> matrix.</summary>
    internal static float[] Transpose(float[] m, int rows, int cols)
    {
        float[] t = new float[m.Length];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) t[c * rows + r] = m[r * cols + c];
        return t;
    }

    /// <summary>A weight as an F32 tensor the caller owns.</summary>
    internal static Tensor ReadF32(IErnie5WeightSource source, string name)
    {
        Tensor t = source.Read(name);
        if (t.DType == DType.F32) return t;
        using (t) return t.CastTo(DType.F32);
    }

    /// <summary>A weight copied to a host F32 array.</summary>
    internal static float[] HostF32(IErnie5WeightSource source, string name)
    {
        using Tensor t = ReadF32(source, name);
        return t.AsReadOnlySpan<float>().ToArray();
    }

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _hca?.Dispose();
        _dsa?.Dispose();
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }
}
