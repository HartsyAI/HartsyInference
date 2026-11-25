using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ernie5;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.LLM.Transformer.Kda;
using HartsyInference.LLM.Transformer.Mhc;

namespace HartsyInference.LLM.Glm5Next;

/// <summary>One glm5_next decoder layer over a <c>[tokens, hc, hidden]</c> residual stream (transformers <c>Glm5NextTextDecoderLayer</c> at 52bac5d0):
/// attention hyper-connection collapse, <c>input_layernorm</c>, KDA or NoPE compressed-pool DSA, expand; feed-forward hyper-connection collapse,
/// <c>post_attention_layernorm</c>, clamped dense or MoE MLP, expand. Host F32 activations over <see cref="IBackend"/> ops.</summary>
public sealed class Glm5NextLayer : IDisposable
{
    private const float L2NormEps = 1e-6f;

    private readonly IBackend _backend;
    private readonly Glm5NextConfig _c;
    private readonly List<Tensor> _owned = [];
    private readonly HyperConnection _attnHc, _ffnHc;
    private readonly Tensor _attnNorm, _ffnNorm;
    private readonly KdaWeights? _kda;
    private readonly DsaAttention? _dsa;
    private readonly MoeFeedForward? _moe;
    private readonly Tensor? _denseGate, _denseUp, _denseDown;

    /// <summary>Builds layer <paramref name="index"/>, reading its tensors from <paramref name="source"/> (the checkpoint's own key names).</summary>
    /// <exception cref="InvalidDataException">A tensor is missing or does not have the shape the config implies.</exception>
    public Glm5NextLayer(IBackend backend, Glm5NextConfig config, int index, IErnie5WeightSource source)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(source);
        _backend = backend;
        _c = config;
        Index = index;
        string p = Glm5NextWeightNames.Layer(index);
        try
        {
            _attnHc = Hc(source, p + "hc_attn_");
            _ffnHc = Hc(source, p + "hc_ffn_");
            _attnNorm = Own(F32(source, p + "input_layernorm.weight", config.Hidden));
            _ffnNorm = Own(F32(source, p + "post_attention_layernorm.weight", config.Hidden));
            if (config.LayerKinds[index] == Glm5NextAttentionKind.Kda) _kda = KdaWeightsOf(source, p + "self_attn.");
            else _dsa = DsaOf(source, p + "self_attn.");
            if (config.SparseMlp[index])
            {
                Dictionary<string, Tensor> moe = new(StringComparer.Ordinal);
                moe[$"{p}mlp.gate.weight"] = Own(source.Read($"{p}mlp.gate.weight"));
                moe[$"{p}mlp.gate.e_score_correction_bias"] = Own(source.Read($"{p}mlp.gate.e_score_correction_bias"));
                foreach (string proj in Glm5NextWeightNames.ExpertProjections)
                    moe[$"{p}mlp.shared_expert.{proj}.weight"] = Own(source.Read($"{p}mlp.shared_experts.{proj}.weight"));
                for (int e = 0; e < config.Experts; e++)
                    foreach (string proj in Glm5NextWeightNames.ExpertProjections)
                        moe[$"{p}mlp.experts.{e}.{proj}.weight"] = Own(source.Read(Glm5NextWeightNames.Expert(index, e, proj)));
                _moe = new MoeFeedForward(MoeSettings(config), config.Hidden, lowVram: false);
                _moe.LoadWeights(moe, p.TrimEnd('.'));
            }
            else
            {
                _denseGate = Own(source.Read(p + "mlp.gate_proj.weight"));
                _denseUp = Own(source.Read(p + "mlp.up_proj.weight"));
                _denseDown = Own(source.Read(p + "mlp.down_proj.weight"));
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Layer index.</summary>
    public int Index { get; }

    /// <summary>KDA shape for <paramref name="c"/>: the forget and output gates' low rank equals the head width.</summary>
    public static KdaSettings KdaSettingsFor(Glm5NextConfig c) => new(c.Hidden, c.Heads, c.KdaHeadDim, c.ConvKernel, c.KdaHeadDim, c.GateLowerBound);

    /// <summary>NoPE MLA shape for <paramref name="c"/>: no rotated slice, no window, no sink; selection comes from <see cref="Glm5NextKpoolIndexer"/>.</summary>
    public static DsaAttentionSettings MlaSettingsFor(Glm5NextConfig c) => new()
    {
        Dim = c.Hidden, Heads = c.Heads, QLoraRank = c.QLoraRank, KvLoraRank = c.KvLoraRank, QkNopeDim = c.QkDim, QkRopeDim = 0, VDim = c.VDim, Window = 0,
        NormEps = c.RmsNormEps,
    };

    /// <summary>The MoE shape: sigmoid scores with a selection-only bias, normalized then scaled weights, and the clamped SwiGLU on routed and shared experts.</summary>
    public static MoeConfig MoeSettings(Glm5NextConfig c) => new()
    {
        NumExperts = c.Experts,
        NumExpertsPerTok = c.ExpertsPerToken,
        MoeIntermediateSize = c.MoeIntermediate,
        NormTopKProb = c.NormTopKProb,
        Scoring = MoeScoring.Sigmoid,
        SharedExpertIntermediateSize = c.SharedExperts * c.MoeIntermediate,
        Activation = ActivationKind.SwigluClamped,
        SwigluLimit = c.SwigluLimit,
        ClampSharedExpert = true,
        RoutedScalingFactor = c.RoutedScalingFactor,
    };

    /// <summary>The hyper-connection formula of <paramref name="c"/>.</summary>
    public static HyperConnectionConfig HcConfigFor(Glm5NextConfig c) =>
        HyperConnectionConfig.GlmFlash(c.HcMult, c.Hidden, c.HcSinkhornIters, c.HcEps, c.RmsNormEps);

    /// <summary>Runs the layer over <paramref name="tokens"/> rows at positions <c>startPos ..</c>, updating <paramref name="stream"/> in place.</summary>
    /// <param name="stream"><c>[tokens, HcMult, Hidden]</c>.</param>
    public void Forward(float[] stream, int tokens, int startPos, Glm5NextState state)
    {
        int d = _c.Hidden;
        float[] x = new float[tokens * d], sub = new float[tokens * d];
        HyperConnectionSublayer.Run(_backend, _attnHc, _attnNorm, d, _c.RmsNormEps, stream, tokens, x, sub, (input, output) => Attention(input, tokens, startPos, state, output));
        HyperConnectionSublayer.Run(_backend, _ffnHc, _ffnNorm, d, _c.RmsNormEps, stream, tokens, x, sub, (input, output) => Mlp(input, tokens, output));
    }

    private void Attention(float[] x, int tokens, int startPos, Glm5NextState state, float[] y)
    {
        if (_dsa is not null)
        {
            _dsa.Forward(x, tokens, startPos, state.Dsa(Index), y);
            return;
        }
        using Tensor input = HostMath.Tensor(x, tokens, _c.Hidden);
        using Tensor output = new(new TensorShape(tokens, _c.Hidden), DType.F32);
        KdaLayer.Forward(_backend, KdaSettingsFor(_c), _kda!, input, output, state.Kda(Index), _c.RmsNormEps, L2NormEps);
        output.AsReadOnlySpan<float>().CopyTo(y);
    }

    private void Mlp(float[] x, int tokens, float[] y)
    {
        int d = _c.Hidden;
        using Tensor input = HostMath.Tensor(x, 1, tokens, d);
        if (_moe is not null)
        {
            using Tensor routed = _moe.Forward(_backend, input, tokens);
            routed.AsReadOnlySpan<float>().CopyTo(y);
            return;
        }
        using Tensor output = new(new TensorShape(1, tokens, d), DType.F32);
        DenseGluFfn.Apply(_backend, input, _denseGate!, _denseUp!, _denseDown!, output, _c.DenseIntermediate, ActivationKind.Silu, 0f, 0f, _c.SwigluLimit);
        output.AsReadOnlySpan<float>().CopyTo(y);
    }

    private HyperConnection Hc(IErnie5WeightSource source, string prefix)
    {
        int hc = _c.HcMult, mixes = (2 + hc) * hc, flat = hc * _c.Hidden;
        float[] fn = Host(source, prefix + "fn", mixes * flat), bias = Host(source, prefix + "base", mixes), scale = Host(source, prefix + "scale", 3);
        return new HyperConnection(_backend, HcConfigFor(_c), fn, scale, bias);
    }

    // The checkpoint keeps q, k and v separate; the shared KDA layer takes one fused projection and one fused conv, in q, k, v order.
    private KdaWeights KdaWeightsOf(IErnie5WeightSource source, string p)
    {
        int h = _c.Hidden, key = _c.KdaKeyDim, r = _c.KdaHeadDim, k = _c.ConvKernel;
        float[] qkv = [.. Host(source, p + "q_proj.weight", key * h), .. Host(source, p + "k_proj.weight", key * h), .. Host(source, p + "v_proj.weight", key * h)];
        float[] taps = [.. Host(source, p + "q_conv1d.weight", key * k), .. Host(source, p + "k_conv1d.weight", key * k), .. Host(source, p + "v_conv1d.weight", key * k)];
        float[] aLog = Host(source, p + "A_log", _c.Heads), dtBias = Host(source, p + "dt_bias", key);
        float[] perChannel = new float[key];
        for (int ch = 0; ch < key; ch++) perChannel[ch] = aLog[ch / _c.KdaHeadDim];
        (float[] up, float[] bias) = KdaLayer.FoldForgetGate(Host(source, p + "f_b_proj.weight", key * r), perChannel, dtBias, key, r);
        return new KdaWeights(
            QkvProj: Own(HostMath.Tensor(qkv, 3 * key, h)),
            ConvTaps: Own(HostMath.Tensor(taps, 3 * key, k)),
            ForgetDown: Own(HostMath.Tensor(Host(source, p + "f_a_proj.weight", r * h), r, h)),
            ForgetUp: Own(HostMath.Tensor(up, key, r)),
            ForgetBias: Own(HostMath.Tensor(bias, key)),
            BetaProj: Own(HostMath.Tensor(Host(source, p + "b_proj.weight", _c.Heads * h), _c.Heads, h)),
            OutGateProj: null,
            OutNorm: Own(F32(source, p + "o_norm.weight", _c.KdaHeadDim)),
            OutProj: Own(source.Read(p + "o_proj.weight")),
            OutGateDown: Own(HostMath.Tensor(Host(source, p + "g_a_proj.weight", r * h), r, h)),
            OutGateUp: Own(HostMath.Tensor(Host(source, p + "g_b_proj.weight", key * r), key, r)));
    }

    private DsaAttention DsaOf(IErnie5WeightSource source, string p)
    {
        DsaAttentionWeights w = new()
        {
            QA = Own(source.Read(p + "q_a_proj.weight")),
            KvA = Own(source.Read(p + "kv_a_proj_with_mqa.weight")),
            QANorm = Own(F32(source, p + "q_a_layernorm.weight", _c.QLoraRank)),
            KvANorm = Own(F32(source, p + "kv_a_layernorm.weight", _c.KvLoraRank)),
            QB = Own(source.Read(p + "q_b_proj.weight")),
            KvB = Own(source.Read(p + "kv_b_proj.weight")),
            WoB = Own(source.Read(p + "o_proj.weight")),
        };
        string ix = p + "indexer.";
        Glm5NextKpoolIndexer indexer = new(_backend, _c, Own(source.Read(ix + "wq_b.weight")), Own(source.Read(ix + "wk.weight")),
            Own(F32(source, ix + "k_norm.weight", _c.IndexHeadDim)), Own(F32(source, ix + "k_norm.bias", _c.IndexHeadDim)),
            Own(source.Read(ix + "weights_proj.weight")), Own(source.Read(ix + "index_kpool_compress_gate")),
            Own(HostMath.Tensor(Host(source, ix + "index_kpool_compress_ape", _c.IndexPool * _c.IndexHeadDim), _c.IndexPool, _c.IndexHeadDim)));
        return new DsaAttention(_backend, MlaSettingsFor(_c), w, rope: null, indexer);
    }

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    /// <summary>Reads a vector as an F32 host tensor of <paramref name="elements"/> values, whatever dtype the checkpoint stores it in.</summary>
    internal static Tensor F32(IErnie5WeightSource source, string name, long elements)
    {
        float[] v = Host(source, name, elements);
        return HostMath.Tensor(v, v.Length);
    }

    private static float[] Host(IErnie5WeightSource source, string name, long elements)
    {
        Tensor raw = source.Read(name);
        Tensor f32 = ProjectionOps.EnsureF32(raw);
        try
        {
            if (f32.ElementCount != elements) throw new InvalidDataException($"'{name}' holds {f32.ElementCount} values; the config implies {elements}.");
            return f32.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            if (!ReferenceEquals(f32, raw)) f32.Dispose();
            raw.Dispose();
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        _dsa?.Dispose();
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }
}
