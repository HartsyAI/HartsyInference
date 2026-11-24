using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Dsa;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>One Hy4 Flash decoder layer (transformers <c>HYV4FlashDecoderLayer</c>): an indexed or sliding gated-MLA attention and a dense or latent-MoE FFN, each between its own
/// identity hyper-connection, with a sandwich norm around every sublayer. The Engram residual, when the layer has one, is added by the model before this runs.</summary>
/// <remarks>Order per sublayer: collapse the stream, RMS norm, sublayer, RMS norm, expand. The MoE block ends in its own <c>combine_output_norm</c> over routed plus shared output; the dense layer ends in
/// <c>post_fc2_norm</c>. Weights are read once and kept (host F32 reference); <see cref="Trace"/> names the stages the HF dump records.</remarks>
public sealed class Hy4FlashLayer : IDisposable
{
    private readonly IBackend _backend;
    private readonly Hy4FlashConfig _cfg;
    private readonly List<Tensor> _owned = [];
    private readonly Hy4FlashHyperConnection _attnHc;
    private readonly Hy4FlashHyperConnection _ffnHc;
    private readonly Tensor _inputNorm;
    private readonly Tensor _attnPostNorm;
    private readonly Tensor _ffnNorm;
    private readonly Tensor _ffnPostNorm;
    private readonly DsaAttention _attention;
    private readonly MoeFeedForward? _moe;
    private readonly Tensor? _denseGate;
    private readonly Tensor? _denseUp;
    private readonly Tensor? _denseDown;
    private int _disposed;

    /// <param name="rope">Rotary rows of this layer's attention (its own base, <c>QkRopeHeadDim / 2</c> wide).</param>
    public Hy4FlashLayer(IBackend backend, Hy4FlashConfig cfg, int index, IHy4FlashWeightSource source, IRopeRowSource rope)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cfg);
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(rope);
        _backend = backend;
        _cfg = cfg;
        Index = index;
        string p = Hy4FlashWeightMap.Layer(index);
        try
        {
            _attnHc = Hc(p + ".hc_attn_layer.hc_pre", source);
            _ffnHc = Hc(p + ".hc_mlp_layer.hc_pre", source);
            _inputNorm = F32(source, p + ".input_layernorm.weight");
            _attnPostNorm = F32(source, p + ".self_attn.post_proj_norm.weight");
            _ffnNorm = F32(source, p + ".post_attention_layernorm.weight");
            if (cfg.SparseFfn[index])
            {
                _ffnPostNorm = F32(source, p + ".mlp.combine_output_norm.weight");
                _moe = BuildMoe(p, source);
            }
            else
            {
                _ffnPostNorm = F32(source, p + ".mlp.post_fc2_norm.weight");
                _denseGate = Own(source.Read(p + ".mlp.gate_proj.weight"));
                _denseUp = Own(source.Read(p + ".mlp.up_proj.weight"));
                _denseDown = Own(source.Read(p + ".mlp.down_proj.weight"));
            }
            _attention = BuildAttention(p, source, rope);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Layer index.</summary>
    public int Index { get; }

    /// <summary>Receives named intermediate arrays (<c>attn_hc_post</c>, <c>attn_hc_in</c>, <c>attn_out</c>, <c>ffn_hc_post</c>, <c>ffn_hc_in</c>, <c>ffn_out</c>) when set.</summary>
    public Action<string, float[]>? Trace { get; set; }

    /// <summary>A fresh cache for this layer: latent rows, and the indexer's token keys on an indexed layer.</summary>
    public DsaAttentionState CreateState(int capacity) => new(capacity, AttentionSettings(_cfg, Index), _cfg.LayerKinds[Index] == Hy4FlashLayerKind.Indexed ? _cfg.IndexHeadDim : 0);

    /// <summary>Runs the layer on <paramref name="stream"/> (<c>[tokens, hc, hidden]</c>) in place.</summary>
    public void Forward(float[] stream, int tokens, int startPos, DsaAttentionState state)
    {
        int h = _cfg.HiddenSize;
        float eps = (float)_cfg.RmsNormEps;

        _attnHc.Gates(stream, tokens, out float[] pre, out float[] post);
        float[] x = _attnHc.Collapse(stream, pre, tokens);
        Trace?.Invoke("attn_hc_post", post.ToArray());
        Trace?.Invoke("attn_hc_in", x.ToArray());
        HostOps.RmsNorm(_backend, x, h, _inputNorm, eps);
        float[] a = new float[tokens * h];
        _attention.Forward(x, tokens, startPos, state, a);
        Trace?.Invoke("attn_out", a.ToArray());
        HostOps.RmsNorm(_backend, a, h, _attnPostNorm, eps);
        _attnHc.Expand(a, stream, post, tokens);

        _ffnHc.Gates(stream, tokens, out pre, out post);
        float[] y = _ffnHc.Collapse(stream, pre, tokens);
        Trace?.Invoke("ffn_hc_post", post.ToArray());
        Trace?.Invoke("ffn_hc_in", y.ToArray());
        HostOps.RmsNorm(_backend, y, h, _ffnNorm, eps);
        float[] f = Ffn(y, tokens);
        _ffnHc.Expand(f, stream, post, tokens);
    }

    /// <summary>Shape of this layer's attention.</summary>
    public static DsaAttentionSettings AttentionSettings(Hy4FlashConfig c, int index) => new()
    {
        Dim = c.HiddenSize,
        Heads = c.HeadsFor(index),
        QLoraRank = c.QLoraRank,
        KvLoraRank = c.KvLoraRank,
        QkNopeDim = c.QkNopeHeadDim,
        QkRopeDim = c.QkRopeHeadDim,
        VDim = c.VHeadDim,
        Window = c.LayerKinds[index] == Hy4FlashLayerKind.Sliding ? c.SlidingWindow : 0,
        NormEps = (float)c.RmsNormEps,
        RopePairing = RotaryPairing.Interleaved,
        GateFromHidden = true,
    };

    /// <summary>The routed-expert block: sigmoid scores with a selection-only bias, latent experts with the clamped SwiGLU, one plain shared expert.</summary>
    public static MoeConfig MoeSettings(Hy4FlashConfig c) => new()
    {
        NumExperts = c.NRoutedExperts,
        NumExpertsPerTok = c.NumExpertsPerTok,
        MoeIntermediateSize = c.MoeIntermediateSize,
        NormTopKProb = c.NormTopkProb,
        Scoring = MoeScoring.Sigmoid,
        SharedExpertIntermediateSize = c.NSharedExperts * c.MoeIntermediateSize,
        Activation = ActivationKind.SwigluClamped,
        SwigluLimit = (float)c.SwigluLimit,
        RoutedScalingFactor = (float)c.RoutedScalingFactor,
        LatentSize = c.MoeHiddenSize,
        LatentNorm = MoeLatentNorm.None,
    };

    private float[] Ffn(float[] x, int tokens)
    {
        int h = _cfg.HiddenSize;
        float[] result = new float[tokens * h];
        using Tensor input = HostMath.Tensor(x, 1, tokens, h);
        if (_moe is not null)
        {
            using Tensor routed = _moe.Forward(_backend, input, tokens);
            routed.AsReadOnlySpan<float>().CopyTo(result);
        }
        else
        {
            using Tensor output = new(new TensorShape(1, tokens, h), DType.F32);
            DenseGluFfn.Apply(_backend, input, _denseGate!, _denseUp!, _denseDown!, output, _cfg.IntermediateSize, ActivationKind.Silu, 0f, 0f);
            output.AsReadOnlySpan<float>().CopyTo(result);
        }
        // Dense layers trace before their post norm, the MoE block after its own (the reference module boundaries).
        if (_moe is null) Trace?.Invoke("ffn_out", result.ToArray());
        HostOps.RmsNorm(_backend, result, h, _ffnPostNorm, (float)_cfg.RmsNormEps);
        if (_moe is not null) Trace?.Invoke("ffn_out", result.ToArray());
        return result;
    }

    private DsaAttention BuildAttention(string p, IHy4FlashWeightSource source, IRopeRowSource rope)
    {
        string a = p + ".self_attn.";
        DsaAttentionSettings settings = AttentionSettings(_cfg, Index);
        DsaAttentionWeights weights = new()
        {
            QA = Own(source.Read(a + "q_a_proj.weight")),
            QANorm = F32(source, a + "q_a_layernorm.weight"),
            QB = Own(source.Read(a + "q_b_proj.weight")),
            KvA = Own(source.Read(a + "kv_a_proj_with_mqa.weight")),
            KvANorm = F32(source, a + "kv_a_layernorm.weight"),
            KvB = Own(source.Read(a + "kv_b_proj.weight")),
            WoB = Own(source.Read(a + "o_proj.weight")),
            OGate = Own(source.Read(a + "linear_gate.weight")),
            Sink = F32(source, a + "learnable_sink_param"),
        };
        IDsaKeySelector selector;
        if (_cfg.LayerKinds[Index] == Hy4FlashLayerKind.Sliding)
        {
            selector = new DsaWindowSelector(_cfg.SlidingWindow);
        }
        else
        {
            string i = a + "compressed_indexer.";
            Hy4FlashIndexerWeights indexer = new()
            {
                QueryProjection = Own(source.Read(i + "query_projection.weight")),
                HeadWeightProjection = Own(source.Read(i + "head_weight_projection.weight")),
                TokenKeyProjection = Own(source.Read(i + "token_key_projection.weight")),
                TokenKeyNorm = F32(source, i + "token_key_norm.weight"),
                KeyValueProjection = F32(source, i + "key_value_projection.weight"),
                KeyGateProjection = F32(source, i + "key_gate_projection.weight"),
                IntraGroupApe = F32(source, i + "intra_group_ape"),
                KeyNorm = F32(source, i + "key_norm.weight"),
            };
            Hy4FlashIndexerSettings indexerSettings = new()
            {
                Heads = _cfg.IndexNHeads, HeadDim = _cfg.IndexHeadDim, RopeDim = _cfg.QkRopeHeadDim, Kpool = _cfg.IndexKpool,
                TopK = _cfg.IndexTopk, NormEps = (float)_cfg.RmsNormEps,
            };
            selector = new Hy4FlashGroupIndexer(_backend, indexerSettings, indexer, rope, _cfg.HiddenSize, _cfg.QLoraRank);
        }
        return new DsaAttention(_backend, settings, weights, rope, selector);
    }

    private MoeFeedForward BuildMoe(string p, IHy4FlashWeightSource source)
    {
        string m = p + ".mlp.";
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal)
        {
            [m + "gate.weight"] = F32(source, m + "gate.weight"),
            [m + "gate.e_score_correction_bias"] = F32(source, m + "gate.e_score_correction_bias"),
            [m + "pre_expert_proj.weight"] = Own(source.Read(m + "routed_expert_down_proj.weight")),
            [m + "post_expert_proj.weight"] = Own(source.Read(m + "routed_expert_up_proj.weight")),
            [m + "shared_expert.gate_proj.weight"] = Own(source.Read(m + "shared_experts.gate_proj.weight")),
            [m + "shared_expert.up_proj.weight"] = Own(source.Read(m + "shared_experts.up_proj.weight")),
            [m + "shared_expert.down_proj.weight"] = Own(source.Read(m + "shared_experts.down_proj.weight")),
        };
        Tensor gateUp = Own(source.Read(m + "experts.gate_up_proj")), down = Own(source.Read(m + "experts.down_proj"));
        StackedExperts.SplitGateUp(w, gateUp, m + "experts.{0}.gate_proj.weight", m + "experts.{0}.up_proj.weight", _cfg.NRoutedExperts,
            _cfg.MoeIntermediateSize, _cfg.MoeHiddenSize);
        StackedExperts.Split(w, down, m + "experts.{0}.down_proj.weight", _cfg.NRoutedExperts, _cfg.MoeHiddenSize, _cfg.MoeIntermediateSize);
        MoeFeedForward moe = new(MoeSettings(_cfg), _cfg.HiddenSize, lowVram: false);
        moe.LoadWeights(w, p);
        return moe;
    }

    private Hy4FlashHyperConnection Hc(string prefix, IHy4FlashWeightSource source)
    {
        using Tensor scale = source.Read(prefix + ".hc_scale"), bias = source.Read(prefix + ".hc_base");
        Tensor fn = F32(source, prefix + ".hc_fn");
        return new Hy4FlashHyperConnection(_backend, _cfg.HcMult, _cfg.HiddenSize, (float)_cfg.HcEps, (float)_cfg.HcMagnitude, (float)_cfg.RmsNormEps,
            fn, bias.AsReadOnlySpan<float>().ToArray(), scale.AsReadOnlySpan<float>().ToArray(), head: false);
    }

    private Tensor F32(IHy4FlashWeightSource source, string name)
    {
        Tensor t = source.Read(name);
        if (t.DType == DType.F32) return Own(t);
        using (t) return Own(ProjectionOps.EnsureF32(t));
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
        _attention?.Dispose();
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }
}
