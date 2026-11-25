using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Transformer.Mhc;

namespace HartsyInference.LLM.Maxwell;

/// <summary>One Maxwell decoder layer: block-sparse attention and a dense or MoE FFN, each wrapped in its own hyper-connection over the
/// <c>[tokens, hc, hidden]</c> residual stream.</summary>
/// <remarks>The hyper-connection is the ERNIE-5.5 formula (mixes with the epsilon outside the square root, head with it inside, post gate times two);
/// <c>mapping_proj</c> is stored <c>[mixes, hc * hidden]</c>, the layout the engine's module takes, so it is not transposed.</remarks>
internal sealed class MaxwellLayer : IDisposable
{
    private readonly IBackend _backend;
    private readonly MaxwellConfig _cfg;
    private readonly List<Tensor> _owned = [];
    private readonly HyperConnection _attnHc;
    private readonly HyperConnection _ffnHc;
    private readonly Tensor _attnNorm;
    private readonly Tensor _ffnNorm;
    private readonly MaxwellAttention _attention;
    private readonly MoeFeedForward? _moe;
    private readonly Tensor? _denseGate;
    private readonly Tensor? _denseUp;
    private readonly Tensor? _denseDown;
    private int _disposed;

    /// <summary>Reads layer <paramref name="index"/>'s weights from <paramref name="source"/>; a failure disposes what it read.</summary>
    public MaxwellLayer(IBackend backend, MaxwellConfig cfg, int index, IModelWeightSource source, MaxwellRope rope)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cfg);
        ArgumentNullException.ThrowIfNull(source);
        _backend = backend;
        _cfg = cfg;
        Index = index;
        string prefix = MaxwellWeightNames.Layer(index);
        try
        {
            _attnHc = BuildHc(backend, cfg, prefix, MaxwellWeightNames.AttnHc, source);
            _ffnHc = BuildHc(backend, cfg, prefix, MaxwellWeightNames.FfnHc, source);
            _attnNorm = Own(WeightReads.ReadF32(source, MaxwellWeightNames.Key(prefix, MaxwellWeightNames.AttnNorm)));
            _ffnNorm = Own(WeightReads.ReadF32(source, MaxwellWeightNames.Key(prefix, MaxwellWeightNames.FfnNorm)));
            _attention = new MaxwellAttention(backend, cfg, prefix, source, rope);
            string[] ffn = MaxwellWeightNames.FfnLeaves;
            if (cfg.IsDenseLayer(index))
            {
                _denseGate = Own(source.Read(MaxwellWeightNames.DenseFfn(prefix, ffn[0])));
                _denseUp = Own(source.Read(MaxwellWeightNames.DenseFfn(prefix, ffn[1])));
                _denseDown = Own(source.Read(MaxwellWeightNames.DenseFfn(prefix, ffn[2])));
            }
            else
            {
                Dictionary<string, Tensor> moe = new(StringComparer.Ordinal);
                foreach ((string checkpoint, string engine) in MoeKeys(cfg, prefix)) moe[engine] = Own(source.Read(checkpoint));
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

    /// <summary>Layer index.</summary>
    public int Index { get; }

    /// <summary>The MoE block, or null for a dense-FFN layer.</summary>
    internal MoeFeedForward? Moe => _moe;

    /// <summary>The MoE block of every layer past the dense ones.</summary>
    public static MoeConfig MoeSettings(MaxwellConfig cfg) => new()
    {
        NumExperts = cfg.NumExperts,
        NumExpertsPerTok = cfg.NumExpertsPerTok,
        MoeIntermediateSize = cfg.MoeIntermediateSize,
        NormTopKProb = true,
        Scoring = MoeScoring.Sigmoid,
        SharedExpertIntermediateSize = cfg.SharedExpertIntermediateSize,
        Activation = ActivationKind.Silu,
        FirstDenseLayers = cfg.DenseLayers.Count,
    };

    /// <summary>The hyper-connection formula for <paramref name="cfg"/>.</summary>
    public static HyperConnectionConfig HcConfig(MaxwellConfig cfg) =>
        HyperConnectionConfig.Ernie55(cfg.HcMult, cfg.HiddenSize, cfg.HcSinkhornIters, (float)MaxwellConfig.HcEpsilon,
            (float)MaxwellConfig.HcEpsilon, 2f, (float)cfg.RmsNormEps);

    /// <summary>Runs the layer on <paramref name="stream"/> (<c>[tokens, hc, hidden]</c>) in place.</summary>
    public void Forward(float[] stream, int tokens, int startPos, MaxwellSequenceState state)
    {
        int d = _cfg.HiddenSize;
        float[] x = new float[tokens * d], sub = new float[tokens * d];
        float eps = (float)_cfg.RmsNormEps;
        HyperConnectionSublayer.Run(_backend, _attnHc, _attnNorm, d, eps, stream, tokens, x, sub,
            (input, output) => _attention.Forward(input, tokens, startPos, state, Index, output));
        HyperConnectionSublayer.Run(_backend, _ffnHc, _ffnNorm, d, eps, stream, tokens, x, sub, (input, output) => Ffn(input, tokens, output));
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
        using Tensor output = new(new TensorShape(1, tokens, d), DType.F32);
        DenseGluFfn.Apply(_backend, input, _denseGate!, _denseUp!, _denseDown!, output, _cfg.IntermediateSize, ActivationKind.Silu, 0f, 0f);
        output.AsReadOnlySpan<float>().CopyTo(y);
    }

    // The checkpoint's own names mapped to the names MoeFeedForward reads (the load-time remap Ernie and MiniMax use too).
    private static IEnumerable<(string Checkpoint, string Engine)> MoeKeys(MaxwellConfig cfg, string prefix)
    {
        string mlp = $"{prefix}.mlp";
        (string Maxwell, string Engine)[] projs = [("w1.weight", "gate_proj.weight"), ("w3.weight", "up_proj.weight"), ("w2.weight", "down_proj.weight")];
        yield return (MaxwellWeightNames.Key(prefix, MaxwellWeightNames.Gate), $"{mlp}.gate.weight");
        yield return (MaxwellWeightNames.Key(prefix, MaxwellWeightNames.ExpertBias), $"{mlp}.gate.e_score_correction_bias");
        foreach ((string leaf, string engine) in projs) yield return (MaxwellWeightNames.SharedExpert(prefix, leaf), $"{mlp}.shared_expert.{engine}");
        for (int e = 0; e < cfg.NumExperts; e++)
            foreach ((string leaf, string engine) in projs) yield return (MaxwellWeightNames.Expert(prefix, e, leaf), $"{mlp}.experts.{e}.{engine}");
    }

    private static HyperConnection BuildHc(IBackend backend, MaxwellConfig cfg, string prefix, string module, IModelWeightSource source)
    {
        int hc = cfg.HcMult, mixes = (2 + hc) * hc, flat = hc * cfg.HiddenSize;
        string[] leaves = MaxwellWeightNames.HcLeaves;
        float[] fn = WeightReads.HostF32(source, MaxwellWeightNames.Hc(prefix, module, leaves[0]));
        if (fn.Length != (long)mixes * flat) throw new InvalidDataException($"{prefix}.{module} mapper holds {fn.Length} values, not {mixes} x {flat}.");
        float[] bias = WeightReads.HostF32(source, MaxwellWeightNames.Hc(prefix, module, leaves[1]));
        float[] scale = new float[3];
        for (int i = 0; i < 3; i++)
        {
            float[] alpha = WeightReads.HostF32(source, MaxwellWeightNames.Hc(prefix, module, leaves[2 + i]));
            if (alpha.Length != 1) throw new InvalidDataException($"{prefix}.{module}.{leaves[2 + i]} must hold one value.");
            scale[i] = alpha[0];
        }
        return new HyperConnection(backend, HcConfig(cfg), fn, scale, bias);
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
