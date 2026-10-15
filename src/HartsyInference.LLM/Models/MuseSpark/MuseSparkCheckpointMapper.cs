using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Models.MuseSpark;

/// <summary>Maps a HuggingFace Muse Spark text checkpoint onto the names and conventions <see cref="GenericTransformer"/> loads, the same
/// ones llama.cpp's muse-spark GGUF carries: centered norms get their <c>+1</c>, the router weight absorbs the output multiplier, the
/// MetaP Q scale becomes a synthesized <c>q_norm</c> weight (<c>k_norm</c> is ones), each residual gate resolves to its
/// <c>alpha</c>/<c>beta</c> vectors, and the fused <c>gate_up_proj</c>/<c>down_proj</c> expert stacks split into per-expert weights.</summary>
/// <remarks>Projection weights keep their checkpoint dtype unless <c>widenToF32</c> is set (the F32-only CPU backend); every derived
/// tensor is a new F32 tensor, and the per-expert weights are views that keep their stacked source alive.</remarks>
public static class MuseSparkCheckpointMapper
{
    /// <summary>The reference's floor on <c>alpha²</c> of a residual gate.</summary>
    public const float ResidualGateEps = 1e-3f;

    private static readonly string[] OffsetNorms =
        ["input_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight"];

    /// <summary>Returns the transformer's weight dictionary (prefix <c>model</c>, head <c>lm_head.weight</c>) for <paramref name="hf"/>, a
    /// <c>MuseSparkForConditionalGeneration</c> (<c>model.language_model.*</c>) or bare text-model (<c>model.*</c>) state dict.</summary>
    /// <exception cref="KeyNotFoundException">A tensor the config requires is missing.</exception>
    public static Dictionary<string, Tensor> Map(IReadOnlyDictionary<string, Tensor> hf, MuseSparkHfConfig config, bool widenToF32 = false)
    {
        ArgumentNullException.ThrowIfNull(hf);
        ArgumentNullException.ThrowIfNull(config);
        TransformerConfig cfg = config.Transformer;
        MoeConfig moe = cfg.Moe ?? throw new ArgumentException("A Muse Spark config always carries its MoE block.", nameof(config));
        string src = hf.ContainsKey("model.language_model.embed_tokens.weight") ? "model.language_model" : "model";
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal);
        Tensor Pass(string key) => widenToF32 ? GenericTransformer.EnsureF32(hf[key]) : hf[key];

        w["model.embed_tokens.weight"] = Pass($"{src}.embed_tokens.weight");
        w["model.norm.weight"] = GenericTransformer.EnsureF32(hf[$"{src}.norm.weight"]);
        if (!cfg.TieWordEmbeddings) w["lm_head.weight"] = Pass("lm_head.weight");

        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string from = $"{src}.layers.{i}", to = $"model.layers.{i}";
            foreach (string norm in OffsetNorms) w[$"{to}.{norm}"] = AddOne(hf[$"{from}.{norm}"]);
            foreach (string proj in (string[])["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"])
                w[$"{to}.self_attn.{proj}.weight"] = Pass($"{from}.self_attn.{proj}.weight");
            w[$"{to}.self_attn.q_norm.weight"] = Filled(cfg.HeadDim, config.QkScaleFactor);
            w[$"{to}.self_attn.k_norm.weight"] = Filled(cfg.HeadDim, 1f);
            (w[$"{to}.post_attention_residual_gate.alpha"], w[$"{to}.post_attention_residual_gate.beta"]) =
                ResolveResidualGate(hf[$"{from}.post_attention_residual_gate.gate"], config.ResidualGateTemperature);
            (w[$"{to}.post_feedforward_residual_gate.alpha"], w[$"{to}.post_feedforward_residual_gate.beta"]) =
                ResolveResidualGate(hf[$"{from}.post_feedforward_residual_gate.gate"], config.ResidualGateTemperature);

            if (!cfg.IsMoeLayer(i))
            {
                foreach (string proj in (string[])["gate_proj", "up_proj", "down_proj"])
                    w[$"{to}.mlp.{proj}.weight"] = Pass($"{from}.mlp.{proj}.weight");
                continue;
            }
            w[$"{to}.mlp.gate.weight"] = Scaled(hf[$"{from}.mlp.gate.weight"], config.OutputMultiplier);
            w[$"{to}.mlp.gate.e_score_correction_bias"] = GenericTransformer.EnsureF32(hf[$"{from}.mlp.gate.e_score_correction_bias"]);
            w[$"{to}.mlp.pre_expert_proj.weight"] = Pass($"{from}.mlp.pre_expert_proj.weight");
            w[$"{to}.mlp.post_expert_proj.weight"] = Pass($"{from}.mlp.post_expert_proj.weight");
            w[$"{to}.mlp.pre_expert_norm.weight"] = AddOne(hf[$"{from}.mlp.pre_expert_norm.weight"]);
            w[$"{to}.mlp.post_expert_norm.weight"] = AddOne(hf[$"{from}.mlp.experts.post_expert_norm.weight"]);
            SplitExperts(w, Pass($"{from}.mlp.experts.gate_up_proj"), Pass($"{from}.mlp.experts.down_proj"), to, moe,
                moe.ExpertWidth(cfg.HiddenSize));
        }
        return w;
    }

    /// <summary>Splits the fused <c>[E, 2·I, H]</c> gate/up stack (gate rows first) and the <c>[E, H, I]</c> down stack into per-expert views.</summary>
    private static void SplitExperts(Dictionary<string, Tensor> w, Tensor gateUp, Tensor down, string to, MoeConfig moe, int width)
    {
        int e = moe.NumExperts, inter = moe.MoeIntermediateSize;
        if (gateUp.ElementCount != (long)e * 2 * inter * width || down.ElementCount != (long)e * width * inter)
            throw new InvalidDataException($"{to}: expert stacks {gateUp.Shape} / {down.Shape} do not match {e} experts of {width}→{inter}.");
        Tensor gateUpRows = gateUp.Reshape(new TensorShape((long)e * 2 * inter, width));
        Tensor downRows = down.Reshape(new TensorShape((long)e * width, inter));
        for (int x = 0; x < e; x++)
        {
            w[$"{to}.mlp.experts.{x}.gate_proj.weight"] = gateUpRows.SliceRows((long)x * 2 * inter, inter);
            w[$"{to}.mlp.experts.{x}.up_proj.weight"] = gateUpRows.SliceRows((long)x * 2 * inter + inter, inter);
            w[$"{to}.mlp.experts.{x}.down_proj.weight"] = downRows.SliceRows((long)x * width, width);
        }
    }

    /// <summary><c>beta = σ(g/T)</c>, <c>alpha = sqrt(max(σ(−g/T)·(1 + beta), eps))</c>, evaluated in double and stored F32.</summary>
    private static (Tensor Alpha, Tensor Beta) ResolveResidualGate(Tensor gate, float temperature)
    {
        Tensor g = GenericTransformer.EnsureF32(gate);
        ReadOnlySpan<float> src = g.AsReadOnlySpan<float>();
        Tensor alpha = new(new TensorShape(src.Length), DType.F32);
        Tensor beta = new(new TensorShape(src.Length), DType.F32);
        Span<float> a = alpha.AsSpan<float>(), b = beta.AsSpan<float>();
        for (int i = 0; i < src.Length; i++)
        {
            double x = (double)(src[i] / temperature);
            double bi = 1.0 / (1.0 + Math.Exp(-x));
            double alphaSq = 1.0 / (1.0 + Math.Exp(x)) * (1.0 + bi);
            b[i] = (float)bi;
            a[i] = (float)Math.Sqrt(Math.Max(alphaSq, ResidualGateEps));
        }
        if (!ReferenceEquals(g, gate)) g.Dispose();
        return (alpha, beta);
    }

    private static Tensor AddOne(Tensor t) => Affine(t, 1f, 1f);

    private static Tensor Scaled(Tensor t, float scale) => Affine(t, scale, 0f);

    /// <summary>A new F32 copy of <paramref name="t"/> holding <c>t·scale + shift</c>.</summary>
    private static Tensor Affine(Tensor t, float scale, float shift)
    {
        Tensor f = GenericTransformer.EnsureF32(t);
        Tensor result = new(f.Shape, DType.F32);
        ReadOnlySpan<float> s = f.AsReadOnlySpan<float>();
        Span<float> d = result.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) d[i] = s[i] * scale + shift;
        if (!ReferenceEquals(f, t)) f.Dispose();
        return result;
    }

    private static Tensor Filled(int n, float value)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        t.AsSpan<float>().Fill(value);
        return t;
    }
}
