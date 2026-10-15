using System.Text.Json;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Models.MuseSpark;

/// <summary>A HuggingFace Muse Spark text config (<c>muse_spark</c> with a nested <c>text_config</c>, or a bare <c>muse_spark_text</c>)
/// as the <see cref="TransformerConfig"/> <see cref="GenericTransformer"/> runs, plus the MetaP scalars
/// <see cref="MuseSparkCheckpointMapper"/> folds into the weights.</summary>
/// <remarks>The transformer ties RoPE to the sliding-window layers, as llama.cpp's muse-spark converter does, so a config whose
/// <c>layer_rope_theta</c> disagrees with its <c>layer_types</c>, or uses a per-layer theta other than the global one, is refused
/// rather than run with different positions. Dense MLP layers are accepted only as a leading block.</remarks>
public sealed record MuseSparkHfConfig
{
    /// <summary>The architecture the transformer runs (split-half RoPE, since HF stores Q/K in rotate_half layout).</summary>
    public required TransformerConfig Transformer { get; init; }

    /// <summary>MetaP multiplier on Q after the scaleless QK-norm; the mapper writes it into a synthesized <c>q_norm</c> weight.</summary>
    public required float QkScaleFactor { get; init; }

    /// <summary>MetaP multiplier on the router logits (folded into the router weight) and on the final logits.</summary>
    public required float OutputMultiplier { get; init; }

    /// <summary>Temperature dividing each residual gate parameter before its sigmoid.</summary>
    public required float ResidualGateTemperature { get; init; }

    /// <summary>Reads a checkpoint's <c>config.json</c>.</summary>
    /// <exception cref="NotSupportedException">The config is not Muse Spark, or uses a layer layout the transformer does not run.</exception>
    public static MuseSparkHfConfig FromJson(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;
        string modelType = root.TryGetProperty("model_type", out JsonElement mt) ? mt.GetString() ?? "" : "";
        if (modelType is not ("muse_spark" or "muse_spark_text"))
            throw new NotSupportedException($"config.json model_type is '{modelType}', not muse_spark.");
        JsonElement t = root.TryGetProperty("text_config", out JsonElement nested) ? nested : root;

        int hidden = Int(t, "hidden_size", 8192);
        int layers = Int(t, "num_hidden_layers", 62);
        int heads = Int(t, "num_attention_heads", 128);
        float rmsEps = Float(t, "rms_norm_eps", 1e-5f);
        float postEps = Float(t, "post_norm_eps", 1e-8f);
        float theta = 500_000f;
        if (t.TryGetProperty("rope_parameters", out JsonElement rope) && rope.ValueKind == JsonValueKind.Object)
        {
            string ropeType = rope.TryGetProperty("rope_type", out JsonElement rt) ? rt.GetString() ?? "default" : "default";
            if (ropeType != "default") throw new NotSupportedException($"Muse Spark rope_type '{ropeType}' is not supported.");
            theta = Float(rope, "rope_theta", theta);
        }

        bool[] global = new bool[layers];
        string[]? layerTypes = Strings(t, "layer_types");
        if (layerTypes is not null && layerTypes.Length != layers)
            throw new NotSupportedException($"layer_types has {layerTypes.Length} entries for {layers} layers.");
        for (int i = 0; i < layers; i++)
            global[i] = layerTypes is not null ? layerTypes[i] == "full_attention" : (layers - 1 - i) % 4 == 0;
        if (t.TryGetProperty("layer_rope_theta", out JsonElement thetas) && thetas.ValueKind == JsonValueKind.Array)
        {
            if (thetas.GetArrayLength() < layers)
                throw new NotSupportedException($"layer_rope_theta has {thetas.GetArrayLength()} entries for {layers} layers.");
            for (int i = 0; i < layers; i++)
            {
                double layerTheta = thetas[i].GetDouble();
                bool nope = layerTheta == 0d;
                if (nope != global[i] || (!nope && Math.Abs(layerTheta - theta) > theta * 1e-6))
                    throw new NotSupportedException(
                        $"Layer {i}: RoPE theta {layerTheta} on a {(global[i] ? "full" : "sliding")}-attention layer; "
                        + "Muse Spark runs RoPE (global theta) on exactly the sliding layers.");
            }
        }

        int firstDense = 0;
        string[]? mlpTypes = Strings(t, "mlp_layer_types");
        if (mlpTypes is not null)
        {
            while (firstDense < mlpTypes.Length && mlpTypes[firstDense] == "dense") firstDense++;
            for (int i = firstDense; i < mlpTypes.Length; i++)
            {
                if (mlpTypes[i] != "sparse")
                    throw new NotSupportedException(
                        $"Layer {i} is a '{mlpTypes[i]}' MLP after the first sparse layer; only a leading dense block is supported.");
            }
        }

        MoeConfig moe = new()
        {
            NumExperts = Int(t, "num_local_experts", 256),
            NumExpertsPerTok = Int(t, "num_experts_per_tok", 8),
            MoeIntermediateSize = Int(t, "moe_intermediate_size", 4096),
            Scoring = MoeScoring.Sigmoid,
            NormTopKProb = true,
            RoutedScalingFactor = Float(t, "routed_scaling_factor", 1f),
            FirstDenseLayers = firstDense,
            LatentSize = Int(t, "moe_hidden_size", 4096),
            LatentNormEps = rmsEps,
            PostExpertNorm = true,
            PostExpertNormEps = postEps,
        };
        float outputMultiplier = Float(t, "output_multiplier", 0.1767766952966369f);
        TransformerConfig transformer = new()
        {
            HiddenSize = hidden,
            NumLayers = layers,
            NumHeads = heads,
            NumKvHeads = Int(t, "num_key_value_heads", 16),
            HeadDim = Int(t, "head_dim", hidden / heads),
            IntermediateSize = Int(t, "intermediate_size", 16384),
            VocabSize = Int(t, "vocab_size", 202_048),
            MaxPositionEmbeddings = Int(t, "max_position_embeddings", 1_048_576),
            RopeTheta = theta,
            RmsNormEps = rmsEps,
            AttentionBias = Bool(t, "attention_bias", false),
            QkNorm = true,
            TieWordEmbeddings = Bool(t, "tie_word_embeddings", false),
            Rope = RopeStyle.SplitHalf,
            Activation = ActivationKind.Silu,
            SandwichNorm = true,
            NoRopeOnGlobalLayers = true,
            GlobalLayers = global,
            SlidingWindow = Int(t, "sliding_window", 2048),
            FinalLogitSoftcap = Float(t, "final_logit_softcapping", 20f),
            LogitScale = 1f / outputMultiplier,
            Architecture = "muse-spark",
            Moe = moe,
            EmbeddingRmsNorm = true,
            AttnHeadOutputNorm = true,
            AttnOutputGate = true,
            PostAttnNormWeightless = true,
            PostFfnNormScalesInput = true,
            PostNormEps = postEps,
            ResidualGates = true,
        };
        return new MuseSparkHfConfig
        {
            Transformer = transformer,
            QkScaleFactor = Float(t, "qk_scale_factor", 5.4730064863875f),
            OutputMultiplier = outputMultiplier,
            ResidualGateTemperature = Float(t, "residual_gate_temperature", 0.3f),
        };
    }

    private static int Int(JsonElement e, string name, int fallback) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;

    private static float Float(JsonElement e, string name, float fallback) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.Number ? (float)v.GetDouble() : fallback;

    private static bool Bool(JsonElement e, string name, bool fallback) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : fallback;

    private static string[]? Strings(JsonElement e, string name)
    {
        if (!e.TryGetProperty(name, out JsonElement v) || v.ValueKind != JsonValueKind.Array) return null;
        string[] values = new string[v.GetArrayLength()];
        for (int i = 0; i < values.Length; i++) values[i] = v[i].GetString() ?? "";
        return values;
    }
}
