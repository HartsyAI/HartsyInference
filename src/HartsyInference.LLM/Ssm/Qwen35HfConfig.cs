namespace HartsyInference.LLM.Ssm;

/// <summary>The <c>text_config</c> of a HuggingFace Qwen3.5 checkpoint, as far as the text trunk needs it.</summary>
public sealed record Qwen35HfConfig
{
    public required int HiddenSize { get; init; }

    public required int NumLayers { get; init; }

    public required int VocabSize { get; init; }

    public required int NumHeads { get; init; }

    public required int NumKvHeads { get; init; }

    public required int HeadDim { get; init; }

    public required int FullAttentionInterval { get; init; }

    public required int LinearConvKernel { get; init; }

    public required int LinearKeyHeadDim { get; init; }

    public required int LinearValueHeadDim { get; init; }

    public required int LinearNumKeyHeads { get; init; }

    public required int LinearNumValueHeads { get; init; }

    public float PartialRotaryFactor { get; init; } = 0.25f;

    public float RopeTheta { get; init; } = 10_000_000f;

    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>Reads a checkpoint's <c>config.json</c> (a conditional-generation wrapper with a nested <c>text_config</c>, or a bare text config).</summary>
    public static Qwen35HfConfig FromJson(string json)
    {
        using System.Text.Json.JsonDocument doc = System.Text.Json.JsonDocument.Parse(json);
        System.Text.Json.JsonElement root = doc.RootElement;
        System.Text.Json.JsonElement t = root.TryGetProperty("text_config", out System.Text.Json.JsonElement nested) ? nested : root;
        float theta = 10_000_000f, partial = 0.25f;
        if (t.TryGetProperty("rope_parameters", out System.Text.Json.JsonElement rope))
        {
            if (rope.TryGetProperty("rope_theta", out System.Text.Json.JsonElement th)) theta = (float)th.GetDouble();
            if (rope.TryGetProperty("partial_rotary_factor", out System.Text.Json.JsonElement pf)) partial = (float)pf.GetDouble();
        }
        else if (t.TryGetProperty("partial_rotary_factor", out System.Text.Json.JsonElement pf2))
        {
            partial = (float)pf2.GetDouble();
        }
        return new Qwen35HfConfig
        {
            HiddenSize = t.GetProperty("hidden_size").GetInt32(), NumLayers = t.GetProperty("num_hidden_layers").GetInt32(),
            VocabSize = t.GetProperty("vocab_size").GetInt32(), NumHeads = t.GetProperty("num_attention_heads").GetInt32(),
            NumKvHeads = t.GetProperty("num_key_value_heads").GetInt32(), HeadDim = t.GetProperty("head_dim").GetInt32(),
            FullAttentionInterval = t.GetProperty("full_attention_interval").GetInt32(),
            LinearConvKernel = t.GetProperty("linear_conv_kernel_dim").GetInt32(),
            LinearKeyHeadDim = t.GetProperty("linear_key_head_dim").GetInt32(),
            LinearValueHeadDim = t.GetProperty("linear_value_head_dim").GetInt32(),
            LinearNumKeyHeads = t.GetProperty("linear_num_key_heads").GetInt32(),
            LinearNumValueHeads = t.GetProperty("linear_num_value_heads").GetInt32(),
            PartialRotaryFactor = partial, RopeTheta = theta,
            RmsNormEps = t.TryGetProperty("rms_norm_eps", out System.Text.Json.JsonElement eps) ? (float)eps.GetDouble() : 1e-6f,
        };
    }
}
