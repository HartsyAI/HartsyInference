using System.Text.Json;
using HartsyInference.LLM.Ssm;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Fish Audio Drama 3 (<c>model_type: fish_omni</c>, speech template <c>reference-bracket-v1</c>), parsed from its
/// <c>config.json</c> with the checks fish-speech's <c>FishOmniConfig</c> applies: a Qwen3.5-MoE hybrid text backbone, the
/// codebook-MTP tower, and the semantic token band the main token is drawn from.</summary>
public sealed record FishAudioDrama3Config
{
    /// <summary>The only speech template Drama 3 ships with.</summary>
    public const string SpeechTemplateVersion = "reference-bracket-v1";

    /// <summary>Most rows a prompt may hold, as the hosted server allows.</summary>
    public const int MaxPromptRows = 8192;

    /// <summary>Frames generated when the request sets no cap and the text runs as one chunk.</summary>
    public const int DefaultMaxNewFrames = 2048;

    /// <summary>Repetition-aware sampling window over the latest main tokens, fixed by the training recipe.</summary>
    public const int RasWindow = 16;

    public required Qwen35HfConfig Backbone { get; init; }

    public required FishAudioDrama3TowerConfig Tower { get; init; }

    /// <summary>Vocab id of <c>&lt;|semantic:0|&gt;</c>.</summary>
    public required int SemanticStartId { get; init; }

    /// <summary>Vocab id of the last semantic token.</summary>
    public required int SemanticEndId { get; init; }

    /// <summary>Vocab id of <c>&lt;|im_end|&gt;</c>, which ends a voice turn.</summary>
    public required int EosId { get; init; }

    public FishAudioDrama3Quantization Quantization { get; init; }

    /// <summary>Inputs per GPTQ scale group (int4 builds).</summary>
    public int GptqGroupSize { get; init; } = 128;

    /// <summary>Edge of an FP8 scale block (FP8 builds).</summary>
    public int Fp8BlockSize { get; init; } = 128;

    // Sampling defaults of the official Fish Audio SDK, which fish-speech uses for Drama 3; the RAS retry is fixed.
    public float Temperature { get; init; } = 0.7f;
    public float TopP { get; init; } = 0.7f;
    public int TopK { get; init; } = 30;
    public float RasTemperature { get; init; } = 1.0f;
    public float RasTopP { get; init; } = 0.95f;
    public int RasTopK { get; init; } = 30;

    public int SampleRate { get; init; } = 44_100;

    public int NumCodebooks => Tower.NumCodebooks;

    public int CodebookSize => Tower.VocabSize;

    /// <summary>True for an id in the semantic band.</summary>
    public bool IsSemantic(int id) => id >= SemanticStartId && id <= SemanticEndId;

    // Text-config flags fish-speech pins: (key, value when absent, required value).
    private static readonly (string Key, bool Default, bool Required)[] FixedTextFlags =
    [
        ("rotary_interleaved", false, false), ("use_qk_norm", true, true), ("attn_output_gate", true, true),
        ("layernorm_zero_centered_gamma", true, true), ("shared_expert_gate", true, true),
    ];

    /// <summary>Reads and validates a Drama 3 <c>config.json</c>.</summary>
    public static FishAudioDrama3Config FromJson(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;
        if (Str(root, "model_type") != "fish_omni")
        {
            throw new NotSupportedException($"model_type '{Str(root, "model_type")}' is not a Drama 3 (fish_omni) checkpoint.");
        }
        if (root.TryGetProperty("audio_encoder_config", out JsonElement encoder) && encoder.ValueKind != JsonValueKind.Null)
        {
            throw new NotSupportedException("Drama 3 audio_encoder_config is not supported.");
        }
        if (Str(root, "speech_template_version") != SpeechTemplateVersion)
        {
            throw new NotSupportedException($"Drama 3 speech_template_version must be {SpeechTemplateVersion}.");
        }
        JsonElement text = root.GetProperty("text_config");
        ValidateText(text);
        (FishAudioDrama3Quantization quantization, int group, int block) = ReadQuantization(root);
        int eos = root.TryGetProperty("eos_token_id", out JsonElement e) && e.ValueKind == JsonValueKind.Number ? e.GetInt32()
            : text.TryGetProperty("eos_token_id", out JsonElement te) && te.ValueKind == JsonValueKind.Number ? te.GetInt32()
            : throw new InvalidDataException("The Drama 3 config has no eos_token_id.");
        FishAudioDrama3Config cfg = new()
        {
            // fish-speech reads an absent tie_word_embeddings as false (and refuses true, checked above).
            Backbone = Qwen35HfConfig.FromTextConfig(text) with { TieWordEmbeddings = false },
            Tower = FishAudioDrama3TowerConfig.FromJson(root.GetProperty("audio_decoder_config")),
            SemanticStartId = root.GetProperty("semantic_start_token_id").GetInt32(),
            SemanticEndId = root.GetProperty("semantic_end_token_id").GetInt32(),
            EosId = eos,
            Quantization = quantization,
            GptqGroupSize = group,
            Fp8BlockSize = block,
        };
        cfg.Validate();
        return cfg;
    }

    /// <summary>Throws when the backbone, tower and token band do not fit together.</summary>
    public void Validate()
    {
        Tower.Validate();
        if (!Backbone.IsMoe) throw new NotSupportedException("The Drama 3 backbone is a mixture of experts; this config has none.");
        if (SemanticStartId < 0 || SemanticStartId > SemanticEndId || SemanticEndId >= Backbone.VocabSize)
        {
            throw new InvalidDataException("The Drama 3 semantic band must lie inside the text vocabulary.");
        }
        if (IsSemantic(EosId)) throw new InvalidDataException("The Drama 3 eos_token_id lies inside the semantic band.");
        if (Tower.Dim != Backbone.HiddenSize) throw new InvalidDataException("The Drama 3 tower and backbone widths differ.");
        if (Tower.VocabSize != SemanticEndId - SemanticStartId + 1)
        {
            throw new InvalidDataException("The Drama 3 tower vocabulary must match the semantic token band.");
        }
    }

    private static void ValidateText(JsonElement text)
    {
        string? type = Str(text, "model_type");
        if (type is not ("fish_omni_text" or "qwen3_5_moe_text"))
        {
            throw new NotSupportedException($"Drama 3 text_config.model_type '{type}' is not a Qwen3.5-MoE text model.");
        }
        if (text.TryGetProperty("mlp_layer_types", out JsonElement mlp) && mlp.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement layer in mlp.EnumerateArray())
            {
                if (layer.GetString() != "moe") throw new NotSupportedException("Every Drama 3 text_config.mlp_layer_types entry must be moe.");
            }
        }
        foreach ((string key, bool fallback, bool required) in FixedTextFlags)
        {
            bool value = text.TryGetProperty(key, out JsonElement v) ? v.GetBoolean() : fallback;
            if (value != required) throw new NotSupportedException($"Drama 3 text_config.{key} must be {required}.");
        }
        if (text.TryGetProperty("tie_word_embeddings", out JsonElement tie) && tie.GetBoolean())
        {
            throw new NotSupportedException("The Drama 3 backbone keeps a separate lm_head; tie_word_embeddings must be false.");
        }
        JsonElement rope = text.TryGetProperty("rope_parameters", out JsonElement r) ? r : text;
        string ropeType = Str(rope, "rope_type") ?? Str(rope, "type") ?? "default";
        if (ropeType != "default") throw new NotSupportedException($"Drama 3 rope_type '{ropeType}' is not supported.");
    }

    private static (FishAudioDrama3Quantization Method, int Group, int Block) ReadQuantization(JsonElement root)
    {
        if (!root.TryGetProperty("quantization_config", out JsonElement q) || q.ValueKind == JsonValueKind.Null)
        {
            return (FishAudioDrama3Quantization.None, 128, 128);
        }
        string? method = Str(q, "quant_method");
        switch (method)
        {
            case "bsgemm-moe":
            {
                int[] block = q.TryGetProperty("weight_block_size", out JsonElement b) && b.ValueKind == JsonValueKind.Array
                    ? [.. b.EnumerateArray().Select(static x => x.GetInt32())] : [];
                if (block.Length != 2 || block[0] != block[1] || block[0] <= 0)
                {
                    throw new NotSupportedException("Drama 3 bsgemm-moe needs a square weight_block_size.");
                }
                return (FishAudioDrama3Quantization.Fp8Block, 128, block[0]);
            }
            case "drama3-int4":
            {
                if (q.TryGetProperty("bits", out JsonElement bits) && bits.GetInt32() != 4)
                {
                    throw new NotSupportedException("Drama 3 int4 must be 4-bit.");
                }
                if (q.TryGetProperty("sym", out JsonElement sym) && !sym.GetBoolean())
                {
                    throw new NotSupportedException("Only symmetric Drama 3 int4 weights are supported.");
                }
                if (q.TryGetProperty("desc_act", out JsonElement act) && act.GetBoolean())
                {
                    throw new NotSupportedException("Drama 3 int4 weights with activation order (desc_act) are not supported.");
                }
                int group = q.TryGetProperty("group_size", out JsonElement g) ? g.GetInt32() : 128;
                return (FishAudioDrama3Quantization.Int4Gptq, group, 128);
            }
            default:
                throw new NotSupportedException($"Drama 3 quant_method '{method}' is not supported; expected bsgemm-moe or drama3-int4.");
        }
    }

    private static string? Str(JsonElement e, string name) =>
        e.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
}
