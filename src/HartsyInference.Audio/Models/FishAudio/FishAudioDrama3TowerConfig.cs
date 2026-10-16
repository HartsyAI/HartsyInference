using System.Text.Json;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Drama 3's codebook-MTP audio decoder (<c>audio_decoder_config</c>, decoder type <c>codebook_mtp_shared</c>): shared
/// enrichment layers plus one residual layer per codebook after the first, each a code-embedding mix and one gated-attention
/// transformer layer with its own KV cache over time.</summary>
public sealed record FishAudioDrama3TowerConfig
{
    public required int Dim { get; init; }

    public required int IntermediateSize { get; init; }

    public required int NumHeads { get; init; }

    public required int NumKvHeads { get; init; }

    public required int HeadDim { get; init; }

    public required float NormEps { get; init; }

    public required float RopeBase { get; init; }

    public required float RotaryPercent { get; init; }

    /// <summary>Codes per codebook; equals the width of the semantic token band.</summary>
    public required int VocabSize { get; init; }

    public required int NumCodebooks { get; init; }

    /// <summary>Shared enrichment layers run before the residual layers.</summary>
    public required int SharedLayers { get; init; }

    /// <summary>Codebooks after the first whose embeddings the backbone input also carries.</summary>
    public required int BackboneCodebooks { get; init; }

    /// <summary>Width of the interleaved partial rotary embedding.</summary>
    public int RotaryDim => (int)(HeadDim * RotaryPercent);

    /// <summary>Residual layers: one per codebook after the first.</summary>
    public int ResidualLayers => NumCodebooks - 1;

    // Flags fish-speech pins for this decoder: (key, required value as JSON text).
    private static readonly (string Key, string Value)[] FixedFlags =
    [
        ("decoder_type", "\"codebook_mtp_shared\""), ("rotary_interleaved", "true"), ("attention_output_gate", "true"),
        ("layernorm_zero_centered_gamma", "true"), ("use_moe", "false"), ("mtp_use_swa", "false"),
        ("attention_qkv_bias", "false"), ("attention_o_bias", "false"),
    ];

    /// <summary>Reads and validates <c>audio_decoder_config</c>.</summary>
    public static FishAudioDrama3TowerConfig FromJson(JsonElement d)
    {
        foreach ((string key, string value) in FixedFlags)
        {
            if (d.TryGetProperty(key, out JsonElement actual) && actual.GetRawText() != value)
            {
                throw new NotSupportedException($"Drama 3 audio_decoder_config.{key} must be {value}; got {actual.GetRawText()}.");
            }
        }
        if (d.TryGetProperty("dropout", out JsonElement dropout) && dropout.GetDouble() != 0)
        {
            throw new NotSupportedException("Drama 3 audio_decoder_config.dropout must be 0.");
        }
        int dim = d.GetProperty("dim").GetInt32();
        if (d.TryGetProperty("text_dim", out JsonElement textDim) && textDim.GetInt32() != dim)
        {
            throw new NotSupportedException("Drama 3 audio_decoder_config.text_dim must equal dim.");
        }
        int shared = d.GetProperty("shared_enrichment_layers").GetInt32();
        if (!d.TryGetProperty("n_layer", out JsonElement layers) || layers.GetInt32() != shared)
        {
            throw new NotSupportedException("Drama 3 audio_decoder_config.n_layer must equal shared_enrichment_layers.");
        }
        int backbone = d.GetProperty("vq_backbone_codebooks").GetInt32();
        string? placement = d.TryGetProperty("vq_feature_placement", out JsonElement p) ? p.GetString() : null;
        if (placement is not ("backbone_partial" or "mtp_shared_pre") || (placement == "mtp_shared_pre" && backbone != 0))
        {
            throw new NotSupportedException($"Drama 3 vq_feature_placement '{placement}' is not supported.");
        }
        FishAudioDrama3TowerConfig cfg = new()
        {
            Dim = dim,
            IntermediateSize = d.GetProperty("intermediate_size").GetInt32(),
            NumHeads = d.GetProperty("n_head").GetInt32(),
            NumKvHeads = d.GetProperty("n_local_heads").GetInt32(),
            HeadDim = d.GetProperty("head_dim").GetInt32(),
            NormEps = (float)d.GetProperty("norm_eps").GetDouble(),
            RopeBase = (float)d.GetProperty("rope_base").GetDouble(),
            RotaryPercent = (float)d.GetProperty("rotary_percent").GetDouble(),
            VocabSize = d.GetProperty("vocab_size").GetInt32(),
            NumCodebooks = d.GetProperty("num_codebooks").GetInt32(),
            SharedLayers = shared,
            BackboneCodebooks = backbone,
        };
        cfg.Validate();
        return cfg;
    }

    /// <summary>Throws when the geometry is not one the tower can run.</summary>
    public void Validate()
    {
        if (NumHeads % NumKvHeads != 0 || RotaryDim <= 0 || RotaryDim > HeadDim || RotaryDim % 2 != 0)
        {
            throw new InvalidDataException("Drama 3 tower attention geometry is invalid.");
        }
        if (NumCodebooks < 2 || BackboneCodebooks < 0 || BackboneCodebooks >= NumCodebooks)
        {
            throw new InvalidDataException("Drama 3 tower codebook counts are invalid.");
        }
        if (SharedLayers < 1)
        {
            throw new InvalidDataException("Drama 3 needs at least one shared enrichment layer.");
        }
    }
}
