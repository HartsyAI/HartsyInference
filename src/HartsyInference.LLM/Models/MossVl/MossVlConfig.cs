using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Models.MossVl;

/// <summary>The vision tower of a MOSS-VL checkpoint (<c>vision_config</c>): a Qwen3-VL-style ViT whose merger reads the last block and the deepstack taps.</summary>
public sealed record MossVlVisionConfig
{
    public required int Depth { get; init; }

    public required int HiddenSize { get; init; }

    public required int IntermediateSize { get; init; }

    public required int NumHeads { get; init; }

    public required int InChannels { get; init; }

    public required int PatchSize { get; init; }

    public required int SpatialMergeSize { get; init; }

    public required int TemporalPatchSize { get; init; }

    public required int NumPositionEmbeddings { get; init; }

    public required int OutHiddenSize { get; init; }

    /// <summary>Block indices whose outputs the merger concatenates with the last block (deepstack).</summary>
    public required IReadOnlyList<int> DeepstackIndexes { get; init; }

    public required string HiddenAct { get; init; }
}

/// <summary>A HuggingFace MOSS-VL config (<c>moss_vl</c>): a 48-layer text decoder with gated cross-attention layers that read the vision tower's features, and the ViT that produces them.</summary>
public sealed record MossVlConfig
{
    public const string ModelType = "moss_vl";

    /// <summary>The text decoder. <see cref="TransformerConfig.CrossAttnLayers"/> holds the gated cross-attention layer indices.</summary>
    public required TransformerConfig Text { get; init; }

    public required MossVlVisionConfig Vision { get; init; }

    /// <summary>The per-axis section widths of the interleaved M-RoPE over the rotary pairs (<c>[24, 20, 20]</c>: temporal, height, width).</summary>
    public required IReadOnlyList<int> MropeSection { get; init; }

    public required bool MropeInterleaved { get; init; }

    public required int ImageTokenId { get; init; }

    public required int VideoTokenId { get; init; }

    public required int VisionStartTokenId { get; init; }

    public required int VisionEndTokenId { get; init; }

    /// <summary>Reads a parsed MOSS-VL <c>config.json</c>. Rejects any other <c>model_type</c> by name.</summary>
    /// <exception cref="HartsyInferenceException">The model type is not <c>moss_vl</c>, or a required field is missing.</exception>
    public static MossVlConfig FromJson(JsonElement root)
    {
        HfConfigFields top = new(root, "config", "MOSS-VL");
        string? modelType = top.OptionalString("model_type");
        if (!string.Equals(modelType, ModelType, StringComparison.Ordinal))
            throw new HartsyInferenceException($"MossVlConfig: config.model_type is '{modelType}', not '{ModelType}'.");

        HfConfigFields text = top.Object("text_config");
        HfConfigFields vision = top.Object("vision_config");
        HfConfigFields rope = text.Object("rope_scaling");

        int hiddenSize = text.Int("hidden_size");
        int numHeads = text.Int("num_attention_heads");
        int headDim = text.Int("head_dim", hiddenSize / numHeads);
        int[] crossLayers = text.Ints("cross_attention_layers");

        TransformerConfig textConfig = new()
        {
            HiddenSize = hiddenSize,
            NumLayers = text.Int("num_hidden_layers"),
            NumHeads = numHeads,
            NumKvHeads = text.Int("num_key_value_heads", numHeads),
            HeadDim = headDim,
            IntermediateSize = text.Int("intermediate_size"),
            VocabSize = text.Int("vocab_size"),
            MaxPositionEmbeddings = text.Int("max_position_embeddings", 32_768),
            RopeTheta = (float)text.Double("rope_theta", 1_000_000d),
            RmsNormEps = (float)text.Double("rms_norm_eps", 1e-6d),
            AttentionBias = text.Bool("attention_bias", false),
            TieWordEmbeddings = top.Bool("tie_word_embeddings", false),
            // Per-head Q/K RMSNorm is part of the architecture (the modeling code registers q_norm/k_norm on every layer).
            QkNorm = true,
            CrossAttnLayers = new HashSet<int>(crossLayers),
        };

        MossVlVisionConfig visionConfig = new()
        {
            Depth = vision.Int("depth"),
            HiddenSize = vision.Int("hidden_size"),
            IntermediateSize = vision.Int("intermediate_size"),
            NumHeads = vision.Int("num_heads"),
            InChannels = vision.Int("in_channels", 3),
            PatchSize = vision.Int("patch_size"),
            SpatialMergeSize = vision.Int("spatial_merge_size"),
            TemporalPatchSize = vision.Int("temporal_patch_size", 1),
            NumPositionEmbeddings = vision.Int("num_position_embeddings"),
            OutHiddenSize = vision.Int("out_hidden_size"),
            DeepstackIndexes = vision.Ints("deepstack_visual_indexes"),
            HiddenAct = vision.OptionalString("hidden_act") ?? "gelu_pytorch_tanh",
        };

        return new MossVlConfig
        {
            Text = textConfig,
            Vision = visionConfig,
            MropeSection = rope.Ints("mrope_section"),
            MropeInterleaved = rope.Bool("mrope_interleaved", false),
            ImageTokenId = top.Int("image_token_id"),
            VideoTokenId = top.Int("video_token_id"),
            VisionStartTokenId = top.Int("vision_start_token_id"),
            VisionEndTokenId = top.Int("vision_end_token_id"),
        };
    }
}
