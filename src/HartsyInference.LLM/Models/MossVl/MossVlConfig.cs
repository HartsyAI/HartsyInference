using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;

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

    /// <summary>Builds the config from a MOSS-VL GGUF's metadata (<c>moss_vl.*</c>), for a GGUF that arrives without the checkpoint's
    /// <c>config.json</c>. The special token ids are looked up in <paramref name="tokenizer"/>, which knows the file's vocabulary.</summary>
    /// <exception cref="HartsyInferenceException">The file is not <c>moss_vl</c>, or a field or special token is missing.</exception>
    public static MossVlConfig FromGguf(GgufMetadata metadata, ILlmTokenizer tokenizer)
    {
        const string P = "moss_vl.";
        string? architecture = metadata.GetString("general.architecture");
        if (!string.Equals(architecture, ModelType, StringComparison.Ordinal))
            throw new HartsyInferenceException($"MossVlConfig: the GGUF architecture is '{architecture}', not '{ModelType}'.");
        int Need(string key)
        {
            int value = metadata.GetInteger(P + key, -1);
            return value >= 0 ? value : throw new HartsyInferenceException($"MossVlConfig: the GGUF has no '{P + key}'.");
        }
        int[] NeedArray(string key) =>
            metadata.GetIntegerArray(P + key) ?? throw new HartsyInferenceException($"MossVlConfig: the GGUF has no '{P + key}'.");
        int Token(string literal) =>
            tokenizer.SpecialId(literal) ?? throw new HartsyInferenceException($"MossVlConfig: the GGUF tokenizer has no '{literal}' token.");

        int hidden = Need("embedding_length");
        int heads = Need("attention.head_count");
        TransformerConfig text = new()
        {
            HiddenSize = hidden,
            NumLayers = Need("block_count"),
            NumHeads = heads,
            NumKvHeads = metadata.GetInteger(P + "attention.head_count_kv", heads),
            HeadDim = metadata.GetInteger(P + "attention.key_length", hidden / heads),
            IntermediateSize = Need("feed_forward_length"),
            VocabSize = Need("vocab_size"),
            MaxPositionEmbeddings = metadata.GetInteger(P + "context_length", 32_768),
            RopeTheta = metadata.GetFloat32(P + "rope.freq_base", 1_000_000f),
            RmsNormEps = metadata.GetFloat32(P + "attention.layer_norm_rms_epsilon", 1e-6f),
            AttentionBias = false,
            TieWordEmbeddings = false,
            QkNorm = true,
            CrossAttnLayers = new HashSet<int>(NeedArray("cross_layer_indexes")),
        };
        MossVlVisionConfig vision = new()
        {
            Depth = Need("vision.depth"),
            HiddenSize = Need("vision.embedding_length"),
            IntermediateSize = Need("vision.feed_forward_length"),
            NumHeads = Need("vision.num_heads"),
            InChannels = 3,
            PatchSize = Need("vision.patch_size"),
            SpatialMergeSize = Need("vision.spatial_merge_size"),
            TemporalPatchSize = metadata.GetInteger(P + "vision.temporal_patch_size", 1),
            NumPositionEmbeddings = Need("vision.num_position_embeddings"),
            OutHiddenSize = Need("vision.out_embedding_length"),
            DeepstackIndexes = NeedArray("vision.deepstack_indexes"),
            HiddenAct = "gelu_pytorch_tanh",
        };
        return new MossVlConfig
        {
            Text = text,
            Vision = vision,
            MropeSection = [Need("rope.mrope_section_t"), Need("rope.mrope_section_h"), Need("rope.mrope_section_w")],
            MropeInterleaved = metadata.GetBool(P + "rope.mrope_interleaved", true),
            ImageTokenId = Token("<|image_pad|>"),
            VideoTokenId = Token("<|video_pad|>"),
            VisionStartTokenId = Token("<|vision_start|>"),
            VisionEndTokenId = Token("<|vision_end|>"),
        };
    }

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
