using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Multimodal;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Kimi.Vision;

/// <summary>What a Kimi K3.1 directory needs to take images: the MoonViT-V2 encoder, the image geometry and the media pad id, read from the checkpoint,
/// or the reason it takes none.</summary>
/// <remarks>Image input is refused by name, with <see cref="Refusal"/>, whenever one of the three is missing or does not agree with the others. The
/// text decoder loads either way.</remarks>
public sealed class KimiVisionInput : IDisposable
{
    /// <summary>The literals the Kimi processor writes around an image; <see cref="MediaPad"/> stands for one vision token.</summary>
    public const string MediaBegin = "<|media_begin|>", MediaContent = "<|media_content|>", MediaPad = "<|media_pad|>", MediaEnd = "<|media_end|>";

    private IVlmImageEncoder? _encoder;

    private KimiVisionInput(IVlmImageEncoder? encoder, KimiImageSpec? image, int? padId, string? refusal)
    {
        _encoder = encoder;
        Image = image;
        PadId = padId;
        Refusal = refusal;
    }

    /// <summary>Whether the directory takes images.</summary>
    public bool Accepts => Refusal is null;

    /// <summary>Why the directory takes no images, or null when it does.</summary>
    public string? Refusal { get; }

    /// <summary>The MoonViT-V2 encoder with its projector, or null when the directory takes no images.</summary>
    public IVlmImageEncoder? Encoder => _encoder;

    /// <summary>The image geometry, or null when the directory takes no images.</summary>
    public KimiImageSpec? Image { get; }

    /// <summary>The id of <see cref="MediaPad"/>, or null when the directory takes no images.</summary>
    public int? PadId { get; }

    /// <summary>The processor's prompt for one image: its size, then the pad that stands for its vision tokens.</summary>
    public static string MediaPrompt(int width, int height) => $"{MediaBegin}image {width}x{height}{MediaContent}{MediaPad}{MediaEnd}";

    /// <summary>Reads the checkpoint in <paramref name="directory"/>: the media pad id from <c>config.json</c> and the tokenizer, the vision tower from
    /// <c>vision_config</c> and its tensors, and the geometry from <c>preprocessor_config.json</c>.</summary>
    /// <param name="textHidden">The decoder's hidden width, which the projector must emit.</param>
    /// <param name="xtmlTemplate">Whether the chat template is the XTML port, the one that renders image placeholders.</param>
    public static KimiVisionInput Load(string directory, ILlmTokenizer tokenizer, int textHidden, bool xtmlTemplate)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        ArgumentNullException.ThrowIfNull(tokenizer);
        try
        {
            return Build(directory, tokenizer, textHidden, xtmlTemplate);
        }
        catch (Exception ex) when (ex is KeyNotFoundException or InvalidDataException or InvalidOperationException or NotSupportedException
            or ArgumentException or FormatException or JsonException or IOException or HartsyInferenceException)
        {
            return Refused(ex.Message);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _encoder?.Dispose();
        _encoder = null;
    }

    private static KimiVisionInput Build(string directory, ILlmTokenizer tokenizer, int textHidden, bool xtmlTemplate)
    {
        if (!xtmlTemplate) return Refused("the chat template is not the XTML template, which renders the image placeholders");
        using JsonDocument config = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "config.json")));
        JsonElement root = config.RootElement;
        if (!root.TryGetProperty("media_placeholder_token_id", out JsonElement padValue) || !padValue.TryGetInt32(out int configPad))
            return Refused("config.json names no media_placeholder_token_id");
        foreach (string literal in new[] { MediaBegin, MediaContent, MediaPad, MediaEnd })
        {
            if (tokenizer.SpecialId(literal) is null) return Refused($"the tokenizer has no '{literal}' token");
        }
        int padId = tokenizer.SpecialId(MediaPad)!.Value;
        if (padId != configPad) return Refused($"the tokenizer's '{MediaPad}' is {padId}, but config.json names {configPad}");
        if (!root.TryGetProperty("vision_config", out JsonElement visionJson)) return Refused("config.json has no vision_config");
        KimiVisionConfig vision = ReadVisionConfig(visionJson);
        if (vision.TextHidden != textHidden) return Refused($"the projector emits {vision.TextHidden} values, but the decoder's hidden width is {textHidden}");

        string preprocessorPath = Path.Combine(directory, "preprocessor_config.json");
        if (!File.Exists(preprocessorPath)) return Refused("preprocessor_config.json is missing");
        using JsonDocument preprocessor = JsonDocument.Parse(File.ReadAllText(preprocessorPath));
        if (!preprocessor.RootElement.TryGetProperty("media_proc_cfg", out JsonElement cfg))
            return Refused("preprocessor_config.json has no media_proc_cfg");
        KimiImageSpec spec = KimiImageSpec.Read(cfg);
        if (spec.PatchSize != vision.PatchSize || spec.MergeKernelSize != vision.MergeSize)
            return Refused("the processor's patch and merge sizes differ from the vision tower's");

        using KimiK31CheckpointWeights source = KimiK31CheckpointWeights.Open(directory);
        KimiVisionWeights weights = KimiVisionWeightLoader.Load(vision, source);
        MoonViTEncoder encoder;
        try
        {
            encoder = new MoonViTEncoder(vision, weights);
        }
        catch
        {
            weights.Dispose();
            throw;
        }
        return new KimiVisionInput(encoder, spec, padId, refusal: null);
    }

    // Every field the tower reads from vision_config. The rotary base, the norm epsilon and the image side are not stored there, so they keep the
    // published architecture's values.
    private static KimiVisionConfig ReadVisionConfig(JsonElement v)
    {
        JsonElement merge = v.GetProperty("merge_kernel_size");
        int mergeSize = merge.EnumerateArray().First().GetInt32();
        if (merge.GetArrayLength() != 2 || merge.EnumerateArray().Any(m => m.GetInt32() != mergeSize))
            throw new NotSupportedException("vision_config's merge_kernel_size is not square.");
        int grid = v.GetProperty("init_pos_emb_height").GetInt32();
        if (v.GetProperty("init_pos_emb_width").GetInt32() != grid)
            throw new NotSupportedException("vision_config's position table is not square.");
        return new KimiVisionConfig
        {
            Hidden = v.GetProperty("vt_hidden_size").GetInt32(),
            Heads = v.GetProperty("vt_num_attention_heads").GetInt32(),
            QkvHidden = v.GetProperty("qkv_hidden_size").GetInt32(),
            MlpDim = v.GetProperty("vt_intermediate_size").GetInt32(),
            Layers = v.GetProperty("vt_num_hidden_layers").GetInt32(),
            PatchSize = v.GetProperty("patch_size").GetInt32(),
            MergeSize = mergeSize,
            InitPosGrid = grid,
            TextHidden = v.GetProperty("text_hidden_size").GetInt32(),
            ProjectorEpsilon = (float)v.GetProperty("projector_ln_eps").GetDouble(),
            BlockNormEpsilon = KimiVisionConfig.Published.BlockNormEpsilon,
            RopeTheta = KimiVisionConfig.Published.RopeTheta,
            ImageSize = KimiVisionConfig.Published.ImageSize,
        };
    }

    private static KimiVisionInput Refused(string reason) =>
        new(encoder: null, image: null, padId: null, refusal: $"the vision tower cannot take images: {reason}");
}
