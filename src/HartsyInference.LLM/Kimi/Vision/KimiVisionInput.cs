using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Multimodal;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Kimi.Vision;

/// <summary>What a Kimi K3.1 directory needs to take images: the MoonViT-V2 encoder, the image geometry and the media pad id, read from the
/// checkpoint on the first image request.</summary>
/// <remarks>Load reads nothing. <see cref="Route"/> loads the tower once and keeps the result; a checkpoint whose tower, placeholder or geometry does
/// not load is refused by name there, and later calls repeat that refusal without reading again. The text decoder loads either way.</remarks>
public sealed class KimiVisionInput : IDisposable
{
    /// <summary>The literals the Kimi processor writes around an image; <see cref="MediaPad"/> stands for one vision token.</summary>
    public const string MediaBegin = "<|media_begin|>", MediaContent = "<|media_content|>", MediaPad = "<|media_pad|>", MediaEnd = "<|media_end|>";

    private readonly string _directory;
    private readonly ILlmTokenizer _tokenizer;
    private readonly int _textHidden;
    private readonly bool _xtmlTemplate;
    private readonly Func<string, IKimiK31WeightStore> _openWeights;
    private readonly Lock _gate = new();
    private bool _resolved;
    private KimiVisionRoute? _route;
    private string? _refusal;
    private bool _disposed;

    private KimiVisionInput(string directory, ILlmTokenizer tokenizer, int textHidden, bool xtmlTemplate,
        Func<string, IKimiK31WeightStore> openWeights)
    {
        _directory = directory;
        _tokenizer = tokenizer;
        _textHidden = textHidden;
        _xtmlTemplate = xtmlTemplate;
        _openWeights = openWeights;
    }

    /// <summary>Whether <see cref="Route"/> has loaded or refused the tower. False until the first image request.</summary>
    internal bool IsResolved
    {
        get
        {
            lock (_gate) return _resolved;
        }
    }

    /// <summary>The processor's prompt for one image: its size, then the pad that stands for its vision tokens.</summary>
    public static string MediaPrompt(int width, int height) => $"{MediaBegin}image {width}x{height}{MediaContent}{MediaPad}{MediaEnd}";

    /// <summary>Records where the checkpoint in <paramref name="directory"/> keeps its vision tower. Reads no file until
    /// <see cref="Route"/>.</summary>
    /// <param name="textHidden">The decoder's hidden width, which the projector must emit.</param>
    /// <param name="xtmlTemplate">Whether the chat template is the XTML port, the one that renders image placeholders.</param>
    public static KimiVisionInput Load(string directory, ILlmTokenizer tokenizer, int textHidden, bool xtmlTemplate) =>
        Load(directory, tokenizer, textHidden, xtmlTemplate, KimiK31CheckpointWeights.Open);

    /// <summary>Records the checkpoint like <see cref="Load(string, ILlmTokenizer, int, bool)"/>, reading its tower from the store
    /// <paramref name="openWeights"/> opens.</summary>
    internal static KimiVisionInput Load(string directory, ILlmTokenizer tokenizer, int textHidden, bool xtmlTemplate,
        Func<string, IKimiK31WeightStore> openWeights)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(openWeights);
        return new KimiVisionInput(directory, tokenizer, textHidden, xtmlTemplate, openWeights);
    }

    /// <summary>The image route. The first call loads the tower; later calls return the same route.</summary>
    /// <exception cref="NotSupportedException">The checkpoint cannot take images; the message gives the reason.</exception>
    /// <exception cref="ObjectDisposedException">This input has been disposed.</exception>
    public KimiVisionRoute Route()
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_resolved)
            {
                Resolve();
                _resolved = true;
            }
            return _route ?? throw new NotSupportedException($"Kimi K3.1 image input is not supported: {_refusal}");
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            _route?.Encoder.Dispose();
            _route = null;
        }
    }

    // Reads the tower once. A failure becomes the refusal; the text decoder never sees it.
    private void Resolve()
    {
        try
        {
            _route = Build();
        }
        catch (Exception ex) when (ex is KeyNotFoundException or InvalidDataException or InvalidOperationException or NotSupportedException
            or ArgumentException or FormatException or JsonException or IOException or HartsyInferenceException)
        {
            _refusal = $"the vision tower cannot take images: {ex.Message}";
        }
    }

    private KimiVisionRoute Build()
    {
        if (!_xtmlTemplate) throw Refuse("the chat template is not the XTML template, which renders the image placeholders");
        using JsonDocument config = JsonDocument.Parse(File.ReadAllText(Path.Combine(_directory, "config.json")));
        JsonElement root = config.RootElement;
        if (!root.TryGetProperty("media_placeholder_token_id", out JsonElement padValue) || !padValue.TryGetInt32(out int configPad))
            throw Refuse("config.json names no media_placeholder_token_id");
        foreach (string literal in new[] { MediaBegin, MediaContent, MediaPad, MediaEnd })
        {
            if (_tokenizer.SpecialId(literal) is null) throw Refuse($"the tokenizer has no '{literal}' token");
        }
        int padId = _tokenizer.SpecialId(MediaPad)!.Value;
        if (padId != configPad) throw Refuse($"the tokenizer's '{MediaPad}' is {padId}, but config.json names {configPad}");
        if (!root.TryGetProperty("vision_config", out JsonElement visionJson)) throw Refuse("config.json has no vision_config");
        KimiVisionConfig vision = ReadVisionConfig(visionJson);
        if (vision.TextHidden != _textHidden)
            throw Refuse($"the projector emits {vision.TextHidden} values, but the decoder's hidden width is {_textHidden}");

        string preprocessorPath = Path.Combine(_directory, "preprocessor_config.json");
        if (!File.Exists(preprocessorPath)) throw Refuse("preprocessor_config.json is missing");
        using JsonDocument preprocessor = JsonDocument.Parse(File.ReadAllText(preprocessorPath));
        if (!preprocessor.RootElement.TryGetProperty("media_proc_cfg", out JsonElement cfg))
            throw Refuse("preprocessor_config.json has no media_proc_cfg");
        KimiImageSpec spec = KimiImageSpec.Read(cfg);
        if (spec.PatchSize != vision.PatchSize || spec.MergeKernelSize != vision.MergeSize)
            throw Refuse("the processor's patch and merge sizes differ from the vision tower's");

        using IKimiK31WeightStore source = _openWeights(_directory);
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
        return new KimiVisionRoute(encoder, spec, padId);
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

    // The refusal of Build, caught by Resolve and kept by name.
    private static NotSupportedException Refuse(string reason) => new(reason);
}
