using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Features;

/// <summary>Qwen3.5's vision tower (<c>qwen3_5_vision</c>: the Qwen3-VL ViT without deepstack, exact-GELU merger) and its
/// image processor (HF <c>Qwen2VLImageProcessorFast</c> values: factor 32, 65 536–16 777 216 pixels, mean/std 0.5).
/// Found in the encoder's own weights (<c>model.visual.*</c> or <c>visual.*</c>, as Comfy-Org's <c>qwen3.5_*</c> files and
/// the HF release carry it) or in a separate safetensors file with the same keys.</summary>
public sealed class Qwen35VisionTower : IDisposable
{
    /// <summary>HF <c>Qwen/Qwen3.5-9B</c> <c>preprocessor_config.json</c> <c>size.shortest_edge</c>.</summary>
    public const int MinPixels = 65_536;

    /// <summary>HF <c>size.longest_edge</c>.</summary>
    public const int MaxPixels = 16_777_216;

    private static readonly string[] Prefixes = ["model.visual.", "visual."];
    private static readonly float[] HalfMeanStd = [0.5f, 0.5f, 0.5f];

    private readonly IDisposable? _owner;
    private int _disposed;

    private Qwen35VisionTower(Qwen3VlVisionEncoder encoder, Qwen3VlVisionConfig config, IDisposable? owner)
    {
        Encoder = encoder;
        Config = config;
        _owner = owner;
        Processor = new Qwen3VlImageProcessor(config, MinPixels, MaxPixels, imageMean: HalfMeanStd, imageStd: HalfMeanStd);
    }

    /// <summary>The tower.</summary>
    public Qwen3VlVisionEncoder Encoder { get; }

    /// <summary>The tower's shape.</summary>
    public Qwen3VlVisionConfig Config { get; }

    /// <summary>Smart-resize and patch layout for the tower.</summary>
    public Qwen3VlImageProcessor Processor { get; }

    /// <summary>The tower in <paramref name="encoderWeights"/> when they carry one, else in <paramref name="overridePath"/>,
    /// else null. <paramref name="visionConfigJson"/> is the encoder's <c>config.json</c> when it has one.</summary>
    public static Qwen35VisionTower? Resolve(IReadOnlyDictionary<string, Tensor>? encoderWeights, string? overridePath,
        string? visionConfigJson, string family)
    {
        if (encoderWeights is not null && Bare(encoderWeights) is Dictionary<string, Tensor> own)
        {
            Logs.Info($"[{family}] Qwen3.5 vision tower: in the text encoder's weights.");
            return Build(own, visionConfigJson, null);
        }
        if (string.IsNullOrWhiteSpace(overridePath))
            return null;
        if (!File.Exists(overridePath) && !Directory.Exists(overridePath))
            throw new FileNotFoundException($"{family}: vision tower '{overridePath}' does not exist.", overridePath);
        IReadOnlyList<string> files = Directory.Exists(overridePath) ? HfCheckpointDirectory.WeightFiles(overridePath) : [overridePath];
        CheckpointSource source = CheckpointSource.OpenShards(files);
        try
        {
            Dictionary<string, Tensor> bare = Bare(source.Weights)
                ?? throw new InvalidDataException($"{family}: '{overridePath}' holds no visual.* or model.visual.* weights.");
            string? config = Directory.Exists(overridePath) ? Path.Combine(overridePath, HfCheckpointDirectory.ConfigFileName) : null;
            Logs.Info($"[{family}] Qwen3.5 vision tower: {Path.GetFileName(Path.TrimEndingDirectorySeparator(overridePath))}.");
            return Build(bare, config is not null && File.Exists(config) ? config : null, source);
        }
        catch
        {
            source.Dispose();
            throw;
        }
    }

    /// <summary>The <c>visual.*</c> subtree re-rooted to bare keys, or null when there is none.</summary>
    private static Dictionary<string, Tensor>? Bare(IReadOnlyDictionary<string, Tensor> weights)
    {
        foreach (string prefix in Prefixes)
        {
            if (!weights.ContainsKey(prefix + "patch_embed.proj.weight"))
                continue;
            Dictionary<string, Tensor> bare = new(StringComparer.Ordinal);
            foreach ((string key, Tensor value) in weights)
            {
                if (key.StartsWith(prefix, StringComparison.Ordinal))
                    bare[key[prefix.Length..]] = value;
            }
            return bare;
        }
        return null;
    }

    private static Qwen35VisionTower Build(Dictionary<string, Tensor> bare, string? configJson, IDisposable? owner)
    {
        Qwen3VlVisionConfig? fromJson = null;
        if (configJson is not null)
        {
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(configJson));
            if (doc.RootElement.TryGetProperty("vision_config", out JsonElement vision))
                fromJson = Qwen3VlVisionConfig.FromJson(vision);
        }
        Qwen3VlVisionConfig config = (fromJson ?? Qwen3VlVisionConfig.FromWeights(bare)) with { MergerExactGelu = true };
        if (config.DeepstackVisualIndexes.Length != 0)
            throw new NotSupportedException("A Qwen3.5 vision tower has no deepstack taps; this checkpoint declares some.");
        Qwen3VlVisionEncoder encoder = new(config);
        encoder.LoadWeights(bare);
        return new Qwen35VisionTower(encoder, config, owner);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Encoder.Dispose();
        _owner?.Dispose();
    }
}
