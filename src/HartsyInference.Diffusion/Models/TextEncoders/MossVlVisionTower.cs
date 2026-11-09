using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>MOSS-VL-Realtime's vision tower: the Qwen3-VL ViT (identical blocks, position interpolation and rotary) with
/// MOSS's unified deepstack merger, read from the checkpoint's <c>model.visual.*</c> weights and its
/// <c>vision_config</c>. Takes preprocessed patches in merge-block order.</summary>
public sealed class MossVlVisionTower : IDisposable
{
    private const string Prefix = "model.visual.";

    private readonly IDisposable? _owner;
    private readonly List<Tensor> _converted;
    private int _disposed;

    private MossVlVisionTower(Qwen3VlVisionEncoder encoder, Qwen3VlVisionConfig config, IDisposable? owner, List<Tensor> converted)
    {
        Encoder = encoder;
        Config = config;
        _owner = owner;
        _converted = converted;
    }

    /// <summary>The tower.</summary>
    public Qwen3VlVisionEncoder Encoder { get; }

    /// <summary>The tower's shape, as read from <c>vision_config</c>.</summary>
    public Qwen3VlVisionConfig Config { get; }

    /// <summary>Reads a MOSS-VL <c>vision_config</c> object: the Qwen3-VL field names, with MOSS's merger layout and its exact GELU.</summary>
    public static Qwen3VlVisionConfig ConfigFromJson(JsonElement visionConfig) =>
        Qwen3VlVisionConfig.FromJson(visionConfig) with { UnifiedDeepstackMerger = true, MergerExactGelu = true };

    /// <summary>Opens the vision tower of a MOSS-VL checkpoint directory (its <c>config.json</c> and safetensors shards).</summary>
    /// <exception cref="InvalidDataException">The checkpoint holds no <c>model.visual.*</c> weights.</exception>
    public static MossVlVisionTower Open(string checkpointDirectory)
    {
        string configPath = Path.Combine(checkpointDirectory, HfCheckpointDirectory.ConfigFileName);
        Qwen3VlVisionConfig config;
        using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(configPath)))
            config = ConfigFromJson(doc.RootElement.GetProperty("vision_config"));

        CheckpointSource source = CheckpointSource.OpenShards(HfCheckpointDirectory.WeightFiles(checkpointDirectory));
        List<Tensor> converted = [];
        try
        {
            Dictionary<string, Tensor> bare = new(StringComparer.Ordinal);
            foreach ((string key, Tensor value) in source.Weights)
            {
                if (!key.StartsWith(Prefix, StringComparison.Ordinal))
                    continue;
                // The CPU path computes in F32; a BF16 checkpoint is widened once here and the copies are owned by the tower.
                Tensor f32 = TensorCasts.EnsureF32(value);
                if (!ReferenceEquals(f32, value)) converted.Add(f32);
                bare[key[Prefix.Length..]] = f32;
            }
            if (!bare.ContainsKey("patch_embed.proj.weight"))
                throw new InvalidDataException($"'{checkpointDirectory}' holds no {Prefix}* vision weights.");

            Qwen3VlVisionEncoder encoder = new(config);
            encoder.LoadWeights(bare);
            return new MossVlVisionTower(encoder, config, source, converted);
        }
        catch
        {
            foreach (Tensor t in converted) t.Dispose();
            source.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Encoder.Dispose();
        foreach (Tensor t in _converted) t.Dispose();
        _owner?.Dispose();
    }
}
