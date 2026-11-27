using System.Text.Json;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Duo;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>The files of a Duo release in the upstream bundle layout under one release folder: <c>realization/dcp</c> (the LoRA),
/// <c>realization/backbone</c> (the MiniMax-H3 Ref2VA shards), <c>codec/video_vae/model.safetensors</c> and
/// <c>codec/audio_vae/diffusion_pytorch_model.safetensors</c>. The distilled release ships the LoRA alone, so the causal release's
/// backbone and codec go beside it. A request's VAE overrides replace the codec files.</summary>
internal sealed record DuoAssets(string LoraDirectory, IReadOnlyList<string> BackboneShards, string VideoVae, string AudioVae)
{
    /// <summary>Resolves the release from <paramref name="checkpointPath"/>: the release folder, its <c>realization/dcp</c> folder, or a
    /// file in that folder.</summary>
    /// <exception cref="FileNotFoundException">The LoRA or a codec file is missing.</exception>
    /// <exception cref="DirectoryNotFoundException">The backbone folder is missing.</exception>
    /// <exception cref="NotSupportedException">The release is of the other stage, or the backbone folder is not the released shard set.</exception>
    public static DuoAssets Resolve(string checkpointPath, ComponentOverrides? components, bool distilled = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        string dcp = DcpFolder(checkpointPath);
        string? realization = Path.GetDirectoryName(dcp);
        string? root = realization is null ? null : Path.GetDirectoryName(realization);
        if (realization is null || root is null)
        {
            throw new DirectoryNotFoundException($"'{dcp}' is not '<release>/realization/dcp', so the backbone and codec cannot be located.");
        }
        RefuseOtherStage(realization, distilled ? "dmd" : "causal");
        return new DuoAssets(dcp, Backbone(Path.Combine(realization, "backbone")),
            Codec(components?.VideoVae ?? components?.Vae, Path.Combine(root, "codec", "video_vae", "model.safetensors"), "video VAE"),
            Codec(components?.AudioVae, Path.Combine(root, "codec", "audio_vae", "diffusion_pytorch_model.safetensors"), "audio VAE"));
    }

    /// <summary>Where the backbone of the release at <paramref name="checkpointPath"/> belongs: the release folder, its LoRA folder or a
    /// file in it. A LoRA folder on disk is known by its <c>.metadata</c> under any name, as <see cref="Resolve"/> reads it; before a
    /// download, <c>realization/dcp</c> is known by name.</summary>
    public static string BackboneFolder(string checkpointPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        string full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(checkpointPath));
        string? lora = File.Exists(full) ? Path.GetDirectoryName(full) : full;
        if (lora is not null && File.Exists(Path.Combine(lora, ".metadata")) && Path.GetDirectoryName(lora) is { } beside)
        {
            return Path.Combine(beside, "backbone");
        }
        foreach (string? dcp in new[] { full, Path.GetDirectoryName(full) })
        {
            string? parent = dcp is null ? null : Path.GetDirectoryName(dcp);
            if (parent is not null && Named(dcp!, "dcp") && Named(parent, "realization"))
            {
                return Path.Combine(parent, "backbone");
            }
        }
        return Path.Combine(full, "realization", "backbone");
    }

    private static bool Named(string folder, string name) =>
        string.Equals(Path.GetFileName(folder), name, StringComparison.OrdinalIgnoreCase);

    private static string DcpFolder(string checkpointPath)
    {
        string full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(checkpointPath));
        string[] candidates = File.Exists(full)
            ? [Path.GetDirectoryName(full)!]
            : [full, Path.Combine(full, "realization", "dcp")];
        return candidates.FirstOrDefault(c => File.Exists(Path.Combine(c, ".metadata")))
            ?? throw new FileNotFoundException(
                $"No Duo LoRA at '{full}': expected the release folder, or its realization/dcp folder, holding realization/dcp/.metadata.");
    }

    private static void RefuseOtherStage(string realization, string expected)
    {
        string manifest = Path.Combine(realization, "complete.json");
        if (!File.Exists(manifest))
        {
            return;
        }
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(manifest));
        string? stage = document.RootElement.TryGetProperty("stage", out JsonElement value) ? value.GetString() : null;
        if (!string.Equals(stage, expected, StringComparison.Ordinal))
        {
            throw new NotSupportedException(
                $"{manifest} names stage '{stage}'; this model runs a '{expected}' release. Pick duo-v1-distill for 'dmd' and a causal "
                + "model for 'causal'.");
        }
    }

    private static IReadOnlyList<string> Backbone(string folder)
    {
        if (!Directory.Exists(folder))
        {
            throw new DirectoryNotFoundException(
                $"Duo needs the MiniMax-H3 Ref2VA backbone shards (about 66 GB) in '{folder}'.");
        }
        if (File.Exists(Path.Combine(folder, DuoRepackManifest.FileName)))
        {
            throw new NotSupportedException(
                $"'{folder}' is a DuoRepack output, already rewritten; merging into it again would be wrong. Put the release's own shards there.");
        }
        IReadOnlyList<string> shards = HfCheckpointDirectory.WeightFiles(folder);
        if (shards.Count < 2)
        {
            throw new NotSupportedException(
                $"'{folder}' holds one weight file. The released backbone is sharded and stores qkv rows per head; a single file may be a "
                + "repack in [Q; K; V] order, which this recipe would reorder again.");
        }
        return shards;
    }

    private static string Codec(string? overridden, string bundled, string what)
    {
        string path = overridden ?? bundled;
        return File.Exists(path) ? path : throw new FileNotFoundException($"Duo {what} not found at '{path}'.", path);
    }
}
