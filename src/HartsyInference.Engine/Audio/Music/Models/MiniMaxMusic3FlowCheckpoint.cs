using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Audio;

/// <summary>Opens a user-selected MiniMax-Music-3 flow-transformer checkpoint — whatever container and layout it is in — as the dictionaries the model loads.</summary>
/// <remarks>The layout is judged by tensor names (<see cref="MiniMaxMusic3DitLayouts"/>), never by file name, so a repack named anything loads and a file that only looks right is refused. Accepted: a single safetensors or GGUF file, any shard of a diffusers shard set, a shard index, or a folder holding one of those.</remarks>
internal static class MiniMaxMusic3FlowCheckpoint
{
    private const string SupportedLayouts =
        "Supported: the official diffusers transformer (single file, shards or index), the Hartsy repack of it, "
        + "a Comfy-Org single-file DiT (fp32, fp16 or int8 ConvRot) and a GGUF build that keeps the diffusers tensor names, "
        + "in any of those precisions or quantizations.";

    /// <summary>Opens <paramref name="path"/> (file, shard index or folder). The result owns every tensor and mapping; dispose it with the runner.</summary>
    internal static MiniMaxMusic3FlowWeights Open(string path)
    {
        IReadOnlyList<string> files = ResolveFiles(path);
        CheckpointSource source = CheckpointSource.OpenShards(files);
        try
        {
            MiniMaxMusic3DitLayout layout = MiniMaxMusic3DitLayouts.Detect(source.Weights.Keys);
            string label = MiniMaxMusic3DitLayouts.Describe(layout);
            switch (layout)
            {
                case MiniMaxMusic3DitLayout.ComfyDit:
                {
                    (Dictionary<string, Tensor> dit, Dictionary<string, Tensor> condition) =
                        MiniMaxMusic3CheckpointConverter.ConvertComfyDit(source.Weights);
                    return new MiniMaxMusic3FlowWeights(source, dit, condition.Count == 0 ? null : condition, label);
                }
                case MiniMaxMusic3DitLayout.DiffusersDit:
                case MiniMaxMusic3DitLayout.DiffusersDitWithConditionEncoder:
                {
                    (Dictionary<string, Tensor> dit, Dictionary<string, Tensor>? condition) =
                        MiniMaxMusic3CheckpointConverter.ConvertDiffusersDit(source.Weights);
                    return new MiniMaxMusic3FlowWeights(source, dit, condition, label);
                }
                default:
                    throw new NotSupportedException(
                        $"'{Path.GetFileName(path)}' is {label} ({source.Weights.Count} tensors, {source.Format}), "
                        + $"not a MiniMax-Music-3 flow transformer the engine can load. {SupportedLayouts} "
                        + "Remove the file to use the official MiniMaxAI/MiniMax-Music3 weights instead.");
            }
        }
        catch
        {
            source.Dispose();
            throw;
        }
    }

    /// <summary>The weight files behind <paramref name="path"/>: itself, the rest of a shard set, an index's shards, or the checkpoint inside a folder.</summary>
    internal static IReadOnlyList<string> ResolveFiles(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (Directory.Exists(path))
        {
            return ResolveFiles(FindInFolder(path));
        }
        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"MiniMax-Music-3 checkpoint '{path}' does not exist.", path);
        }
        if (path.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
        {
            string folder = Path.GetDirectoryName(Path.GetFullPath(path))!;
            return [.. AudioCheckpoints.ReadShardNames(path).OrderBy(name => name, StringComparer.Ordinal)
                .Select(name => Path.Combine(folder, name))];
        }
        return ShardSetDiscovery.Resolve(path);
    }

    private static string FindInFolder(string folder)
    {
        foreach (string name in new[] { "diffusion_pytorch_model.safetensors.index.json", "diffusion_pytorch_model.safetensors" })
        {
            string candidate = Path.Combine(folder, name);
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }
        string[] weights = [.. Directory.EnumerateFiles(folder, "*.safetensors").Concat(Directory.EnumerateFiles(folder, "*.gguf"))];
        return weights.Length == 1 ? weights[0]
            : throw new FileNotFoundException(
                $"'{folder}' holds {weights.Length} weight files and no diffusion_pytorch_model.safetensors[.index.json]; "
                + "point at the transformer file itself.", folder);
    }
}
