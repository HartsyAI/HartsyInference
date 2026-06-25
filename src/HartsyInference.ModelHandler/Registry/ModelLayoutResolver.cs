using HartsyInference.Core.Logging;

namespace HartsyInference.ModelHandler.Registry;

/// <summary>On-disk shapes a diffusion model can take.</summary>
public enum ModelLayoutKind
{
    /// <summary>One self-contained checkpoint file (the common all-in-one safetensors / GGUF).</summary>
    SingleFile,

    /// <summary>A multi-shard safetensors checkpoint (<c>model-00001-of-0000N.safetensors</c> + index).</summary>
    Sharded,

    /// <summary>A diffusers-style directory: <c>model_index.json</c> plus per-component subfolders
    /// (<c>unet/</c> or <c>transformer/</c>, <c>vae/</c>, <c>text_encoder/</c>, ...).</summary>
    Diffusers,
}

/// <summary>Result of resolving a user-supplied path into a concrete file layout.</summary>
public sealed record ModelLayout
{
    /// <summary>Which on-disk shape was found.</summary>
    public required ModelLayoutKind Kind { get; init; }

    /// <summary>Absolute root: the file itself for <see cref="ModelLayoutKind.SingleFile"/>, otherwise the directory.</summary>
    public required string RootPath { get; init; }

    /// <summary>A single file whose header is representative of the checkpoint for architecture detection
    /// (the bundled file, the first shard, or the transformer/unet component).</summary>
    public required string RepresentativeFile { get; init; }

    /// <summary>Every safetensors file that makes up the checkpoint (1 for single-file).</summary>
    public required IReadOnlyList<string> SafeTensorsFiles { get; init; }
}

/// <summary>Resolves a path (a single checkpoint file, a sharded checkpoint, or a diffusers-layout
/// directory) into a <see cref="ModelLayout"/>. Pairs with <see cref="ModelArchitectureDetector"/>:
/// the resolver decides <i>which files</i> to read, the detector decides <i>what they are</i>.</summary>
public static class ModelLayoutResolver
{
    private static readonly string[] TransformerSubdirs = ["transformer", "unet"];
    private static readonly string[] ComponentSubdirs = ["vae", "text_encoder", "text_encoder_2"];

    /// <summary>Resolves the layout of <paramref name="path"/>.</summary>
    /// <exception cref="FileNotFoundException">The path does not exist.</exception>
    /// <exception cref="InvalidOperationException">A directory contained no usable safetensors files.</exception>
    public static ModelLayout Resolve(string path)
    {
        if (File.Exists(path))
        {
            return new ModelLayout
            {
                Kind = ModelLayoutKind.SingleFile,
                RootPath = path,
                RepresentativeFile = path,
                SafeTensorsFiles = [path],
            };
        }

        if (!Directory.Exists(path))
        {
            throw new FileNotFoundException($"Model path not found: {path}", path);
        }

        // Diffusers layout: model_index.json or a known component subfolder.
        bool hasModelIndex = File.Exists(Path.Combine(path, "model_index.json"));
        string? transformerDir = TransformerSubdirs
            .Select(d => Path.Combine(path, d))
            .FirstOrDefault(Directory.Exists);

        if (hasModelIndex || transformerDir is not null)
        {
            return ResolveDiffusers(path, transformerDir);
        }

        // Flat directory of safetensors: single bundled file or a shard set.
        string[] rootShards = Directory.GetFiles(path, "*.safetensors", SearchOption.TopDirectoryOnly);
        if (rootShards.Length == 0)
        {
            throw new InvalidOperationException($"No .safetensors files found under directory: {path}");
        }

        Array.Sort(rootShards, StringComparer.Ordinal);
        if (rootShards.Length == 1)
        {
            return new ModelLayout
            {
                Kind = ModelLayoutKind.SingleFile,
                RootPath = path,
                RepresentativeFile = rootShards[0],
                SafeTensorsFiles = rootShards,
            };
        }

        Logs.Debug($"ModelLayoutResolver: {rootShards.Length} shards under {path}.");
        return new ModelLayout
        {
            Kind = ModelLayoutKind.Sharded,
            RootPath = path,
            RepresentativeFile = rootShards[0],
            SafeTensorsFiles = rootShards,
        };
    }

    private static ModelLayout ResolveDiffusers(string root, string? transformerDir)
    {
        List<string> files = new List<string>();

        // Representative = the transformer/unet component (its keys carry the architecture signature).
        string? representative = null;
        if (transformerDir is not null)
        {
            string[] tFiles = Directory.GetFiles(transformerDir, "*.safetensors", SearchOption.TopDirectoryOnly);
            Array.Sort(tFiles, StringComparer.Ordinal);
            files.AddRange(tFiles);
            representative = tFiles.FirstOrDefault();
        }

        foreach (string comp in ComponentSubdirs)
        {
            string compDir = Path.Combine(root, comp);
            if (Directory.Exists(compDir))
            {
                files.AddRange(Directory.GetFiles(compDir, "*.safetensors", SearchOption.TopDirectoryOnly));
            }
        }

        representative ??= files.FirstOrDefault();
        if (representative is null)
        {
            throw new InvalidOperationException($"Diffusers layout at {root} contained no component safetensors files.");
        }

        return new ModelLayout
        {
            Kind = ModelLayoutKind.Diffusers,
            RootPath = root,
            RepresentativeFile = representative,
            SafeTensorsFiles = files,
        };
    }
}
