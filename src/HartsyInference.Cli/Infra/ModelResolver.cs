using HartsyInference.Cli.Dispatch;

namespace HartsyInference.Cli.Infra;

/// <summary>Turns a user's <c>--model</c> / <c>--model-path</c> selection into a <see cref="ModelSpec"/> by matching
/// the catalog and locating a local checkpoint (explicit path, raw path, or under the models root).</summary>
public static class ModelResolver
{
    private static readonly Dictionary<Modality, string> ModalitySubdir = new()
    {
        [Modality.Text] = "LLM",
        [Modality.Image] = "Image",
        [Modality.Speech] = "Audio",
        [Modality.Music] = "Audio",
        [Modality.Transcribe] = "Audio",
        [Modality.Vision] = "Vision",
        [Modality.Video] = "Video",
        [Modality.Mesh] = "3D",
        [Modality.World] = "Interactive",
        [Modality.VoiceConvert] = "Audio",
        [Modality.Fx] = "Audio",
    };

    /// <summary>Resolves a model selection for <paramref name="modality"/>. <paramref name="modelArg"/> may be a catalog
    /// id, a local path, or an HF repo id; <paramref name="modelPathArg"/> is an explicit override that wins.</summary>
    public static ModelSpec Resolve(string modelArg, string? modelPathArg, Modality modality)
    {
        CatalogEntry? catalog = ModelCatalog.Find(modelArg);
        string? local = LocateLocal(modelArg, modelPathArg, catalog, modality);
        return new ModelSpec
        {
            Requested = modelArg,
            Modality = modality,
            Catalog = catalog,
            LocalPath = local,
        };
    }

    private static string? LocateLocal(string modelArg, string? modelPathArg, CatalogEntry? catalog, Modality modality)
    {
        if (!string.IsNullOrWhiteSpace(modelPathArg) && (File.Exists(modelPathArg) || Directory.Exists(modelPathArg)))
            return Path.GetFullPath(modelPathArg);

        if (File.Exists(modelArg) || Directory.Exists(modelArg))
            return Path.GetFullPath(modelArg);

        string subdir = ModalitySubdir.TryGetValue(modality, out string? s) ? s : "";
        string id = catalog?.Id ?? modelArg;
        string candidate = Path.Combine(RepoPaths.ModelsRoot(), subdir, id);
        if (Directory.Exists(candidate) || File.Exists(candidate))
            return Path.GetFullPath(candidate);

        return null;
    }
}
