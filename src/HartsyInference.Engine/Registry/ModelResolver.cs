using HartsyInference.Engine.Dispatch;

namespace HartsyInference.Engine.Registry;

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
        [Modality.Embedding] = "Embedding",
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

        // A catalog entry's Assets record where its files ACTUALLY land (TargetSubdir/FileName) — for several
        // families (e.g. image/video's "Stable-Diffusion/<Family>") that's a different folder than the coarse
        // per-modality guess below ("Image"/"Video"). Prefer the authoritative asset path when it's actually on
        // disk; only fall through to the guess for catalog-less entries or ones with no defined Assets.
        if (catalog is { Assets.Count: > 0 })
        {
            string? primary = ModelDownloader.PrimaryLocalPath(catalog);
            if (primary is not null && File.Exists(primary))
                return Path.GetFullPath(primary);
        }

        string subdir = ModalitySubdir.TryGetValue(modality, out string? s) ? s : "";
        string id = catalog?.Id ?? modelArg;
        string candidate = Path.Combine(RepoPaths.ModelsRoot(), subdir, id);
        if (File.Exists(candidate))
            return Path.GetFullPath(candidate);
        if (Directory.Exists(candidate))
        {
            // TextService.LoadInto needs a direct .gguf file path — unlike the folder-checkpoint diffusion/music
            // families, a Text catalog id's Models/LLM/<id>/ directory is never itself a valid LocalPath. Auto-
            // discover the single .gguf inside it (same convention TextService.FindMmproj uses for its sidecar
            // scan); ambiguous (0 or 2+) falls through to null so an Assets-based download or an explicit
            // --model-path resolves it instead, rather than handing the loader a directory it can't open.
            if (modality == Modality.Text)
            {
                string[] ggufs = Directory.GetFiles(candidate, "*.gguf");
                return ggufs.Length == 1 ? Path.GetFullPath(ggufs[0]) : null;
            }
            return Path.GetFullPath(candidate);
        }

        return null;
    }
}
