using HartsyInference.Core.Exceptions;
using HartsyInference.Core.IO;
using HartsyInference.Engine.Dispatch;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Registry;

/// <summary>Turns a user's <c>--model</c> / <c>--model-path</c> selection into a <see cref="ModelSpec"/> by matching the catalog and locating a local checkpoint (explicit path, raw path, or under the models root).</summary>
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
        [Modality.Restore] = "Video",
        [Modality.Decision] = "Decision",
    };

    /// <summary>Resolves a model selection for <paramref name="modality"/>. <paramref name="modelArg"/> may be a catalog id, a local path, or an HF repo id; <paramref name="modelPathArg"/> is an explicit override that wins.</summary>
    public static ModelSpec Resolve(string modelArg, string? modelPathArg, Modality modality)
    {
        CatalogEntry? catalog = ModelCatalog.Find(modelArg);
        string? variant = null;
        // Image and video take a family:variant selector (qwen-image:edit); audio descriptors parse the raw token
        // themselves, so for them an unmatched selector must stay catalog-less.
        if (catalog is null && modality is Modality.Image or Modality.Video)
        {
            ModelSelector selector = ModelSelector.Parse(modelArg);
            if (selector.Variant is not null && ModelCatalog.Find(selector.Id) is CatalogEntry family)
            {
                catalog = family;
                variant = selector.Variant;
            }
        }
        string? local = LocateLocal(modelArg, modelPathArg, catalog, modality);
        return new ModelSpec
        {
            Requested = modelArg,
            Modality = modality,
            Catalog = catalog,
            LocalPath = local,
            Variant = variant,
        };
    }

    private static string? LocateLocal(string modelArg, string? modelPathArg, CatalogEntry? catalog, Modality modality)
    {
        if (!string.IsNullOrWhiteSpace(modelPathArg) && (File.Exists(modelPathArg) || Directory.Exists(modelPathArg)))
            return Path.GetFullPath(modelPathArg);

        if (File.Exists(modelArg) || Directory.Exists(modelArg))
            return Path.GetFullPath(modelArg);

        // The store may spell folders in another case (SwarmUI's root has llm/ and audio/ for LLM/ and Audio/). Those
        // are tried only after every exact spelling missed, so whatever resolved before still resolves the same.
        return LocateUnderModelsRoot(modelArg, catalog, modality, ignoreCase: false)
            ?? LocateUnderModelsRoot(modelArg, catalog, modality, ignoreCase: true);
    }

    /// <summary>The catalog's primary asset, then the per-modality folder guess, under the models root, with every
    /// folder spelled exactly or, with <paramref name="ignoreCase"/>, matched ignoring case.</summary>
    private static string? LocateUnderModelsRoot(
        string modelArg, CatalogEntry? catalog, Modality modality, bool ignoreCase)
    {
        // A catalog entry's Assets record where its files ACTUALLY land (TargetSubdir/FileName) — for several
        // families (e.g. image/video's "Stable-Diffusion/<Family>") that's a different folder than the coarse
        // per-modality guess below ("Image"/"Video"). Prefer the authoritative asset path when it's actually on
        // disk; only fall through to the guess for catalog-less entries or ones with no defined Assets.
        if (catalog is { Assets.Count: > 0 } && ModelDownloader.PrimaryAsset(catalog) is ModelAsset asset
            && ModelDownloader.FindExisting(asset, ignoreCase) is string primary)
        {
            return Path.GetFullPath(primary);
        }

        string subdir = ModalitySubdir.TryGetValue(modality, out string? s) ? s : "";
        string id = catalog?.Id ?? modelArg;
        // Path.Combine(root, "Vision", "") is the Vision FOLDER, which exists — so a blank id used to resolve to a
        // whole modality directory and only fail much later, inside a loader, naming a path the caller never sent.
        if (string.IsNullOrWhiteSpace(id))
        {
            return null;
        }
        string root = RepoPaths.ModelsRoot();
        string relative = Path.Combine(subdir, id);
        string candidate = ignoreCase ? CaseInsensitivePath.ResolveEntry(root, relative) : Path.Combine(root, relative);
        if (File.Exists(candidate))
            return Path.GetFullPath(candidate);
        if (Directory.Exists(candidate))
        {
            // TextService.LoadInto takes a .gguf file or a Hugging Face checkpoint directory. Auto-discover the single
            // .gguf inside (same convention TextService.FindMmproj uses for its sidecar scan); a directory that
            // probes as a Hugging Face checkpoint is itself the LocalPath. Anything else (0 or 2+ ggufs) falls
            // through to null so an Assets-based download or an explicit --model-path resolves it instead.
            if (modality == Modality.Text)
                return ResolveTextDirectory(candidate);
            return Path.GetFullPath(candidate);
        }

        return null;
    }

    /// <summary>A Text model directory: the single .gguf inside it, else the directory itself when it is a Hugging Face checkpoint, else null.</summary>
    /// <exception cref="HartsyInferenceException">The directory holds both a .gguf and a safetensors index, so which one to load is ambiguous.</exception>
    internal static string? ResolveTextDirectory(string directory)
    {
        string[] ggufs = Directory.GetFiles(directory, "*.gguf");
        HfCheckpointInfo? hf = HfCheckpointDirectory.TryProbe(directory);
        if (hf?.IndexPath is not null && ggufs.Length > 0)
        {
            throw new HartsyInferenceException(
                $"'{directory}' holds both a safetensors checkpoint ({Path.GetFileName(hf.IndexPath)}) and {ggufs.Length} "
                + $".gguf file(s); pass --model-path with the .gguf file or with a directory containing only the safetensors checkpoint.");
        }
        if (ggufs.Length == 1)
            return Path.GetFullPath(ggufs[0]);
        return hf is not null && ggufs.Length == 0 ? Path.GetFullPath(directory) : null;
    }
}
