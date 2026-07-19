using HartsyInference.Engine.HuggingFace;

namespace HartsyInference.Engine;

/// <summary>Fetches a catalog model's preset asset set (transformer + text encoder + VAE + …) from HuggingFace into the
/// correct models-root folders, so a selected model is always runnable. The mechanism only — the confirm prompt and
/// progress rendering belong to the caller.</summary>
public static class ModelDownloader
{
    /// <summary>The on-disk path an asset resolves to under the models root.</summary>
    public static string TargetPath(ModelAsset asset) =>
        Path.Combine(RepoPaths.ModelsRoot(), asset.TargetSubdir, asset.FileName);

    /// <summary>The subset of <paramref name="entry"/>'s assets not already present on disk.</summary>
    public static IReadOnlyList<ModelAsset> MissingAssets(CatalogEntry entry) =>
        entry.Assets.Where(a => !File.Exists(TargetPath(a))).ToList();

    /// <summary>The local path of the model's primary file (its transformer/checkpoint), or null when the entry has no
    /// assets. Used as the resolved <c>LocalPath</c> once the set is present.</summary>
    public static string? PrimaryLocalPath(CatalogEntry entry)
    {
        ModelAsset? primary = entry.Assets.FirstOrDefault(a => a.Role == "transformer")
            ?? (entry.Assets.Count > 0 ? entry.Assets[0] : null);
        return primary is null ? null : TargetPath(primary);
    }

    /// <summary>Downloads <paramref name="assets"/> to their target paths, reporting per-file progress (0..1).</summary>
    public static async Task DownloadAsync(IReadOnlyList<ModelAsset> assets, Action<ModelAsset, double>? onProgress, CancellationToken ct)
    {
        using HuggingFaceClient client = new HuggingFaceClient();
        foreach (ModelAsset asset in assets)
        {
            IProgress<double> p = new Progress<double>(fraction => onProgress?.Invoke(asset, fraction));
            await client.DownloadFileAsync(asset.Repo, asset.RepoPath, TargetPath(asset), p, ct).ConfigureAwait(false);
        }
    }
}
