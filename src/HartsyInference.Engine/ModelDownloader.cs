using System.Collections.Concurrent;
using HartsyInference.Engine.HuggingFace;

namespace HartsyInference.Engine;

/// <summary>Fetches a catalog model's preset asset set (transformer + text encoder + VAE + …) from HuggingFace into the
/// correct models-root folders, so a selected model is always runnable. The mechanism only — the confirm prompt and
/// progress rendering belong to the caller. Concurrent requests for the same asset serialize on a per-target lock so
/// two loads never race the same file; each download stages atomically and (when a hash is registered) SHA-256-verifies
/// before it appears at its canonical path.</summary>
public static class ModelDownloader
{
    /// <summary>One gate per target path — the async equivalent of the extension's per-canonical-name lock set.</summary>
    private static readonly ConcurrentDictionary<string, SemaphoreSlim> _gates = new(StringComparer.OrdinalIgnoreCase);

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

    /// <summary>Downloads <paramref name="assets"/> to their target paths, reporting per-file progress (0..1). Each
    /// asset is fetched under its own lock, skipped if already present, and SHA-256-verified when a hash is set.</summary>
    public static async Task DownloadAsync(IReadOnlyList<ModelAsset> assets, Action<ModelAsset, double>? onProgress, CancellationToken ct)
    {
        using HuggingFaceClient client = new HuggingFaceClient();
        foreach (ModelAsset asset in assets)
            await EnsureAsync(client, asset, allowDownload: true, onProgress, ct).ConfigureAwait(false);
    }

    /// <summary>Ensures a single side model is on disk (downloading + verifying under its lock) and returns its local
    /// path. The default behavior is strict: if the file is not present, this throws before any network call.
    /// Extension or caller flows that explicitly opt in can pass <c>downloadIfMissing: true</c>.</summary>
    public static async Task<string> EnsureSideModelAsync(ModelAsset asset, Action<ModelAsset, double>? onProgress, CancellationToken ct)
    {
        return await EnsureSideModelAsync(asset, downloadIfMissing: false, onProgress, ct).ConfigureAwait(false);
    }

    /// <summary>Ensures a single side model is on disk and returns its local path.
    /// Set <paramref name="downloadIfMissing"/> to <c>true</c> to download it from HuggingFace when missing.</summary>
    public static async Task<string> EnsureSideModelAsync(ModelAsset asset, bool downloadIfMissing, Action<ModelAsset, double>? onProgress, CancellationToken ct)
    {
        using HuggingFaceClient client = new HuggingFaceClient();
        await EnsureAsync(client, asset, downloadIfMissing, onProgress, ct).ConfigureAwait(false);
        return TargetPath(asset);
    }

    /// <summary>Ensures a single <paramref name="asset"/> is present, downloading it if needed under a per-target lock.</summary>
    public static async Task EnsureAsync(HuggingFaceClient client, ModelAsset asset, bool allowDownload, Action<ModelAsset, double>? onProgress, CancellationToken ct)
    {
        string target = TargetPath(asset);
        SemaphoreSlim gate = _gates.GetOrAdd(target, _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            // Re-check under the lock: a concurrent request may have finished the download while we waited.
            if (File.Exists(target))
                return;

            if (!allowDownload)
            {
                throw new FileNotFoundException(
                    $"Model asset '{asset.Role}' is missing locally: {target}. Download it first with the catalog pull workflow "
                    + $"for '{asset.Repo}'.");
            }

            IProgress<double> progress = new Progress<double>(fraction => onProgress?.Invoke(asset, fraction));
            await client.DownloadFileAsync(asset.Repo, asset.RepoPath, target, progress, asset.Sha256, ct).ConfigureAwait(false);
            if (!string.IsNullOrWhiteSpace(asset.Sha256))
            {
                // DownloadFileAsync has already streamed and verified these exact bytes. Persisting that result here
                // keeps the first profile-aware request from reading a multi-gigabyte checkpoint a second time.
                await Planning.VideoCheckpointHashCache.RecordVerifiedSha256Async(target, asset.Sha256)
                    .ConfigureAwait(false);
            }
        }
        finally
        {
            gate.Release();
        }
    }
}
