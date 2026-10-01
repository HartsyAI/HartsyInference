using System.Collections.Concurrent;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.IO;
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

    /// <summary>The on-disk path an asset resolves to under the models root: the file that exists, else where a
    /// download should put it. Folders spelled in another case on a case-sensitive filesystem (SwarmUI's <c>llm/</c>
    /// for the catalog's <c>LLM/</c>) are matched only after every exact spelling missed, so a file found exactly is
    /// always the one returned, and a download lands in the folder that already exists instead of a second one.</summary>
    public static string TargetPath(ModelAsset asset) =>
        FindExisting(asset, ignoreCase: false) ?? FindExisting(asset, ignoreCase: true)
            ?? CaseInsensitivePath.ResolveFile(RepoPaths.ModelsRoot(), Path.Combine(asset.TargetSubdir, asset.FileName));

    /// <summary>The asset's file when it is on disk under its canonical name or a legacy name, in that order, each
    /// spelled exactly or, with <paramref name="ignoreCase"/>, matched ignoring case; null when none exists.</summary>
    internal static string? FindExisting(ModelAsset asset, bool ignoreCase)
    {
        string root = RepoPaths.ModelsRoot();
        string? found = Existing(root, Path.Combine(asset.TargetSubdir, asset.FileName), ignoreCase);
        // The canonical name moved (usually to match what SwarmUI downloads) and this install still holds the
        // file under the old one. Resolving to it beats re-fetching multi-gigabyte bytes we already have, and
        // keeps the strict no-download callers (MageFlowRecipe and friends) from reporting it missing.
        for (int i = 0; found is null && i < asset.LegacyTargetNames.Count; i++)
        {
            found = Existing(root, Path.Combine(asset.TargetSubdir, asset.LegacyTargetNames[i]), ignoreCase);
        }
        return found;
    }

    /// <summary>The subset of <paramref name="entry"/>'s assets not already present on disk.</summary>
    public static IReadOnlyList<ModelAsset> MissingAssets(CatalogEntry entry) =>
        entry.Assets.Where(a => !File.Exists(TargetPath(a))).ToList();

    /// <summary>The local path of the model's primary file (its transformer/checkpoint), or null when the entry has no
    /// assets. Used as the resolved <c>LocalPath</c> once the set is present.</summary>
    public static string? PrimaryLocalPath(CatalogEntry entry) =>
        PrimaryAsset(entry) is ModelAsset primary ? TargetPath(primary) : null;

    /// <summary>The model's primary asset (its transformer/checkpoint), or null when the entry has no assets.</summary>
    internal static ModelAsset? PrimaryAsset(CatalogEntry entry) =>
        entry.Assets.FirstOrDefault(a => a.Role == "transformer") ?? (entry.Assets.Count > 0 ? entry.Assets[0] : null);

    /// <summary>Downloads <paramref name="assets"/> to their target paths, reporting per-file progress (0..1). Each
    /// asset is fetched under its own lock, skipped if already present, and SHA-256-verified when a hash is set.</summary>
    public static async Task DownloadAsync(IReadOnlyList<ModelAsset> assets, Action<ModelAsset, double>? onProgress, CancellationToken ct)
    {
        using HuggingFaceClient client = new HuggingFaceClient();
        foreach (ModelAsset asset in assets)
            await EnsureAsync(client, asset, allowDownload: true, onProgress, ct).ConfigureAwait(false);
    }

    /// <summary>Ensures a single side model is on disk (downloading + verifying under its lock) and returns its local
    /// path. A missing asset is fetched unless <c>paths.sideModelAutofetch</c> is off; a caller that must never reach
    /// the network regardless passes <c>downloadIfMissing: false</c> explicitly.</summary>
    public static async Task<string> EnsureSideModelAsync(ModelAsset asset, Action<ModelAsset, double>? onProgress, CancellationToken ct)
    {
        return await EnsureSideModelAsync(asset, EngineKnobs.SideModelAutofetch.Value, onProgress, ct).ConfigureAwait(false);
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
            string audioRoot = Audio.AudioModelRoot.Location();
            // Ignoring case: on Windows and macOS the target keeps the catalog's Audio/ spelling for the same folder.
            if (target.StartsWith(audioRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            {
                // A converted checkpoint may stand in for this asset; linking it beats downloading the original.
                HartsyInference.Audio.Cache.AudioStandIns.Resync(audioRoot);
                if (File.Exists(target))
                    return;
            }

            if (!allowDownload)
            {
                // allowDownload false can come from the setting or from a caller that must not reach the network;
                // saying the wrong one sends the operator to flip a setting that would change nothing.
                string why = EngineKnobs.SideModelAutofetch.Value
                    ? "this caller does not download"
                    : "paths.sideModelAutofetch is off";
                throw new FileNotFoundException(
                    $"Model asset '{asset.Role}' is missing locally: {target}, and fetching it is disabled ({why}). "
                    + $"Download it from '{asset.Repo}' ({asset.RepoPath}).");
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

    private static string? Existing(string root, string relativePath, bool ignoreCase)
    {
        string path = ignoreCase
            ? CaseInsensitivePath.ResolveFile(root, relativePath)
            : Path.Combine(root, relativePath);
        return File.Exists(path) ? path : null;
    }
}
