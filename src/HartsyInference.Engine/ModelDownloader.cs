using System.Collections.Concurrent;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.IO;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.HuggingFace;
using HartsyInference.ModelAssets.PyTorch;

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
            ?? CaseInsensitivePath.ResolveFile(RepoPaths.ModelsRoot(), Path.Combine(ModelFolders.MapSubdir(asset.TargetSubdir), asset.FileName));

    /// <summary>The asset's file when it is on disk under its canonical name or a legacy name, in that order, each
    /// spelled exactly or, with <paramref name="ignoreCase"/>, matched ignoring case; null when none exists.</summary>
    internal static string? FindExisting(ModelAsset asset, bool ignoreCase)
    {
        string root = RepoPaths.ModelsRoot();
        string? found = ExistingInFolders(root, asset.TargetSubdir, asset.FileName, ignoreCase);
        // The canonical name moved (usually to match what SwarmUI downloads) and this install still holds the
        // file under the old one. Resolving to it beats re-fetching multi-gigabyte bytes we already have, and
        // keeps the strict no-download callers (MageFlowRecipe and friends) from reporting it missing.
        for (int i = 0; found is null && i < asset.LegacyTargetNames.Count; i++)
        {
            found = ExistingInFolders(root, asset.TargetSubdir, asset.LegacyTargetNames[i], ignoreCase);
        }
        return found;
    }

    /// <summary>The safetensors repack of an asset that lists one, when it is on disk beside where the pickle goes (same name, extension <c>.safetensors</c>); null for an asset with no repack or whose repack is not there.</summary>
    internal static string? FindRepacked(ModelAsset asset)
    {
        if (asset.Repack is null)
            return null;
        ModelAsset repacked = asset with { TargetName = Path.ChangeExtension(asset.FileName, RepackedCheckpoint.Extension), LegacyTargetNames = [] };
        return FindExisting(repacked, ignoreCase: false) ?? FindExisting(repacked, ignoreCase: true);
    }

    /// <summary>Where a repack of an asset is saved: beside <see cref="TargetPath"/>.</summary>
    internal static string RepackedTargetPath(ModelAsset asset) => RepackedCheckpoint.SiblingOf(TargetPath(asset));

    /// <summary>The file under each spelling of the asset's folder, in the order <see cref="ModelFolders.Spellings"/> gives.</summary>
    private static string? ExistingInFolders(string root, string targetSubdir, string fileName, bool ignoreCase)
    {
        foreach (string folder in ModelFolders.Spellings(targetSubdir))
        {
            string? found = Existing(root, Path.Combine(folder, fileName), ignoreCase);
            if (found is not null)
                return found;
        }
        return null;
    }

    /// <summary>The subset of <paramref name="entry"/>'s assets not already present on disk.</summary>
    public static IReadOnlyList<ModelAsset> MissingAssets(CatalogEntry entry) => MissingAssets(entry, null);

    /// <summary>
    /// The subset of <paramref name="entry"/>'s assets present neither at their catalog path nor, for an entry whose loader reads <see cref="CatalogEntry.LoadsFromFolder"/>, in
    /// <paramref name="checkpointFolder"/> (an existing folder named with <c>--model-path</c>) under the asset's exact file name: such a folder is the install, wherever the catalog would
    /// have put it. The folder is ignored for every other entry, whose loaders look their side files up under the catalog folders. The match is by the catalog's file name only (no
    /// <c>.ckpt</c> spelling, no <see cref="ModelAsset.LegacyTargetNames"/>), so a folder of other spellings is still asked about.
    /// </summary>
    public static IReadOnlyList<ModelAsset> MissingAssets(CatalogEntry entry, string? checkpointFolder)
    {
        string? folder = entry.LoadsFromFolder && checkpointFolder is not null && Directory.Exists(checkpointFolder) ? checkpointFolder : null;
        return entry.Assets
            .Where(a => !(folder is not null && File.Exists(Path.Combine(folder, a.FileName))) && !File.Exists(TargetPath(a)) && FindRepacked(a) is null)
            .ToList();
    }

    /// <summary>The local path of the model's primary file (its transformer/checkpoint), or null when the entry has no
    /// assets. Used as the resolved <c>LocalPath</c> once the set is present.</summary>
    public static string? PrimaryLocalPath(CatalogEntry entry) =>
        PrimaryAsset(entry) is ModelAsset primary ? LocalPath(primary) : null;

    /// <summary>The file a loader reads for <paramref name="asset"/>: where it is on disk (<see cref="TargetPath"/>), else its repack beside that when the asset lists one and only the repack was fetched, else where a download would put it.</summary>
    internal static string LocalPath(ModelAsset asset)
    {
        string target = TargetPath(asset);
        return File.Exists(target) ? target : FindRepacked(asset) ?? target;
    }

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

    /// <summary>Ensures a single <paramref name="asset"/> is present, downloading it if needed under a per-target lock. An asset that lists a repack is satisfied by the repack, and fetches it from the mirror
    /// (<c>paths.weightsMirrorUrl</c>) before its upstream file; an asset with a <see cref="ModelAsset.Url"/> is fetched from that URL.</summary>
    public static Task EnsureAsync(HuggingFaceClient client, ModelAsset asset, bool allowDownload, Action<ModelAsset, double>? onProgress, CancellationToken ct) =>
        EnsureAsync(client, asset, allowDownload, onProgress, null, EngineKnobs.WeightsMirrorUrl.Value, ct);

    /// <summary>The fetch itself, with the HTTP client for mirror and URL downloads and the mirror's base URL supplied so a test can stand in for the network. A null <paramref name="assetHttp"/> makes a client without credentials when one is needed.</summary>
    internal static async Task EnsureAsync(HuggingFaceClient client, ModelAsset asset, bool allowDownload, Action<ModelAsset, double>? onProgress,
        HttpClient? assetHttp, string? mirrorBase, CancellationToken ct)
    {
        string target = TargetPath(asset);
        SemaphoreSlim gate = _gates.GetOrAdd(target, _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            // Re-check under the lock: a concurrent request may have finished the download while we waited.
            if (File.Exists(target) || FindRepacked(asset) is not null)
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
            HttpClient? owned = null;
            try
            {
                if (AssetUrlDownload.MirrorUrl(asset, mirrorBase) is string mirrorUrl)
                {
                    HttpClient http = assetHttp ?? (owned = NewAssetClient());
                    string repacked = RepackedTargetPath(asset);
                    try
                    {
                        await AssetUrlDownload.DownloadAsync(http, mirrorUrl, $"{asset.Repo}/{asset.RepoPath} (mirror repack)", repacked, progress, asset.Repack!.Sha256, ct).ConfigureAwait(false);
                        await Planning.VideoCheckpointHashCache.RecordVerifiedSha256Async(repacked, asset.Repack.Sha256).ConfigureAwait(false);
                        return;
                    }
                    catch (Exception ex) when (ex is HttpRequestException or InvalidOperationException or IOException or ArgumentException
                        || (ex is OperationCanceledException && !ct.IsCancellationRequested))
                    {
                        Logs.Warning($"[Download] The mirror did not give '{asset.RepoPath}' ({ex.GetType().Name}: {ex.Message}); fetching the upstream file instead.");
                    }
                }
                if (asset.Url is not null)
                {
                    await AssetUrlDownload.DownloadAsync(assetHttp ?? (owned = NewAssetClient()), asset.Url, $"{asset.Repo}/{asset.RepoPath}", target, progress, asset.Sha256, ct).ConfigureAwait(false);
                }
                else
                {
                    await client.DownloadFileAsync(asset.Repo, asset.RepoPath, target, progress, asset.Sha256, ct).ConfigureAwait(false);
                }
            }
            finally
            {
                owned?.Dispose();
            }
            if (!string.IsNullOrWhiteSpace(asset.Sha256))
            {
                // The download has already streamed and verified these exact bytes. Persisting that result here
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

    /// <summary>An HTTP client for mirror and release-asset downloads. It carries no credentials: the Hugging Face token is not sent to another host.</summary>
    private static HttpClient NewAssetClient()
    {
        HttpClient http = new();
        http.DefaultRequestHeaders.UserAgent.ParseAdd("HartsyInference/1.0");
        return http;
    }

    /// <summary>The variant of <paramref name="entry"/> named <paramref name="variantId"/> (case-insensitive).</summary>
    /// <exception cref="ArgumentException">The entry has no such variant; the message lists the ones it has.</exception>
    public static CatalogVariant FindVariant(CatalogEntry entry, string variantId)
    {
        ArgumentNullException.ThrowIfNull(entry);
        return entry.Variants.FirstOrDefault(v => string.Equals(v.Id, variantId, StringComparison.OrdinalIgnoreCase))
            ?? throw new ArgumentException(
                $"'{entry.Id}' has no variant '{variantId}'. Variants: {string.Join(", ", entry.Variants.Select(v => v.Id))}.", nameof(variantId));
    }

    /// <summary>The files a folder variant downloads (its index, weight shards and support files), each pinned to the variant's
    /// revision and carrying the hash the set lists for it. A variant without an index set, such as a single GGUF, is the entry's
    /// default asset set instead (<see cref="CatalogEntry.Assets"/>).</summary>
    public static IReadOnlyList<ModelAsset> VariantAssets(CatalogEntry entry, string variantId)
    {
        CatalogShardSet set = FindVariant(entry, variantId).Source;
        if (set.IndexFile is null && set.SupportFiles.Count == 0)
            return entry.Assets;
        return [.. set.AllFiles().Select(file => new ModelAsset
        {
            Repo = set.Repo, RepoPath = file, TargetSubdir = set.TargetSubdir,
            Role = file.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) ? "transformer" : "variant file",
            Sha256 = set.Sha256ByFile.TryGetValue(file, out string? sha) ? sha : null,
        })];
    }

    /// <summary>The folder a downloaded folder variant lives in (where a loader is pointed), or null for a variant that is a single
    /// file (use <see cref="PrimaryLocalPath"/>) or whose index is not on disk yet.</summary>
    public static string? VariantFolder(CatalogEntry entry, string variantId)
    {
        CatalogShardSet set = FindVariant(entry, variantId).Source;
        if (set.IndexFile is null) return null;
        ModelAsset? index = VariantAssets(entry, variantId).FirstOrDefault(a => a.RepoPath == set.IndexFile);
        string? path = index is null ? null : TargetPath(index);
        return path is not null && File.Exists(path) ? Path.GetDirectoryName(path) : null;
    }

    /// <summary>Downloads <paramref name="variantId"/> of <paramref name="entry"/>: a folder variant's files at the variant's pinned
    /// revision into its own folder, a single-file variant the entry's default set. Files already present are kept; each download
    /// is staged to a <c>.tmp</c> file, SHA-256-verified when the set lists a hash, and moved into place under a per-target lock.
    /// When it finishes, every shard the index names is checked to exist.</summary>
    /// <param name="onProgress">Called with each file and its 0..1 progress.</param>
    /// <exception cref="InvalidDataException">The index names a shard that is not on disk after the download.</exception>
    public static async Task DownloadVariantAsync(CatalogEntry entry, string variantId, Action<ModelAsset, double>? onProgress,
        CancellationToken ct)
    {
        CatalogShardSet set = FindVariant(entry, variantId).Source;
        if (set.IndexFile is null && set.SupportFiles.Count == 0)
        {
            await DownloadAsync(MissingAssets(entry), onProgress, ct).ConfigureAwait(false);
            return;
        }
        using HuggingFaceClient client = new();
        foreach (ModelAsset asset in VariantAssets(entry, variantId))
            await EnsureRevisionAsync(client, asset, set.Revision, onProgress, ct).ConfigureAwait(false);
        VerifyIndexedShards(entry, variantId);
    }

    /// <summary>Downloads one file at a pinned revision under the per-target lock, unless it is already present.</summary>
    private static async Task EnsureRevisionAsync(HuggingFaceClient client, ModelAsset asset, string revision,
        Action<ModelAsset, double>? onProgress, CancellationToken ct)
    {
        string target = TargetPath(asset);
        SemaphoreSlim gate = _gates.GetOrAdd(target, _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (File.Exists(target)) return;
            IProgress<double> progress = new Progress<double>(fraction => onProgress?.Invoke(asset, fraction));
            await client.DownloadRevisionAsync(asset.Repo, asset.RepoPath, revision, target, progress, asset.Sha256, ct).ConfigureAwait(false);
        }
        finally
        {
            gate.Release();
        }
    }

    /// <summary>Checks that every file the folder variant's safetensors index names is on disk.</summary>
    private static void VerifyIndexedShards(CatalogEntry entry, string variantId)
    {
        string? folder = VariantFolder(entry, variantId);
        string? indexFile = FindVariant(entry, variantId).Source.IndexFile;
        if (folder is null || indexFile is null) return;
        using System.Text.Json.JsonDocument doc = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(Path.Combine(folder, indexFile)));
        if (!doc.RootElement.TryGetProperty("weight_map", out System.Text.Json.JsonElement map)) return;
        foreach (string shard in map.EnumerateObject().Select(p => p.Value.GetString()!).Distinct(StringComparer.Ordinal))
        {
            if (!File.Exists(Path.Combine(folder, shard)))
                throw new InvalidDataException($"The index of '{entry.Id}' variant '{variantId}' names '{shard}', which is not in '{folder}'.");
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
