using HartsyInference.Audio.Cache;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.HuggingFace;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Registry;

namespace HartsyInference.Engine.Services;

/// <summary>Resolves a model spec to its weight files and downloads them without loading the model.
///
/// <para>The file list comes from the same descriptor the loader uses, so an install fetches exactly what the
/// first generation will ask for. Families that have not declared a list yet are reported as unsupported
/// rather than silently succeeding — before this existed, installing an engine-managed audio model did nothing
/// at all and still returned success.</para></summary>
public sealed class ModelPrefetchService : IModelPrefetchService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal ModelPrefetchService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public async Task<ModelPrefetchResult> PrefetchAsync(ModelSpec spec, IProgress<AudioFetchProgress>? progress = null,
        CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(spec);
        if (spec.Modality == Modality.Mesh)
        {
            return await PrefetchMeshAsync(spec, progress, cancel).ConfigureAwait(false);
        }
        AudioModelSelector selector = AudioModelSelector.Parse(spec);
        try
        {
            return spec.Modality switch
            {
                Modality.Transcribe => await PrefetchSttAsync(selector, progress, cancel).ConfigureAwait(false),
                Modality.Speech => await PrefetchTtsAsync(selector, progress, cancel).ConfigureAwait(false),
                Modality.Music => await PrefetchMusicAsync(selector, progress, cancel).ConfigureAwait(false),
                _ => ModelPrefetchResult.Unsupported(selector.Id, $"the {spec.Modality} path has no declared weight list yet"),
            };
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][Prefetch] '{selector.Id}:{selector.Variant}' failed: {ex.Message}", ex);
            throw;
        }
    }

    /// <summary>Fetches the catalog files of a 3D model that are absent and fetchable, the primary file last. A file that cannot be fetched (gated, refused, unreachable, corrupt) comes back as a typed blocker and the rest are still fetched; only a cancellation throws.</summary>
    private static async Task<ModelPrefetchResult> PrefetchMeshAsync(ModelSpec spec, IProgress<AudioFetchProgress>? progress, CancellationToken cancel)
    {
        using HuggingFaceClient client = new();
        return await PrefetchMeshAsync(spec, progress, (asset, token) => ModelDownloader.EnsureAsync(client, asset, allowDownload: true, onProgress: null, token), cancel)
            .ConfigureAwait(false);
    }

    /// <summary>The fetch itself, with the per-file download supplied so a test can stand in for the network.</summary>
    internal static async Task<ModelPrefetchResult> PrefetchMeshAsync(
        ModelSpec spec, IProgress<AudioFetchProgress>? progress, Func<ModelAsset, CancellationToken, Task> fetch, CancellationToken cancel)
    {
        MeshAssetService assets = new();
        MeshInventory inventory = assets.GetInventory(spec);
        MeshAssetStatus status = inventory.Status;
        if (status.ModelBlocker is MeshAssetBlocker whole)
        {
            return new ModelPrefetchResult(false, $"'{status.ModelId}' cannot be pre-downloaded: {whole.Message}", []) { Blockers = [whole] };
        }
        CatalogEntry entry = spec.Catalog ?? ModelCatalog.Find(status.ModelId)!;
        ModelAsset? primary = ModelDownloader.PrimaryAsset(entry);
        List<MeshAssetBlocker> blockers = [];
        List<ModelAsset> toFetch = [];
        for (int i = 0; i < status.Assets.Count; i++)
        {
            MeshAssetInfo info = status.Assets[i];
            if (info.Present)
            {
                continue;
            }
            if (info.Blocker is MeshAssetBlockerKind kind)
            {
                blockers.Add(new MeshAssetBlocker(info.Role, kind, info.Instructions ?? string.Empty));
            }
            else if (inventory.Catalog[i] is ModelAsset asset)
            {
                toFetch.Add(asset);
            }
        }
        toFetch = [.. toFetch.Where(asset => asset != primary), .. toFetch.Where(asset => asset == primary)];
        int fetched = 0;
        if (toFetch.Count > 0)
        {
            Logs.Info($"[Mesh][Prefetch] Fetching {toFetch.Count} file(s) for '{status.ModelId}'.");
            foreach (ModelAsset asset in toFetch)
            {
                progress?.Report(new AudioFetchProgress(asset.FileName, fetched, toFetch.Count));
                try
                {
                    await fetch(asset, cancel).ConfigureAwait(false);
                    fetched++;
                }
                catch (OperationCanceledException) when (cancel.IsCancellationRequested)
                {
                    throw;
                }
                catch (Exception ex) when (ex is HartsyInferenceException or HttpRequestException or IOException or InvalidOperationException or OperationCanceledException)
                {
                    Logs.Warning($"[Mesh][Prefetch] '{asset.RepoPath}' of '{status.ModelId}' was not fetched: {ex.Message}");
                    blockers.Add(FetchFailure(asset, ex));
                }
            }
            progress?.Report(new AudioFetchProgress(toFetch[^1].FileName, fetched, toFetch.Count));
        }
        string message = blockers.Count == 0
            ? $"Fetched {fetched} file(s) for '{status.ModelId}'."
            : $"Fetched {fetched} of {toFetch.Count} file(s) for '{status.ModelId}'; {blockers.Count} file(s) need your action: {string.Join("; ", blockers.Select(b => b.Role))}.";
        string? primaryPath = primary is not null && ModelDownloader.TargetPath(primary) is string path && File.Exists(path) ? path : null;
        List<string> present = [.. inventory.Paths.Where(p => p is not null).Select(p => p!)];
        present.AddRange(toFetch.Select(ModelDownloader.TargetPath).Where(File.Exists));
        return new ModelPrefetchResult(true, message, [.. present.Distinct(StringComparer.OrdinalIgnoreCase)], primaryPath) { Blockers = blockers };
    }

    /// <summary>The blocker for a file whose download failed: a 401 or 403 means the repository is gated (or its terms are not accepted yet); anything else is a failed fetch.</summary>
    internal static MeshAssetBlocker FetchFailure(ModelAsset asset, Exception error)
    {
        bool refused = error.Message.Contains("HTTP 401", StringComparison.Ordinal) || error.Message.Contains("HTTP 403", StringComparison.Ordinal)
            || error is HttpRequestException { StatusCode: System.Net.HttpStatusCode.Unauthorized or System.Net.HttpStatusCode.Forbidden };
        // A file fetched from a URL (a release asset) is not behind a Hugging Face gate: a refusal there is a failed fetch.
        if (refused && asset.Url is null)
        {
            return new MeshAssetBlocker(asset.Role, MeshAssetBlockerKind.Gated, MeshAssetService.GatedInstructions(asset.Repo));
        }
        // A fixed sentence per kind of failure: the exception text can hold a server's error page or a local path, and is in the log.
        string reason = error switch
        {
            InvalidOperationException => "the download did not match its pinned hash",
            IOException => "the file could not be written to disk",
            OperationCanceledException => "the server did not answer in time",
            _ => "the server could not be reached or refused the request",
        };
        return new MeshAssetBlocker(asset.Role, MeshAssetBlockerKind.FetchFailed, $"{asset.Url ?? $"{asset.Repo}/{asset.RepoPath}"} was not fetched: {reason}. Check the connection and the disk, then fetch again.");
    }

    private static async Task<ModelPrefetchResult> PrefetchTtsAsync(AudioModelSelector selector,
        IProgress<AudioFetchProgress>? progress, CancellationToken cancel)
    {
        TtsModelDescriptor descriptor = TtsCatalog.Resolve(selector.Id);
        if (descriptor.ResolveFiles is null)
        {
            return ModelPrefetchResult.Unsupported(selector.Id, "its weight list is still implicit in the load path");
        }
        // A voice-selects-weights family (Piper) ships one file per voice, so the variant is the voice.
        string repo = descriptor.ResolveRepo(selector.Variant);
        IReadOnlyList<AudioModelFile> files = await descriptor.ResolveFiles(selector.Variant, cancel).ConfigureAwait(false);
        Logs.Info($"[Audio][Prefetch] Fetching {files.Count} file(s) for '{selector.Id}:{selector.Variant}' from '{repo}'.");
        IReadOnlyDictionary<string, string> fetched = await AudioModelCache
            .FetchAllAsync(repo, files, category: "tts", progress: progress, ct: cancel).ConfigureAwait(false);
        return new ModelPrefetchResult(true, $"Fetched {fetched.Count} file(s) for '{selector.Id}:{selector.Variant}' from '{repo}'.",
            [.. fetched.Values], PrimaryPathOf(files, fetched));
    }

    private static async Task<ModelPrefetchResult> PrefetchMusicAsync(AudioModelSelector selector,
        IProgress<AudioFetchProgress>? progress, CancellationToken cancel)
    {
        MusicModelDescriptor descriptor = MusicCatalog.Resolve(selector.Id);
        if (!descriptor.ManagesOwnWeights)
        {
            return ModelPrefetchResult.Unsupported(selector.Id, "it loads a checkpoint you place yourself");
        }
        if (descriptor.ResolveFiles is null)
        {
            return ModelPrefetchResult.Unsupported(selector.Id, "its weight list is still implicit in the load path");
        }
        string repo = descriptor.CacheKey(selector);
        IReadOnlyList<AudioModelFile> files = await descriptor.ResolveFiles(selector, cancel).ConfigureAwait(false);
        Logs.Info($"[Audio][Prefetch] Fetching {files.Count} file(s) for '{selector.Id}:{selector.Variant}' from '{repo}'.");
        IReadOnlyDictionary<string, string> fetched = await AudioModelCache
            .FetchAllAsync(repo, files, category: "music", progress: progress, ct: cancel).ConfigureAwait(false);
        return new ModelPrefetchResult(true, $"Fetched {fetched.Count} file(s) for '{selector.Id}:{selector.Variant}' from '{repo}'.",
            [.. fetched.Values], PrimaryPathOf(files, fetched));
    }

    /// <summary>The last required file's resolved path — the same one <see cref="AudioModelCache.FetchAllAsync"/>
    /// deliberately fetches last, so "the primary landed" and "everything landed" mean the same thing.</summary>
    private static string? PrimaryPathOf(IReadOnlyList<AudioModelFile> files, IReadOnlyDictionary<string, string> fetched)
    {
        for (int i = files.Count - 1; i >= 0; i--)
        {
            if (files[i].Required && fetched.TryGetValue(files[i].Name, out string? path))
            {
                return path;
            }
        }
        return null;
    }

    private static async Task<ModelPrefetchResult> PrefetchSttAsync(AudioModelSelector selector,
        IProgress<AudioFetchProgress>? progress, CancellationToken cancel)
    {
        SttModelDescriptor descriptor = SttCatalog.Resolve(selector.Id);
        if (descriptor.ResolveFiles is null)
        {
            return ModelPrefetchResult.Unsupported(selector.Id, "its weight list is still implicit in the load path");
        }
        string repo = descriptor.ResolveRepo(selector.Variant);
        IReadOnlyList<AudioModelFile> files = await descriptor.ResolveFiles(selector.Variant, cancel).ConfigureAwait(false);
        Logs.Info($"[Audio][Prefetch] Fetching {files.Count} file(s) for '{selector.Id}:{selector.Variant}' from '{repo}'.");
        IReadOnlyDictionary<string, string> fetched = await AudioModelCache
            .FetchAllAsync(repo, files, category: "stt", progress: progress, ct: cancel).ConfigureAwait(false);
        return new ModelPrefetchResult(true, $"Fetched {fetched.Count} file(s) for '{selector.Id}:{selector.Variant}' from '{repo}'.",
            [.. fetched.Values], PrimaryPathOf(files, fetched));
    }
}
