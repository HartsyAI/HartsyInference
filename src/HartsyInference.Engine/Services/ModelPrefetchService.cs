using HartsyInference.Audio.Cache;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;

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
