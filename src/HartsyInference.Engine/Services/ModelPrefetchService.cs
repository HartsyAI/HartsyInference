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
            [.. fetched.Values]);
    }
}
