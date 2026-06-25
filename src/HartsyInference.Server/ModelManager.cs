using System.Collections.Concurrent;
using HartsyInference.Core.Backends;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelHandler.Registry;
using HartsyInference.Tokenizers;

namespace HartsyInference.Server;

/// <summary>Owns the server's loaded diffusion pipelines and the shared backend/tokenizer. Bridges the
/// HTTP layer to <see cref="PipelineFactory"/> (architecture detection + construction) and
/// <see cref="ModelRegistry.LoadAsync"/> (HuggingFace download + cache). Pipelines are cached by id and
/// reused across requests.</summary>
public sealed class ModelManager : IDisposable
{
    private readonly IBackend _backend;
    private readonly ModelRegistry _registry;
    private readonly ModelCacheStore _cache;
    private readonly ClipTokenizer _clipTokenizer = new ClipTokenizer(); // embedded OpenAI CLIP vocab/merges
    private readonly ConcurrentDictionary<string, DiffusionPipelineBase> _pipelines = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Creates the manager over a compute backend.</summary>
    public ModelManager(IBackend backend, HartsyInferenceServerOptions options)
    {
        _backend = backend;
        _registry = new ModelRegistry();
        _cache = options.ModelCacheDirectory is null ? new ModelCacheStore() : new ModelCacheStore(options.ModelCacheDirectory);
    }

    /// <summary>Currently loaded model ids.</summary>
    public IReadOnlyCollection<string> LoadedModels => _pipelines.Keys.ToArray();

    /// <summary>True if <paramref name="modelId"/> is loaded.</summary>
    public bool IsLoaded(string modelId) => _pipelines.ContainsKey(modelId);

    /// <summary>Loads a model from a local path or HuggingFace repo id and caches the constructed pipeline.
    /// Returns the architecture that was detected.</summary>
    public async Task<string> LoadAsync(string modelIdOrPath, CancellationToken ct)
    {
        if (_pipelines.ContainsKey(modelIdOrPath)) return "already-loaded";

        // Resolve to a local checkpoint path (downloading from HuggingFace if needed).
        string localPath;
        if (File.Exists(modelIdOrPath) || Directory.Exists(modelIdOrPath))
        {
            localPath = modelIdOrPath;
        }
        else
        {
            LoadedModel loaded = await _registry.LoadAsync(modelIdOrPath, _cache, ct: ct).ConfigureAwait(false);
            localPath = loaded.Info.LocalPath;
            // Free the registry's mmap copy — PipelineFactory re-reads + converts the file itself.
            _registry.Unload(modelIdOrPath);
        }

        ModelArchitecture arch = PipelineFactory.DetectArchitecture(localPath);
        DiffusionPipelineBase pipeline = PipelineFactory.LoadAuto(localPath, _backend);
        _pipelines[modelIdOrPath] = pipeline;
        return arch.ToString();
    }

    /// <summary>Unloads and disposes a cached pipeline.</summary>
    public bool Unload(string modelId)
    {
        if (_pipelines.TryRemove(modelId, out DiffusionPipelineBase? pipeline))
        {
            pipeline.Dispose();
            return true;
        }
        return false;
    }

    /// <summary>Generates one image from a text prompt using a loaded SDXL pipeline. Other architectures
    /// throw <see cref="NotSupportedException"/> (server image generation currently covers the SDXL path
    /// that <see cref="PipelineFactory"/> constructs end-to-end).</summary>
    public (byte[] rgb, int width, int height, int seed) GenerateImage(
        string modelId, ImageGenerationRequest req, Action<GenerationProgress>? onProgress = null)
    {
        if (!_pipelines.TryGetValue(modelId, out DiffusionPipelineBase? pipeline))
            throw new InvalidOperationException($"Model '{modelId}' is not loaded. POST /v1/models/load first.");

        if (pipeline is not SdxlPipeline sdxl)
            throw new NotSupportedException($"Image generation over HTTP currently supports SDXL pipelines; '{modelId}' is {pipeline.GetType().Name}.");

        (int width, int height) = ParseSize(req.Size);
        string prompt = req.Prompt;
        string negative = req.NegativePrompt ?? "";

        int[] tokensL = _clipTokenizer.Encode(prompt);
        int[] negL = _clipTokenizer.Encode(negative);
        int[] tokensG = _clipTokenizer.Encode(prompt);
        int[] negG = _clipTokenizer.Encode(negative);
        int eosG = ClipTokenizer.FindEosPosition(tokensG);
        int negEosG = ClipTokenizer.FindEosPosition(negG);

        TextToImageRequest request = new TextToImageRequest
        {
            Prompt = prompt,
            NegativePrompt = negative,
            Width = width,
            Height = height,
            Steps = req.Steps,
            CfgScale = req.CfgScale,
            Seed = req.Seed < 0 ? null : (int)req.Seed,
            ClipSkip = req.ClipSkip,
        };

        return sdxl.GenerateFromTokens(tokensL, negL, tokensG, negG, eosG, negEosG, request, onProgress);
    }

    private static (int width, int height) ParseSize(string? size)
    {
        if (string.IsNullOrWhiteSpace(size)) return (1024, 1024);
        string[] parts = size.Split('x', 'X');
        if (parts.Length == 2 && int.TryParse(parts[0], out int w) && int.TryParse(parts[1], out int h))
            return (w, h);
        return (1024, 1024);
    }

    public void Dispose()
    {
        foreach (DiffusionPipelineBase p in _pipelines.Values) p.Dispose();
        _pipelines.Clear();
        _registry.Dispose();
    }
}
