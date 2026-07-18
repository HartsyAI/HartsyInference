using HartsyInference.Engine.Requests;
using System.Collections.Concurrent;
using HartsyInference.Core.Backends;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Ssm;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelHandler.Gguf;
using HartsyInference.ModelHandler.Gguf.KeyMappers;
using HartsyInference.Engine.Registry;
using HartsyInference.ModelHandler.Registry;
using HartsyInference.Tokenizers;

namespace HartsyInference.Engine;

/// <summary>Owns the server's loaded diffusion pipelines and LLM/SSM generation pipelines, plus the shared
/// backend/tokenizer. Bridges the HTTP layer to <see cref="PipelineFactory"/> (diffusion architecture
/// detection + construction), the LLM/SSM loaders (<see cref="GgufLanguageModel"/>/<see cref="SsmLanguageModel"/>,
/// mirroring the exact pattern the SwarmUI-LLMAssistant extension's local provider uses), and
/// <see cref="ModelRegistry.LoadAsync"/> (HuggingFace download + cache). Pipelines are cached by id and
/// reused across requests.</summary>
public sealed class ModelManager : IDisposable
{
    private readonly IBackend _backend;
    private readonly EngineOptions _options;
    private readonly InferenceQueue _queue;
    private readonly ModelRegistry _registry;
    private readonly ModelCacheStore _cache;
    private readonly ClipTokenizer _clipTokenizer = new ClipTokenizer(); // embedded OpenAI CLIP vocab/merges
    private readonly ConcurrentDictionary<string, DiffusionPipelineBase> _pipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly ConcurrentDictionary<string, LlmSession> _llmSessions = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>A loaded chat model — either a dense/MoE transformer (Pool+Scheduler) or an SSM (SsmPipeline,
    /// no batching — out of scope for the paged-cache/continuous-batching rework, see
    /// docs/Checklists/LLM_DECODE_PERF_GRIND.md Phase 3/4 notes), exactly one pair populated.</summary>
    private sealed class LlmSession : IDisposable
    {
        public GgufLanguageModel? Model;
        public PagedKvPool? Pool;
        public DynamicBatchScheduler? Scheduler;
        public SsmLanguageModel? SsmModel;
        public SsmGenerationPipeline? SsmPipeline;
        public void Dispose() { Scheduler?.Dispose(); Pool?.Dispose(); Model?.Dispose(); SsmModel?.Dispose(); }
    }

    /// <summary>Creates the manager over a compute backend. <paramref name="queue"/> is the SAME
    /// <see cref="InferenceQueue"/> instance diffusion image generation uses — every chat model's
    /// <see cref="DynamicBatchScheduler"/> routes its GPU-touching work through it too, so the whole server
    /// never runs more than one physical GPU operation at a time on the shared backend (see
    /// <see cref="DynamicBatchScheduler"/>'s "Backend exclusivity" doc), while still batching every
    /// concurrently-submitted chat request into that one operation.</summary>
    public ModelManager(IBackend backend, EngineOptions options, InferenceQueue queue)
    {
        _backend = backend;
        _options = options;
        _queue = queue;
        _registry = new ModelRegistry();
        _cache = options.ModelCacheDirectory is null ? new ModelCacheStore() : new ModelCacheStore(options.ModelCacheDirectory);
    }

    /// <summary>Currently loaded model ids (diffusion + LLM/SSM).</summary>
    public IReadOnlyCollection<string> LoadedModels => [.. _pipelines.Keys, .. _llmSessions.Keys];

    /// <summary>True if <paramref name="modelId"/> is loaded (diffusion or LLM/SSM).</summary>
    public bool IsLoaded(string modelId) => _pipelines.ContainsKey(modelId) || _llmSessions.ContainsKey(modelId);

    /// <summary>True if <paramref name="modelId"/> is a loaded LLM/SSM chat model (as opposed to diffusion).</summary>
    public bool IsChatModel(string modelId) => _llmSessions.ContainsKey(modelId);

    /// <summary>Model ids whose serving loop has died (<see cref="DynamicBatchScheduler.IsLoopAlive"/> is
    /// false) despite the model still being "loaded" — every request against one of these will hang forever
    /// (the scheduler's background loop is what drains new admissions; once it's gone, nothing ever will).
    /// Its own decode-round/admission containment (see that class's doc) means this should normally stay
    /// empty even after individual request failures — a non-empty result here means a specific model's chat
    /// traffic needs a reload, not just "some requests are failing." SSM sessions have no loop to die, so
    /// they're never reported here. Intended for a readiness probe, not a per-request check.</summary>
    public IReadOnlyCollection<string> UnhealthyChatModels =>
        [.. _llmSessions.Where(kv => kv.Value.Scheduler is { IsLoopAlive: false }).Select(kv => kv.Key)];

    // Diffusion mappers registered in GgufKeyMapperRegistry.BuildRegistry() — a small, closed list that
    // changes far less often than the LLM side. Used to derive "is this GGUF architecture a language model"
    // WITHOUT loading any tensor data: GetByArchitecture(arch) is a metadata-only dictionary lookup, and if
    // the matched mapper is one of these, the file is diffusion, not LLM.
    //
    // This must be a real upfront check, not a "try the LLM loader, catch and fall back" strategy — that
    // was tried first and caused a real incident: GgufLanguageModel.Load's expensive step (GgufModelLoader
    // reading + dequantizing every tensor to F32/F16) runs BEFORE GgufConfigFactory's cheap architecture
    // validation, so attempting to load a multi-GB diffusion GGUF (e.g. Flux) down the LLM path fully
    // materializes its tensors — 10+ GB of RAM — before anything has a chance to throw. That OOM-killed the
    // test process along with unrelated processes on the box. Never attempt a heavy loader speculatively;
    // decide which loader to use from cheap metadata alone.
    private static readonly HashSet<Type> DiffusionMapperTypes =
    [
        typeof(ChromaRadianceKeyMapper), typeof(ZetaChromaKeyMapper), typeof(FluxKeyMapper), typeof(Flux2KeyMapper),
        typeof(SdxlKeyMapper), typeof(Sd3KeyMapper), typeof(Sd15KeyMapper), typeof(FLiteKeyMapper), typeof(ChromaKeyMapper),
        typeof(AuraFlowKeyMapper), typeof(ZImageKeyMapper), typeof(ErnieImageKeyMapper), typeof(HunyuanImageKeyMapper),
        typeof(QwenImageKeyMapper),
    ];

    /// <summary>Loads a model from a local path or HuggingFace repo id and caches the constructed pipeline.
    /// Returns the architecture that was detected. Routes to the LLM/SSM loader ONLY when a cheap GGUF
    /// metadata peek (<see cref="GgufLoader"/>, header/tensor-descriptors only — no weight-data read) finds
    /// an architecture that is either a known SSM architecture or registered in
    /// <see cref="GgufKeyMapperRegistry"/> under a NON-diffusion mapper; every other case (unrecognized
    /// architecture, or a diffusion mapper match) goes straight to diffusion detection, never attempting the
    /// LLM loader speculatively. See <see cref="DiffusionMapperTypes"/> for why this can't be a
    /// try-then-fall-back strategy.</summary>
    public async Task<string> LoadAsync(string modelIdOrPath, CancellationToken ct)
    {
        if (IsLoaded(modelIdOrPath)) return "already-loaded";

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
            // Free the registry's mmap copy — the LLM loader / PipelineFactory re-reads the file itself.
            _registry.Unload(modelIdOrPath);
        }

        string? ggufArch = TryPeekGgufArchitecture(localPath);
        if (ggufArch is not null && SsmLanguageModel.IsSsmArchitecture(ggufArch))
        {
            SsmLanguageModel ssm = SsmLanguageModel.Load(localPath, ggufArch);
            _llmSessions[modelIdOrPath] = new LlmSession
            {
                SsmModel = ssm,
                SsmPipeline = new SsmGenerationPipeline(ssm.Model, ssm.Tokenizer, _backend, ssm.Template),
            };
            return ssm.Architecture;
        }
        if (ggufArch is not null && IsLikelyLlmArchitecture(ggufArch))
        {
            GgufLanguageModel model = GgufLanguageModel.Load(localPath, dequantizeToF32: _backend is not Cuda.CudaBackend);
            if (_backend is Cuda.CudaBackend cuda) cuda.PreloadWeights(model.Transformer.EnumerateWeights());

            TransformerConfig cfg = model.Config;
            int[] headDimPerLayer = new int[cfg.NumLayers];
            for (int i = 0; i < cfg.NumLayers; i++) headDimPerLayer[i] = cfg.HeadDimFor(i);
            int maxPages = ComputeKvPoolPageCount(cfg.NumKvHeads, headDimPerLayer, _options.KvPageSize, _options.KvPoolBytesBudget);
            PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, headDimPerLayer, _options.KvPageSize, maxPages);
            DynamicBatchScheduler scheduler = new(model.Transformer, model.Tokenizer, _backend, pool, model.Template,
                gpuGate: RunUnderInferenceQueue);

            _llmSessions[modelIdOrPath] = new LlmSession { Model = model, Pool = pool, Scheduler = scheduler };
            return model.Architecture;
        }

        ModelArchitecture arch = PipelineFactory.DetectArchitecture(localPath);
        DiffusionPipelineBase pipeline = PipelineFactory.LoadAuto(localPath, _backend);
        _pipelines[modelIdOrPath] = pipeline;
        return arch.ToString();
    }

    /// <summary>Converts a VRAM byte budget into a page count sized to THIS model's actual KV dimensions
    /// (<see cref="PagedKvPool"/> pre-allocates every page up front, F32, K+V, across every layer, so this
    /// must be exact, not a rough estimate). A fixed page count is unsafe across different model shapes —
    /// see <see cref="EngineOptions.KvPoolBytesBudget"/>'s doc for the real incident this
    /// fixes (a 1024-page fixed default OOM'd loading a model with wider heads/more layers than the one it
    /// was tuned against).</summary>
    public static int ComputeKvPoolPageCount(int numKvHeads, int[] headDimPerLayer, int pageSize, long bytesBudget)
    {
        long bytesPerPage = 0;
        foreach (int headDim in headDimPerLayer)
            bytesPerPage += (long)numKvHeads * headDim * pageSize * sizeof(float) * 2; // ×2 for K and V
        return (int)Math.Max(8, bytesBudget / Math.Max(1, bytesPerPage)); // floor of 8 pages so a tiny budget still admits at least one short sequence
    }

    /// <summary>Cheap GGUF metadata-only peek at <c>general.architecture</c> (header/tensor descriptors,
    /// no weight-data read) so loading can route to the LLM/SSM vs. diffusion path before committing to
    /// either heavy loader. Returns null for non-GGUF files (safetensors diffusion checkpoints etc.) or on
    /// any read failure — callers treat null as "not GGUF, go straight to diffusion detection."</summary>
    private static string? TryPeekGgufArchitecture(string path)
    {
        try
        {
            using GgufLoader probe = new();
            probe.Load(path);
            string? arch = probe.Metadata.GetString("general.architecture");
            return string.IsNullOrWhiteSpace(arch) ? null : arch;
        }
        catch
        {
            return null;
        }
    }

    /// <summary>True when <paramref name="arch"/> is registered in <see cref="GgufKeyMapperRegistry"/> under
    /// a mapper that is NOT one of the known diffusion mappers. False for unrecognized architectures too —
    /// safest default is "don't attempt the LLM loader" (falls through to diffusion detection, which does
    /// its own cheap header-only check and fails fast with a clear error for a truly unrecognized file).</summary>
    private static bool IsLikelyLlmArchitecture(string arch)
    {
        IGgufKeyMapper? mapper = GgufKeyMapperRegistry.GetByArchitecture(arch);
        return mapper is not null && !DiffusionMapperTypes.Contains(mapper.GetType());
    }

    /// <summary>Runs the given synchronous GPU-touching work under the shared <see cref="InferenceQueue"/>
    /// (the "backend exclusivity" gate <see cref="DynamicBatchScheduler"/> needs — see its class doc).</summary>
    private Task RunUnderInferenceQueue(Action work) => _queue.EnqueueAsync(() => { work(); return Task.FromResult(true); }, CancellationToken.None);

    /// <summary>Submits one chat-completion request to a loaded LLM/SSM model. Dense/MoE transformer models
    /// go through their <see cref="DynamicBatchScheduler"/> (dynamically batched with any other concurrently
    /// in-flight request against the same model); SSM models have no batching path (out of scope — see
    /// <see cref="LlmSession"/> doc) and run one at a time through the shared queue directly, same as before.</summary>
    public Task<GenerationResult> GenerateChatAsync(string modelId, GenerationRequest request, Action<int>? onToken, CancellationToken ct)
    {
        if (!_llmSessions.TryGetValue(modelId, out LlmSession? session))
            throw new InvalidOperationException($"Model '{modelId}' is not a loaded chat model. POST /v1/models/load first.");
        return session.Scheduler is not null
            ? session.Scheduler.SubmitAsync(request, onToken, ct)
            : _queue.EnqueueAsync(() => Task.FromResult(session.SsmPipeline!.Generate(request, onToken, ct)), ct);
    }

    /// <summary>Unloads and disposes a cached pipeline (diffusion or LLM/SSM).</summary>
    public bool Unload(string modelId)
    {
        if (_pipelines.TryRemove(modelId, out DiffusionPipelineBase? pipeline))
        {
            pipeline.Dispose();
            return true;
        }
        if (_llmSessions.TryRemove(modelId, out LlmSession? session))
        {
            session.Dispose();
            return true;
        }
        return false;
    }

    /// <summary>Generates one image from a text prompt using a loaded SDXL pipeline. Other architectures
    /// throw <see cref="NotSupportedException"/> (image generation currently covers the SDXL path
    /// that <see cref="PipelineFactory"/> constructs end-to-end).</summary>
    public (byte[] rgb, int width, int height, int seed) GenerateImage(
        string modelId, ImageRequest req, Action<GenerationProgress>? onProgress = null)
    {
        if (!_pipelines.TryGetValue(modelId, out DiffusionPipelineBase? pipeline))
            throw new InvalidOperationException($"Model '{modelId}' is not loaded. Load it first.");

        // VALIDATION-PENDING: verify vs diffusers FluxPipeline true_cfg. When a Flux dispatch
        // branch is added here, tokenize req.NegativePrompt with the CLIP-L + T5 tokenizers
        // (same as the positive prompt) and pass the negative tokens to
        // FluxPipeline.GenerateFromTokens along with trueCfgScale: req.CfgScale (Flux ignores
        // the embedded-guidance CfgScale field otherwise), and an empty negative prompt must stay
        // single-pass (doTrueCfg gates on trueCfgScale > 1 AND a non-null negative T5 stream).
        if (pipeline is not SdxlPipeline sdxl)
            throw new NotSupportedException($"Image generation over HTTP currently supports SDXL pipelines; '{modelId}' is {pipeline.GetType().Name}.");

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
            Width = req.Width,
            Height = req.Height,
            Steps = req.Steps,
            CfgScale = req.CfgScale,
            Seed = req.Seed < 0 ? null : (int)req.Seed,
            ClipSkip = req.ClipSkip,
        };

        return sdxl.GenerateFromTokens(tokensL, negL, tokensG, negG, eosG, negEosG, request, onProgress);
    }

    public void Dispose()
    {
        foreach (DiffusionPipelineBase p in _pipelines.Values) p.Dispose();
        _pipelines.Clear();
        foreach (LlmSession s in _llmSessions.Values) s.Dispose();
        _llmSessions.Clear();
        _registry.Dispose();
    }
}
