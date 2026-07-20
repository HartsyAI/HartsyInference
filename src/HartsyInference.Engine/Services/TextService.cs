using System.Collections.Concurrent;
using System.IO;
using System.Threading.Channels;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Multimodal;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Services;

/// <summary>Text-generation service: owns per-device model slots, GGUF/SSM loading, chat-template application,
/// sampling, the multimodal VLM path, and token streaming — all against the native <see cref="TextRequest"/>
/// contract. Lifted from SwarmUI's HartsyLocalLLMProvider with the host-app coupling stripped.</summary>
public sealed class TextService : ITextService, IDisposable
{
    /// <summary>OpenAI-CLIP normalization for the mllama image processor (splice encoders expose their own).</summary>
    private static readonly float[] MllamaMean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] MllamaStd = [0.26862954f, 0.26130258f, 0.27577711f];

    /// <summary>Minimum free-RAM-to-file-size ratio required before loading a GGUF — load dequantizes tensors the
    /// GPU path can't consume onto host buffers atop the mmap, so peak host usage exceeds the file size (~1.5-2x
    /// observed); 2.5x is a safety margin so a big model fails cleanly instead of OOM-killing the process.</summary>
    private const double RamHeadroomMultiplier = 2.5;

    /// <summary>How long <see cref="Unload"/> waits for an in-flight generation before giving up on a slot. Long
    /// enough to cover a full completion, bounded so a host's "free memory" call can never hang forever.</summary>
    private const int UnloadWaitSeconds = 120;

    private readonly InferenceEngine _engine;
    private readonly ConcurrentDictionary<string, TextDeviceSlot> _slots = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal TextService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public async Task<TextResult> GenerateAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default)
    {
        try
        {
            GenOutcome outcome = await RunAsync(spec, request, sink: null, cancel).ConfigureAwait(false);
            return new TextResult
            {
                Text = outcome.Text,
                Stop = outcome.Stop,
                PromptTokens = outcome.PromptTokens,
                CompletionTokens = outcome.CompletionTokens,
            };
        }
        catch (OperationCanceledException)
        {
            Logs.Debug("Text generation cancelled.");
            throw;
        }
        catch (Exception ex)
        {
            Logs.Error($"Text generation failed: {ex.Message}", ex);
            throw;
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<TextChunk> StreamAsync(ModelSpec spec, TextRequest request,
        [EnumeratorCancellation] CancellationToken cancel = default)
    {
        Channel<TextChunk> channel = Channel.CreateUnbounded<TextChunk>();
        Task worker = Task.Run(
            async () =>
            {
                try
                {
                    void Sink(TextChunk chunk) => channel.Writer.TryWrite(chunk);
                    GenOutcome outcome = await RunAsync(spec, request, Sink, cancel).ConfigureAwait(false);
                    channel.Writer.TryWrite(new TextChunk { Kind = TextChunkKind.Result, Text = outcome.Text });
                    channel.Writer.TryWrite(new TextChunk { Kind = TextChunkKind.StopReason, Stop = outcome.Stop });
                }
                catch (OperationCanceledException)
                {
                    Logs.Debug("Text stream cancelled.");
                    channel.Writer.TryWrite(new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopReason.Cancelled });
                }
                catch (Exception ex)
                {
                    Logs.Error($"Text stream failed: {ex.Message}", ex);
                    channel.Writer.TryWrite(new TextChunk { Kind = TextChunkKind.StopReason, Stop = StopReason.Error });
                }
                finally
                {
                    channel.Writer.Complete();
                }
            },
            cancel);
        await foreach (TextChunk chunk in channel.Reader.ReadAllAsync(CancellationToken.None).ConfigureAwait(false))
            yield return chunk;
        await worker.ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public int CountTokens(ModelSpec spec, string text)
    {
        string content = text ?? "";
        // Never load a model just to count: use a loaded slot's tokenizer if one is free, preferring the slot that
        // holds this spec's model, else fall back to a cheap 4-chars-per-token heuristic.
        int? counted = TryCountWith(spec.LocalPath, content) ?? TryCountWith(null, content);
        return counted ?? Math.Max(1, (content.Length + 3) / 4);
    }

    private int? TryCountWith(string? preferredPath, string text)
    {
        foreach (TextDeviceSlot slot in _slots.Values)
        {
            if (preferredPath is not null && !string.Equals(slot.LoadedPath, preferredPath, StringComparison.OrdinalIgnoreCase))
                continue;
            if (!slot.Lock.Wait(0))
                continue;
            try
            {
                ILlmTokenizer? tokenizer = slot.Model?.Tokenizer ?? slot.SsmModel?.Tokenizer;
                if (tokenizer is not null)
                    return tokenizer.EncodeOrdinary(text).Length;
            }
            catch (Exception ex)
            {
                Logs.Debug($"CountTokens tokenizer failed on {slot.LoadedPath}: {ex.Message}");
            }
            finally
            {
                slot.Lock.Release();
            }
        }
        return null;
    }

    private async Task<GenOutcome> RunAsync(ModelSpec spec, TextRequest request, Action<TextChunk>? sink, CancellationToken cancel)
    {
        string deviceKey = NormalizeDeviceKey(request.Device);
        TextDeviceSlot slot = _slots.GetOrAdd(deviceKey, static _ => new TextDeviceSlot());
        await slot.Lock.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            return await Task.Run(() => RunCore(slot, deviceKey, spec, request, sink, cancel), cancel).ConfigureAwait(false);
        }
        finally
        {
            slot.Lock.Release();
        }
    }

    private GenOutcome RunCore(TextDeviceSlot slot, string deviceKey, ModelSpec spec, TextRequest request,
        Action<TextChunk>? sink, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        LoadInto(slot, deviceKey, spec, request);
        try
        {
            ImageData? image = LastImage(request);
            if (image is not null && (slot.SpliceVision is not null || slot.MllamaVision is not null))
                return RunVision(slot, request, image, sink, cancel);
            return RunText(slot, request, sink, cancel);
        }
        finally
        {
            if (request.AlwaysFreeMemory == true)
                UnloadSlot(slot);
        }
    }

    private static GenOutcome RunText(TextDeviceSlot slot, TextRequest request, Action<TextChunk>? sink, CancellationToken cancel)
    {
        ILlmTokenizer tokenizer = slot.SsmModel is not null ? slot.SsmModel.Tokenizer : slot.Model!.Tokenizer;
        IChatTemplate template = slot.SsmModel is not null ? slot.SsmModel.Template : slot.Model!.Template;
        bool rawCompletion = NeedsRawCompletion(template, tokenizer);
        GenerationRequest genRequest = BuildRequest(request, rawCompletion, tokenizer);

        Action<int>? onToken = null;
        if (sink is not null)
        {
            List<int> acc = [];
            int emitted = 0;
            onToken = id =>
            {
                cancel.ThrowIfCancellationRequested();
                acc.Add(id);
                string full = tokenizer.Decode(acc);
                if (full.Length > emitted)
                {
                    sink(new TextChunk { Kind = TextChunkKind.Chunk, Text = full[emitted..] });
                    emitted = full.Length;
                }
            };
        }

        GenerationResult result = slot.SsmPipeline is not null
            ? slot.SsmPipeline.Generate(genRequest, onToken, cancel)
            : slot.Pipeline!.Generate(genRequest, onToken, cancel);

        StopReason stop = result.StoppedOnStopToken ? StopReason.Stop : StopReason.Length;
        return new GenOutcome(result.Text, stop, result.PromptTokens, result.TokenIds.Count);
    }

    private static GenOutcome RunVision(TextDeviceSlot slot, TextRequest request, ImageData image, Action<TextChunk>? sink, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        string question = VisionQuestion(request);
        SamplingOptions sampling = BuildVisionSampling(request);
        int maxTokens = request.MaxTokens > 0 ? request.MaxTokens : 512;
        string answer;
        if (slot.MllamaVision is not null)
        {
            using Tensor px = VlmImagePreprocessor.Preprocess(image.Rgb, image.Width, image.Height,
                slot.MllamaVision.ImageSize, MllamaMean, MllamaStd);
            answer = new MllamaGenerator(slot.Model!, slot.MllamaVision, slot.Backend!).Generate(px, question, maxTokens, sampling);
        }
        else
        {
            IVlmImageEncoder vision = slot.SpliceVision!;
            using Tensor px = VlmImagePreprocessor.Preprocess(image.Rgb, image.Width, image.Height,
                vision.ImageSize, vision.ImageMean, vision.ImageStd);
            answer = new MultimodalGenerator(slot.Model!, vision, slot.Backend!).Generate(px, question, maxTokens, sampling);
        }
        sink?.Invoke(new TextChunk { Kind = TextChunkKind.Chunk, Text = answer });
        int completion = slot.Model!.Tokenizer.EncodeOrdinary(answer).Length;
        return new GenOutcome(answer, StopReason.Stop, 0, completion);
    }

    private void LoadInto(TextDeviceSlot slot, string deviceKey, ModelSpec spec, TextRequest request)
    {
        string? path = spec.LocalPath;
        if (string.IsNullOrEmpty(path))
            throw new HartsyInferenceException(
                "Text model has no local path. Pass a .gguf file via the model spec (looked under " +
                $"'{RepoPaths.ModelsRoot()}').");
        if ((slot.Model is not null || slot.SsmModel is not null) && slot.LoadedPath == path)
            return;
        UnloadSlot(slot);
        EnsureRamHeadroomFor(path);
        IBackend backend = slot.Backend ??= CreateBackendFor(deviceKey);
        bool isCpu = backend is not CudaBackend;
        string architecture = PeekArchitecture(path);
        if (SsmLanguageModel.IsSsmArchitecture(architecture))
        {
            slot.SsmModel = SsmLanguageModel.Load(path, architecture);
            slot.SsmPipeline = new SsmGenerationPipeline(slot.SsmModel.Model, slot.SsmModel.Tokenizer, backend, slot.SsmModel.Template);
            slot.LoadedPath = path;
            Logs.Info($"[TextService] Loaded GGUF SSM model '{Path.GetFileName(path)}' ({architecture}) on {deviceKey}.");
            return;
        }
        // The engine's on-disk quant is honored as-is; LowVramQuant here is the "keep quant compressed on-device"
        // toggle (any non-empty value enables it) — the loader takes a bool, not a target quant string.
        bool lowVram = !string.IsNullOrEmpty(request.LowVramQuant);
        slot.Model = GgufLanguageModel.Load(path, lowVram, dequantizeToF32: isCpu);
        if (backend is CudaBackend)
            backend.PreloadWeights(slot.Model.Transformer.EnumerateWeights());
        slot.Pipeline = new TextGenerationPipeline(slot.Model.Transformer, slot.Model.Tokenizer, backend, slot.Model.Template);
        slot.LoadedPath = path;
        LoadVisionInto(slot, path);
        Logs.Info($"[TextService] Loaded GGUF model '{Path.GetFileName(path)}' ({slot.Model.Architecture}) on {deviceKey}"
            + (slot.VisionPath is not null ? $" + vision '{Path.GetFileName(slot.VisionPath)}'." : "."));
    }

    /// <summary>Pairs the loaded text model with a sidecar mmproj GGUF (if present) and loads the matching vision
    /// encoder: cross-attention <see cref="MllamaVisionEncoder"/> for Llama-3.2-Vision, else a splice encoder
    /// (Qwen2.5-VL vs SigLIP). A bad mmproj degrades to text-only rather than failing the load.</summary>
    private static void LoadVisionInto(TextDeviceSlot slot, string textPath)
    {
        string? mmproj = FindMmproj(textPath);
        if (mmproj is null)
            return;
        try
        {
            bool isMllama = slot.Model!.Architecture == "mllama" || slot.Model.Config.CrossAttnLayers.Count > 0;
            if (isMllama)
                slot.MllamaVision = MllamaVisionEncoder.Load(mmproj);
            else
                slot.SpliceVision = IsQwen25Vl(mmproj) ? Qwen25VlEncoder.Load(mmproj) : SiglipVlmEncoder.Load(mmproj);
            slot.VisionPath = mmproj;
        }
        catch (Exception ex)
        {
            slot.SpliceVision = null;
            slot.MllamaVision = null;
            slot.VisionPath = null;
            Logs.Warning($"[TextService] Failed to load vision encoder '{Path.GetFileName(mmproj)}': {ex.Message}. Model stays text-only.");
        }
    }

    /// <inheritdoc/>
    public bool Unload(string? device = null)
    {
        if (!string.IsNullOrWhiteSpace(device))
        {
            return _slots.TryGetValue(NormalizeDeviceKey(device), out TextDeviceSlot? slot) && UnloadDeviceSlot(slot);
        }
        bool freed = false;
        foreach (TextDeviceSlot slot in _slots.Values)
        {
            freed |= UnloadDeviceSlot(slot);
        }
        return freed;
    }

    /// <summary>Takes the slot's generation lock so the release cannot race an in-flight request, then frees the model
    /// AND the slot's backend — <see cref="UnloadSlot"/> alone deliberately keeps the device context alive for the next
    /// load, which is not enough when the host is reclaiming memory.</summary>
    private static bool UnloadDeviceSlot(TextDeviceSlot slot)
    {
        if (!slot.Lock.Wait(TimeSpan.FromSeconds(UnloadWaitSeconds)))
        {
            Logs.Warning($"[TextService] Unload timed out waiting on an in-flight generation ({UnloadWaitSeconds}s) — "
                + $"'{slot.LoadedPath}' stays resident.");
            return false;
        }
        try
        {
            bool freed = UnloadSlot(slot);
            if (slot.Backend is not null)
            {
                try { slot.Backend.Dispose(); }
                catch (Exception ex) { Logs.Debug($"[TextService] Backend dispose on unload failed: {ex.Message}"); }
                slot.Backend = null;
                freed = true;
            }
            return freed;
        }
        finally
        {
            slot.Lock.Release();
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Unload();
        _slots.Clear();
    }

    /// <summary>Frees the slot's loaded model, keeping its backend/device alive. Caller holds <c>slot.Lock</c>.
    /// Returns whether a model was actually resident.</summary>
    private static bool UnloadSlot(TextDeviceSlot slot)
    {
        slot.SpliceVision?.Dispose();
        slot.SpliceVision = null;
        slot.MllamaVision?.Dispose();
        slot.MllamaVision = null;
        slot.VisionPath = null;
        if ((slot.Model is not null || slot.SsmModel is not null) && slot.Backend is CudaBackend cuda)
        {
            try { cuda.FreeAllDeviceMemory(); }
            catch (Exception ex) { Logs.Debug($"[TextService] FreeAllDeviceMemory failed: {ex.Message}"); }
        }
        slot.Pipeline = null;
        slot.Model?.Dispose();
        slot.Model = null;
        slot.SsmPipeline = null;
        slot.SsmModel?.Dispose();
        slot.SsmModel = null;
        bool hadModel = slot.LoadedPath is not null;
        slot.LoadedPath = null;
        // A GGUF load leaves multi-GB dequantized host buffers (and the closed mmap's pages) reachable only via
        // finalizers; without forcing a collection here, free host RAM shrinks monotonically across sequential
        // model loads until the process restarts. Ported verbatim from the provider — measured, not defensive.
        if (hadModel)
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }
        return hadModel;
    }

    /// <summary>Refuses to load when there isn't enough free host RAM to survive dequantization, so a big model
    /// fails with a clear error instead of OOM-killing the process. No-op when <c>/proc/meminfo</c> is absent.</summary>
    private static void EnsureRamHeadroomFor(string path)
    {
        long availableKb = ReadAvailableMemoryKb();
        if (availableKb <= 0)
            return;
        long fileBytes;
        try { fileBytes = new FileInfo(path).Length; }
        catch (Exception ex) { Logs.Debug($"[TextService] Could not stat '{path}': {ex.Message}"); return; }
        double availableBytes = availableKb * 1024.0;
        double requiredBytes = fileBytes * RamHeadroomMultiplier;
        if (availableBytes < requiredBytes)
        {
            throw new HartsyInferenceException(
                $"Not enough free host RAM to safely load '{Path.GetFileName(path)}' ({fileBytes / 1024.0 / 1024 / 1024:0.0} GB file): "
                + $"{availableBytes / 1024 / 1024 / 1024:0.0} GB free, need ~{requiredBytes / 1024 / 1024 / 1024:0.0} GB headroom for dequantization. "
                + "Free RAM or use a smaller quant, then retry — loading anyway risks crashing the whole process.");
        }
    }

    /// <summary>MemAvailable from /proc/meminfo in KiB, or 0 when unavailable (non-Linux).</summary>
    private static long ReadAvailableMemoryKb()
    {
        try
        {
            foreach (string line in File.ReadLines("/proc/meminfo"))
            {
                if (line.StartsWith("MemAvailable:", StringComparison.Ordinal))
                {
                    string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    return long.TryParse(parts[1], out long kb) ? kb : 0;
                }
            }
        }
        catch (Exception ex)
        {
            Logs.Debug($"[TextService] /proc/meminfo unreadable: {ex.Message}");
        }
        return 0;
    }

    /// <summary>Cheap <c>general.architecture</c> read (metadata only) to route a GGUF to the transformer or SSM loader.</summary>
    private static string PeekArchitecture(string path)
    {
        using GgufLoader probe = new GgufLoader();
        probe.Load(path);
        return probe.Metadata.GetString("general.architecture") ?? "";
    }

    /// <summary>Finds a sidecar mmproj GGUF next to a text model (any *.gguf whose name contains "mmproj"),
    /// preferring an f16 projector. Null if none.</summary>
    private static string? FindMmproj(string textPath)
    {
        string? dir = Path.GetDirectoryName(textPath);
        if (string.IsNullOrEmpty(dir))
            return null;
        string? best = null;
        foreach (string f in Directory.EnumerateFiles(dir, "*.gguf"))
        {
            string name = Path.GetFileName(f);
            if (!name.Contains("mmproj", StringComparison.OrdinalIgnoreCase))
                continue;
            if (best is null || name.Contains("f16", StringComparison.OrdinalIgnoreCase))
                best = f;
        }
        return best;
    }

    /// <summary>True if the mmproj is a Qwen2/Qwen2.5-VL merger — via <c>clip.projector_type</c> metadata (robust
    /// to file names), falling back to the filename.</summary>
    private static bool IsQwen25Vl(string mmprojPath)
    {
        try
        {
            using GgufLoader probe = new GgufLoader();
            probe.Load(mmprojPath);
            string proj = (probe.Metadata.GetString("clip.projector_type") ?? "").ToLowerInvariant();
            if (proj.Contains("qwen")) { return true; }
            if (proj.Length > 0) { return false; }
        }
        catch (Exception ex)
        {
            Logs.Debug($"[TextService] mmproj projector_type probe failed: {ex.Message}");
        }
        return Path.GetFileName(mmprojPath).Replace(".", "").Replace("-", "").Contains("qwen2", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>True when the model has no usable chat template (ChatML fallback but the tokenizer never registered
    /// the <c>&lt;|im_start|&gt;</c> special token — base/non-instruct checkpoints), so generation must feed the
    /// latest user text straight to the tokenizer instead of applying a template that would throw.</summary>
    private static bool NeedsRawCompletion(IChatTemplate template, ILlmTokenizer tokenizer)
        => template is ChatMlTemplate && tokenizer.SpecialId("<|im_start|>") is null;

    /// <summary>Builds the engine's <see cref="GenerationRequest"/> from the native request; raw-completion feeds the
    /// last user message's plain text through <see cref="GenerationRequest.RawTokenIds"/>.</summary>
    private static GenerationRequest BuildRequest(TextRequest request, bool rawCompletion, ILlmTokenizer tokenizer)
    {
        SamplingOptions sampling = BuildSampling(request);
        // Base/non-instruct checkpoints have no chat-template slot to teach the <tool_call> convention in, so
        // grammar-hardening the tool span can't reach the raw path.
        if (!rawCompletion && request.Tools is { Count: > 0 })
            sampling = sampling with { JsonModeSentinel = "<tool_call>" };
        GenerationRequest genRequest = new GenerationRequest
        {
            MaxTokens = request.MaxTokens > 0 ? request.MaxTokens : 4096,
            Sampling = sampling,
            GraphDecode = request.GraphDecode,
            SpeculativeDecode = request.SpeculativeDecode,
        };
        if (rawCompletion)
        {
            string rawText = LastUserText(request);
            return genRequest with { RawTokenIds = tokenizer.EncodeOrdinary(rawText) };
        }
        return genRequest with
        {
            Messages = [.. request.Messages.Select(m => new ChatMessage(RoleName(m.Role), m.Content ?? ""))],
            SystemPrompt = request.SystemPrompt,
        };
    }

    /// <summary>Per-request sampler: temperature/top-p/seed from the request; top-k/min-p/repetition-penalty are the
    /// request's backend-tuning knobs (null → filter off).</summary>
    private static SamplingOptions BuildSampling(TextRequest request) => SamplingOptions.Default with
    {
        Temperature = (float)Math.Max(0, request.Temperature),
        TopP = (float)(request.TopP > 0 ? request.TopP : 1.0),
        TopK = Math.Max(0, request.TopK ?? 0),
        MinP = (float)Math.Max(0, request.MinP ?? 0),
        RepetitionPenalty = (float)(request.RepetitionPenalty is > 0 ? request.RepetitionPenalty.Value : 1.0),
        Seed = request.Seed >= 0 ? (ulong)request.Seed : 0,
        Greedy = request.Temperature <= 0 || request.Greedy,
    };

    /// <summary>Vision-path sampler: keeps the user's temperature/top-p/seed but drops top-k and floors the
    /// temperature to the VLM-tuned 0.4 (small quantized VLMs hallucinate under aggressive top-k).</summary>
    private static SamplingOptions BuildVisionSampling(TextRequest request) => SamplingOptions.Default with
    {
        Temperature = request.Temperature <= 0 ? 0f : (float)Math.Max(0.4, request.Temperature),
        TopP = (float)(request.TopP > 0 ? request.TopP : 0.9),
        TopK = 0,
        MinP = (float)Math.Max(0, request.MinP ?? 0),
        RepetitionPenalty = (float)(request.RepetitionPenalty is > 0 ? request.RepetitionPenalty.Value : 1.0),
        Seed = request.Seed >= 0 ? (ulong)request.Seed : 1,
        Greedy = request.Temperature <= 0 || request.Greedy,
    };

    private static string RoleName(TextRole role) => role switch
    {
        TextRole.System => "system",
        TextRole.Assistant => "assistant",
        TextRole.Tool => "tool",
        _ => "user",
    };

    private static string LastUserText(TextRequest request)
    {
        for (int i = request.Messages.Count - 1; i >= 0; i--)
        {
            if (request.Messages[i].Role == TextRole.User)
                return request.Messages[i].Content ?? "";
        }
        return request.Messages.Count > 0 ? request.Messages[^1].Content ?? "" : "";
    }

    /// <summary>The most recent user-message image attachment, or null. The VLM generators take one image.</summary>
    private static ImageData? LastImage(TextRequest request)
    {
        for (int i = request.Messages.Count - 1; i >= 0; i--)
        {
            TextMessage m = request.Messages[i];
            if (m.Role == TextRole.User && m.Images is { Count: > 0 })
                return m.Images[^1];
        }
        return null;
    }

    private static string VisionQuestion(TextRequest request)
    {
        string q = LastUserText(request);
        return string.IsNullOrWhiteSpace(q) ? "Describe this image in detail." : q;
    }

    /// <summary>Normalizes a requested device string to a slot key: blank → primary; bare "cuda" → cuda:0; else the
    /// lowercased key as-is.</summary>
    private string NormalizeDeviceKey(string? device)
    {
        if (string.IsNullOrWhiteSpace(device))
            return PrimaryDeviceKey();
        string key = device.Trim().ToLowerInvariant();
        return key == "cuda" ? "cuda:0" : key;
    }

    /// <summary>This service's primary device key, derived from the engine's backend selector.</summary>
    private string PrimaryDeviceKey()
    {
        string resolved = BackendFactory.Resolve(_engine.BackendSelector);
        return resolved == "cpu" ? "cpu" : "cuda:0";
    }

    /// <summary>Creates the compute backend for a device key ("cpu" / "cuda:{ordinal}").</summary>
    private static IBackend CreateBackendFor(string deviceKey)
    {
        string key = (deviceKey ?? "cuda").Trim().ToLowerInvariant();
        if (key == "cpu")
            return BackendFactory.Create("cpu");
        if (key.StartsWith("cuda"))
        {
            int ordinal = 0;
            int colon = key.IndexOf(':');
            if (colon >= 0 && int.TryParse(key[(colon + 1)..], out int n))
                ordinal = n;
            return BackendFactory.CreateCuda(ordinal);
        }
        throw new HartsyInferenceException($"Local LLM device '{deviceKey}' is not supported — choose CUDA or CPU.");
    }

    /// <summary>The outcome of one generation: full text, stop reason, and token counts.</summary>
    private readonly record struct GenOutcome(string Text, StopReason Stop, int PromptTokens, int CompletionTokens);
}
