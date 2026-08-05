using System.Collections.Concurrent;
using System.IO;
using System.Threading.Channels;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
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
        // Gate BY ORDINAL, before the load: the weight upload must not run concurrently with a same-device
        // sibling's generation either, and resolving the ordinal from the key means a layer-split load doesn't
        // have to construct a throwaway backend just to name its device.
        // Device gate INSIDE slot.Lock (gate is always innermost): an LLM slot and an image generation on the
        // same GPU are two backends on one device — state-isolated, but not yet audited for concurrent execution.
        using IDisposable gate = DeviceGate.AcquireForCudaOrdinal(GateOrdinalFor(deviceKey), cancel);
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
            // LLaVA-NeXT's unpad merge step needs the ORIGINAL image aspect ratio, which the shared fixed-square
            // VlmImagePreprocessor would destroy — it gets the raw native-resolution tensor instead and does its
            // own resize/pad/tile internally (see LlavaNextEncoder.Encode).
            using Tensor px = vision is LlavaNextEncoder
                ? LlavaNextImagePreprocessor.RawToNativeTensor(image.Rgb, image.Width, image.Height)
                : VlmImagePreprocessor.Preprocess(image.Rgb, image.Width, image.Height, vision.ImageSize, vision.ImageMean, vision.ImageStd);
            // Qwen3.5-VL rides the SSM Qwen35Model backbone (slot.SsmModel), which MultimodalGenerator (typed to
            // GenericTransformer) can't drive — use the dedicated SSM VLM generator.
            answer = slot.SsmModel is not null
                ? new Qwen35VlGenerator((Qwen35Model)slot.SsmModel.Model, slot.SsmModel.Tokenizer, vision, slot.Backend!).Generate(px, question, maxTokens, sampling)
                : new MultimodalGenerator(slot.Model!, vision, slot.Backend!).Generate(px, question, maxTokens, sampling);
        }
        sink?.Invoke(new TextChunk { Kind = TextChunkKind.Chunk, Text = answer });
        int completion = (slot.SsmModel is not null ? slot.SsmModel.Tokenizer : slot.Model!.Tokenizer).EncodeOrdinary(answer).Length;
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
        string[] shardDevices = ResolveShardDevices(deviceKey);
        ValidateShardDevices(shardDevices);
        UnloadSlot(slot);
        EnsureRamHeadroomFor(path);
        string architecture0 = PeekArchitecture(path);
        if (shardDevices.Length >= 2 && !SsmLanguageModel.IsSsmArchitecture(architecture0))
        {
            LoadSharded(slot, deviceKey, path, request, shardDevices);
            return;
        }
        if (shardDevices.Length >= 2)
        {
            // SSM recurrent state has no per-layer-crossing story, so layer-split isn't offered for it — this
            // used to fall back with no signal at all that the rest of the device list was being ignored.
            // A request-level composite key ("cuda:2+cuda:3") additionally hits a real footgun below: plain
            // CreateBackendFor(deviceKey) naively splits on the FIRST colon, so it reads "2+cuda:3" as the
            // ordinal, fails to parse, and silently defaults to ordinal 0 — NOT shardDevices[0]. Resolve the
            // fallback device explicitly (first requested device for a composite key; unchanged otherwise, since
            // an engine-placement-driven deviceKey here is already a single valid device) and log THAT value,
            // so the warning always describes where the load actually lands.
            string fallbackDevice = deviceKey.Contains('+') ? shardDevices[0] : deviceKey;
            Logs.Warning($"[TextService] '{deviceKey}' requests a {shardDevices.Length}-way split, but "
                + $"'{architecture0}' is an SSM/recurrent architecture — layer-split isn't supported for it. "
                + $"Loading on '{fallbackDevice}' only; the rest of the device list is ignored.");
            deviceKey = fallbackDevice;
        }
        IBackend backend = slot.Backend ??= CreateBackendFor(deviceKey);
        bool isCpu = backend is not CudaBackend;
        string architecture = architecture0;
        if (SsmLanguageModel.IsSsmArchitecture(architecture))
        {
            slot.SsmModel = SsmLanguageModel.Load(path, architecture);
            slot.SsmPipeline = new SsmGenerationPipeline(slot.SsmModel.Model, slot.SsmModel.Tokenizer, backend, slot.SsmModel.Template);
            slot.LoadedPath = path;
            LoadVisionInto(slot, path);   // qwen35 ships a Qwen3.5-VL mmproj sidecar; other SSM archs have none.
            Logs.Info($"[TextService] Loaded GGUF SSM model '{Path.GetFileName(path)}' ({architecture}) on {deviceKey}."
                + (slot.VisionPath is not null ? $" + vision '{Path.GetFileName(slot.VisionPath)}'." : "."));
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

    /// <summary>The CUDA ordinal this slot's generation gate keys on; -1 for CPU (ungated). For a layer-split
    /// key the LAST stage's device is used — that stage owns the logits and is the slot's <c>Backend</c>, and
    /// taking exactly one gate keeps the ordering deadlock-free.</summary>
    private static int GateOrdinalFor(string deviceKey)
    {
        string last = deviceKey.Contains('+')
            ? deviceKey.Split('+', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)[^1]
            : deviceKey;
        if (!last.StartsWith("cuda", StringComparison.OrdinalIgnoreCase))
        {
            return -1;
        }
        int colon = last.IndexOf(':');
        return colon >= 0 && int.TryParse(last[(colon + 1)..], out int ordinal) ? ordinal : 0;
    }

    /// <summary>The shard-device list in effect for <paramref name="deviceKey"/>: a request-level
    /// <c>"cuda:0+cuda:1"</c> composite wins; else the engine placement's <c>ShardDevices</c> applies to the
    /// primary slot; else empty (single-device).</summary>
    private string[] ResolveShardDevices(string deviceKey)
    {
        if (deviceKey.Contains('+'))
        {
            return deviceKey.Split('+', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        }
        if (_engine.Placement.ShardDevices.Count >= 2 && deviceKey == PrimaryDeviceKey())
        {
            return [.. _engine.Placement.ShardDevices];
        }
        return [];
    }

    /// <summary>Rejects a shard list containing a non-CUDA device. <see cref="LoadSharded"/> always loads the
    /// GGUF with <c>dequantizeToF32: false</c> (every CUDA backend handles quantized formats directly), so a
    /// CPU stage would receive quantized tensors a CPU backend can't compute on (CPU requires F32). No-op for
    /// fewer than 2 devices — single-device loads already pick <c>dequantizeToF32: isCpu</c> correctly.</summary>
    private static void ValidateShardDevices(string[] shardDevices)
    {
        if (shardDevices.Length < 2)
            return;
        foreach (string device in shardDevices)
        {
            if (!device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase))
            {
                throw new HartsyInferenceException(
                    $"LLM layer-split sharding requires every stage to be a CUDA device — got '{device}' in "
                    + $"[{string.Join(", ", shardDevices)}]. A CPU stage isn't supported here: the sharded loader "
                    + "keeps GGUF tensors quantized for CUDA backends, and a CPU backend needs them dequantized "
                    + "to F32. Use a single CPU-only device (no '+') instead of mixing CPU into a shard list.");
            }
        }
    }

    /// <summary>Layer-split load: plans layer ranges across <paramref name="shardDevices"/> (explicit engine
    /// ratios win, else free-VRAM proportional), builds one backend per stage (slot-owned), and hands the
    /// placement to the pipeline. VRAM pooling — a model larger than any single card runs across them. The
    /// vision sidecar is deliberately skipped when sharded (the VLM generators drive the FULL model on one
    /// backend, defeating the split); SSM never reaches here.</summary>
    private void LoadSharded(TextDeviceSlot slot, string deviceKey, string path, TextRequest request, string[] shardDevices)
    {
        // A previous single-device load may have left a backend on this slot (engine-level placement flips
        // rebuild slots, but a request-level composite key can only collide with itself).
        if (slot.Backend is not null && slot.Placement is null)
        {
            try { slot.Backend.Dispose(); }
            catch (Exception ex) { Logs.Debug($"[TextService] Disposing pre-shard backend failed: {ex.Message}"); }
            slot.Backend = null;
        }
        bool lowVram = !string.IsNullOrEmpty(request.LowVramQuant);
        slot.Model = GgufLanguageModel.Load(path, lowVram, dequantizeToF32: false);
        int layers = slot.Model.Transformer.Config.NumLayers;
        long totalBytes = 0;
        foreach (Tensor t in slot.Model.Transformer.EnumerateWeights(includeRedundantSplits: false))
        {
            totalBytes += Tensor.ComputeByteSize(t.Shape, t.DType);
        }
        IReadOnlyList<float>? ratios = deviceKey.Contains('+') ? null : _engine.Placement.ShardRatios;
        IReadOnlyList<LlmStagePlan> plan = PlacementPlanner.LlmSplitPlan(
            shardDevices, ratios, layers, totalBytes / Math.Max(1, layers));

        List<LlmStage> stages = new(plan.Count);
        foreach (LlmStagePlan stagePlan in plan)
        {
            stages.Add(new LlmStage(CreateBackendFor(stagePlan.Device), stagePlan.StartLayer, stagePlan.EndLayer));
        }
        LlmPlacement placement = new([.. stages]);
        slot.Placement = placement;
        slot.Backend = placement.LastBackend;
        slot.ExtraStageBackends = [.. stages.Select(s => s.Backend).Where(b => !ReferenceEquals(b, placement.LastBackend))];
        slot.Pipeline = new TextGenerationPipeline(slot.Model.Transformer, slot.Model.Tokenizer,
            placement.LastBackend, slot.Model.Template, placement);
        slot.LoadedPath = path;
        if (FindMmproj(path) is not null)
        {
            Logs.Warning("[TextService] Vision sidecar found but skipped: VLM generation is not layer-split-aware "
                + "yet — load the model on a single device for image questions.");
        }
        Logs.Info($"[TextService] Loaded GGUF model '{Path.GetFileName(path)}' ({slot.Model.Architecture}) "
            + $"layer-split across {string.Join(" + ", plan.Select(p => $"{p.Device}[{p.StartLayer},{p.EndLayer})"))}.");
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
            // slot.Model is null for the SSM backbone (Qwen3.5-VL) — guard the mllama probe accordingly.
            bool isMllama = slot.Model?.Architecture == "mllama" || (slot.Model?.Config.CrossAttnLayers.Count ?? 0) > 0;
            if (isMllama)
                slot.MllamaVision = MllamaVisionEncoder.Load(mmproj);
            else if (IsQwen3Vl(mmproj))   // must precede IsQwen25Vl (which greedily matches any "qwen" projector).
                slot.SpliceVision = Qwen3VlEncoder.Load(mmproj);
            else if (IsLlavaNext(mmproj))
                slot.SpliceVision = LlavaNextEncoder.Load(mmproj);
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
            if (slot.ExtraStageBackends is not null)
            {
                foreach (IBackend stage in slot.ExtraStageBackends)
                {
                    try { stage.Dispose(); }
                    catch (Exception ex) { Logs.Debug($"[TextService] Stage backend dispose on unload failed: {ex.Message}"); }
                }
                slot.ExtraStageBackends = null;
                freed = true;
            }
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
        if (slot.Model is not null || slot.SsmModel is not null)
        {
            if (slot.Backend is CudaBackend cuda)
            {
                try { cuda.FreeAllDeviceMemory(); }
                catch (Exception ex) { Logs.Debug($"[TextService] FreeAllDeviceMemory failed: {ex.Message}"); }
            }
            if (slot.ExtraStageBackends is not null)
            {
                foreach (IBackend stage in slot.ExtraStageBackends)
                {
                    if (stage is not CudaBackend stageCuda) continue;
                    try { stageCuda.FreeAllDeviceMemory(); }
                    catch (Exception ex) { Logs.Debug($"[TextService] Stage FreeAllDeviceMemory failed: {ex.Message}"); }
                }
            }
        }
        slot.Placement = null;
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

    /// <summary>Finds a sidecar mmproj GGUF paired with a text model: prefers a candidate whose real
    /// (symlink-resolved) source directory matches the text model's own — this is the authoritative signal and
    /// survives a flat model directory where every file is a symlink back into its own real per-model subfolder.
    /// Once that check has a real directory to compare against, its answer is final (match or no match) — it is
    /// never overridden by the flat-folder guess below, because <see cref="ResolveRealDirectory"/> succeeds for
    /// any existing file (symlink or not), so "no match" here already means "no companion mmproj exists", not
    /// "inconclusive". Falling through anyway (the pre-2026-07-25-fix-round-two behavior) mislabeled every plain
    /// text model in a symlinked model layout as vision-capable, since the flat-folder guess matches ANY mmproj
    /// symlink sharing the directory, regardless of which family it actually belongs to — confirmed live testing
    /// 2026-07-25 second pass (every model in <c>Models/llm/</c> reported <c>vision: true</c>). The flat-folder
    /// fallback is now reached only when real-path resolution itself fails outright (I/O error, permission
    /// denial) — a true "no signal at all" case, unlike a clean negative from the primary check.</summary>
    public static string? FindMmproj(string textPath)
    {
        string? dir = Path.GetDirectoryName(textPath);
        if (string.IsNullOrEmpty(dir))
            return null;
        string? realTextDir = ResolveRealDirectory(textPath);
        if (realTextDir is not null)
        {
            foreach (string f in Directory.EnumerateFiles(dir, "*.gguf"))
            {
                string name = Path.GetFileName(f);
                if (!name.Contains("mmproj", StringComparison.OrdinalIgnoreCase))
                    continue;
                if (string.Equals(ResolveRealDirectory(f), realTextDir, StringComparison.Ordinal))
                    return f;
            }
            return null;
        }
        // Last resort — resolution itself failed, so there's no directory signal to trust at all. Prefer a
        // candidate whose filename stem shares a prefix with the text model's own (cheap disambiguation when
        // several VLM families' mmprojs sit loose in the same folder), falling back to "any f16" only when
        // nothing shares a name.
        string stem = Path.GetFileNameWithoutExtension(textPath);
        string? best = null;
        string? bestNamed = null;
        foreach (string f in Directory.EnumerateFiles(dir, "*.gguf"))
        {
            string name = Path.GetFileName(f);
            if (!name.Contains("mmproj", StringComparison.OrdinalIgnoreCase))
                continue;
            if (bestNamed is null && SharesNamePrefix(stem, Path.GetFileNameWithoutExtension(f)))
                bestNamed = f;
            if (best is null || name.Contains("f16", StringComparison.OrdinalIgnoreCase))
                best = f;
        }
        return bestNamed ?? best;
    }

    /// <summary>True if the shorter of the two stems is a prefix of the longer, case-insensitively, at least 4
    /// characters — a cheap same-family signal (e.g. "llava-v1.5-7b" / "llava-v1.5-7b-mmproj-model-f16").</summary>
    private static bool SharesNamePrefix(string a, string b)
    {
        string shorter = a.Length <= b.Length ? a : b;
        string longer = a.Length <= b.Length ? b : a;
        return shorter.Length >= 4 && longer.StartsWith(shorter, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Resolves <paramref name="path"/> through any symlink chain to its final target's containing
    /// directory. Null on failure (not a symlink and doesn't exist, permission error, etc) — callers must treat
    /// null as "no signal", not "no match".</summary>
    private static string? ResolveRealDirectory(string path)
    {
        try
        {
            FileSystemInfo? target = File.ResolveLinkTarget(path, returnFinalTarget: true);
            string real = target?.FullName ?? Path.GetFullPath(path);
            return Path.GetDirectoryName(real);
        }
        catch (Exception ex)
        {
            Logs.Debug($"[TextService] Symlink resolution failed for '{path}': {ex.Message}");
            return null;
        }
    }

    /// <summary>True if the mmproj is a LLaVA-NeXT/1.6 anyres checkpoint — distinguished from plain LLaVA-1.5
    /// (same CLIP tower + <c>mm.0</c>/<c>mm.2</c> projector tensors) by the presence of the
    /// <c>clip.vision.image_grid_pinpoints</c> metadata key, which only anyres checkpoints carry.</summary>
    private static bool IsLlavaNext(string mmprojPath)
    {
        try
        {
            using GgufLoader probe = new GgufLoader();
            probe.Load(mmprojPath);
            return probe.Metadata.ContainsKey("clip.vision.image_grid_pinpoints");
        }
        catch (Exception ex)
        {
            Logs.Debug($"[TextService] mmproj image_grid_pinpoints probe failed: {ex.Message}");
            return false;
        }
    }

    /// <summary>True if the mmproj is a Qwen3-VL / Qwen3.5-VL merger — <c>clip.projector_type = qwen3vl_merger</c>.
    /// Probed BEFORE <see cref="IsQwen25Vl"/> (whose "qwen" substring test would otherwise claim it).</summary>
    private static bool IsQwen3Vl(string mmprojPath)
    {
        try
        {
            using GgufLoader probe = new GgufLoader();
            probe.Load(mmprojPath);
            string proj = (probe.Metadata.GetString("clip.projector_type") ?? "").ToLowerInvariant();
            if (proj.Contains("qwen3vl") || proj.Contains("qwen3_vl")) return true;
            if (proj.Length > 0) return false;
        }
        catch (Exception ex)
        {
            Logs.Debug($"[TextService] mmproj qwen3vl probe failed: {ex.Message}");
        }
        return Path.GetFileName(mmprojPath).Replace(".", "").Replace("-", "").Contains("qwen3vl", StringComparison.OrdinalIgnoreCase);
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
            EnableThinking = request.EnableThinking,
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
        // 1.1 (not 1.0/off) matches MultimodalGenerator's own intended fallback — its doc comment says small
        // quantized VLMs are repetition-prone, but that fallback (`sampling ?? new SamplingOptions { ... 1.1 }`)
        // is unreachable since this method always supplies a non-null SamplingOptions, short-circuiting it.
        RepetitionPenalty = (float)(request.RepetitionPenalty is > 0 ? request.RepetitionPenalty.Value : 1.1),
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
        // Carry the engine's ordinal through: otherwise a 'cuda:1' engine renders on GPU 1 while its LLM lands on GPU 0.
        return resolved == "cpu" ? "cpu" : $"cuda:{BackendFactory.ParseOrdinal(_engine.BackendSelector)}";
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
