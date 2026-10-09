using System.Collections.Concurrent;
using System.IO;
using System.Text;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.OutputParsing;
using HartsyInference.LLM.Transformer;
using HartsyInference.LLM.Multimodal;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Services;

/// <summary>Text-generation service: owns per-device model slots, GGUF/SSM loading, chat-template application, sampling, the multimodal VLM path, and token streaming — all against the native <see cref="TextRequest"/> contract. Lifted from SwarmUI's HartsyLocalLLMProvider with the host-app coupling stripped.</summary>
public sealed class TextService : ITextService, IDisposable
{
    /// <summary>OpenAI-CLIP normalization for the mllama image processor (splice encoders expose their own).</summary>
    private static readonly float[] MllamaMean = [0.48145466f, 0.4578275f, 0.40821073f];
    private static readonly float[] MllamaStd = [0.26862954f, 0.26130258f, 0.27577711f];

    /// <summary>Minimum free-RAM-to-file-size ratio required before loading a GGUF — load dequantizes tensors the GPU path can't consume onto host buffers atop the mmap, so peak host usage exceeds the file size (~1.5-2x observed); 2.5x is a safety margin so a big model fails cleanly instead of OOM-killing the process.</summary>
    private const double RamHeadroomMultiplier = 2.5;

    /// <summary>Free-RAM-to-file-size ratio for the part of a GGUF a quantized-capable backend keeps compressed: those weights are read through
    /// the mmap, so only the file's own pages plus runtime overhead are needed. Quantized tensors outside the backend's supported set are
    /// dequantized to F32 and are added on top (<see cref="DequantizedHostBytes"/>).</summary>
    private const double QuantizedResidentHeadroomMultiplier = 1.15;

    /// <summary>How long <see cref="Unload"/> waits for an in-flight generation before giving up on a slot. Long enough to cover a full completion, bounded so a host's "free memory" call can never hang forever.</summary>
    private const int UnloadWaitSeconds = 120;

    /// <summary>Sequences a pool-less scheduled model (the V4.1 host) decodes at once. Its states own their memory and there are no pages to bound them, so the
    /// bound is this count.</summary>
    private const int PoolLessMaxActive = 4;

    private static long _requestCounter;

    /// <summary>The deployments this service has loaded or tried to load, in first-deployed order. Guarded by <see cref="_deploymentsGate"/>.</summary>
    private readonly List<DeploymentRecord> _deployedList = [];
    private readonly object _deploymentsGate = new();

    /// <summary>One deployment's state. The state moves only along <see cref="DeploymentStateMachine"/>'s transitions, under <see cref="_deploymentsGate"/>.</summary>
    private sealed class DeploymentRecord(string deploymentId, ModelSpec spec, string deviceKey)
    {
        public string DeploymentId { get; } = deploymentId;
        public ModelSpec Spec { get; set; } = spec;
        public string DeviceKey { get; set; } = deviceKey;
        public DeploymentState State { get; set; } = DeploymentState.Unloaded;
        public string? Problem { get; set; }

        /// <summary>Counts the loads started under this id. A load completes the record only while it is still the latest (see <see cref="Complete"/>).</summary>
        public int Attempt { get; set; }
    }

    private readonly InferenceEngine _engine;
    private readonly ConcurrentDictionary<string, TextDeviceSlot> _slots = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal TextService(InferenceEngine engine) => _engine = engine;

    /// <summary>How long an unload waits for a slot's scheduled requests before it gives up and leaves the model resident. Tests shorten it.</summary>
    internal TimeSpan UnloadLeaseWait { get; set; } = TimeSpan.FromSeconds(UnloadWaitSeconds);

    /// <inheritdoc/>
    public async Task<DeploymentStatus> DeployAsync(DeploymentRequest request, CancellationToken cancel = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(request.DeploymentId);
        string deviceKey = NormalizeDeviceKey(request.Device);
        DeploymentRecord record = BeginDeployment(request.DeploymentId, request.Model, deviceKey, out int attempt);
        TextDeviceSlot slot = _slots.GetOrAdd(deviceKey, static _ => new TextDeviceSlot());
        await slot.Lock.WaitAsync(cancel).ConfigureAwait(false);
        try
        {
            // A later deploy of this id began while this one waited for the device: that one owns the record, so this one loads nothing.
            if (IsLatest(record, attempt))
            {
                // A load that replaces the device's model waits for scheduled requests on it first, as a request does (see PrepareSlot).
                if (slot.HasLeases && !string.Equals(slot.LoadedPath, request.Model.LocalPath, StringComparison.OrdinalIgnoreCase)
                    && !slot.WaitForLeases(TimeSpan.FromSeconds(UnloadWaitSeconds)))
                {
                    throw new HartsyInferenceException($"Scheduled requests on '{slot.LoadedPath}' did not finish within {UnloadWaitSeconds}s; retry the deployment.");
                }
                await Task.Run(() =>
                {
                    using IDisposable gate = DeviceGate.AcquireAllOrdinals(GateOrdinalsFor(deviceKey), cancel);
                    LoadInto(slot, deviceKey, request.Model, new TextRequest { Messages = [] });
                }, cancel).ConfigureAwait(false);
                lock (_deploymentsGate) Complete(record, attempt, DeploymentState.Ready, problem: null);
            }
        }
        catch (OperationCanceledException)
        {
            lock (_deploymentsGate) Complete(record, attempt, DeploymentState.Failed, "the deployment was cancelled before its model loaded");
            throw;
        }
        catch (Exception ex)
        {
            Logs.Error($"[TextService] Deployment '{request.DeploymentId}' did not load: {ex.Message}", ex);
            lock (_deploymentsGate) Complete(record, attempt, DeploymentState.Failed, ex.Message);
        }
        finally
        {
            slot.Lock.Release();
        }
        lock (_deploymentsGate) return Snapshot(record);
    }

    /// <inheritdoc/>
    public IReadOnlyList<DeploymentStatus> Deployments
    {
        get
        {
            lock (_deploymentsGate)
            {
                // A Ready deployment whose device no longer holds its model was unloaded through Unload: record that.
                foreach (DeploymentRecord record in _deployedList)
                {
                    bool held = _slots.TryGetValue(record.DeviceKey, out TextDeviceSlot? slot)
                        && string.Equals(slot.LoadedPath, record.Spec.LocalPath, StringComparison.OrdinalIgnoreCase);
                    if (record.State == DeploymentState.Ready && !held) Retire(record);
                }
                return [.. _deployedList.Select(Snapshot)];
            }
        }
    }

    /// <inheritdoc/>
    public DeploymentCapacity? Capacity(string deploymentId)
    {
        DeploymentRecord? record;
        lock (_deploymentsGate) record = _deployedList.Find(r => r.DeploymentId == deploymentId);
        if (record is null) return null;
        _slots.TryGetValue(record.DeviceKey, out TextDeviceSlot? slot);
        DynamicBatchScheduler? scheduler = slot?.Scheduler;
        PagedKvPool? pool = slot?.SchedulerPool;
        lock (_deploymentsGate)
        {
            return new DeploymentCapacity(record.State, scheduler?.ActiveCount ?? 0, scheduler?.QueuedCount ?? 0, scheduler?.MaxActive ?? 0,
                pool?.FreePageCount, pool?.MaxPages);
        }
    }

    /// <summary>Records a deployment as Loading. The device's other ready deployments are retired, since this load replaces its model, and a previous load of the same id is
    /// retired first. Returns the record the load then completes, and in <paramref name="attempt"/> the number of this load under its id.</summary>
    private DeploymentRecord BeginDeployment(string deploymentId, ModelSpec spec, string deviceKey, out int attempt)
    {
        lock (_deploymentsGate)
        {
            foreach (DeploymentRecord other in _deployedList.Where(r => r.DeviceKey == deviceKey && r.DeploymentId != deploymentId
                && (r.State == DeploymentState.Ready || r.State == DeploymentState.Degraded)))
            {
                Retire(other);
            }
            DeploymentRecord? record = _deployedList.Find(r => r.DeploymentId == deploymentId);
            if (record is null)
            {
                record = new DeploymentRecord(deploymentId, spec, deviceKey);
                _deployedList.Add(record);
            }
            else
            {
                Retire(record);
            }
            record.Spec = spec;
            record.DeviceKey = deviceKey;
            record.Problem = null;
            Transition(record, DeploymentState.Loading);
            attempt = ++record.Attempt;
            return record;
        }
    }

    /// <summary>Whether load <paramref name="attempt"/> is still the latest one started under <paramref name="record"/>'s id.</summary>
    private bool IsLatest(DeploymentRecord record, int attempt)
    {
        lock (_deploymentsGate) return record.Attempt == attempt && record.State == DeploymentState.Loading;
    }

    /// <summary>Ends load <paramref name="attempt"/>: moves the record from Loading to <paramref name="to"/>, but only while that load is still the latest under its id. A
    /// load that a later deploy of the same id overtook changes nothing, so neither its completion nor its failure can hit a transition the state machine refuses.</summary>
    private static void Complete(DeploymentRecord record, int attempt, DeploymentState to, string? problem)
    {
        if (record.Attempt != attempt || record.State != DeploymentState.Loading) return;
        record.Problem = problem;
        Transition(record, to);
    }

    /// <summary>Moves a deployment along the allowed transitions to <see cref="DeploymentState.Unloaded"/>; a degraded one drains first.</summary>
    private static void Retire(DeploymentRecord record)
    {
        while (record.State != DeploymentState.Unloaded)
            Transition(record, record.State == DeploymentState.Degraded ? DeploymentState.Draining : DeploymentState.Unloaded);
    }

    /// <summary>Moves a deployment to <paramref name="to"/>, refusing a transition the state machine does not allow.</summary>
    private static void Transition(DeploymentRecord record, DeploymentState to)
    {
        if (!DeploymentStateMachine.CanTransition(record.State, to))
            throw new InvalidOperationException($"Deployment '{record.DeploymentId}' cannot go from {record.State} to {to}.");
        record.State = to;
    }

    private static DeploymentStatus Snapshot(DeploymentRecord record) =>
        new(record.DeploymentId, record.Spec.Requested, record.DeviceKey, record.State, record.Problem);

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
                ToolCall = outcome.ToolCall,
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
    public IAsyncEnumerable<TextChunk> StreamAsync(ModelSpec spec, TextRequest request, CancellationToken cancel = default)
        => TextStreamPump.Run(
            async (sink, token) =>
            {
                GenOutcome outcome = await RunAsync(spec, request, sink, token).ConfigureAwait(false);
                return
                [
                    new TextChunk { Kind = TextChunkKind.Result, Text = outcome.Text },
                    new TextChunk { Kind = TextChunkKind.StopReason, Stop = outcome.Stop },
                ];
            },
            cancel);

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
                ILlmTokenizer? tokenizer = slot.Model?.Tokenizer ?? slot.SsmModel?.Tokenizer ?? slot.DeepSeekV41?.Tokenizer;
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
        // The tenant comes from the caller's identity when the request names none, so per-tenant state is keyed the same way for every route.
        request = request with { TenantId = TenantContext.Resolve(request.TenantId) };
        long diagnosticId = _engine.StartDiagnostics();
        string deviceKey = NormalizeDeviceKey(request.Device);
        TextDeviceSlot slot = _slots.GetOrAdd(deviceKey, static _ => new TextDeviceSlot());
        await slot.Lock.WaitAsync(cancel).ConfigureAwait(false);
        DynamicBatchScheduler? scheduler;
        try
        {
            // Loads the model under the lock and, for a request the scheduler takes, leases the scheduler before the lock is released.
            scheduler = await Task.Run(() => PrepareSlot(slot, deviceKey, spec, request, diagnosticId, cancel), cancel).ConfigureAwait(false);
            if (scheduler is null)
            {
                // The pipeline route generates under the lock, as it always has.
                return await Task.Run(() => RunPipeline(slot, deviceKey, request, sink, diagnosticId, cancel), cancel).ConfigureAwait(false);
            }
        }
        finally
        {
            slot.Lock.Release();
        }
        // The scheduled route runs without the slot lock, so concurrent requests join the same batch.
        try
        {
            return await RunScheduledAsync(scheduler, slot, request, sink, diagnosticId, cancel).ConfigureAwait(false);
        }
        finally
        {
            slot.ExitLease();
        }
    }

    /// <summary>Loads the request's model onto <paramref name="slot"/>. Returns the scheduler when the request takes the scheduled route, with a lease on it already taken; null when it takes the pipeline route. Caller holds <see cref="TextDeviceSlot.Lock"/>.</summary>
    private DynamicBatchScheduler? PrepareSlot(TextDeviceSlot slot, string deviceKey, ModelSpec spec, TextRequest request, long diagnosticId, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        // A load that replaces the model waits for the scheduled requests running on it. This wait happens before the device gate is taken,
        // because those requests' rounds need the gate to finish: waiting under it would deadlock.
        if (slot.HasLeases && ReplacesLoadedModel(slot, spec.LocalPath)
            && !slot.WaitForLeases(TimeSpan.FromSeconds(UnloadWaitSeconds)))
        {
            throw new HartsyInferenceException($"Scheduled requests on '{slot.LoadedPath}' did not finish within {UnloadWaitSeconds}s; retry the request.");
        }
        // Gate BY ORDINAL, before the load: the weight upload must not run concurrently with a same-device
        // sibling's generation either, and resolving the ordinal from the key means a layer-split load doesn't
        // have to construct a throwaway backend just to name its device.
        // Device gate INSIDE slot.Lock (gate is always innermost): an LLM slot and an image generation on the
        // same GPU are two backends on one device, state-isolated but not yet audited for concurrent execution.
        using IDisposable gate = DeviceGate.AcquireAllOrdinals(GateOrdinalsFor(deviceKey), cancel);
        LoadInto(slot, deviceKey, spec, request);
        _engine.ReportDiagnostic(diagnosticId, Diagnostics.InferenceDiagnosticKind.ModelReady, backend: slot.Backend);
        if (slot.Scheduler is null || !ScheduledRouteAllowed(request, LastImage(request) is not null)) return null;
        slot.EnterLease();
        return slot.Scheduler;
    }

    /// <summary>The pipeline route: generates on the slot's pipeline under the device gate. Caller holds <see cref="TextDeviceSlot.Lock"/>. Scheduled requests still running on the slot drain first, because the pipeline must not share the model with them.</summary>
    private GenOutcome RunPipeline(TextDeviceSlot slot, string deviceKey, TextRequest request, Action<TextChunk>? sink, long diagnosticId, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        if (!slot.WaitForLeases(TimeSpan.FromSeconds(UnloadWaitSeconds)))
        {
            throw new HartsyInferenceException($"Scheduled requests on '{slot.LoadedPath}' did not finish within {UnloadWaitSeconds}s; retry the request.");
        }
        GenOutcome outcome = default;
        Action generate = () =>
        {
            // The round gate (when the server sets one) is taken before the device gate, the order scheduled rounds take them in, so the two cannot deadlock.
            using IDisposable gate = DeviceGate.AcquireAllOrdinals(GateOrdinalsFor(deviceKey), cancel);
            try
            {
                ImageData? image = LastImage(request);
                outcome = image is not null && (slot.SpliceVision is not null || slot.MllamaVision is not null)
                    ? RunVision(slot, request, image, sink, cancel)
                    : RunText(slot, request, sink, diagnosticId, cancel);
            }
            finally
            {
                // UnloadSlot throws while scheduled requests hold leases. None can here: this method drained them above, and none start while the caller holds the slot lock.
                if (request.AlwaysFreeMemory == true)
                    UnloadSlot(slot);
            }
        };
        // With continuous batching on, the API does not queue text requests itself, so a pipeline request takes the engine's pipeline gate here: the same
        // server-wide queue that image work and the scheduled rounds use.
        // Blocking is safe: RunAsync runs this method on a thread-pool worker (Task.Run) with no synchronization context for the gate's continuation to need.
        if (EngineKnobs.ContinuousBatching.Value && _engine.PipelineGate is { } pipelineGate)
            pipelineGate(generate).GetAwaiter().GetResult();
        else
            generate();
        _engine.ReportDiagnostic(diagnosticId, Diagnostics.InferenceDiagnosticKind.RequestCompleted, outcome.CompletionTokens);
        return outcome;
    }

    /// <summary>Whether a request may take a slot's scheduler. A prefix-cache request stays on the pipeline, since the scheduler keeps no retained sequences; an image goes to the vision path; and AlwaysFreeMemory unloads the slot after its request, which only the pipeline does.</summary>
    internal static bool ScheduledRouteAllowed(TextRequest request, bool hasImage) =>
        request.PrefixCacheKey is not { Length: > 0 } && request.AlwaysFreeMemory != true && !hasImage;

    /// <summary>The continuous-batching scheduler for a model just loaded on <paramref name="slot"/>, or null when the knob is off or the model cannot batch its decode (the pipeline serves it then). Rounds take the device gate on <paramref name="gateOrdinals"/>; a host-backed model passes none, and its rounds run ungated.</summary>
    private DynamicBatchScheduler? CreateScheduler(TextDeviceSlot slot, IGenerationModel model, ILlmTokenizer tokenizer,
        IChatTemplate template, PagedKvPool? pool, IReadOnlyList<int> gateOrdinals)
    {
        if (!EngineKnobs.ContinuousBatching.Value || !model.Capabilities.SupportsBatchDecode) return null;
        int[] ordinals = [.. gateOrdinals.Where(o => o >= 0)];
        // A device round takes the engine's round gate (the server's shared queue, when one is set) and then the device gate: the order the pipeline
        // uses too, so the two cannot deadlock. A host-backed round takes neither.
        Func<Action, Task>? gate = ordinals.Length == 0 ? null : work => RoundGate(() =>
        {
            using IDisposable held = DeviceGate.AcquireAllOrdinals(ordinals, CancellationToken.None);
            work();
        });
        slot.SchedulerPool = pool;
        // A pool-less model has nothing else to bound its sequences by; a pooled one is bounded by its pages.
        return new DynamicBatchScheduler(model, tokenizer, pool, template, gate,
            maxActiveSequences: pool is null ? PoolLessMaxActive : int.MaxValue);
    }

    /// <summary>Runs one unit of device work through the engine's round gate, or directly when the engine has none.</summary>
    private Task RoundGate(Action work)
    {
        if (_engine.GpuRoundGate is { } gate) return gate(work);
        work();
        return Task.CompletedTask;
    }

    /// <summary>The scheduler for a freshly loaded GGUF transformer, drawing from a KV pool sized by the engine's KV budget. Allocates nothing when the knob is off.</summary>
    private DynamicBatchScheduler? CreateGgufScheduler(TextDeviceSlot slot, string deviceKey, GgufLanguageModel model, IBackend backend)
    {
        if (!EngineKnobs.ContinuousBatching.Value) return null;
        TransformerConfig cfg = model.Config;
        int[] headDimPerLayer = new int[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++) headDimPerLayer[i] = cfg.HeadDimFor(i);
        EngineOptions options = _engine.Options;
        int maxPages = PagedKvPool.PageCountForBudget(cfg.NumKvHeads, headDimPerLayer, options.KvPageSize, options.KvPoolBytesBudget);
        PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, headDimPerLayer, options.KvPageSize, maxPages);
        DynamicBatchScheduler? scheduler = CreateScheduler(slot, new GenericTransformerModel(model.Transformer, backend), model.Tokenizer,
            model.Template, pool, [.. GateOrdinalsFor(deviceKey)]);
        if (scheduler is null) pool.Dispose();
        return scheduler;
    }

    private GenOutcome RunText(TextDeviceSlot slot, TextRequest request, Action<TextChunk>? sink, long diagnosticId, CancellationToken cancel)
    {
        using GenerationRequestWiring run = BeginText(slot, request, sink, diagnosticId, cancel, scheduled: false);
        // Opt-in (null key = today's behavior, unchanged): checked OUT of the slot's store so a second concurrent
        // request on the same busy key finds nothing and falls back to this same uncached path, and checked back
        // IN from `finally` below whatever the outcome; success, a filter stop, or a genuine exception all leave
        // `reuse` in a state TextGenerationPipeline.Generate already decided is safe to store (see its doc).
        RetainedSequence? reuse = request.PrefixCacheKey is { Length: > 0 } cacheKey && slot.SsmPipeline is null
            ? (slot.PrefixCache ??= NewPrefixCacheStore()).Checkout(cacheKey) ?? new RetainedSequence()
            : null;
        GenerationResult result;
        try
        {
            result = slot.SsmPipeline is not null ? slot.SsmPipeline.Generate(run.Request, run.OnToken, run.Generation)
                : slot.Pipeline!.Generate(run.Request, reuse, run.OnToken, run.Generation);
        }
        catch (OperationCanceledException) when (run.FilterSink is { Stopped: true } && !cancel.IsCancellationRequested)
        {
            return run.FilterStopOutcome();
        }
        finally
        {
            if (reuse is not null)
            {
                slot.PrefixCache!.CheckIn(request.PrefixCacheKey!, reuse);
            }
        }
        return run.Finish(result);
    }

    /// <summary>The scheduler route: the request joins <paramref name="scheduler"/>'s batch and completes with its own result. Parsing, the filter and the stop reason are the pipeline route's. The token callback never throws on this route: it runs inside a decode round shared with other requests, and a throw would fail all of them. The scheduler evicts a cancelled or filter-stopped request on its next round instead.</summary>
    private async Task<GenOutcome> RunScheduledAsync(DynamicBatchScheduler scheduler, TextDeviceSlot slot, TextRequest request,
        Action<TextChunk>? sink, long diagnosticId, CancellationToken cancel)
    {
        using GenerationRequestWiring run = BeginText(slot, request, sink, diagnosticId, cancel, scheduled: true);
        GenOutcome outcome;
        try
        {
            GenerationResult result = await scheduler.SubmitAsync(run.Request, run.OnToken, run.Generation).ConfigureAwait(false);
            outcome = run.Finish(result);
        }
        catch (OperationCanceledException) when (run.FilterSink is { Stopped: true } && !cancel.IsCancellationRequested)
        {
            outcome = run.FilterStopOutcome();
        }
        _engine.ReportDiagnostic(diagnosticId, Diagnostics.InferenceDiagnosticKind.RequestCompleted, outcome.CompletionTokens);
        return outcome;
    }

    /// <summary>The wiring both routes share for one text request: the request the engine sees, the callback that feeds token ids to the parser and out as chunks, and the filter that can stop generation early.</summary>
    private GenerationRequestWiring BeginText(TextDeviceSlot slot, TextRequest request, Action<TextChunk>? sink, long diagnosticId, CancellationToken cancel, bool scheduled)
    {
        ILlmTokenizer tokenizer = slot.SsmModel is not null ? slot.SsmModel.Tokenizer
            : slot.DeepSeekV41 is not null ? slot.DeepSeekV41.Tokenizer
            : slot.TpCheckpoint is not null ? slot.TpCheckpoint.Tokenizer : slot.Model!.Tokenizer;
        IChatTemplate template = slot.SsmModel is not null ? slot.SsmModel.Template
            : slot.DeepSeekV41 is not null ? slot.DeepSeekV41.Template
            : slot.TpCheckpoint is not null ? slot.TpCheckpoint.Template : slot.Model!.Template;
        bool rawCompletion = NeedsRawCompletion(template, tokenizer);
        GenerationRequest genRequest = BuildRequest(request, rawCompletion, tokenizer);
        ITextStreamFilter? filter = _engine.CreateTextStreamFilter(request);
        // A filter stops generation through its own linked source so the stop maps to ToolCall, not Cancelled.
        CancellationTokenSource? stopSource = filter is null ? null : CancellationTokenSource.CreateLinkedTokenSource(cancel);
        GenerationRequestWiring run = new() { Request = genRequest, Generation = stopSource?.Token ?? cancel, StopSource = stopSource };
        if (diagnosticId != 0 || filter is not null)
        {
            run.Request = genRequest with { OnPrefillCompleted = count =>
            {
                run.PromptTokens = count;
                _engine.ReportDiagnostic(diagnosticId, Diagnostics.InferenceDiagnosticKind.PrefillCompleted, count);
            } };
        }
        if (scheduled && sink is { } emitStatus)
        {
            // A request that waits behind others says where it stands, as a status chunk.
            run.Request = run.Request with { OnQueued = position => emitStatus(new TextChunk { Kind = TextChunkKind.Status, Status = new TextStatus("queued", position) }) };
        }
        if (sink is not null || filter is not null)
        {
            run.Parser = CreateParser(template, tokenizer, run.Request, request, rawCompletion);
            Action<TextChunk> chunkSink = sink!;
            if (filter is not null)
            {
                run.FilterSink = new TextFilterSink(filter, sink, stopSource!.Cancel);
                chunkSink = run.FilterSink.Handle;
            }
            run.Emit = new ParsedEventTranslator(chunkSink, Interlocked.Increment(ref _requestCounter)).Handle;
            run.OnToken = id =>
            {
                if (run.Generation.IsCancellationRequested)
                {
                    // The pipeline stops a cancelled request by throwing here; a scheduled one must not throw (see RunScheduledAsync).
                    if (!scheduled) run.Generation.ThrowIfCancellationRequested();
                    return;
                }
                _engine.ReportDiagnostic(diagnosticId, Diagnostics.InferenceDiagnosticKind.TokenGenerated, ++run.Count);
                run.Parser!.Push(id, run.Emit!);
                // A filter stop takes effect on this token, before the next decode step runs.
                if (!scheduled && run.FilterSink is { Stopped: true }) run.Generation.ThrowIfCancellationRequested();
            };
        }
        else if (diagnosticId != 0)
        {
            run.OnToken = _ => _engine.ReportDiagnostic(diagnosticId, Diagnostics.InferenceDiagnosticKind.TokenGenerated, ++run.Count);
        }
        return run;
    }

    /// <summary>One text request's wiring, shared by the pipeline and scheduler routes (see <see cref="BeginText"/>). Disposes its filter's linked cancellation source.</summary>
    private sealed class GenerationRequestWiring : IDisposable
    {
        public GenerationRequest Request { get; set; } = null!;
        public CancellationToken Generation { get; init; }
        public CancellationTokenSource? StopSource { get; init; }
        public TextFilterSink? FilterSink { get; set; }
        public IOutputParser? Parser { get; set; }
        public Action<ParsedEvent>? Emit { get; set; }
        public Action<int>? OnToken { get; set; }
        public int PromptTokens { get; set; }
        public int Count { get; set; }

        /// <summary>The outcome of a completed request: the parser is flushed, then the filter's text and any tool call replace the raw decode when a filter is present.</summary>
        public GenOutcome Finish(GenerationResult result)
        {
            if (Parser is not null) Parser.Finish(Emit!);
            StopReason stop = result.StoppedOnStopToken ? StopReason.Stop : StopReason.Length;
            if (FilterSink is null)
                return new GenOutcome(result.Text, stop, result.PromptTokens, result.TokenIds.Count);
            FilterSink.End();
            return new GenOutcome(FilterSink.Text, FilterSink.ToolCall is null ? stop : StopReason.ToolCall,
                result.PromptTokens, result.TokenIds.Count, FilterSink.ToolCall);
        }

        /// <summary>The outcome of a request the filter stopped: ToolCall only when a call was completed; a bare filter stop is a natural end of the turn.</summary>
        public GenOutcome FilterStopOutcome()
        {
            StopReason stop = FilterSink!.ToolCall is null ? StopReason.Stop : StopReason.ToolCall;
            return new GenOutcome(FilterSink.Text, stop, PromptTokens, Count, FilterSink.ToolCall);
        }

        public void Dispose() => StopSource?.Dispose();
    }

    /// <summary>The structured parser when the model's template exposes one, else the passthrough that keeps plain-decode streaming.</summary>
    private static IOutputParser CreateParser(IChatTemplate template, ILlmTokenizer tokenizer, GenerationRequest genRequest,
        TextRequest request, bool rawCompletion)
    {
        // The parser's initial state must come from the message list the template renders (system prompt included).
        if (!rawCompletion && template is ChatTemplateEncoderAdapter adapter && genRequest.EffectiveMessages() is { } messages)
        {
            try
            {
                return adapter.CreateParser(tokenizer, messages, request.EnableThinking, genRequest.Tools);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                Logs.Warning($"Structured output parser unavailable, streaming plain text: {ex.Message}");
            }
        }
        return new PassthroughOutputParser(tokenizer);
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
            answer = new MllamaGenerator(slot.Model!, slot.MllamaVision, slot.Backend!, slot.Placement).Generate(px, question, maxTokens, sampling);
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
                : new MultimodalGenerator(slot.Model!, vision, slot.Backend!, slot.Placement).Generate(px, question, maxTokens, sampling);
        }
        sink?.Invoke(new TextChunk { Kind = TextChunkKind.Chunk, Text = answer });
        int completion = (slot.SsmModel is not null ? slot.SsmModel.Tokenizer : slot.Model!.Tokenizer).EncodeOrdinary(answer).Length;
        return new GenOutcome(answer, StopReason.Stop, 0, completion);
    }

    /// <summary>The Hugging Face directory path: loads a DeepSeek-V4.1 checkpoint onto the host reference model. The weights live on the host, so the slot gets a CPU backend whatever device was asked for.</summary>
    private void LoadHfDirectory(TextDeviceSlot slot, string deviceKey, HfCheckpointInfo checkpoint, string path)
    {
        HfTextDirectoryLoader.RequireSupported(checkpoint);
        if (KeepsLoadedModel(slot, path, hfDirectory: true))
            return;
        UnloadSlot(slot);
        // after the unload, so the memory of a model this load replaces counts as free
        EnsureRamHeadroomForDeepSeekV41(checkpoint.Root);
        if (slot.ExtraStageBackends is not null)
        {
            foreach (IBackend stage in slot.ExtraStageBackends)
            {
                try { stage.Dispose(); }
                catch (Exception ex) { Logs.Debug($"[TextService] Disposing stale stage backend failed: {ex.Message}"); }
            }
            slot.ExtraStageBackends = null;
        }
        if (slot.Backend is not null)
        {
            try { slot.Backend.Dispose(); }
            catch (Exception ex) { Logs.Debug($"[TextService] Disposing stale backend failed: {ex.Message}"); }
            slot.Backend = null;
        }
        slot.Placement = null;
        if (!string.Equals(BackendFactory.Kind(deviceKey), "cpu", StringComparison.OrdinalIgnoreCase))
            Logs.Warning($"[TextService] DeepSeek-V4.1 runs on the host reference model only; '{deviceKey}' is ignored and the model loads on the CPU.");
        IBackend backend = CreateBackendFor("cpu");
        try
        {
            slot.DeepSeekV41 = HfTextDirectoryLoader.Load(checkpoint, backend);
        }
        catch
        {
            backend.Dispose();
            throw;
        }
        slot.Backend = backend;
        slot.Pipeline = new TextGenerationPipeline(slot.DeepSeekV41.Generation, slot.DeepSeekV41.Tokenizer, slot.DeepSeekV41.Template);
        // Host-backed and CPU-only: its rounds run ungated, and its sequence states own their storage (no KV pool).
        slot.Scheduler = CreateScheduler(slot, slot.DeepSeekV41.Generation, slot.DeepSeekV41.Tokenizer, slot.DeepSeekV41.Template, pool: null, gateOrdinals: []);
        slot.CacheWeightCastsApplied = null;
        slot.PreloadRedundantWeightSplitsApplied = null;
        slot.LoadedPath = path;
        Logs.Info($"[TextService] Loaded DeepSeek-V4.1 '{Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar))}' on the host reference model (CPU).");
    }

    /// <summary>Refuses a V4.1 load that would not fit: the weights stay as mapped checkpoint bytes the kernel can drop and re-read, so what must fit in RAM is the working set (sequence state, prefill activations, small widened tensors and Engram row caches) plus a fixed margin.</summary>
    private static void EnsureRamHeadroomForDeepSeekV41(string directory)
    {
        long availableKb = ReadAvailableMemoryKb();
        if (availableKb <= 0)
            return;
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(directory);
        IReadOnlyDictionary<DeepSeekV41WeightClass, long> bytes = checkpoint.Weights.BytesByClass;
        long denseStored = bytes[DeepSeekV41WeightClass.Dense] + bytes[DeepSeekV41WeightClass.Embed] + bytes[DeepSeekV41WeightClass.Head];
        double requiredBytes = DeepSeekV41WorkingMemory.AnonymousBytes(checkpoint.Config, denseStored, HfTextDirectoryLoader.LoadOptions) + 4.0 * 1024 * 1024 * 1024;
        double availableBytes = availableKb * 1024.0;
        if (availableBytes < requiredBytes)
        {
            throw new HartsyInferenceException(
                $"Not enough free host RAM to safely load DeepSeek-V4.1 '{directory}': {availableBytes / 1024 / 1024 / 1024:0.0} GB free, "
                + $"need ~{requiredBytes / 1024 / 1024 / 1024:0.0} GB of working memory (the {denseStored / 1024.0 / 1024 / 1024:0.0} GB of dense weights are read from the checkpoint files and also benefit from free page cache). "
                + "Free RAM, then retry — loading anyway risks crashing the whole process.");
        }
    }

    private void LoadInto(TextDeviceSlot slot, string deviceKey, ModelSpec spec, TextRequest request)
    {
        string? path = spec.LocalPath;
        if (string.IsNullOrEmpty(path))
            throw new HartsyInferenceException(
                $"No checkpoint found for model '{spec.Requested}'. Pass a .gguf file via the model spec " +
                $"(looked under '{RepoPaths.ModelsRoot()}').");
        if (ProbeHfDirectory(path) is { } hfCheckpoint)
        {
            LoadHfDirectory(slot, deviceKey, hfCheckpoint, path);
            return;
        }
        if (KeepsLoadedModel(slot, path, hfDirectory: false))
        {
            LogLoadTimeSettingMismatch(slot, deviceKey, "CacheWeightCasts", request.CacheWeightCasts, slot.CacheWeightCastsApplied);
            LogLoadTimeSettingMismatch(slot, deviceKey, "PreloadRedundantWeightSplits", request.PreloadRedundantWeightSplits, slot.PreloadRedundantWeightSplitsApplied);
            return;
        }
        string[] shardDevices = ResolveShardDevices(deviceKey);
        ValidateShardDevices(shardDevices);
        UnloadSlot(slot);
        string architecture0 = PeekArchitecture(path);
        // TP claims ShardDevices as its rank list (ValidatePlacement enforces Count == degree and excludes
        // every other multi-device mode). This branch MUST come before the layer-split one — without it a
        // TP config would silently layer-split, pass token parity AND the both-cards VRAM check, and read
        // as green while testing nothing (the DiT-mosaic lesson).
        int tpDegree = _engine.Placement.TensorParallelDegree;
        if (tpDegree > 1 && !deviceKey.Contains('+') && shardDevices.Length == tpDegree
            && !SsmLanguageModel.IsSsmArchitecture(architecture0))
        {
            EnsureRamHeadroomFor(path, dequantizesEverything: true);
            LoadTensorParallel(slot, path, request, shardDevices);
            return;
        }
        if (shardDevices.Length >= 2 && tpDegree <= 1 && !SsmLanguageModel.IsSsmArchitecture(architecture0))
        {
            EnsureRamHeadroomFor(path, dequantizesEverything: false);
            LoadSharded(slot, deviceKey, path, request, shardDevices);
            return;
        }
        // Build from the selector as written, not the slot key: CanonicalDeviceKey spells a bare "vulkan" as
        // "vulkan:0", which reads as an EXPLICIT ordinal and pins loader index 0 instead of ranking.
        string buildSelector = string.IsNullOrWhiteSpace(request.Device)
            ? PrimaryDeviceKey()
            : request.Device.Trim().ToLowerInvariant();
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
            buildSelector = fallbackDevice;
        }
        IBackend backend = slot.Backend ??= CreateBackendFor(buildSelector);
        ApplyCacheWeightCastsOverride(slot, request, [backend]);
        // A backend that cannot read quantized weights needs them dequantized on the way in. Asking the backend
        // what it supports rather than what class it is means Vulkan gets the right answer the moment it publishes
        // SupportsQuantized, instead of silently paying an F32 expansion forever because it is not CUDA.
        bool dequantize = !backend.Capabilities.SupportsQuantized;
        string architecture = architecture0;
        bool ssm = SsmLanguageModel.IsSsmArchitecture(architecture);
        EnsureRamHeadroomFor(path, dequantizesEverything: dequantize || ssm);
        if (ssm)
        {
            slot.SsmModel = SsmLanguageModel.Load(path, architecture);
            slot.SsmPipeline = new SsmGenerationPipeline(slot.SsmModel.Model, slot.SsmModel.Tokenizer, backend, slot.SsmModel.Template);
            // SSM has its own loader with no EnumerateWeights/redundant-split concept at all.
            slot.PreloadRedundantWeightSplitsApplied = null;
            slot.LoadedPath = path;
            LoadVisionInto(slot, path);   // qwen35 ships a Qwen3.5-VL mmproj sidecar; other SSM archs have none.
            Logs.Info($"[TextService] Loaded GGUF SSM model '{Path.GetFileName(path)}' ({architecture}) on {deviceKey}."
                + (slot.VisionPath is not null ? $" + vision '{Path.GetFileName(slot.VisionPath)}'." : "."));
            return;
        }
        // The engine's on-disk quant is honored as-is; LowVramQuant here is the "keep quant compressed on-device"
        // toggle (any non-empty value enables it) — the loader takes a bool, not a target quant string.
        bool lowVram = !string.IsNullOrEmpty(request.LowVramQuant);
        slot.Model = GgufLanguageModel.Load(path, lowVram, dequantizeToF32: dequantize);
        // Unconditional: PreloadWeights is a no-op on a backend with no device memory, and a backend that HAS
        // device memory wants its weights resident — gating on the class meant Vulkan re-uploaded every weight
        // over PCIe on every op.
        // includeRedundantSplits defaults true here (unlike LoadSharded below and PreloadDecodeWeights, which
        // always pass false) to preserve this path's long-standing behavior for every existing caller; a request
        // can opt out via PreloadRedundantWeightSplits — see its doc comment on TextRequest for the measured cost.
        bool preloadRedundantSplits = request.PreloadRedundantWeightSplits ?? true;
        backend.PreloadWeights(slot.Model.Transformer.EnumerateWeights(preloadRedundantSplits));
        slot.PreloadRedundantWeightSplitsApplied = preloadRedundantSplits;
        slot.Pipeline = new TextGenerationPipeline(slot.Model.Transformer, slot.Model.Tokenizer, backend, slot.Model.Template);
        slot.Scheduler = CreateGgufScheduler(slot, deviceKey, slot.Model, backend);
        slot.LoadedPath = path;
        LoadVisionInto(slot, path);
        Logs.Info($"[TextService] Loaded GGUF model '{Path.GetFileName(path)}' ({slot.Model.Architecture}) on {deviceKey}"
            + (slot.VisionPath is not null ? $" + vision '{Path.GetFileName(slot.VisionPath)}'." : "."));
    }

    /// <summary>Whether a load of <paramref name="path"/> keeps the model <paramref name="slot"/> holds. Every load path reloads by this rule, and the lease wait
    /// before a load uses it too (<see cref="ReplacesLoadedModel"/>), so the wait and the reload cannot disagree. A Hugging Face directory keeps only a V4.1 model
    /// loaded from it; a GGUF keeps any model loaded from it. Paths compare ordinally, since a case-only difference names another file on a case-sensitive filesystem.</summary>
    private static bool KeepsLoadedModel(TextDeviceSlot slot, string path, bool hfDirectory) =>
        string.Equals(slot.LoadedPath, path, StringComparison.Ordinal) && (hfDirectory ? slot.DeepSeekV41 is not null
            : slot.Model is not null || slot.SsmModel is not null || slot.TpTransformer is not null || slot.DeepSeekV41 is not null);

    /// <summary>Whether loading <paramref name="path"/> onto <paramref name="slot"/> frees the model it holds, decided as <see cref="LoadInto"/> decides it. A missing
    /// path frees nothing: the load refuses it first.</summary>
    private static bool ReplacesLoadedModel(TextDeviceSlot slot, string? path) =>
        !string.IsNullOrEmpty(path) && !KeepsLoadedModel(slot, path, hfDirectory: ProbeHfDirectory(path) is not null);

    /// <summary>The Hugging Face checkpoint at <paramref name="path"/>, or null when it is not such a directory (a GGUF file, say).</summary>
    private static HfCheckpointInfo? ProbeHfDirectory(string path) => Directory.Exists(path) ? HfCheckpointDirectory.TryProbe(path) : null;

    /// <summary>The slot serving <paramref name="device"/>, or null before anything ran there. For tests that hold a slot's lease as a running scheduled request does.</summary>
    internal TextDeviceSlot? SlotFor(string? device) => _slots.TryGetValue(NormalizeDeviceKey(device), out TextDeviceSlot? slot) ? slot : null;

    /// <summary>Every CUDA ordinal a load+generate on <paramref name="deviceKey"/> can touch: each stage device of a layer-split (request-level composite key or engine-placement <c>ShardDevices</c>), else the single device. Gating only the logits stage left the other stage devices open to same-device siblings; <see cref="DeviceGate.AcquireAllOrdinals"/> acquires ascending, so multi-gate stays deadlock-free.</summary>
    private IEnumerable<int> GateOrdinalsFor(string deviceKey)
    {
        string[] shard = ResolveShardDevices(deviceKey);
        if (shard.Length >= 2)
        {
            foreach (string device in shard)
            {
                yield return GateOrdinalFor(device);
            }
        }
        else
        {
            yield return GateOrdinalFor(deviceKey);
        }
    }

    /// <summary>The device ordinal <paramref name="deviceKey"/> gates on; -1 for CPU (ungated). For a composite layer-split key the LAST stage's device is returned — single-selector callers get that selector's own ordinal (multi-gate callers use <see cref="GateOrdinalsFor"/>).</summary>
    /// <remarks>Any device backend gates, not only CUDA: a second Vulkan generation on one card contends for the same
    /// VRAM as a second CUDA one.</remarks>
    private static int GateOrdinalFor(string deviceKey)
    {
        string last = deviceKey.Contains('+')
            ? deviceKey.Split('+', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)[^1]
            : deviceKey;
        if (!BackendFactory.IsDeviceKind(BackendFactory.Kind(last)))
        {
            return -1;
        }
        return BackendFactory.ParseOrdinal(last);
    }

    /// <summary>The shard-device list in effect for <paramref name="deviceKey"/>: a request-level <c>"cuda:0+cuda:1"</c> composite wins; else the engine placement's <c>ShardDevices</c> applies to the primary slot; else empty (single-device).</summary>
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

    /// <summary>Rejects a shard list containing a non-CUDA device. <see cref="LoadSharded"/> always loads the GGUF with <c>dequantizeToF32: false</c> (every CUDA backend handles quantized formats directly), so a CPU stage would receive quantized tensors a CPU backend can't compute on (CPU requires F32). No-op for fewer than 2 devices — single-device loads already pick <c>dequantizeToF32: isCpu</c> correctly.</summary>
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

    /// <summary>Layer-split load: plans layer ranges across <paramref name="shardDevices"/> (explicit engine ratios win, else free-VRAM proportional), builds one backend per stage (slot-owned), and hands the placement to the pipeline. VRAM pooling — a model larger than any single card runs across them. The vision sidecar loads the same as the unsharded path (<see cref="LoadVisionInto"/>): both VLM generators (<see cref="MllamaGenerator"/>'s per-stage cross-attention-state peer copy, <see cref="MultimodalGenerator"/>'s plain staged embeds handoff) drive <see cref="GenericTransformer.ForwardEmbedsStaged"/> across the full placement rather than the single last-stage backend, so the split is preserved for image questions too. SSM never reaches here (layer-split isn't offered for recurrent architectures).</summary>
    private void LoadSharded(TextDeviceSlot slot, string deviceKey, string path, TextRequest request, string[] shardDevices)
    {
        // A previous load left this slot's backends alive (UnloadSlot keeps contexts for a same-config reload),
        // but the shard path builds fresh stage backends below — anything kept here would leak its CUDA context.
        // A previous SHARDED load parks its non-last stages in ExtraStageBackends; dispose those too, not just
        // the last-stage backend, or every sharded model switch leaks one context per stage.
        if (slot.ExtraStageBackends is not null)
        {
            foreach (IBackend stage in slot.ExtraStageBackends)
            {
                try { stage.Dispose(); }
                catch (Exception ex) { Logs.Debug($"[TextService] Disposing pre-shard stage backend failed: {ex.Message}"); }
            }
            slot.ExtraStageBackends = null;
        }
        if (slot.Backend is not null)
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
        ApplyCacheWeightCastsOverride(slot, request, stages.Select(s => s.Backend));
        // LoadSharded's own preload above (line ~490) always passes includeRedundantSplits: false, unconditionally
        // — TextRequest.PreloadRedundantWeightSplits is only read by the single-device path.
        slot.PreloadRedundantWeightSplitsApplied = false;
        slot.Pipeline = new TextGenerationPipeline(slot.Model.Transformer, slot.Model.Tokenizer,
            placement.LastBackend, slot.Model.Template, placement);
        slot.LoadedPath = path;
        LoadVisionInto(slot, path);
        Logs.Info($"[TextService] Loaded GGUF model '{Path.GetFileName(path)}' ({slot.Model.Architecture}) "
            + $"layer-split across {string.Join(" + ", plan.Select(p => $"{p.Device}[{p.StartLayer},{p.EndLayer})"))}"
            + (slot.VisionPath is not null ? $" + vision '{Path.GetFileName(slot.VisionPath)}'." : "."));
    }

    /// <summary>Tensor-parallel load (<c>TensorParallelDegree</c> ≥ 2): every rank device gets a weight SHARD (column/row-split per layer, embed/head on rank 0) and the forward all-reduces at the two per-layer seams. Latency-oriented sibling of <see cref="LoadSharded"/> (which pools VRAM by layer range) — the two are mutually exclusive by validation. Vision sidecars are not supported under TP v1 (logged, not loaded).</summary>
    private void LoadTensorParallel(TextDeviceSlot slot, string path, TextRequest request, string[] rankDevices)
    {
        // Same stale-backend cleanup as LoadSharded: fresh rank backends are built below, so anything the
        // slot kept alive (single-device backend OR a previous shard/TP load's stages) would leak its context.
        if (slot.ExtraStageBackends is not null)
        {
            foreach (IBackend stage in slot.ExtraStageBackends)
            {
                try { stage.Dispose(); }
                catch (Exception ex) { Logs.Debug($"[TextService] Disposing pre-TP stage backend failed: {ex.Message}"); }
            }
            slot.ExtraStageBackends = null;
        }
        if (slot.Backend is not null)
        {
            try { slot.Backend.Dispose(); }
            catch (Exception ex) { Logs.Debug($"[TextService] Disposing pre-TP backend failed: {ex.Message}"); }
            slot.Backend = null;
        }
        bool lowVram = !string.IsNullOrEmpty(request.LowVramQuant);
        GgufLanguageModel.TpCheckpoint checkpoint = GgufLanguageModel.LoadForTensorParallel(path, lowVram);
        List<IBackend> backends = [.. rankDevices.Select(CreateBackendFor)];
        ApplyCacheWeightCastsOverride(slot, request, backends);
        // Tensor-parallel weights come from TensorParallelTransformer.EnumerateRankWeights, a wholly different
        // method with no redundant-split concept — TextRequest.PreloadRedundantWeightSplits doesn't apply here.
        slot.PreloadRedundantWeightSplitsApplied = null;
        ICollectiveComm comm = CollectiveComm.Create(backends);
        TensorParallelTransformer tp = new(checkpoint.Config, new TpPlacement(backends, comm));
        tp.LoadWeights(checkpoint.Weights, "model");
        for (int rank = 0; rank < backends.Count; rank++)
        {
            backends[rank].PreloadWeights(tp.EnumerateRankWeights(rank));
        }
        slot.TpCheckpoint = checkpoint;
        slot.TpTransformer = tp;
        slot.TpComm = comm;
        slot.Backend = backends[0];
        slot.ExtraStageBackends = [.. backends.Skip(1)];
        slot.Pipeline = new TextGenerationPipeline(tp, checkpoint.Tokenizer, backends[0], checkpoint.Template);
        slot.LoadedPath = path;
        if (FindMmproj(path) is not null)
        {
            Logs.Warning($"[TextService] '{Path.GetFileName(path)}' ships a vision sidecar — NOT loaded: vision is unsupported under tensor parallelism v1 (text-only).");
        }
        Logs.Info($"[TensorParallel] active degree={rankDevices.Length} devices=[{string.Join(", ", rankDevices)}] "
            + $"comm={comm.Transport} for '{Path.GetFileName(path)}' (column/row-split per layer, 2 all-reduces/layer).");
    }

    /// <summary>Pairs the loaded text model with a sidecar mmproj GGUF (if present) and loads the matching vision encoder: cross-attention <see cref="MllamaVisionEncoder"/> for Llama-3.2-Vision, else a splice encoder (Qwen2.5-VL vs SigLIP). A bad mmproj degrades to text-only rather than failing the load.</summary>
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

    /// <summary>Takes the slot's generation lock so the release cannot race an in-flight request, then frees the model AND the slot's backend — <see cref="UnloadSlot"/> alone deliberately keeps the device context alive for the next load, which is not enough when the host is reclaiming memory.</summary>
    private bool UnloadDeviceSlot(TextDeviceSlot slot)
    {
        if (!slot.Lock.Wait(TimeSpan.FromSeconds(UnloadWaitSeconds)))
        {
            Logs.Warning($"[TextService] Unload timed out waiting on an in-flight generation ({UnloadWaitSeconds}s) — "
                + $"'{slot.LoadedPath}' stays resident.");
            return false;
        }
        try
        {
            // Draining: the scheduler takes no more work and fails what is waiting; the requests it is decoding run on to their end below.
            slot.Scheduler?.CancelQueued();
            // Scheduled requests run without the slot lock: let them finish before the model they run on is freed.
            if (!slot.WaitForLeases(UnloadLeaseWait))
            {
                // The model stays resident and keeps serving, so its scheduler takes requests again.
                slot.Scheduler?.ResumeAdmission();
                Logs.Warning($"[TextService] Unload timed out waiting on scheduled requests ({UnloadLeaseWait.TotalSeconds:0.#}s) - "
                    + $"'{slot.LoadedPath}' stays resident.");
                return false;
            }
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

    /// <summary>A fresh prefix-cache store sized from the <c>vram.prefixCache*</c> knobs, for a slot's first request that opts in.</summary>
    private static RetainedSequenceStore NewPrefixCacheStore() =>
        new(EngineKnobs.PrefixCacheMaxEntries.Value, EngineKnobs.PrefixCacheMaxBytes.Value);

    /// <summary>Logs once per slot (debug level) when a request explicitly asks for a load-time-only setting
    /// (<see cref="TextRequest.CacheWeightCasts"/>, <see cref="TextRequest.PreloadRedundantWeightSplits"/>) that
    /// differs from what is actually in force on an ALREADY-loaded slot — e.g. a non-voice caller loaded this
    /// device's slot first with the default, so a later voice request's VRAM-saving override is silently a no-op
    /// without a reload. <paramref name="requested"/> null means the caller didn't ask, so there is nothing to
    /// compare (no mismatch is possible by leaving it to the slot's existing setting). "Once" via
    /// <see cref="TextDeviceSlot.LoggedSettingMismatches"/> — otherwise every turn of a long voice call would
    /// repeat the identical line.</summary>
    private static void LogLoadTimeSettingMismatch(TextDeviceSlot slot, string deviceKey, string settingName, bool? requested, bool? applied)
    {
        if (requested is { } value && applied is { } inForce && value != inForce && slot.LoggedSettingMismatches.Add(settingName))
        {
            Logs.Debug($"[TextService] '{settingName}' requested {value} for the already-loaded slot on "
                + $"{deviceKey}, but {inForce} has been in force since that slot's backend was created — "
                + "takes effect only on the next load; reload the slot (or restart on this device) to apply it.");
        }
    }

    /// <summary>Applies <see cref="TextRequest.CacheWeightCasts"/> to every backend just created for a slot (a
    /// no-op when the request leaves it null — the backend's own default stands), and records the resulting
    /// effective value on <paramref name="slot"/> for <see cref="LoadInto"/>'s later-request mismatch check.</summary>
    private static void ApplyCacheWeightCastsOverride(TextDeviceSlot slot, TextRequest request, IEnumerable<IBackend> backends)
    {
        List<IBackend> list = [.. backends];
        if (request.CacheWeightCasts is { } value)
        {
            foreach (IBackend backend in list)
            {
                backend.CacheWeightCasts = value;
            }
        }
        slot.CacheWeightCastsApplied = list.Count > 0 ? list[0].CacheWeightCasts : null;
    }

    /// <inheritdoc/>
    public async Task TrimMemoryPool(string? device = null)
    {
        IEnumerable<TextDeviceSlot> targets = string.IsNullOrWhiteSpace(device)
            ? _slots.Values
            : _slots.TryGetValue(NormalizeDeviceKey(device), out TextDeviceSlot? found) ? [found] : [];
        foreach (TextDeviceSlot slot in targets)
        {
            // Best-effort and non-blocking: a slot mid-generation is skipped rather than waited on, since this is
            // background hygiene (pool slack an idle point returns to the driver), never something a live request
            // should queue behind.
            // A lease is taken only while the slot lock is held, so none can start between these two checks; that is what makes them race-free.
            if (slot.HasLeases || !await slot.Lock.WaitAsync(0).ConfigureAwait(false))
            {
                continue;
            }
            try
            {
                slot.Backend?.TrimMemoryPool();
                if (slot.ExtraStageBackends is not null)
                {
                    foreach (IBackend stage in slot.ExtraStageBackends)
                    {
                        stage.TrimMemoryPool();
                    }
                }
            }
            catch (Exception ex)
            {
                Logs.Debug($"[TextService] TrimMemoryPool failed on a slot: {ex.Message}");
            }
            finally
            {
                slot.Lock.Release();
            }
        }
    }

    /// <summary>Frees the slot's loaded model, keeping its backend/device alive. Caller holds <c>slot.Lock</c>. Returns whether a model was actually resident.</summary>
    private static bool UnloadSlot(TextDeviceSlot slot)
    {
        // A scheduled request runs on the scheduler without the slot lock. Every caller that frees a model drains those first
        // (see PrepareSlot, UnloadDeviceSlot and RunPipeline), so one still running here is a bug, not a wait.
        if (slot.HasLeases)
            throw new HartsyInferenceException($"Scheduled requests are still running on '{slot.LoadedPath}'; retry once they finish.");
        // Stopped before the model it drives is freed: its loop must not touch a freed model.
        slot.Scheduler?.Dispose();
        slot.Scheduler = null;
        slot.SchedulerPool?.Dispose();
        slot.SchedulerPool = null;
        // Disposed BEFORE FreeAllDeviceMemory below: a retained entry's KV Tensors reference this backend's
        // device allocations directly, and FreeAllDeviceMemory resets the backend's allocator wholesale — a
        // Tensor.Dispose() call after that would free an already-invalidated pointer.
        if (slot.PrefixCache is not null)
        {
            slot.PrefixCache.Dispose();
            slot.PrefixCache = null;
        }
        slot.SpliceVision?.Dispose();
        slot.SpliceVision = null;
        slot.MllamaVision?.Dispose();
        slot.MllamaVision = null;
        slot.VisionPath = null;
        if (slot.Model is not null || slot.SsmModel is not null || slot.TpTransformer is not null || slot.DeepSeekV41 is not null)
        {
            if (slot.Backend is not null)
            {
                try { slot.Backend.FreeAllDeviceMemory(); }
                catch (Exception ex) { Logs.Debug($"[TextService] FreeAllDeviceMemory failed: {ex.Message}"); }
            }
            if (slot.ExtraStageBackends is not null)
            {
                foreach (IBackend stage in slot.ExtraStageBackends)
                {
                    try { stage.FreeAllDeviceMemory(); }
                    catch (Exception ex) { Logs.Debug($"[TextService] Stage FreeAllDeviceMemory failed: {ex.Message}"); }
                }
            }
        }
        slot.Placement = null;
        slot.Pipeline = null;
        slot.TpTransformer?.Dispose();
        slot.TpTransformer = null;
        slot.TpComm?.Dispose();
        slot.TpComm = null;
        slot.TpCheckpoint?.Dispose();
        slot.TpCheckpoint = null;
        slot.Model?.Dispose();
        slot.Model = null;
        slot.DeepSeekV41?.Dispose();
        slot.DeepSeekV41 = null;
        slot.SsmPipeline = null;
        slot.SsmModel?.Dispose();
        slot.SsmModel = null;
        slot.CacheWeightCastsApplied = null;
        slot.PreloadRedundantWeightSplitsApplied = null;
        slot.LoggedSettingMismatches.Clear();
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

    /// <summary>Refuses to load when there isn't enough free host RAM to survive the load, so a big model fails with a clear error instead of OOM-killing the process. A load that dequantizes everything needs <see cref="RamHeadroomMultiplier"/> times the file; one that keeps supported quants compressed needs <see cref="QuantizedResidentHeadroomMultiplier"/> times the file plus the F32 size of any quantized tensor it still has to expand. No-op when <c>/proc/meminfo</c> is absent.</summary>
    private static void EnsureRamHeadroomFor(string path, bool dequantizesEverything)
    {
        long availableKb = ReadAvailableMemoryKb();
        if (availableKb <= 0)
            return;
        long fileBytes;
        try { fileBytes = new FileInfo(path).Length; }
        catch (Exception ex) { Logs.Debug($"[TextService] Could not stat '{path}': {ex.Message}"); return; }
        double availableBytes = availableKb * 1024.0;
        double requiredBytes = RequiredHostRamBytes(path, fileBytes, dequantizesEverything, out double expandedBytes);
        if (availableBytes < requiredBytes)
        {
            throw new HartsyInferenceException(
                $"Not enough free host RAM to safely load '{Path.GetFileName(path)}' ({fileBytes / 1024.0 / 1024 / 1024:0.0} GB file): "
                + $"{availableBytes / 1024 / 1024 / 1024:0.0} GB free, need ~{requiredBytes / 1024 / 1024 / 1024:0.0} GB headroom"
                + (dequantizesEverything
                    ? $" ({RamHeadroomMultiplier}x the file for dequantization). "
                    : $" ({QuantizedResidentHeadroomMultiplier}x the file plus {expandedBytes / 1024 / 1024 / 1024:0.0} GB of tensors expanded to F32). ")
                + "Free RAM or use a smaller quant, then retry — loading anyway risks crashing the whole process.");
        }
    }

    /// <summary>Free host RAM a GGUF load needs: <see cref="RamHeadroomMultiplier"/> times the file when everything is dequantized, else
    /// <see cref="QuantizedResidentHeadroomMultiplier"/> times the file plus the F32 size of the tensors still expanded (read from the header).</summary>
    internal static double RequiredHostRamBytes(string path, long fileBytes, bool dequantizesEverything, out double expandedBytes)
    {
        expandedBytes = 0;
        if (dequantizesEverything)
            return fileBytes * RamHeadroomMultiplier;
        using GgufLoader probe = new GgufLoader();
        probe.Load(path);
        expandedBytes = DequantizedHostBytes(probe.Descriptors.Values);
        return fileBytes * QuantizedResidentHeadroomMultiplier + expandedBytes;
    }

    /// <summary>Host bytes of F32 copies the load builds even for a quantized-capable backend: every quantized tensor it cannot keep compressed,
    /// plus the token and per-layer embedding tables, which <c>GenericTransformer.LoadWeights</c> always widens.</summary>
    internal static double DequantizedHostBytes(IEnumerable<GgufTensorDescriptor> tensors)
    {
        double bytes = 0;
        foreach (GgufTensorDescriptor tensor in tensors)
        {
            bool widenedEmbedding = tensor.Name is "token_embd.weight" or "per_layer_token_embd.weight";
            bool expanded = tensor.DType.IsQuantized && !GgufLanguageModel.KeepsQuantizedOnGpu(tensor.DType.Name);
            if ((expanded || widenedEmbedding) && tensor.DType != DType.F32)
                bytes += tensor.Shape.ElementCount * 4.0;
        }
        return bytes;
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

    /// <summary>Finds a sidecar mmproj GGUF paired with a text model: prefers a candidate whose real (symlink-resolved) source directory matches the text model's own — this is the authoritative signal and survives a flat model directory where every file is a symlink back into its own real per-model subfolder. Once that check has a real directory to compare against, its answer is final (match or no match) — it is never overridden by the flat-folder guess below, because <see cref="ResolveRealDirectory"/> succeeds for any existing file (symlink or not), so "no match" here already means "no companion mmproj exists", not "inconclusive". Falling through anyway (the pre-2026-07-25-fix-round-two behavior) mislabeled every plain text model in a symlinked model layout as vision-capable, since the flat-folder guess matches ANY mmproj symlink sharing the directory, regardless of which family it actually belongs to — confirmed live testing 2026-07-25 second pass (every model in <c>Models/llm/</c> reported <c>vision: true</c>). The flat-folder fallback is now reached only when real-path resolution itself fails outright (I/O error, permission denial) — a true "no signal at all" case, unlike a clean negative from the primary check.</summary>
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

    /// <summary>True if the shorter of the two stems is a prefix of the longer, case-insensitively, at least 4 characters — a cheap same-family signal (e.g. "llava-v1.5-7b" / "llava-v1.5-7b-mmproj-model-f16").</summary>
    private static bool SharesNamePrefix(string a, string b)
    {
        string shorter = a.Length <= b.Length ? a : b;
        string longer = a.Length <= b.Length ? b : a;
        return shorter.Length >= 4 && longer.StartsWith(shorter, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Resolves <paramref name="path"/> through any symlink chain to its final target's containing directory. Null on failure (not a symlink and doesn't exist, permission error, etc) — callers must treat null as "no signal", not "no match".</summary>
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

    /// <summary>True if the mmproj is a LLaVA-NeXT/1.6 anyres checkpoint — distinguished from plain LLaVA-1.5 (same CLIP tower + <c>mm.0</c>/<c>mm.2</c> projector tensors) by the presence of the <c>clip.vision.image_grid_pinpoints</c> metadata key, which only anyres checkpoints carry.</summary>
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

    /// <summary>True if the mmproj is a Qwen3-VL / Qwen3.5-VL merger — <c>clip.projector_type = qwen3vl_merger</c>. Probed BEFORE <see cref="IsQwen25Vl"/> (whose "qwen" substring test would otherwise claim it).</summary>
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

    /// <summary>True if the mmproj is a Qwen2/Qwen2.5-VL merger — via <c>clip.projector_type</c> metadata (robust to file names), falling back to the filename.</summary>
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

    /// <summary>True when the model has no usable chat template (ChatML fallback but the tokenizer never registered the <c>&lt;|im_start|&gt;</c> special token — base/non-instruct checkpoints), so generation must feed the latest user text straight to the tokenizer instead of applying a template that would throw.</summary>
    private static bool NeedsRawCompletion(IChatTemplate template, ILlmTokenizer tokenizer)
        => template is ChatMlTemplate && tokenizer.SpecialId("<|im_start|>") is null;

    /// <summary>Builds the engine's <see cref="GenerationRequest"/> from the native request; raw-completion feeds the last user message's plain text through <see cref="GenerationRequest.RawTokenIds"/>.</summary>
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
            PrefixCacheCapacityHint = request.PrefixCacheCapacityHint,
        };
        if (rawCompletion)
        {
            string rawText = LastUserText(request);
            return genRequest with { RawTokenIds = tokenizer.EncodeOrdinary(rawText) };
        }
        return genRequest with
        {
            Messages = [.. request.Messages.Select(ToChatMessage)],
            SystemPrompt = request.SystemPrompt,
            Tools = ToToolSpecs(request.Tools),
        };
    }

    private static ChatMessage ToChatMessage(TextMessage m) => new(RoleName(m.Role), m.Content ?? "")
    {
        ToolCalls = m.ToolCalls is { Count: > 0 }
            ? [.. m.ToolCalls.Select(c => new ChatToolCall(c.Id, c.Name, c.Arguments) { Namespace = c.Namespace })]
            : null,
        ToolCallId = m.ToolCallId,
        Name = m.Name,
    };

    /// <summary>OpenAI-form tool JSON (<c>{"type":"function","function":{name,description,parameters}}</c>) for the chat template; null when no tools are offered. A schema that is not valid JSON fails here, before any model work.</summary>
    private static IReadOnlyList<ToolSpec>? ToToolSpecs(IReadOnlyList<ToolDefinition>? tools)
    {
        if (tools is not { Count: > 0 }) return null;
        List<ToolSpec> specs = new(tools.Count);
        foreach (ToolDefinition tool in tools) specs.Add(new ToolSpec(ToolJson(tool)));
        return specs;
    }

    private static string ToolJson(ToolDefinition tool)
    {
        using MemoryStream buffer = new();
        using (Utf8JsonWriter writer = new(buffer))
        {
            writer.WriteStartObject();
            writer.WriteString("type", "function");
            writer.WritePropertyName("function");
            writer.WriteStartObject();
            writer.WriteString("name", tool.Name);
            writer.WriteString("description", tool.Description);
            writer.WritePropertyName("parameters");
            writer.WriteRawValue(string.IsNullOrWhiteSpace(tool.JsonSchema) ? "{}" : tool.JsonSchema);
            writer.WriteEndObject();
            writer.WriteEndObject();
        }
        return Encoding.UTF8.GetString(buffer.GetBuffer(), 0, (int)buffer.Length);
    }

    /// <summary>Per-request sampler: temperature/top-p/seed from the request; top-k/min-p/repetition-penalty are the request's backend-tuning knobs (null → filter off).</summary>
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

    /// <summary>Vision-path sampler: keeps the user's temperature/top-p/seed but drops top-k and floors the temperature to the VLM-tuned 0.4 (small quantized VLMs hallucinate under aggressive top-k).</summary>
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

    /// <summary>Normalizes a requested device string to a slot key: blank → primary; anything that names a device →
    /// its concrete <c>kind:ordinal</c> form; else the lowercased key as-is.</summary>
    /// <remarks>The key identifies a SLOT and is what <see cref="GateOrdinalsFor"/> reads, so it has to be concrete
    /// on both counts. Canonicalizing only "cuda" was consistent while CUDA was the only device kind a slot could
    /// name. <c>auto</c> is the sharper case: <see cref="CreateBackendFor"/> resolves it and can build a GPU backend,
    /// while <c>auto</c> as a gate ordinal reads as CPU — so the slot would run unserialized against the very device
    /// it shares, and would sit in a second slot beside the concrete key naming that same device.</remarks>
    private string NormalizeDeviceKey(string? device)
    {
        if (string.IsNullOrWhiteSpace(device))
            return PrimaryDeviceKey();
        return BackendFactory.CanonicalDeviceKey(device);
    }

    /// <summary>This service's primary device key, derived from the engine's backend selector.</summary>
    /// <remarks>Carries the engine's KIND as well as its ordinal. This used to read "not cpu, therefore cuda", so a
    /// <c>vulkan</c> engine silently ran its LLM on CUDA — on a machine with no CUDA device that is a driver error
    /// instead of a generation, and on a mixed box it is a generation on the wrong backend entirely.</remarks>
    private string PrimaryDeviceKey()
    {
        string resolved = BackendFactory.Resolve(_engine.BackendSelector);
        // Carry the engine's ordinal through: otherwise a 'cuda:1' engine renders on GPU 1 while its LLM lands on GPU 0.
        // Only a device kind carries one — 'auto:1' resolving to CPU on a GPU-less host must stay plain "cpu".
        return BackendFactory.IsDeviceKind(resolved)
            ? BackendFactory.WithOrdinal(resolved, BackendFactory.ParseOrdinal(_engine.BackendSelector))
            : resolved;
    }

    /// <summary>Creates the compute backend for a device key ("cpu" / "cuda:{ordinal}" / "vulkan:{ordinal}"), carrying the engine's VRAM policy onto it.</summary>
    /// <remarks>Instance rather than static so the policy lands here too: text slots build their own backends instead
    /// of going through <c>EnsureBackend</c>, which used to leave them on the environment's policy while every
    /// image/video backend honoured the host's configured one.</remarks>
    private IBackend CreateBackendFor(string deviceKey)
    {
        string key = (deviceKey ?? "cuda").Trim().ToLowerInvariant();
        if (!BackendFactory.IsValidSelector(key))
        {
            throw new HartsyInferenceException(
                $"Local LLM device '{deviceKey}' is not a backend selector — expected one of "
                + $"{string.Join(", ", BackendFactory.ValidSelectors)}, optionally with a ':{{ordinal}}' suffix.");
        }
        IBackend backend = BackendFactory.Create(key);
        _engine.ApplyVramPolicy(backend);
        return backend;
    }

    /// <summary>The outcome of one generation: full text, stop reason, token counts, and the tool call a stream filter completed (null without one).</summary>
    private readonly record struct GenOutcome(string Text, StopReason Stop, int PromptTokens, int CompletionTokens, NativeToolCall? ToolCall = null);
}
