using System.Threading.Channels;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.Tokenizers;

namespace HartsyInference.LLM.Generation;

/// <summary>True continuous-batching scheduler: unlike the static-batch design it replaces (which took a
/// fixed request list up front and ran it to completion before returning anything), this admits requests at
/// ANY time via <see cref="SubmitAsync"/> and evicts each sequence the moment it finishes/stops/cancels,
/// rather than waiting for the whole cohort. A single dedicated background loop owns the model/backend and
/// every active sequence's mutable state exclusively — external callers only ever touch a thread-safe
/// <see cref="Channel{T}"/>, never the model directly — which is what makes it safe to call
/// <see cref="SubmitAsync"/> from multiple concurrent callers even though the underlying backend is not
/// itself safely re-entrant (see <c>InferenceQueue</c>'s doc comment on that constraint, which this
/// sidesteps by construction rather than by serializing callers).
///
/// <para>Each round: (1) drain newly-submitted requests, prefilling and admitting each one (single-sequence
/// prefill, matching the design this replaces — chunked/batched prefill is a further throughput
/// optimization, not required for correctness, and is left as a documented follow-up); (2) evict any
/// sequence that is cancelled, just hit a stop token, or hit its token limit — BEFORE running a decode round,
/// so a sequence never wastes a batched step after it should have stopped; (3) run one batched decode step
/// (<see cref="GenericTransformer.ForwardBatchDecode"/>) over every remaining active sequence. KV storage
/// comes from a <see cref="PagedKvPool"/> shared across every active sequence — admission fails fast with
/// <see cref="KvPoolExhaustedException"/> when the pool has no room, rather than blocking or evicting
/// something else (reject policy, matching the pool's own design).</para>
///
/// <para><b>Backend exclusivity:</b> multiple concurrent <see cref="SubmitAsync"/> callers is exactly the
/// point (that's what lets requests batch together), but the shared <see cref="IBackend"/> instance is NOT
/// itself safely re-entrant (one CUDA stream, non-thread-safe activation/weight caches) — and on a server
/// that also runs diffusion image generation through the SAME backend instance via a separate queue, this
/// scheduler's GPU work must never overlap with THAT either. So every GPU-touching step (prefill, one
/// decode round) is gated through the optional <paramref name="gpuGate"/> — pass the server's existing
/// <c>InferenceQueue</c> (shared with diffusion) to keep the whole server down to one physical GPU operation
/// at a time, while still batching every concurrently-submitted chat request into that one operation. This
/// does NOT serialize chat requests the way routing each whole request through the queue would (that was
/// tried first and rejected — it would gate one call to <see cref="SubmitAsync"/> at a time, so a second
/// request could never even be ADMITTED into the batch until the first one's entire generation finished,
/// defeating the purpose); only the actual GPU round is gated, and rounds already contain every request that
/// arrived since the last one.</para></summary>
public sealed class DynamicBatchScheduler : IBatchScheduler, IDisposable
{
    private readonly GenericTransformer _model;
    private readonly ILlmTokenizer _tokenizer;
    private readonly IChatTemplate _template;
    private readonly IBackend _backend;
    private readonly PagedKvPool _pool;
    private readonly HashSet<int> _stopIds;
    private readonly Channel<PendingRequest> _incoming;
    private readonly CancellationTokenSource _shutdown = new();
    private readonly Task _loopTask;
    private readonly Func<Action, Task>? _gpuGate;

    private sealed class PendingRequest
    {
        public required GenerationRequest Request;
        public Action<int>? OnToken;
        public required CancellationToken Ct;
        public required TaskCompletionSource<GenerationResult> Completion;
    }

    /// <summary>One active sequence's per-request state. <see cref="Cache"/> is <see cref="IKvCache"/> (not
    /// concretely <see cref="PagedKvCache"/>) so a sequence admitted while the scheduler is otherwise idle can
    /// use a dedicated <see cref="FixedKvCache"/> instead — see the graph-decode retrofit design in
    /// <c>docs/Checklists/LLM_DECODE_PERF_GRIND.md</c>'s "NEW PLAN" section. <see cref="GraphSession"/> is
    /// non-null only for such a sequence, only while it's still eligible for graph replay (see
    /// <see cref="RunLoopAsync"/>'s one-way retirement). <see cref="Dispose"/> is intentionally the ONLY way
    /// callers free this sequence's resources (never call <c>seq.Cache.Dispose()</c> directly) so a session
    /// can never be forgotten at a disposal call site.</summary>
    private sealed class ActiveSeq : IDisposable
    {
        public required PendingRequest Pending;
        public required int[] PromptIds;
        public required IKvCache Cache;
        public required SamplerChain Sampler;
        public required HashSet<int> Stops;
        public required List<int> Generated;
        public int Next;
        public GraphDecodeSession? GraphSession;

        public void Dispose()
        {
            Cache.Dispose();
            GraphSession?.Dispose();
        }
    }

    /// <summary>Creates a scheduler backed by <paramref name="pool"/> (shared across every sequence this
    /// scheduler admits — size it for the concurrency you want to support, not per-request).
    /// <paramref name="gpuGate"/>, when supplied, wraps every GPU-touching step (prefill, one decode round)
    /// — pass a function that runs its argument through the server's shared backend-exclusivity queue (see
    /// class doc "Backend exclusivity"); omit it only when nothing else can contend for the same backend
    /// instance (e.g. tests, or a CPU-only deployment with no diffusion sharing the process).</summary>
    public DynamicBatchScheduler(GenericTransformer model, ILlmTokenizer tokenizer, IBackend backend,
        PagedKvPool pool, IChatTemplate? template = null, Func<Action, Task>? gpuGate = null)
    {
        _model = model;
        _tokenizer = tokenizer;
        _backend = backend;
        _pool = pool;
        _template = template ?? new ChatMlTemplate();
        _stopIds = [.. tokenizer.StopIds];
        _incoming = Channel.CreateUnbounded<PendingRequest>();
        _gpuGate = gpuGate;
        _loopTask = Task.Run(RunLoopAsync);
        // RunLoopAsync's own decode-round/admission try/catches are the primary defense (they isolate a
        // failure to the sequences actually involved and keep the loop running) — this continuation is
        // defense-in-depth for the residual case where something still escapes those and the loop itself
        // dies: without it, a background Task's fault is completely silent (nothing awaits _loopTask), so
        // the model's scheduler would go dark with zero log evidence of why.
        _loopTask.ContinueWith(
            t => Logs.Error("DynamicBatchScheduler: background loop terminated unexpectedly", t.Exception!),
            TaskContinuationOptions.OnlyOnFaulted | TaskContinuationOptions.ExecuteSynchronously);
    }

    /// <summary>True while the background loop is still running. Goes false once it exits — cleanly on
    /// <see cref="Dispose"/>, or (should every containment in <see cref="RunLoopAsync"/> somehow fail to
    /// catch something) if it faults. Callers that track a model's serving health (e.g. an HTTP readiness
    /// check) should treat <c>false</c> after a successful construction as "this model's chat traffic is
    /// dead and won't recover without reloading the model."</summary>
    public bool IsLoopAlive => !_loopTask.IsCompleted;

    /// <summary>Test-only fault injection: when set, invoked once per decode round with that round's feeder
    /// count; a non-null return is thrown instead of running the round. Exists so fault-isolation behavior
    /// (a round failing without killing the loop or affecting unrelated sequences) can be tested
    /// deterministically — reproducing a REAL backend crash (like the CPU-MoE AccessViolationException that
    /// motivated this containment) isn't something a fast, safe unit test can do on demand. Null in
    /// production; never read unless a test sets it.</summary>
    internal Func<int, Exception?>? TestFaultInjector { get; set; }

    /// <summary>Test-only override for the "does this architecture/backend support graph decode" check in
    /// <see cref="AdmitAndPrefill"/> — the CPU backend never satisfies this for real (graph decode is
    /// CUDA-only), so a CPU-backend unit test can't otherwise reach the solo-admission graph-capture path at
    /// all. Null in production (falls through to the real <c>_model.SupportsGraphDecode(_backend)</c> check).</summary>
    internal bool? TestForceSupportsGraphDecode { get; set; }

    /// <summary>Test-only fault injection for <see cref="CaptureGraphSession"/>: when set, invoked once per
    /// capture attempt; a non-null return is thrown instead of actually capturing, exercising the circuit
    /// breaker (<see cref="_graphCaptureUnavailable"/>) without needing a real CUDA capture failure. Null in
    /// production.</summary>
    internal Func<Exception?>? TestGraphCaptureFailureInjector { get; set; }

    private async Task RunGpuWork(Action work)
    {
        if (_gpuGate is null) { work(); return; }
        await _gpuGate(work).ConfigureAwait(false);
    }

    public Task<GenerationResult> SubmitAsync(GenerationRequest request, Action<int>? onToken, CancellationToken ct)
    {
        TaskCompletionSource<GenerationResult> tcs = new(TaskCreationOptions.RunContinuationsAsynchronously);
        PendingRequest pending = new() { Request = request, OnToken = onToken, Ct = ct, Completion = tcs };
        if (!_incoming.Writer.TryWrite(pending))
            tcs.SetException(new ObjectDisposedException(nameof(DynamicBatchScheduler)));
        return tcs.Task;
    }

    private async Task RunLoopAsync()
    {
        List<ActiveSeq> active = [];
        try
        {
            while (!_shutdown.IsCancellationRequested)
            {
                await DrainIncomingAsync(active).ConfigureAwait(false);

                if (active.Count == 0)
                {
                    await WaitForWorkOrShutdown().ConfigureAwait(false);
                    continue;
                }

                // Filter BEFORE decoding: a cancelled/stopped/limit-reached sequence never gets fed into a
                // wasted batched step. Iterate in reverse so RemoveAt is safe mid-loop.
                List<ActiveSeq> feeders = new(active.Count);
                List<ActiveSeq> evicted = [];
                for (int i = active.Count - 1; i >= 0; i--)
                {
                    ActiveSeq seq = active[i];
                    if (seq.Pending.Ct.IsCancellationRequested)
                    {
                        seq.Pending.Completion.TrySetCanceled(seq.Pending.Ct);
                        evicted.Add(seq);
                        active.RemoveAt(i);
                        continue;
                    }
                    bool stoppedNow = seq.Stops.Contains(seq.Next);
                    bool atLimit = seq.Generated.Count >= seq.Pending.Request.MaxTokens;
                    if (stoppedNow || atLimit)
                    {
                        CompleteSeq(seq, stoppedNow);
                        evicted.Add(seq);
                        active.RemoveAt(i);
                        continue;
                    }
                    feeders.Add(seq);
                }
                if (evicted.Count > 0)
                {
                    // A sequence's Dispose() may touch real GPU resources (currently a PagedKvCache's gather
                    // scratch tensors; later a solo sequence's dedicated FixedKvCache buffers and, once
                    // graph-decode sessions exist, a captured CUDA graph + its device buffers) — batch every
                    // eviction this round into ONE gated call so disposal never races concurrent GPU work
                    // from another gated caller (diffusion, another model's scheduler), same rationale as
                    // every other GPU-touching step here.
                    await RunGpuWork(() => { foreach (ActiveSeq seq in evicted) seq.Dispose(); }).ConfigureAwait(false);
                }
                if (feeders.Count == 0) continue;

                // Dispatch: a lone feeder carrying a captured graph replays it (one launch, no per-round
                // kernel-issuance overhead); anything else — multiple feeders, or a solo feeder with no
                // session — takes the existing eager batched path. A feeder that has a session but is NOT
                // alone this round gets it retired (disposed, one-way — see ActiveSeq.GraphSession's doc)
                // before the eager round runs, so it never sits around going stale.
                Action work = feeders.Count == 1 && feeders[0].GraphSession is { } gs
                    ? () => ReplayGraphRound(feeders[0], gs)
                    : () =>
                    {
                        foreach (ActiveSeq seq in feeders)
                        {
                            if (seq.GraphSession is null) continue;
                            seq.GraphSession.Dispose();
                            seq.GraphSession = null;
                        }
                        RunDecodeRound(feeders);
                    };

                try
                {
                    Exception? injected = TestFaultInjector?.Invoke(feeders.Count);
                    if (injected is not null) throw injected;
                    await RunGpuWork(work).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    // A decode round writes every feeder's KV cache in one native call; once ANY exception
                    // escapes mid-round, none of those caches can be trusted (partially-written K/V, cache
                    // position potentially out of sync with what was actually computed), so there's no safe
                    // way to keep just some of them going. Fail every sequence THIS round touched and free
                    // its pages back to the pool, but — critically — keep the LOOP running: without this,
                    // one bad request (or one architecture-specific kernel bug) would silently wedge this
                    // model's scheduler forever, hanging every future request to it with no crash and no
                    // log line to explain why (see the granitemoe CPU-MoE AccessViolationException that
                    // motivated this — that one specific case is still uncatchable/process-fatal by CLR
                    // design, but an ordinary C# exception from anywhere else in the decode path is not,
                    // and previously got the same silent-wedge treatment).
                    Logs.Error($"DynamicBatchScheduler: decode round failed for {feeders.Count} sequence(s), failing them and continuing", ex);
                    foreach (ActiveSeq seq in feeders)
                    {
                        seq.Pending.Completion.TrySetException(ex);
                        active.Remove(seq);
                    }
                    await RunGpuWork(() => { foreach (ActiveSeq seq in feeders) seq.Dispose(); }).ConfigureAwait(false);
                }
            }
        }
        finally
        {
            foreach (ActiveSeq seq in active)
                seq.Pending.Completion.TrySetException(new ObjectDisposedException(nameof(DynamicBatchScheduler)));
            if (active.Count > 0)
                await RunGpuWork(() => { foreach (ActiveSeq seq in active) seq.Dispose(); }).ConfigureAwait(false);
        }
    }

    private async Task DrainIncomingAsync(List<ActiveSeq> active)
    {
        while (_incoming.Reader.TryRead(out PendingRequest? pending))
        {
            if (pending.Ct.IsCancellationRequested) { pending.Completion.TrySetCanceled(pending.Ct); continue; }
            // Recomputed fresh for EACH dequeued request, not once per drain burst — active.Count grows as
            // this loop admits earlier requests in the same burst, so a request 2nd-or-later in a burst that
            // arrived all at once correctly sees itself as non-solo.
            bool solo = active.Count == 0;
            try
            {
                ActiveSeq? seq = null;
                await RunGpuWork(() => seq = AdmitAndPrefill(pending, solo)).ConfigureAwait(false);
                active.Add(seq!);
            }
            catch (Exception ex)
            {
                Logs.Error("DynamicBatchScheduler: admission/prefill failed for one request", ex);
                pending.Completion.TrySetException(ex);
            }
        }
    }

    private async Task WaitForWorkOrShutdown()
    {
        try
        {
            await _incoming.Reader.WaitToReadAsync(_shutdown.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            // Shutdown requested while idle — loop condition re-checks _shutdown.IsCancellationRequested next.
        }
    }

    /// <summary>True once a CUDA-graph capture has failed once for this scheduler's model. A capture failure
    /// is architecture/backend-determined, not request-specific — it will fail identically for every future
    /// solo-eligible request against this same model, so this is a circuit breaker (stop attempting capture
    /// at all, avoiding a repeat of whatever it costs to fail) rather than a per-request retry.</summary>
    private bool _graphCaptureUnavailable;

    private ActiveSeq AdmitAndPrefill(PendingRequest pending, bool solo)
    {
        GenerationRequest req = pending.Request;
        int[] promptIds = BuildPromptIds(req);
        if (promptIds.Length == 0) throw new ArgumentException("Request produced zero tokens.");
        HashSet<int> stops = _stopIds;
        if (req.StopTokenIds is not null) { stops = [.. _stopIds]; foreach (int s in req.StopTokenIds) stops.Add(s); }

        TransformerConfig cfg = _model.Config;
        // Only a request admitted while the scheduler is otherwise idle, and only when it's eligible for the
        // SAME reasons TextGenerationPipeline.Generate's graph-decode dispatch requires (greedy,
        // SupportsGraphDecode), plus non-JSON-mode (that pipeline's own graph-decode step bypasses the CPU
        // sampler chain entirely — including JsonGrammarStep — so combining the two would silently produce
        // non-JSON output; excluded here even though the existing single-sequence pipeline doesn't exclude
        // it, since the server is a much larger audience for this gap than the CLI). Gets a dedicated
        // FixedKvCache instead of drawing from the shared pool — see DynamicBatchScheduler's class doc and
        // ActiveSeq's GraphSession field doc for why this is safe and why it's a one-way admission decision
        // (never converted later).
        bool graphEligible = solo && !_graphCaptureUnavailable
            && req.Sampling.Greedy && !req.Sampling.JsonMode
            && (req.GraphDecode ?? (Environment.GetEnvironmentVariable("HARTSY_GRAPH_DECODE") == "1"))
            && (TestForceSupportsGraphDecode ?? _model.SupportsGraphDecode(_backend));

        // Throws KvPoolExhaustedException if the pool can't fit the prompt (PagedKvCache path only) —
        // propagates to the caller's SubmitAsync task as a fault, the reject policy PagedKvPool documents.
        IKvCache cache = graphEligible
            ? new FixedKvCache(cfg.NumLayers, 1, cfg.NumKvHeads, HeadDimPerLayer(cfg), promptIds.Length + req.MaxTokens + 1)
            : new PagedKvCache(_pool);
        try
        {
            List<int> generated = new(req.MaxTokens);
            SamplerChain sampler = SamplerChain.FromOptions(req.Sampling, _tokenizer, cfg.VocabSize);
            int next;
            using (Tensor hidden = _model.Forward(_backend, promptIds, 0, cache))
            using (Tensor logits = _model.ProjectLogits(_backend, hidden, promptIds.Length))
                next = sampler.Next(LastRow(logits, promptIds.Length, cfg.VocabSize), generated);

            GraphDecodeSession? session = null;
            if (graphEligible)
            {
                try
                {
                    session = CaptureGraphSession((FixedKvCache)cache, promptIds.Length, next, req.Sampling.RepetitionPenalty);
                }
                catch (Exception captureEx)
                {
                    // SupportsGraphDecode said this architecture/backend combination SHOULD work — a capture
                    // failure past that point is a genuine, unexpected gap, not a routine per-request
                    // condition. Trip the breaker so no future request repeats this failure, then let it
                    // propagate — DrainIncomingAsync's existing catch already fails just this one request
                    // cleanly without wedging the loop, matching TextGenerationPipeline's own "opt-in, clear
                    // error, not silent mis-generation" philosophy for this same failure mode.
                    _graphCaptureUnavailable = true;
                    Logs.Error("DynamicBatchScheduler: CUDA-graph capture failed for a solo-eligible sequence; disabling further graph-decode admission for this model", captureEx);
                    throw;
                }
            }

            return new ActiveSeq
            {
                Pending = pending, PromptIds = promptIds, Cache = cache, Sampler = sampler,
                Stops = stops, Generated = generated, Next = next, GraphSession = session,
            };
        }
        catch
        {
            cache.Dispose();
            throw;
        }
    }

    private static int[] HeadDimPerLayer(TransformerConfig cfg)
    {
        int[] a = new int[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++) a[i] = cfg.HeadDimFor(i);
        return a;
    }

    /// <summary>Captures one CUDA graph for greedy decode against <paramref name="cache"/> — mirrors
    /// <see cref="TextGenerationPipeline"/>'s <c>GenerateGraphDecode</c> capture step exactly (same device
    /// buffer allocations, same capture call), just returning a <see cref="GraphDecodeSession"/> that
    /// outlives one method call instead of local variables scoped to one generation. On failure, disposes
    /// whatever was already allocated before rethrowing — no partial session leaks.</summary>
    private GraphDecodeSession CaptureGraphSession(FixedKvCache cache, int promptLen, int firstToken, float repetitionPenalty)
    {
        // Checked before any real backend call so a test can simulate a capture failure without needing an
        // actual CUDA-capable backend (see TestGraphCaptureFailureInjector's doc) — every call attempted
        // after this point genuinely touches the backend, so a production capture failure is always real.
        Exception? injected = TestGraphCaptureFailureInjector?.Invoke();
        if (injected is not null) throw injected;

        // Found live testing this retrofit: a genuinely COLD model's first-ever solo admission reliably
        // failed capture with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED inside DenseFfn's weight projection —
        // prefill's multi-token GEMM shape apparently promotes weights lazily via a different path than the
        // single-token GEMV shape decode uses, so the FIRST-EVER single-token forward pass for this model
        // can trigger a non-stream-ordered "auto-promote to GPU" allocation, which CUDA forbids mid-capture.
        // Every later solo admission is unaffected (once promoted, a weight stays promoted for the model's
        // lifetime) — this is a one-time, first-request-only cost. Forcing that promotion via a real
        // single-token forward pass HERE, before capture starts, using a throwaway cache (weight promotion
        // is cached on the model/backend, not tied to any specific cache instance, so this warms up exactly
        // what the real capture will need without touching the real sequence's cache/position state at all)
        // moves the failure-prone allocation safely outside the capture region. This is a pre-existing gap in
        // TextGenerationPipeline.GenerateGraphDecode too (confirmed: reproduces there identically, no
        // scheduler code involved) — not something this retrofit introduced, but this is the first code path
        // that realistically hits it on a cold model (a single CLI process rarely captures graph decode
        // completely cold the way a freshly-loaded server model's first request does).
        // Every weight touched by a single-token-shaped (GEMV) forward pass needs this — not just the
        // per-layer MLP/attention projections but also the LM head (ProjectLogits), which is a SEPARATE
        // call from Forward and needs its own single-token-shaped warm-up (prefill's own ProjectLogits call
        // uses promptLen>1 positions — a GEMM shape — so it doesn't warm the GEMV path either).
        TransformerConfig warmupCfg = _model.Config;
        using (FixedKvCache warmup = new(warmupCfg.NumLayers, 1, warmupCfg.NumKvHeads, HeadDimPerLayer(warmupCfg), maxSequenceLength: 2))
        {
            using Tensor warmupHidden = _model.Forward(_backend, [firstToken], 0, warmup);
            _model.ProjectLogits(_backend, warmupHidden, 1).Dispose();
        }

        Tensor embedTable = _model.EnsureEmbedResidentForGraphDecode(_backend);
        (Tensor cosTable, Tensor sinTable) = _model.EnsureRopeTableForGraphDecode(_backend, cache.MaxSequenceLength);
        ulong devicePos = _backend.AllocDevicePos();
        ulong deviceTokenId = _backend.AllocDeviceTokenId();
        ulong history = _backend.AllocDeviceHistory(cache.MaxSequenceLength);
        ulong historyCount = _backend.AllocDeviceCounter();
        object? graph = null;
        try
        {
            int pos = promptLen;
            _backend.WriteDeviceTokenId(deviceTokenId, firstToken);
            _backend.WriteDevicePos(devicePos, pos + 1, pos);
            _backend.WriteDeviceCounter(historyCount, 0);
            graph = _backend.CaptureGraph(() =>
                _model.ForwardGraphDecodeStep(_backend, embedTable, cache, cosTable, sinTable, devicePos, deviceTokenId,
                    history, historyCount, repetitionPenalty));
            return new GraphDecodeSession(_backend, graph!, devicePos, deviceTokenId, history, historyCount, pos);
        }
        catch
        {
            if (graph is not null) _backend.DisposeGraph(graph);
            _backend.FreeDevicePos(devicePos);
            _backend.FreeDeviceTokenId(deviceTokenId);
            _backend.FreeDeviceHistory(history);
            _backend.FreeDeviceCounter(historyCount);
            throw;
        }
    }

    private void RunDecodeRound(List<ActiveSeq> feeders)
    {
        TransformerConfig cfg = _model.Config;
        int bn = feeders.Count;
        int[] tokens = new int[bn];
        int[] positions = new int[bn];
        IKvCache[] caches = new IKvCache[bn];
        for (int b = 0; b < bn; b++)
        {
            ActiveSeq seq = feeders[b];
            tokens[b] = seq.Next;
            positions[b] = seq.Cache.CurrentLength;
            caches[b] = seq.Cache;
            seq.Generated.Add(seq.Next);
            seq.Pending.OnToken?.Invoke(seq.Next);
        }

        using Tensor embeds = new(new TensorShape(1, bn, cfg.HiddenSize), DType.F32);
        _model.EmbedLookup(embeds, tokens);
        using Tensor hidden = _model.ForwardBatchDecode(_backend, embeds, positions, caches);
        using Tensor logits = _model.ProjectLogits(_backend, hidden, bn);
        for (int b = 0; b < bn; b++)
            feeders[b].Next = feeders[b].Sampler.Next(LastRow(logits, b + 1, cfg.VocabSize), feeders[b].Generated);
    }

    /// <summary>One round for a lone sequence with a captured graph — mirrors
    /// <see cref="TextGenerationPipeline"/>'s <c>GenerateGraphDecode</c> per-step loop body exactly (add the
    /// current token, invoke the callback, THEN replay to get the next one) so the scheduler's existing
    /// stop/MaxTokens eviction check — which runs BEFORE this round, on <see cref="ActiveSeq.Next"/> — stays
    /// exactly as correct here as it already is for <see cref="RunDecodeRound"/>. <see cref="IKvCache.AdvanceLength"/>
    /// is called explicitly here (unlike the CLI path, which never needs to since it never mixes a
    /// <see cref="FixedKvCache"/> into a later eager multi-sequence round) — required so
    /// <see cref="ActiveSeq.Cache"/>'s position bookkeeping stays correct if this sequence is later joined by
    /// another and falls back to <see cref="RunDecodeRound"/>'s eager path, which reads
    /// <see cref="IKvCache.CurrentLength"/> to compute that round's position.</summary>
    private void ReplayGraphRound(ActiveSeq seq, GraphDecodeSession gs)
    {
        seq.Generated.Add(seq.Next);
        seq.Pending.OnToken?.Invoke(seq.Next);
        int next = gs.Replay();
        gs.Pos++;
        gs.WriteNextPos();
        seq.Cache.AdvanceLength(1);
        seq.Next = next;
    }

    /// <summary>Builds the final result and completes the request's task. Does NOT dispose <paramref name="seq"/>
    /// — callers batch every evicted sequence's disposal into one gated <see cref="RunGpuWork"/> call (see
    /// <see cref="RunLoopAsync"/>) rather than disposing inline here, one at a time, ungated.</summary>
    private void CompleteSeq(ActiveSeq seq, bool stopped)
    {
        GenerationResult result = new()
        {
            TokenIds = seq.Generated,
            Text = _tokenizer.Decode(seq.Generated),
            PromptTokens = seq.PromptIds.Length,
            StoppedOnStopToken = stopped,
        };
        seq.Pending.Completion.TrySetResult(result);
    }

    private int[] BuildPromptIds(GenerationRequest request)
    {
        if (request.RawTokenIds is not null) return [.. request.RawTokenIds];
        if (request.Messages is not null) return _template.Encode(_tokenizer, request.Messages, addGenerationPrompt: true);
        if (request.Prompt is not null)
        {
            List<ChatMessage> messages = new(2);
            if (!string.IsNullOrEmpty(request.SystemPrompt)) messages.Add(ChatMessage.System(request.SystemPrompt));
            messages.Add(ChatMessage.User(request.Prompt));
            return _template.Encode(_tokenizer, messages, addGenerationPrompt: true);
        }
        throw new ArgumentException("Request must set RawTokenIds, Messages, or Prompt.", nameof(request));
    }

    private static unsafe Span<float> LastRow(Tensor logits, int t, int vocab)
    {
        float* p = (float*)logits.DataPointer;
        return new Span<float>(p + (long)(t - 1) * vocab, vocab);
    }

    /// <summary>Stops the background loop and fails every still-active/pending request. Does NOT dispose the
    /// shared <see cref="PagedKvPool"/> (the caller owns it and may share it across schedulers/sessions).</summary>
    public void Dispose()
    {
        _shutdown.Cancel();
        _incoming.Writer.TryComplete();
        try { _loopTask.Wait(TimeSpan.FromSeconds(5)); } catch { /* best-effort shutdown */ }
        _shutdown.Dispose();
    }
}
