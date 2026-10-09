using HartsyInference.Core.Configuration;
using System.Threading.Channels;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Generation;

/// <summary>True continuous-batching scheduler: admits requests at ANY time via <see cref="SubmitAsync"/> and evicts each sequence the moment it finishes/stops/cancels, rather than waiting for the whole cohort like the static-batch design it replaces.</summary>
/// <remarks>
/// <para>A single dedicated background loop owns the model/backend and every active sequence's mutable state exclusively — external callers only ever touch a thread-safe <see cref="Channel{T}"/>, never the model directly — which is what makes it safe to call <see cref="SubmitAsync"/> from multiple concurrent callers even though the underlying backend is not itself safely re-entrant (see <c>InferenceQueue</c>'s doc comment on that constraint, which this sidesteps by construction rather than by serializing callers).</para>
/// <para>Each round: (1) drain newly-submitted requests, prefilling and admitting each one (single-sequence prefill, matching the design this replaces — chunked/batched prefill is a further throughput optimization, not required for correctness, and is left as a documented follow-up); (2) evict any sequence that is cancelled, just hit a stop token, or hit its token limit — BEFORE running a decode round, so a sequence never wastes a batched step after it should have stopped; (3) run one batched decode step (<see cref="GenericTransformer.ForwardBatchDecode"/>) over every remaining active sequence. KV storage comes from a <see cref="PagedKvPool"/> shared across every active sequence — admission fails fast with <see cref="KvPoolExhaustedException"/> when the pool has no room, rather than blocking or evicting something else (reject policy, matching the pool's own design).</para>
/// <para><b>Backend exclusivity:</b> multiple concurrent <see cref="SubmitAsync"/> callers is exactly the point (that's what lets requests batch together), but the shared <see cref="IBackend"/> instance is NOT itself safely re-entrant (one CUDA stream, non-thread-safe activation/weight caches) — and on a server that also runs diffusion image generation through the SAME backend instance via a separate queue, this scheduler's GPU work must never overlap with THAT either. So every GPU-touching step (prefill, one decode round) is gated through the optional <paramref name="gpuGate"/> — pass the server's existing <c>InferenceQueue</c> (shared with diffusion) to keep the whole server down to one physical GPU operation at a time, while still batching every concurrently-submitted chat request into that one operation. This does NOT serialize chat requests the way routing each whole request through the queue would (that was tried first and rejected — it would gate one call to <see cref="SubmitAsync"/> at a time, so a second request could never even be ADMITTED into the batch until the first one's entire generation finished, defeating the purpose); only the actual GPU round is gated, and rounds already contain every request that arrived since the last one.</para>
/// </remarks>
public sealed class DynamicBatchScheduler : IBatchScheduler, IDisposable
{
    private readonly IGenerationModel _model;
    private readonly IGraphDecodable? _graphModel;
    private readonly ILlmTokenizer _tokenizer;
    private readonly IChatTemplate _template;
    private readonly IBackend _backend;
    private readonly PagedKvPool? _pool;
    private readonly HashSet<int> _stopIds;
    private readonly Channel<PendingRequest> _incoming;
    private readonly CancellationTokenSource _shutdown = new();
    private readonly Task _loopTask;
    private readonly Func<Action, Task>? _gpuGate;
    private int _disposed;
    private readonly int _maxQueued;
    private readonly int _maxActive;

    /// <summary>Requests admitted into the waiting queue and not yet admitted or failed; read and written only under <see cref="_waitingGate"/>
    /// except by <see cref="SubmitAsync"/>'s bound check, which is an <see cref="Interlocked"/> counter.</summary>
    private int _queued;

    /// <summary>Submitted requests in arrival order that have not been admitted yet; guarded by <see cref="_waitingGate"/>.</summary>
    private readonly Queue<PendingRequest> _waiting = new();
    private readonly object _waitingGate = new();

    /// <summary>Pages reserved by the active sequences (each reserves its whole prompt-plus-budget footprint at admission); loop-owned, and always zero on a model with no pool.</summary>
    private int _reservedPages;

    /// <summary>Queue places to announce after an admission pass, collected under <see cref="_waitingGate"/> and invoked after it; loop-owned and reused.</summary>
    private readonly List<(Action<int> OnQueued, int Place)> _announcements = [];

    private sealed class PendingRequest
    {
        public required GenerationRequest Request;
        public Action<int>? OnToken;
        public required CancellationToken Ct;
        public required TaskCompletionSource<GenerationResult> Completion;

        /// <summary>The prompt's ids, built once when the request first reaches the head of the queue.</summary>
        public int[]? PromptIds;

        /// <summary>Whether this request has been told its place in the queue; it is told once.</summary>
        public bool Announced;
    }

    /// <summary>One active sequence's per-request state; <see cref="Cache"/> is <see cref="IKvCache"/> rather than concretely <see cref="PagedKvCache"/> so an idle-admitted sequence can use a dedicated <see cref="FixedKvCache"/> instead (see <c>docs/Checklists/LLM_DECODE_PERF_GRIND.md</c>'s "NEW PLAN"). <see cref="GraphSession"/> is non-null only for such a sequence while still eligible for graph replay (see <see cref="RunLoopAsync"/>'s one-way retirement); <see cref="Dispose"/> is intentionally the ONLY way callers free this sequence's resources so a session can never be forgotten at a disposal call site.</summary>
    private sealed class ActiveSeq : IDisposable
    {
        public required PendingRequest Pending;
        public required int[] PromptIds;
        public required ISequenceState Cache;
        public required SamplerChain Sampler;
        public required HashSet<int> Stops;
        public required List<int> Generated;
        public int Next;
        public GraphDecodeSession? GraphSession;

        /// <summary>The result the request completes with once this sequence is released; null for a cancelled request.</summary>
        public GenerationResult? Result;

        /// <summary>Pages this sequence reserved at admission; returned to <see cref="_reservedPages"/> when it leaves the active set.</summary>
        public int ReservedPages;

        public void Dispose()
        {
            Cache.Dispose();
            GraphSession?.Dispose();
        }
    }

    /// <summary><paramref name="pool"/> is shared across every sequence this scheduler admits (size it for the concurrency you want, not per-request); <paramref name="gpuGate"/>, when supplied, wraps every GPU-touching step through the server's shared backend-exclusivity queue (see class doc "Backend exclusivity") — omit only when nothing else can contend for the same backend instance.</summary>
    public DynamicBatchScheduler(GenericTransformer model, ILlmTokenizer tokenizer, IBackend backend,
        PagedKvPool? pool, IChatTemplate? template = null, Func<Action, Task>? gpuGate = null,
        int maxQueued = DefaultMaxQueued, int maxActiveSequences = int.MaxValue)
        : this(new GenericTransformerModel(model, backend), tokenizer, pool, template, gpuGate, maxQueued, maxActiveSequences)
    {
    }

    /// <summary>Drives any <see cref="IGenerationModel"/>; the scheduler does not own the model. <paramref name="pool"/> is the KV pool its sequences draw from, or null for a model whose sequence states own their storage (the V4.1 host model).</summary>
    /// <param name="maxQueued">Requests that may wait for admission at once; one more is refused with <see cref="SchedulerQueueFullException"/>.</param>
    /// <param name="maxActiveSequences">Sequences decoding at once. A pooled model is bounded by its pages already; a pool-less model (the V4.1 host) is bounded by this alone.</param>
    public DynamicBatchScheduler(IGenerationModel model, ILlmTokenizer tokenizer,
        PagedKvPool? pool, IChatTemplate? template = null, Func<Action, Task>? gpuGate = null,
        int maxQueued = DefaultMaxQueued, int maxActiveSequences = int.MaxValue)
    {
        _maxQueued = Math.Max(1, maxQueued);
        _maxActive = Math.Max(1, maxActiveSequences);
        _model = model;
        _graphModel = model as IGraphDecodable;
        _tokenizer = tokenizer;
        _backend = model.OutputBackend;
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

    /// <summary>True while the background loop is running; goes false cleanly on <see cref="Dispose"/> or if it faults — callers tracking serving health should treat <c>false</c> after successful construction as "this model's chat traffic is dead and won't recover without reloading the model."</summary>
    public bool IsLoopAlive => !_loopTask.IsCompleted;

    /// <summary>Test-only fault injection: when set, invoked once per decode round with that round's feeder count; a non-null return is thrown instead of running the round, so fault-isolation behavior can be tested deterministically without reproducing a real backend crash. Null in production.</summary>
    internal Func<int, Exception?>? TestFaultInjector { get; set; }

    /// <summary>Test-only override for the "does this architecture/backend support graph decode" check in <see cref="AdmitAndPrefill"/> — the CPU backend never satisfies this for real, so a CPU-backend unit test can't otherwise reach the solo-admission graph-capture path. Null in production.</summary>
    internal bool? TestForceSupportsGraphDecode { get; set; }

    /// <summary>Test-only fault injection for <see cref="CaptureGraphSession"/>: when set, invoked once per capture attempt; a non-null return is thrown instead of actually capturing, exercising the <see cref="_graphCaptureUnavailable"/> circuit breaker without a real CUDA capture failure. Null in production.</summary>
    internal Func<Exception?>? TestGraphCaptureFailureInjector { get; set; }

    private async Task RunGpuWork(Action work)
    {
        if (_gpuGate is null) { work(); return; }
        await _gpuGate(work).ConfigureAwait(false);
    }

    /// <summary>The default number of requests that may wait for admission at once.</summary>
    public const int DefaultMaxQueued = 64;

    /// <summary>Queues a request for admission. When <c>maxQueued</c> requests already wait, the request fails at once with <see cref="SchedulerQueueFullException"/>, which the API answers with 429.</summary>
    public Task<GenerationResult> SubmitAsync(GenerationRequest request, Action<int>? onToken, CancellationToken ct)
    {
        TaskCompletionSource<GenerationResult> tcs = new(TaskCreationOptions.RunContinuationsAsynchronously);
        if (Interlocked.Increment(ref _queued) > _maxQueued)
        {
            Interlocked.Decrement(ref _queued);
            tcs.SetException(new SchedulerQueueFullException(_maxQueued));
            return tcs.Task;
        }
        PendingRequest pending = new() { Request = request, OnToken = onToken, Ct = ct, Completion = tcs };
        if (!_incoming.Writer.TryWrite(pending))
        {
            Interlocked.Decrement(ref _queued);
            tcs.SetException(new SchedulerStoppedException());
        }
        return tcs.Task;
    }

    private async Task RunLoopAsync()
    {
        List<ActiveSeq> active = [];
        try
        {
            while (!_shutdown.IsCancellationRequested)
            {
                DrainIncoming();
                await AdmitWaitingAsync(active).ConfigureAwait(false);
                // Shutdown requested while this round admitted or waited: no round runs after it, so every
                // unfinished sequence fails in the finally below instead of producing tokens after Dispose.
                if (_shutdown.IsCancellationRequested) break;
                AnnounceWaiting();

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
                        evicted.Add(seq);
                        Retire(seq);
                        active.RemoveAt(i);
                        continue;
                    }
                    bool stoppedNow = seq.Stops.Contains(seq.Next);
                    bool atLimit = seq.Generated.Count >= seq.Pending.Request.MaxTokens;
                    if (stoppedNow || atLimit)
                    {
                        seq.Result = BuildResult(seq, stoppedNow);
                        evicted.Add(seq);
                        Retire(seq);
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
                    await ReleaseThenCompleteAsync(evicted, static seq =>
                    {
                        if (seq.Result is { } result) seq.Pending.Completion.TrySetResult(result);
                        else seq.Pending.Completion.TrySetCanceled(seq.Pending.Ct);
                    }).ConfigureAwait(false);
                }
                if (feeders.Count == 0) continue;

                // Dispatch: a lone feeder carrying a captured graph replays it (one launch, no per-round
                // kernel-issuance overhead); anything else — multiple feeders, or a solo feeder with no
                // session — takes the existing eager batched path. A feeder that has a session but is NOT
                // alone this round gets it retired (disposed, one-way — see ActiveSeq.GraphSession's doc)
                // before the eager round runs, so it never sits around going stale.
                Action work = feeders.Count == 1 && feeders[0].GraphSession is { } gs
                    ? () => ReplayGraphRound(feeders[0], gs) : () =>
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
                        Retire(seq);
                        active.Remove(seq);
                    }
                    await ReleaseThenCompleteAsync(feeders, seq => seq.Pending.Completion.TrySetException(ex)).ConfigureAwait(false);
                }
            }
        }
        finally
        {
            // Closed first: a submit after the loop stops fails with SchedulerStoppedException instead of waiting for a loop that will never read it.
            _incoming.Writer.TryComplete();
            FailQueued();
            if (active.Count > 0)
                await ReleaseThenCompleteAsync(active, static seq => seq.Pending.Completion.TrySetException(new SchedulerStoppedException())).ConfigureAwait(false);
        }
    }

    /// <summary>Releases <paramref name="seqs"/> in one gated call, then completes each request with <paramref name="complete"/>, even when the release throws. A request
    /// completes only after its sequence is released, so a caller that frees the model once its requests have completed (TextService's slot lease) never leaves a
    /// teardown behind that still needs the device gate.</summary>
    private async Task ReleaseThenCompleteAsync(List<ActiveSeq> seqs, Action<ActiveSeq> complete)
    {
        try
        {
            await RunGpuWork(() => { foreach (ActiveSeq seq in seqs) seq.Dispose(); }).ConfigureAwait(false);
        }
        finally
        {
            foreach (ActiveSeq seq in seqs) complete(seq);
        }
    }

    /// <summary>Moves submitted requests into the waiting queue in arrival order. A request cancelled, or read after shutdown began, leaves here instead. A request is told its
    /// place only if the admission pass that follows leaves it waiting (see <see cref="AnnounceWaiting"/>).</summary>
    private void DrainIncoming()
    {
        lock (_waitingGate)
        {
            while (_incoming.Reader.TryRead(out PendingRequest? pending))
            {
                if (pending.Ct.IsCancellationRequested)
                {
                    Leave();
                    pending.Completion.TrySetCanceled(pending.Ct);
                    continue;
                }
                // Read after shutdown was requested: failed, not admitted, since the loop is about to stop.
                if (_shutdown.IsCancellationRequested)
                {
                    Leave();
                    pending.Completion.TrySetException(new SchedulerStoppedException());
                    continue;
                }
                _waiting.Enqueue(pending);
            }
        }
    }

    /// <summary>Admits waiting requests in arrival order while the head fits. A head that does not fit holds everyone behind it (FIFO). On a pooled model a request fits when its
    /// whole footprint (prompt plus generation budget, in pages) fits beside the pages already reserved, so an admitted sequence never runs out of pages mid-decode. A request
    /// larger than the whole pool is refused now, since waiting can never make it fit.</summary>
    private async Task AdmitWaitingAsync(List<ActiveSeq> active)
    {
        SweepCancelled();
        while (true)
        {
            PendingRequest? head;
            lock (_waitingGate) head = _waiting.Count > 0 ? _waiting.Peek() : null;
            if (head is null) return;

            if (head.Ct.IsCancellationRequested)
            {
                RemoveHead(head);
                head.Completion.TrySetCanceled(head.Ct);
                continue;
            }

            long pages;
            try
            {
                head.PromptIds ??= BuildPromptIds(head.Request);
                pages = _pool is null ? 0 : PagesFor(head.PromptIds.Length, head.Request.MaxTokens, _pool.PageSize);
            }
            catch (Exception ex)
            {
                RemoveHead(head);
                head.Completion.TrySetException(ex);
                continue;
            }
            if (Refusal(head.PromptIds.Length, head.Request.MaxTokens, pages) is { } refusal)
            {
                RemoveHead(head);
                head.Completion.TrySetException(refusal);
                continue;
            }
            if (active.Count >= _maxActive) return;
            if (_pool is not null && _reservedPages + pages > _pool.MaxPages) return;

            RemoveHead(head);
            try
            {
                ActiveSeq? seq = null;
                // Recomputed fresh for EACH admitted request: active.Count grows as this loop admits earlier requests, so a request admitted after them is non-solo.
                bool solo = active.Count == 0;
                await RunGpuWork(() => seq = AdmitAndPrefill(head, solo)).ConfigureAwait(false);
                // At most MaxPages here (Refusal), so the count fits an int.
                seq!.ReservedPages = (int)pages;
                _reservedPages += (int)pages;
                active.Add(seq);
            }
            catch (Exception ex)
            {
                Logs.Error("DynamicBatchScheduler: admission/prefill failed for one request", ex);
                head.Completion.TrySetException(ex);
            }
        }
    }

    /// <summary>Takes every cancelled request off the waiting queue, wherever it stands, and completes it as cancelled, so a caller that gave up stops holding a place in
    /// the bounded queue while the requests ahead of it are still waiting.</summary>
    private void SweepCancelled()
    {
        List<PendingRequest>? cancelled = null;
        lock (_waitingGate)
        {
            bool any = false;
            foreach (PendingRequest pending in _waiting)
                any |= pending.Ct.IsCancellationRequested;
            // One rotation through the queue keeps arrival order for everyone who stays.
            for (int i = any ? _waiting.Count : 0; i > 0; i--)
            {
                PendingRequest pending = _waiting.Dequeue();
                if (pending.Ct.IsCancellationRequested) (cancelled ??= []).Add(pending);
                else _waiting.Enqueue(pending);
            }
        }
        if (cancelled is null) return;
        foreach (PendingRequest pending in cancelled)
        {
            Leave();
            pending.Completion.TrySetCanceled(pending.Ct);
        }
    }

    /// <summary>Tells each request still waiting after an admission pass its place in the queue (1 is next), once. A request admitted in the pass that took it in hears
    /// nothing. The callbacks run after the lock is released, so a slow one holds up neither submitters nor <see cref="FailQueued"/>.</summary>
    private void AnnounceWaiting()
    {
        lock (_waitingGate)
        {
            int place = 0;
            foreach (PendingRequest pending in _waiting)
            {
                place++;
                if (pending.Announced || pending.Request.OnQueued is not { } onQueued) continue;
                pending.Announced = true;
                _announcements.Add((onQueued, place));
            }
        }
        foreach ((Action<int> onQueued, int place) in _announcements)
            onQueued(place);
        _announcements.Clear();
    }

    /// <summary>Takes <paramref name="head"/> off the waiting queue. Does nothing when shutdown has already completed it and taken it off the queue.</summary>
    private void RemoveHead(PendingRequest head)
    {
        lock (_waitingGate)
        {
            if (_waiting.Count == 0 || !ReferenceEquals(_waiting.Peek(), head)) return;
            _waiting.Dequeue();
        }
        Leave();
    }

    /// <summary>Returns a sequence's page reservation when it leaves the active set.</summary>
    private void Retire(ActiveSeq seq) => _reservedPages -= seq.ReservedPages;

    /// <summary>One fewer request waiting: called wherever a submitted request leaves the queue.</summary>
    private void Leave() => Interlocked.Decrement(ref _queued);

    /// <summary>The pages a sequence of this many prompt tokens and generation budget can grow to, at <paramref name="pageSize"/> tokens a page. Long arithmetic, so a huge
    /// budget gives a huge count instead of wrapping to a small or negative one that would pass the pool checks.</summary>
    private static long PagesFor(int promptTokens, int maxTokens, int pageSize) => ((long)promptTokens + maxTokens + 1 + pageSize - 1) / pageSize;

    /// <summary>Why a request can never be admitted, or null when it can wait its turn: a negative budget, a sequence longer than any state can hold, or on a pooled model a
    /// page count that is not positive or exceeds the whole pool. Each is the caller's to fix, so it is an <see cref="ArgumentException"/>, which the API answers with 400.</summary>
    private ArgumentException? Refusal(int promptTokens, int maxTokens, long pages)
    {
        if (maxTokens < 0)
            return new ArgumentException($"max_tokens must not be negative (got {maxTokens}).", nameof(GenerationRequest));
        if ((long)promptTokens + maxTokens + 1 > int.MaxValue)
        {
            return new ArgumentException(
                $"The prompt ({promptTokens} tokens) plus max_tokens ({maxTokens}) is longer than a sequence can hold; lower max_tokens.", nameof(GenerationRequest));
        }
        if (_pool is not null && (pages <= 0 || pages > _pool.MaxPages))
        {
            return new ArgumentException(
                $"The request needs {pages} KV pages of {_pool.PageSize} tokens, but the pool holds {_pool.MaxPages}; shorten the prompt or max_tokens.", nameof(GenerationRequest));
        }
        return null;
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

    /// <summary>True once a CUDA-graph capture has failed once for this scheduler's model; a capture failure is architecture/backend-determined, not request-specific, so this is a circuit breaker rather than a per-request retry.</summary>
    private bool _graphCaptureUnavailable;

    private ActiveSeq AdmitAndPrefill(PendingRequest pending, bool solo)
    {
        GenerationRequest req = pending.Request;
        int[] promptIds = pending.PromptIds ??= BuildPromptIds(req);
        if (promptIds.Length == 0) throw new ArgumentException("Request produced zero tokens.");
        HashSet<int> stops = _stopIds;
        if (req.StopTokenIds is not null) { stops = [.. _stopIds]; foreach (int s in req.StopTokenIds) stops.Add(s); }

        int vocab = _model.Info.VocabSize;
        // Only a request admitted while the scheduler is otherwise idle, and only when it's eligible for the
        // SAME reasons TextGenerationPipeline.Generate's graph-decode dispatch requires (greedy,
        // SupportsGraphDecode), plus non-JSON-mode (that pipeline's own graph-decode step bypasses the CPU
        // sampler chain entirely — including JsonGrammarStep — so combining the two would silently produce
        // non-JSON output; excluded here even though the existing single-sequence pipeline doesn't exclude
        // it, since the server is a much larger audience for this gap than the CLI). Gets a dedicated
        // FixedKvCache instead of drawing from the shared pool — see DynamicBatchScheduler's class doc and
        // ActiveSeq's GraphSession field doc for why this is safe and why it's a one-way admission decision
        // (never converted later).
        bool graphEligible = solo && !_graphCaptureUnavailable && req.Sampling.Greedy && !req.Sampling.HasJsonConstraint
            && (req.GraphDecode ?? EngineKnobs.GraphDecode.Value)
            && _graphModel is not null
            && (TestForceSupportsGraphDecode ?? _graphModel.SupportsGraphDecode(_backend));

        // Throws KvPoolExhaustedException if the pool can't fit the prompt (PagedKvCache path only) —
        // propagates to the caller's SubmitAsync task as a fault, the reject policy PagedKvPool documents.
        // Stays F32 regardless of vram.kvF16: this branch exists specifically BECAUSE graphEligible, and
        // FlashAttentionDev refuses F16-storage KV (v1 scope — see CudaBackend), silently falling back to
        // eager per-token. Honoring the switch here would sabotage the very feature this branch selects for.
        ISequenceState cache = graphEligible
            ? _model.CreateSequenceState(new SequenceStateOptions(promptIds.Length + req.MaxTokens + 1, FullPrecisionKv: true))
            : _model.CreateSequenceState(new SequenceStateOptions(promptIds.Length + req.MaxTokens + 1, _pool));
        try
        {
            List<int> generated = new(req.MaxTokens);
            SamplerChain sampler = SamplerChain.FromOptions(req.Sampling, _tokenizer, vocab);
            int next;
            using (Tensor hidden = _model.Prefill(new PrefillChunk(promptIds, 0), cache))
            using (Tensor logits = _model.ProjectLogits(hidden, promptIds.Length))
                next = sampler.Next(LastRow(logits, promptIds.Length, vocab), generated);

            // Fired once the prompt is in the cache, as TextGenerationPipeline does: the caller reads the prompt length from it.
            req.OnPrefillCompleted?.Invoke(promptIds.Length);

            GraphDecodeSession? session = null;
            if (graphEligible)
            {
                try
                {
                    session = CaptureGraphSession(cache, promptIds.Length, next, req.Sampling.RepetitionPenalty);
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

    /// <summary>Captures one CUDA graph for greedy decode against <paramref name="cache"/>, mirroring <see cref="TextGenerationPipeline"/>'s <c>GenerateGraphDecode</c> capture step but returning a <see cref="GraphDecodeSession"/> that outlives one method call; disposes whatever was already allocated before rethrowing on failure.</summary>
    private GraphDecodeSession CaptureGraphSession(ISequenceState cache, int promptLen, int firstToken, float repetitionPenalty)
    {
        // Checked before any real backend call so a test can simulate a capture failure without needing an
        // actual CUDA-capable backend (see TestGraphCaptureFailureInjector's doc) — every call attempted
        // after this point genuinely touches the backend, so a production capture failure is always real.
        Exception? injected = TestGraphCaptureFailureInjector?.Invoke();
        if (injected is not null) throw injected;

        // Warmup, residency and capture live in the model (see GenericTransformerModel.CaptureDecodeGraph).
        return _graphModel!.CaptureDecodeGraph(cache, promptLen, firstToken, repetitionPenalty);
    }

    private void RunDecodeRound(List<ActiveSeq> feeders)
    {
        int vocab = _model.Info.VocabSize;
        int bn = feeders.Count;
        int[] tokens = new int[bn];
        ISequenceState[] states = new ISequenceState[bn];
        for (int b = 0; b < bn; b++)
        {
            ActiveSeq seq = feeders[b];
            tokens[b] = seq.Next;
            states[b] = seq.Cache;
            seq.Generated.Add(seq.Next);
            seq.Pending.OnToken?.Invoke(seq.Next);
        }

        using Tensor hidden = _model.DecodeBatch(tokens, states);
        using Tensor logits = _model.ProjectLogits(hidden, bn);
        for (int b = 0; b < bn; b++)
            feeders[b].Next = feeders[b].Sampler.Next(LastRow(logits, b + 1, vocab), feeders[b].Generated);
    }

    /// <summary>One round for a lone sequence with a captured graph — mirrors <see cref="TextGenerationPipeline"/>'s <c>GenerateGraphDecode</c> loop body order (add token, invoke callback, THEN replay) so the eviction check on <see cref="ActiveSeq.Next"/> stays correct; commits the replayed token explicitly (<see cref="IGraphDecodable.CommitReplayedStep"/>) so <see cref="ActiveSeq.Cache"/>'s position stays correct if this sequence is later joined by another and falls back to <see cref="RunDecodeRound"/>'s eager path.</summary>
    private void ReplayGraphRound(ActiveSeq seq, GraphDecodeSession gs)
    {
        seq.Generated.Add(seq.Next);
        seq.Pending.OnToken?.Invoke(seq.Next);
        int next = gs.Replay();
        gs.Pos++;
        gs.WriteNextPos();
        _graphModel!.CommitReplayedStep(seq.Cache);
        seq.Next = next;
    }

    /// <summary>The final result of a finished sequence. The request completes with it only after the sequence is released (see <see cref="ReleaseThenCompleteAsync"/>).</summary>
    private GenerationResult BuildResult(ActiveSeq seq, bool stopped) => new()
    {
        TokenIds = seq.Generated,
        Text = _tokenizer.Decode(seq.Generated),
        PromptTokens = seq.PromptIds.Length,
        StoppedOnStopToken = stopped,
    };

    private int[] BuildPromptIds(GenerationRequest request) => PromptBuilder.BuildPromptIds(request, _tokenizer, _template);

    private static unsafe Span<float> LastRow(Tensor logits, int t, int vocab)
    {
        float* p = (float*)logits.DataPointer;
        return new Span<float>(p + (long)(t - 1) * vocab, vocab);
    }

    /// <summary>Stops the background loop and fails every request it has not finished: queued requests fail at once, and active sequences fail when the loop exits. Each fails with <see cref="SchedulerStoppedException"/>, so no caller waits on a loop that will not run. Does NOT dispose the shared <see cref="PagedKvPool"/> (the caller owns it and may share it across schedulers/sessions).</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _shutdown.Cancel();
        _incoming.Writer.TryComplete();
        FailQueued();
        try { _loopTask.Wait(TimeSpan.FromSeconds(5)); } catch { /* best-effort shutdown */ }
        // Only once the loop has stopped: it still reads the token while it runs.
        if (_loopTask.IsCompleted) _shutdown.Dispose();
    }

    /// <summary>Completes every request still waiting, both submitted and in the admission queue: as cancelled when its own token already fired, else with <see cref="SchedulerStoppedException"/>.</summary>
    private void FailQueued()
    {
        while (_incoming.Reader.TryRead(out PendingRequest? pending))
        {
            Leave();
            if (pending.Ct.IsCancellationRequested) pending.Completion.TrySetCanceled(pending.Ct);
            else pending.Completion.TrySetException(new SchedulerStoppedException());
        }
        lock (_waitingGate)
        {
            while (_waiting.Count > 0)
            {
                PendingRequest waiting = _waiting.Dequeue();
                Leave();
                if (waiting.Ct.IsCancellationRequested) waiting.Completion.TrySetCanceled(waiting.Ct);
                else waiting.Completion.TrySetException(new SchedulerStoppedException());
            }
        }
    }
}
