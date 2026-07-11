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

    private sealed class ActiveSeq
    {
        public required PendingRequest Pending;
        public required int[] PromptIds;
        public required PagedKvCache Cache;
        public required SamplerChain Sampler;
        public required HashSet<int> Stops;
        public required List<int> Generated;
        public int Next;
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
                for (int i = active.Count - 1; i >= 0; i--)
                {
                    ActiveSeq seq = active[i];
                    if (seq.Pending.Ct.IsCancellationRequested)
                    {
                        seq.Cache.Dispose();
                        seq.Pending.Completion.TrySetCanceled(seq.Pending.Ct);
                        active.RemoveAt(i);
                        continue;
                    }
                    bool stoppedNow = seq.Stops.Contains(seq.Next);
                    bool atLimit = seq.Generated.Count >= seq.Pending.Request.MaxTokens;
                    if (stoppedNow || atLimit)
                    {
                        CompleteSeq(seq, stoppedNow);
                        active.RemoveAt(i);
                        continue;
                    }
                    feeders.Add(seq);
                }
                if (feeders.Count == 0) continue;

                try
                {
                    Exception? injected = TestFaultInjector?.Invoke(feeders.Count);
                    if (injected is not null) throw injected;
                    await RunGpuWork(() => RunDecodeRound(feeders)).ConfigureAwait(false);
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
                        seq.Cache.Dispose();
                        seq.Pending.Completion.TrySetException(ex);
                        active.Remove(seq);
                    }
                }
            }
        }
        finally
        {
            foreach (ActiveSeq seq in active)
            {
                seq.Cache.Dispose();
                seq.Pending.Completion.TrySetException(new ObjectDisposedException(nameof(DynamicBatchScheduler)));
            }
        }
    }

    private async Task DrainIncomingAsync(List<ActiveSeq> active)
    {
        while (_incoming.Reader.TryRead(out PendingRequest? pending))
        {
            if (pending.Ct.IsCancellationRequested) { pending.Completion.TrySetCanceled(pending.Ct); continue; }
            try
            {
                ActiveSeq? seq = null;
                await RunGpuWork(() => seq = AdmitAndPrefill(pending)).ConfigureAwait(false);
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

    private ActiveSeq AdmitAndPrefill(PendingRequest pending)
    {
        GenerationRequest req = pending.Request;
        int[] promptIds = BuildPromptIds(req);
        if (promptIds.Length == 0) throw new ArgumentException("Request produced zero tokens.");
        HashSet<int> stops = _stopIds;
        if (req.StopTokenIds is not null) { stops = [.. _stopIds]; foreach (int s in req.StopTokenIds) stops.Add(s); }

        TransformerConfig cfg = _model.Config;
        // Throws KvPoolExhaustedException if the pool can't fit the prompt — propagates to the caller's
        // SubmitAsync task as a fault, the reject policy PagedKvPool documents.
        PagedKvCache cache = new(_pool);
        try
        {
            List<int> generated = new(req.MaxTokens);
            SamplerChain sampler = SamplerChain.FromOptions(req.Sampling, _tokenizer, cfg.VocabSize);
            int next;
            using (Tensor hidden = _model.Forward(_backend, promptIds, 0, cache))
            using (Tensor logits = _model.ProjectLogits(_backend, hidden, promptIds.Length))
                next = sampler.Next(LastRow(logits, promptIds.Length, cfg.VocabSize), generated);

            return new ActiveSeq
            {
                Pending = pending, PromptIds = promptIds, Cache = cache, Sampler = sampler,
                Stops = stops, Generated = generated, Next = next,
            };
        }
        catch
        {
            cache.Dispose();
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

    private void CompleteSeq(ActiveSeq seq, bool stopped)
    {
        GenerationResult result = new()
        {
            TokenIds = seq.Generated,
            Text = _tokenizer.Decode(seq.Generated),
            PromptTokens = seq.PromptIds.Length,
            StoppedOnStopToken = stopped,
        };
        seq.Cache.Dispose();
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
