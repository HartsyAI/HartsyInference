using HartsyInference.Core.Configuration;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using System.Diagnostics;
using HartsyInference.LLM.Generation.Speculation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Generation;

/// <summary>End-to-end LLM text generation: chat template → tokenize → GPU-resident prefill → autoregressive decode (per-token sampler chain) → stop on EOS/limit → detokenize; the pipeline does not own the model/tokenizer.</summary>
public sealed class TextGenerationPipeline
{
    private readonly IGenerationModel? _model;
    private readonly ILlmTokenizer _tokenizer;
    private readonly IChatTemplate _template;
    private readonly IBackend _backend;
    private readonly HashSet<int> _stopIds;

    /// <summary>Tensor-parallel transformer when this pipeline runs TP; null otherwise. Mutually exclusive with <see cref="_model"/> — TP has its own forward/logits and no graph/speculative/staged paths.</summary>
    private readonly TensorParallelTransformer? _tp;

    /// <summary><paramref name="template"/> defaults to ChatML when not supplied; <paramref name="placement"/> shards the decoder layers across devices (VRAM pooling), or null keeps the single-backend path byte-identical. Wraps the transformer in a <see cref="GenericTransformerModel"/> that preloads decode weights up front (headroom-guarded): auto-promotion's size floor otherwise leaves the small weights to be re-uploaded on every prefill.</summary>
    public TextGenerationPipeline(GenericTransformer model, ILlmTokenizer tokenizer, IBackend backend,
        IChatTemplate? template = null, LlmPlacement? placement = null)
        : this(new GenericTransformerModel(model, backend, placement, preloadWeights: true), tokenizer, template)
    {
    }

    /// <summary>Drives any <see cref="IGenerationModel"/>; the model decides its own weight residency and the pipeline does not own it.</summary>
    public TextGenerationPipeline(IGenerationModel model, ILlmTokenizer tokenizer, IChatTemplate? template = null)
    {
        _model = model;
        _tokenizer = tokenizer;
        _backend = model.OutputBackend;
        _template = template ?? new ChatMlTemplate();
        _stopIds = [.. tokenizer.StopIds];
    }

    /// <summary>Tensor-parallel pipeline: prefill + eager greedy/sampled decode via <see cref="TensorParallelTransformer.ForwardTp"/>; graph/speculative decode and layer-split staging are structurally unreachable here. The caller preloads per-rank weights; <paramref name="rankZeroBackend"/> is where logits/sampling rows are read.</summary>
    public TextGenerationPipeline(TensorParallelTransformer tp, ILlmTokenizer tokenizer, IBackend rankZeroBackend,
        IChatTemplate? template = null)
    {
        _tp = tp;
        _tokenizer = tokenizer;
        _backend = rankZeroBackend;
        _template = template ?? new ChatMlTemplate();
        _stopIds = [.. tokenizer.StopIds];
    }

    /// <summary>Generates text for <paramref name="request"/>, invoking <paramref name="onToken"/> per produced token; cancelling via <paramref name="ct"/> stops between layers during the prompt prefill and between tokens during decode, and throws, discarding already-produced tokens (rely on <paramref name="onToken"/> for partial output, which still fires for every token generated before cancellation is observed).</summary>
    public GenerationResult Generate(GenerationRequest request, Action<int>? onToken = null, CancellationToken ct = default)
        => Generate(request, reuse: null, onToken, ct);

    /// <summary>Same as <see cref="Generate(GenerationRequest,Action{int},CancellationToken)"/>, but when
    /// <paramref name="reuse"/> is supplied, prefills only the SUFFIX of this call's prompt that diverges from
    /// <paramref name="reuse"/>'s retained token ids (the longest common prefix, always leaving at least the final
    /// prompt token to be prefilled fresh so sampling has a real logits row), instead of always prefilling the
    /// whole prompt from an empty cache — opt-in, bounded-VRAM prefix-KV reuse across calls that share a
    /// conversation. A retained cache too small for this call grows by copying its reusable prefix on device.
    /// <paramref name="reuse"/> is updated in place with this call's full sequence (prompt + generated tokens)
    /// whether the call completes normally or is cancelled during decode — a stream filter's intentional stop and
    /// caller cancellation both leave the cache in a valid, consistently-committed state, so either is safe to keep.
    /// A cancellation during the prompt prefill, which commits nothing, and an exception from the backend itself (an
    /// unknown, possibly-inconsistent state) drop it instead. What is kept is copied down to its length plus
    /// <see cref="GenerationRequest.PrefixCacheHeadroomTokens"/>, and freed instead when even that exceeds
    /// <see cref="GenerationRequest.PrefixCacheMaxBytes"/>. Storage across keys, eviction policy and the busy-key rule
    /// are the caller's responsibility (see
    /// <see cref="RetainedSequenceStore"/>) — this method only reads and mutates the one instance it is given.
    /// Ignored for the tensor-parallel path (<see cref="GenerateTp"/>): per-rank <see cref="KvCache"/>s have no
    /// <see cref="ISequenceState"/> to retain, so <paramref name="reuse"/> is left untouched when <c>_tp</c> is set.</summary>
    public GenerationResult Generate(GenerationRequest request, RetainedSequence? reuse, Action<int>? onToken = null, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        int[] promptIds = BuildPromptIds(request);
        if (promptIds.Length == 0) throw new ArgumentException("Prompt produced zero tokens.", nameof(request));

        int vocab = _tp?.Config.VocabSize ?? _model!.Info.VocabSize;
        SamplerChain sampler = SamplerChain.FromOptions(request.Sampling, _tokenizer, vocab);
        List<int> generated = new(request.MaxTokens);
        HashSet<int> stops = _stopIds;
        if (request.StopTokenIds is not null) { stops = [.. _stopIds]; foreach (int s in request.StopTokenIds) stops.Add(s); }

        if (_tp is not null)
        {
            return GenerateTp(request, promptIds, sampler, stops, onToken, ct);
        }

        // Fixed-capacity KV (O(n) appends, bounded VRAM) sized for the prompt + the requested generation, unless
        // `reuse` already holds one (AcquireCache), in which case its common prefix with `promptIds` is kept —
        // grown by copy when the retained buffer is too small — and only the diverging tail below is prefilled.
        int maxSeq = promptIds.Length + request.MaxTokens + 1;
        (ISequenceState cache, int reusedLen) = AcquireCache(reuse, promptIds, maxSeq, request.PrefixCacheCapacityHint);
        bool committed = false;
        // True only once cache.Length is guaranteed >= promptIds.Length (the first prefill below has returned).
        // The catch block below commits on cancellation ONLY once this is true: the first Prefill observes `ct`
        // between layers and a stopped one commits nothing, so the finally block's cache.Length-vs-promptIds.Length
        // reconciliation would otherwise run against a prompt that never went in.
        bool firstPrefillDone = false;
        // Wall-clock split for the llama.cpp-style timings the API reports: prefill ends when the first logits row is sampled.
        System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();
        double prefillMs = 0;
        try
        {
            bool stopped;
            int next;
            // Logits for the LAST prompt position only: sampling reads a single row (see GenericTransformerModel.Prefill).
            // reusedLen is always < promptIds.Length (AcquireCache clamps it), so this chunk is never empty. Only this
            // forward takes `ct`: decode steps are one token each and already stop between tokens.
            using (Tensor hidden = _model!.Prefill(new PrefillChunk(promptIds.AsMemory(reusedLen), reusedLen, LastRowOnly: true), cache, ct))
            using (Tensor logits = _model.ProjectLogits(hidden, 1))
            {
                Span<float> lastRow = LastRow(logits, 1, vocab);
                next = sampler.Next(lastRow, generated);
            }
            firstPrefillDone = true;
            prefillMs = clock.Elapsed.TotalMilliseconds;

            request.OnPrefillCompleted?.Invoke(promptIds.Length);

            // CUDA-graph decode: collapses the ~600-700 kernel launches/token the plain loop below issues into one
            // cuGraphLaunch/step, removing the CPU launch-issuance bottleneck the perf grind identified as the
            // biggest remaining gap to llama.cpp (docs/Checklists/LLM_DECODE_PERF_GRIND.md Phase 6). Opt-in
            // (env-gated) and scoped to what's actually graph-safe: greedy only (the on-device argmax has no
            // sampler chain yet) and the plain dense GQA/RoPE decoder shape (SupportsGraphDecode) — MoE/MLA/
            // cross-attention/sliding-window models fall through to the verified default loop unchanged.
            bool graphDecodeRequested = request.GraphDecode ?? EngineKnobs.GraphDecode.Value;
            // !HasJsonConstraint: graph decode's on-device argmax bypasses the CPU sampler chain entirely —
            // including any JSON grammar step — so combining the two would silently produce unconstrained output.
            // (Previously missing here even though DynamicBatchScheduler's equivalent admission check already
            // excluded it — now consistent.)
            // Staged v1 keeps decode eager (the adapter reports no graph support): the step graph is a single-backend
            // capture, so per-stage graphs are a measured follow-up, not a default.
            IGraphDecodable? graphModel = _model as IGraphDecodable;
            // A sampled (non-greedy) request can replay a graph too when the backend draws on the device: top-k, temperature, top-p and
            // min-p are done there, which covers the common server defaults; anything else (no top-k, a larger k) stays on the host chain.
            DeviceSamplerConfig? deviceSampler = DeviceSamplerFor(request.Sampling);
            bool useGraphDecode = !request.Sampling.HasJsonConstraint && graphDecodeRequested
                && graphModel is not null && graphModel.SupportsGraphDecode(_backend)
                && (request.Sampling.Greedy || (deviceSampler is not null && graphModel.SupportsDeviceSampling(_backend)));

            // Prompt-lookup speculative decoding: batches a verify pass across several drafted tokens instead of
            // one plain decode step apiece. Mutually exclusive with graph decode (graph decode wins when both are
            // eligible — it's the more mature, unconditionally-faster path). See GenerateSpeculative's doc for why
            // this is restricted to greedy, non-JSON-mode requests.
            bool specDecodeRequested = request.SpeculativeDecode ?? EngineKnobs.SpecDecode.Value;
            bool useSpecDecode = !useGraphDecode && request.Sampling.Greedy && !request.Sampling.HasJsonConstraint
                && _model.Capabilities.SupportsSpeculation && specDecodeRequested;

            if (useGraphDecode)
            {
                stopped = GenerateGraphDecode(request, graphModel!, cache, promptIds.Length, next, generated, stops, onToken, ct,
                    request.Sampling.Greedy ? null : deviceSampler);
            }
            else if (useSpecDecode)
            {
                stopped = GenerateSpeculative(request, cache, promptIds, sampler, next, generated, stops, onToken, ct);
            }
            else
            {
                stopped = false;
                for (int step = 0; step < request.MaxTokens; step++)
                {
                    ct.ThrowIfCancellationRequested();
                    if (stops.Contains(next)) { stopped = true; break; }
                    generated.Add(next);
                    onToken?.Invoke(next);

                    using Tensor hidden = _model.Prefill(new PrefillChunk(new[] { next }, cache.Length), cache);
                    using Tensor logits = _model.ProjectLogits(hidden, 1);
                    Span<float> row = LastRow(logits, 1, vocab);
                    next = sampler.Next(row, generated);
                }
            }

            committed = true;
            double decodeMs = clock.Elapsed.TotalMilliseconds - prefillMs;
            return new GenerationResult
            {
                TokenIds = generated,
                Text = _tokenizer.Decode(generated),
                PromptTokens = promptIds.Length,
                StoppedOnStopToken = stopped,
                ReusedPromptTokens = reusedLen,
                PrefillMilliseconds = prefillMs,
                DecodeMilliseconds = decodeMs,
            };
        }
        catch (OperationCanceledException)
        {
            // NOT always strictly between committed steps: onToken runs (and, via TextService's filter-stop
            // wrapper, can throw) right after generated.Add(next) but BEFORE the Prefill call that actually
            // commits `next` to the cache (eager loop; graph decode's onToken sits at the same spot). So
            // generated.Count can be exactly one ahead of what cache.Length reflects when this fires — the
            // `finally` below reconciles against cache.Length, the one count that is always ground truth,
            // rather than trusting generated.Count here. Committing is conditioned on firstPrefillDone: a
            // cancellation before the prompt itself finished prefilling has no committed state at all (cache.Length
            // is still short of promptIds.Length), and the finally block's reconciliation formula assumes the full
            // prompt is already in — falling through to the genuine-fault branch below discards cleanly instead.
            committed = firstPrefillDone;
            throw;
        }
        finally
        {
            if (reuse is null)
            {
                cache.Dispose();
            }
            else if (committed)
            {
                // Clamped to cache.Length, not generated.Count: a cancellation from inside onToken (a filter or
                // tool-call stop) can leave one token recorded in `generated` whose Prefill never ran. Storing
                // more ids than the cache actually holds would make a later AcquireCache compute a reused-prefix
                // length beyond old.Length, and FixedKvCache.Truncate throws on that.
                int committedGenerated = Math.Clamp(cache.Length - promptIds.Length, 0, generated.Count);
                int[] fullIds = new int[promptIds.Length + committedGenerated];
                promptIds.CopyTo(fullIds, 0);
                generated.CopyTo(0, fullIds, promptIds.Length, committedGenerated);
                Retain(reuse, cache, fullIds, request);
            }
            else
            {
                // A genuine fault (backend exception, not cancellation): don't trust this cache's state.
                Discard(reuse, cache);
            }
        }
    }

    /// <summary>Stores <paramref name="cache"/> in <paramref name="reuse"/> as holding <paramref name="tokenIds"/>,
    /// first copying it down to its length plus the request's headroom when it holds more spare capacity than that,
    /// or frees it instead when the kept size would exceed the request's byte cap.</summary>
    private void Retain(RetainedSequence reuse, ISequenceState cache, int[] tokenIds, GenerationRequest request)
    {
        int headroom = Math.Max(0, request.PrefixCacheHeadroomTokens ?? EngineKnobs.PrefixCacheHeadroomTokens.Value);
        long maxBytes = request.PrefixCacheMaxBytes ?? EngineKnobs.PrefixCacheMaxBytes.Value;
        int target = (int)Math.Min((long)cache.Length + headroom, cache.Capacity);
        // Checked before shrinking, through the same estimator as the check after it: a sequence over the cap is
        // freed outright, never first copied into a buffer that would only be thrown away.
        if (_model!.EstimateSequenceBytes(target) > maxBytes)
        {
            Discard(reuse, cache);
            return;
        }
        ISequenceState kept = cache;
        try
        {
            if (cache.Capacity > target)
            {
                kept = TryResize(cache, target) ?? cache;
            }
        }
        catch
        {
            // A device fault mid-copy surfaces even though every token was already streamed; nothing is retained.
            Discard(reuse, cache);
            throw;
        }
        long keptBytes = _model.EstimateSequenceBytes(kept.Capacity);
        if (keptBytes > maxBytes)
        {
            if (!ReferenceEquals(kept, cache))
            {
                kept.Dispose();
            }
            Discard(reuse, cache);
            return;
        }
        if (!ReferenceEquals(kept, cache))
        {
            // Update only disposes the cache `reuse` itself references; a fresh or grown one is not it.
            if (ReferenceEquals(reuse.Cache, cache))
            {
                reuse.Clear();
            }
            cache.Dispose();
        }
        reuse.Update(kept, tokenIds, keptBytes);
    }

    /// <summary>Frees <paramref name="cache"/> and empties <paramref name="reuse"/>. <paramref name="cache"/> is either
    /// <c>reuse.Cache</c> itself (reused in place) or one <see cref="AcquireCache"/> already detached
    /// <paramref name="reuse"/> from — each is disposed exactly once either way.</summary>
    private static void Discard(RetainedSequence reuse, ISequenceState cache)
    {
        bool same = ReferenceEquals(cache, reuse.Cache);
        reuse.Dispose();
        if (!same)
        {
            cache.Dispose();
        }
    }

    /// <summary>Resolves the cache <see cref="Generate(GenerationRequest,RetainedSequence,Action{int},CancellationToken)"/>
    /// prefills into: <paramref name="reuse"/>'s cache truncated to its common prefix with <paramref name="promptIds"/>
    /// when it is large enough for <paramref name="maxSeq"/>; when it is not, a copy of that prefix in a cache of
    /// exactly <paramref name="maxSeq"/>; else a fresh one sized by <paramref name="capacityHint"/>. An outgrown
    /// <paramref name="reuse"/> cache is disposed before any fresh allocation — never the other way around, so a
    /// failed allocation never leaves <paramref name="reuse"/> pointing at an already-disposed buffer. The returned
    /// length is always LESS than <paramref name="promptIds"/>.Length, even on an exact repeat, so the caller always
    /// prefills a real final token and gets a fresh logits row to sample from; it is also never more than
    /// <c>old.Length</c> (defensive — Generate's own bookkeeping keeps <c>reuse.TokenIds</c> within that bound already,
    /// but <see cref="FixedKvCache.Truncate"/> throws instead of clamping, so a future bug here should degrade to less
    /// reuse, not a crash).</summary>
    private (ISequenceState Cache, int ReusedLen) AcquireCache(RetainedSequence? reuse, int[] promptIds, int maxSeq, int? capacityHint)
    {
        if (reuse?.Cache is { } old)
        {
            int commonLen = Math.Min(CommonPrefixLength(reuse.TokenIds, promptIds), old.Length);
            int reusedLen = Math.Max(0, Math.Min(commonLen, promptIds.Length - 1));
            // Before the grow below, too: a resize copies the committed length, so only the reused prefix moves.
            old.Truncate(reusedLen);
            if (old.Capacity >= maxSeq)
            {
                return (old, reusedLen);
            }
            // Outgrown: copy only the reusable prefix into a cache with room for this call. Exactly maxSeq, not the
            // hint: what is retained is shrunk back when the call ends anyway.
            ISequenceState? grown = null;
            try
            {
                grown = reusedLen > 0 ? TryResize(old, maxSeq) : null;
            }
            finally
            {
                reuse.Dispose();
            }
            if (grown is not null)
            {
                return (grown, reusedLen);
            }
        }
        int capacity = Math.Max(maxSeq, capacityHint ?? 0);
        return (_model!.CreateSequenceState(new SequenceStateOptions(capacity)), 0);
    }

    /// <summary><see cref="IGenerationModel.ResizeSequenceState"/>, or null when the model cannot copy its state or
    /// the device cannot fit the copy; the caller then keeps or rebuilds the original instead.</summary>
    private ISequenceState? TryResize(ISequenceState state, int capacity)
    {
        try
        {
            return _model!.ResizeSequenceState(state, capacity);
        }
        catch (OutOfVramException ex)
        {
            Logs.Debug($"[TextGenerationPipeline] Retained KV not resized to {capacity} tokens: {ex.Message}");
            return null;
        }
    }

    /// <summary>Length of the shared prefix of <paramref name="a"/> and <paramref name="b"/>.</summary>
    private static int CommonPrefixLength(int[] a, int[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        int i = 0;
        while (i < n && a[i] == b[i])
        {
            i++;
        }
        return i;
    }

    /// <summary>Tensor-parallel generation: prefill + eager decode over per-rank KV caches; no graph/speculative decode (structurally unreachable — <c>ForwardTp</c> is the only forward) and no last-row gather optimization yet (prefill projects all rows, correctness-first).</summary>
    private GenerationResult GenerateTp(GenerationRequest request, int[] promptIds, SamplerChain sampler,
        HashSet<int> stops, Action<int>? onToken, CancellationToken ct)
    {
        TransformerConfig cfg = _tp!.Config;
        List<int> generated = new(request.MaxTokens);
        KvCache[] caches = _tp.CreateKvCaches();
        try
        {
            bool stopped = false;
            int next;
            using (Tensor hidden = _tp.ForwardTp(promptIds, 0, caches))
            using (Tensor logits = _tp.ProjectLogits(hidden, promptIds.Length))
            {
                Span<float> lastRow = LastRow(logits, promptIds.Length, cfg.VocabSize);
                next = sampler.Next(lastRow, generated);
            }
            request.OnPrefillCompleted?.Invoke(promptIds.Length);
            for (int step = 0; step < request.MaxTokens; step++)
            {
                ct.ThrowIfCancellationRequested();
                if (stops.Contains(next)) { stopped = true; break; }
                generated.Add(next);
                onToken?.Invoke(next);

                using Tensor hidden = _tp.ForwardTp([next], caches[0].CurrentLength, caches);
                using Tensor logits = _tp.ProjectLogits(hidden, 1);
                next = sampler.Next(LastRow(logits, 1, cfg.VocabSize), generated);
            }
            return new GenerationResult
            {
                TokenIds = generated,
                Text = _tokenizer.Decode(generated),
                PromptTokens = promptIds.Length,
                StoppedOnStopToken = stopped,
            };
        }
        finally
        {
            foreach (KvCache cache in caches) cache.Dispose();
        }
    }

    /// <summary>Greedy decode via one captured CUDA graph, replayed once per token; <paramref name="firstToken"/> is the token already sampled from the prefill's last position.</summary>
    /// <remarks>Device state (position, current token id, the RoPE table, and — when a repetition penalty is requested — the token history) is refreshed OUTSIDE the graph before each replay, which is what makes one capture valid for every step (see IBackend's "Device-side decode position" docs). Repetition penalty is the only sampler stage graph decode replicates (see <see cref="GenericTransformer.ForwardGraphDecodeStep"/> for why temperature/top-k/top-p/min-p are no-ops for a greedy pick); when the request's penalty is 1.0 the history buffers are still allocated but the backend skips the append/penalty kernels entirely. If capture throws (an eligible-looking model hits something the graphed path doesn't support), the exception propagates rather than falling back — this path is opt-in (env-gated), so a gap surfaces as a clear error, not silent mis-generation.</remarks>
    /// <summary>The device sampler configuration for <paramref name="options"/>, or null when the request needs the host chain (top-k outside 1..64, or no temperature to scale by).</summary>
    private static DeviceSamplerConfig? DeviceSamplerFor(SamplingOptions options) =>
        options.TopK is >= 1 and <= 64 && options.Temperature > 0f
            ? new DeviceSamplerConfig(options.TopK, options.Temperature, options.TopP, options.MinP, options.Seed)
            : null;

    private bool GenerateGraphDecode(GenerationRequest request, IGraphDecodable graphModel, ISequenceState cache,
        int promptLen, int firstToken, List<int> generated, HashSet<int> stops, Action<int>? onToken, CancellationToken ct,
        DeviceSamplerConfig? sampler = null)
    {
        GraphDecodeSession session = graphModel.CaptureDecodeGraph(cache, promptLen, firstToken,
            request.Sampling.RepetitionPenalty, sampler);
        try
        {
            int next = firstToken;
            for (int step = 0; step < request.MaxTokens; step++)
            {
                ct.ThrowIfCancellationRequested();
                if (stops.Contains(next)) return true;
                generated.Add(next);
                onToken?.Invoke(next);

                next = session.Replay();
                session.Pos++;
                session.WriteNextPos();   // prep for the NEXT replay
                // Mirrors DynamicBatchScheduler.ReplayGraphRound: the graph advances the DEVICE-side position
                // itself, but cache.Length (host-side, what AcquireCache/the retained-sequence bookkeeping in
                // Generate reads) only moves via this explicit commit. Before prefix-cache reuse, nothing ever
                // read cache.Length again after a graph-decode run (the cache was always disposed), so this had
                // no observable effect either way; it matters now that a retained cache's Length needs to be
                // accurate after a graph-decode turn too.
                graphModel.CommitReplayedStep(cache);
            }
            return false;
        }
        finally
        {
            session.Dispose();
        }
    }

    // Prompt-lookup speculative decoding tuning: small, fixed constants rather than request-level knobs — the
    // technique either pays off on repetitive content or costs nothing (see GenerateSpeculative's doc), so
    // there's no per-request tradeoff worth exposing yet. The drafting itself lives in PromptLookupDraftProvider.
    private const int SpecMaxDraftTokens = 8;
    private static readonly ISpeculativeDraftProvider DefaultSpecDraftProvider = new PromptLookupDraftProvider();

    /// <summary>Chooses the draft provider for each speculative round and whether speculation stays enabled. When null,
    /// speculative decoding drafts with prompt lookup exactly as before. The selector keeps measurements across calls,
    /// so it belongs to one pipeline and must not be shared by concurrent <c>Generate</c> calls.</summary>
    public SpeculationSelector? DraftSelector { get; set; }

    /// <summary>Prompt-lookup speculative decoding: greedy-only, draft-model-free, drafting via n-gram match and verifying the whole draft plus one bonus position in one batched forward pass.</summary>
    /// <remarks>Each round drafts up to <see cref="SpecMaxDraftTokens"/> tokens via <see cref="DraftSelector"/> when one is set, otherwise via <see cref="DefaultSpecDraftProvider"/> (n-gram match against the prompt + generated-so-far) and verifies them in ONE batched forward pass, reusing the same prefill-shaped <see cref="GenericTransformer.Forward"/> call with a short token span at an arbitrary <c>posStart</c> against an already-partially-filled cache. The longest correct prefix (verified against this model's own greedy pick, row by row) is accepted; a rejected or never-drafted token still costs exactly one forward call, same as the eager loop, so this is a pure speedup on repetitive content and a no-op tax otherwise. Every accepted token's history-dependent sampler state (repetition penalty) is computed in the same left-to-right order the eager loop uses, so output is byte-identical to plain greedy decode. Rejected draft tokens' KV entries were already physically written by the verification forward pass (unavoidable — verification needs every candidate present in the batch before any is judged), so <see cref="IKvCache.Truncate"/> rolls them back on partial/zero acceptance.
    /// <para>Requires <see cref="SamplingOptions.Greedy"/> (no order-independent way to reproduce a non-greedy multinomial draw out of sequence) and excludes JSON grammar mode (its incremental state walker isn't designed to roll back mid-token) — both enforced by the caller's dispatch gate, not re-checked here.</para></remarks>
    private bool GenerateSpeculative(GenerationRequest request, ISequenceState cache, int[] promptIds, SamplerChain sampler,
        int firstToken, List<int> generated, HashSet<int> stops, Action<int>? onToken, CancellationToken ct)
    {
        int vocab = _model!.Info.VocabSize;
        int next = firstToken;
        SpeculationSelector? selector = DraftSelector;

        while (generated.Count < request.MaxTokens)
        {
            ct.ThrowIfCancellationRequested();
            if (stops.Contains(next)) return true;

            generated.Add(next);
            onToken?.Invoke(next);
            if (generated.Count >= request.MaxTokens) return false;

            int maxDraft = Math.Min(Math.Min(SpecMaxDraftTokens, _model.Capabilities.MaxSpeculativeDepth),
                request.MaxTokens - generated.Count);
            // A null provider means the selector disabled speculation: the round then verifies zero drafted tokens,
            // which is the plain one-token decode step and emits the same token.
            ISpeculativeDraftProvider? provider = selector is null ? DefaultSpecDraftProvider : selector.Select();
            long roundStart = selector is null ? 0 : Stopwatch.GetTimestamp();
            int[] draft = provider is null ? [] : provider.Propose(promptIds, generated, maxDraft);
            int k = draft.Length;

            int cachePos = cache.Length;
            int[] input = new int[k + 1];
            input[0] = next;
            Array.Copy(draft, 0, input, 1, k);

            using Tensor hidden = _model.Prefill(new PrefillChunk(input, cachePos), cache);
            using Tensor logits = _model.ProjectLogits(hidden, k + 1);

            int accepted = 0;
            int? mismatchNext = null;
            bool sawStop = false;
            for (int i = 0; i < k; i++)
            {
                Span<float> row = RowAt(logits, i, vocab);
                int predicted = sampler.Next(row, generated);
                if (predicted != draft[i]) { mismatchNext = predicted; break; }
                if (stops.Contains(draft[i])) { sawStop = true; break; }
                generated.Add(draft[i]);
                onToken?.Invoke(draft[i]);
                accepted++;
            }

            if (mismatchNext is not null)
            {
                cache.Truncate(cachePos + 1 + accepted);
                RecordRound(provider, k, accepted, accepted + 1, roundStart);
                next = mismatchNext.Value;
                continue;
            }
            if (sawStop)
            {
                cache.Truncate(cachePos + 1 + accepted);
                RecordRound(provider, k, accepted, accepted, roundStart);
                return true;
            }

            // Full draft accepted (k == 0 degenerates to this trivially): cache already holds exactly
            // cachePos + k + 1 entries, matching what was verified — no truncation needed. Row k is a free
            // bonus prediction (already computed by this same forward pass, no extra GPU call).
            RecordRound(provider, k, k, k + 1, roundStart);
            Span<float> bonusRow = RowAt(logits, k, vocab);
            next = sampler.Next(bonusRow, generated);
        }
        return false;

        // Feeds one measured round back to the selector. The caller's clock is read here; the selector never reads one.
        void RecordRound(ISpeculativeDraftProvider? roundProvider, int proposed, int accepted, int emitted, long startTimestamp)
        {
            if (selector is null || roundProvider is null) return;
            double elapsedMs = Stopwatch.GetElapsedTime(startTimestamp).TotalMilliseconds;
            selector.Record(roundProvider, new SpeculationRound(proposed, accepted, emitted, elapsedMs));
        }
    }

    private int[] BuildPromptIds(GenerationRequest request) => PromptBuilder.BuildPromptIds(request, _tokenizer, _template);

    private static unsafe Span<float> RowAt(Tensor logits, int row, int vocab)
    {
        float* p = (float*)logits.DataPointer;
        return new Span<float>(p + (long)row * vocab, vocab);
    }

    private static Span<float> LastRow(Tensor logits, int t, int vocab) => RowAt(logits, t - 1, vocab);
}
