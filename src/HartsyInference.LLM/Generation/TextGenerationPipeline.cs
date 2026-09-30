using HartsyInference.Core.Configuration;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
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

    /// <summary>Generates text for <paramref name="request"/>, invoking <paramref name="onToken"/> per produced token; cancelling via <paramref name="ct"/> stops between tokens and throws, discarding already-produced tokens (rely on <paramref name="onToken"/> for partial output, which still fires for every token generated before cancellation is observed).</summary>
    public GenerationResult Generate(GenerationRequest request, Action<int>? onToken = null, CancellationToken ct = default)
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

        // Fixed-capacity KV (O(n) appends, bounded VRAM) sized for the prompt + the requested generation.
        int maxSeq = promptIds.Length + request.MaxTokens + 1;
        using ISequenceState cache = _model!.CreateSequenceState(new SequenceStateOptions(maxSeq));

        bool stopped = false;
        int next;
        // Logits for the LAST prompt position only: sampling reads a single row (see GenericTransformerModel.Prefill).
        using (Tensor hidden = _model.Prefill(new PrefillChunk(promptIds, 0, LastRowOnly: true), cache))
        using (Tensor logits = _model.ProjectLogits(hidden, 1))
        {
            Span<float> lastRow = LastRow(logits, 1, vocab);
            next = sampler.Next(lastRow, generated);
        }

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
        bool useGraphDecode = request.Sampling.Greedy && !request.Sampling.HasJsonConstraint && graphDecodeRequested
            && graphModel is not null && graphModel.SupportsGraphDecode(_backend);

        // Prompt-lookup speculative decoding: batches a verify pass across several drafted tokens instead of
        // one plain decode step apiece. Mutually exclusive with graph decode (graph decode wins when both are
        // eligible — it's the more mature, unconditionally-faster path). See GenerateSpeculative's doc for why
        // this is restricted to greedy, non-JSON-mode requests.
        bool specDecodeRequested = request.SpeculativeDecode ?? EngineKnobs.SpecDecode.Value;
        bool useSpecDecode = !useGraphDecode && request.Sampling.Greedy && !request.Sampling.HasJsonConstraint
            && _model.Capabilities.SupportsSpeculation && specDecodeRequested;

        if (useGraphDecode)
        {
            stopped = GenerateGraphDecode(request, graphModel!, cache, promptIds.Length, next, generated, stops, onToken, ct);
        }
        else if (useSpecDecode)
        {
            stopped = GenerateSpeculative(request, cache, promptIds, sampler, next, generated, stops, onToken, ct);
        }
        else
        {
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

        return new GenerationResult
        {
            TokenIds = generated,
            Text = _tokenizer.Decode(generated),
            PromptTokens = promptIds.Length,
            StoppedOnStopToken = stopped,
        };
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
    private bool GenerateGraphDecode(GenerationRequest request, IGraphDecodable graphModel, ISequenceState cache,
        int promptLen, int firstToken, List<int> generated, HashSet<int> stops, Action<int>? onToken, CancellationToken ct)
    {
        GraphDecodeSession session = graphModel.CaptureDecodeGraph(cache, promptLen, firstToken,
            request.Sampling.RepetitionPenalty);
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
    // there's no per-request tradeoff worth exposing yet.
    private const int SpecNgramSize = 3;
    private const int SpecMaxDraftTokens = 8;
    private const int SpecMaxLookback = 4096;

    /// <summary>Prompt-lookup speculative decoding: greedy-only, draft-model-free, drafting via n-gram match and verifying the whole draft plus one bonus position in one batched forward pass.</summary>
    /// <remarks>Each round drafts up to <see cref="SpecMaxDraftTokens"/> tokens via <see cref="FindDraftContinuation"/> (n-gram match against the prompt + generated-so-far) and verifies them in ONE batched forward pass, reusing the same prefill-shaped <see cref="GenericTransformer.Forward"/> call with a short token span at an arbitrary <c>posStart</c> against an already-partially-filled cache. The longest correct prefix (verified against this model's own greedy pick, row by row) is accepted; a rejected or never-drafted token still costs exactly one forward call, same as the eager loop, so this is a pure speedup on repetitive content and a no-op tax otherwise. Every accepted token's history-dependent sampler state (repetition penalty) is computed in the same left-to-right order the eager loop uses, so output is byte-identical to plain greedy decode. Rejected draft tokens' KV entries were already physically written by the verification forward pass (unavoidable — verification needs every candidate present in the batch before any is judged), so <see cref="IKvCache.Truncate"/> rolls them back on partial/zero acceptance.
    /// <para>Requires <see cref="SamplingOptions.Greedy"/> (no order-independent way to reproduce a non-greedy multinomial draw out of sequence) and excludes JSON grammar mode (its incremental state walker isn't designed to roll back mid-token) — both enforced by the caller's dispatch gate, not re-checked here.</para></remarks>
    private bool GenerateSpeculative(GenerationRequest request, ISequenceState cache, int[] promptIds, SamplerChain sampler,
        int firstToken, List<int> generated, HashSet<int> stops, Action<int>? onToken, CancellationToken ct)
    {
        int vocab = _model!.Info.VocabSize;
        int next = firstToken;

        while (generated.Count < request.MaxTokens)
        {
            ct.ThrowIfCancellationRequested();
            if (stops.Contains(next)) return true;

            generated.Add(next);
            onToken?.Invoke(next);
            if (generated.Count >= request.MaxTokens) return false;

            int maxDraft = Math.Min(Math.Min(SpecMaxDraftTokens, _model.Capabilities.MaxSpeculativeDepth),
                request.MaxTokens - generated.Count);
            int[] draft = FindDraftContinuation(promptIds, generated, SpecNgramSize, maxDraft);
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
                next = mismatchNext.Value;
                continue;
            }
            if (sawStop)
            {
                cache.Truncate(cachePos + 1 + accepted);
                return true;
            }

            // Full draft accepted (k == 0 degenerates to this trivially): cache already holds exactly
            // cachePos + k + 1 entries, matching what was verified — no truncation needed. Row k is a free
            // bonus prediction (already computed by this same forward pass, no extra GPU call).
            Span<float> bonusRow = RowAt(logits, k, vocab);
            next = sampler.Next(bonusRow, generated);
        }
        return false;
    }

    /// <summary>Searches context for a prior occurrence of the last <paramref name="ngramSize"/> tokens and, if found, returns the up-to-<paramref name="maxDraftLen"/> tokens that followed it as a draft continuation guess.</summary>
    /// <remarks>Deliberately searches OLDEST-match-first rather than nearest-match-first: the nearer a match is to the current position, the less context trails it, so nearest-first pathologically degenerates to single-token drafts on short-period repeats (e.g. a stuck "the the the the..." loop) — the exact case this technique should help most. Oldest-first costs nothing extra since every draft is verified against the real model regardless of which historical match produced it. Returns an empty array whenever no match exists (safe — costs one plain decode step). The search window is capped at <see cref="SpecMaxLookback"/> tokens so a very long generation can't turn this into an O(n²) scan; missing a distant match only forgoes a speedup, it never affects correctness.</remarks>
    private static int[] FindDraftContinuation(int[] promptIds, List<int> generated, int ngramSize, int maxDraftLen)
    {
        int totalLen = promptIds.Length + generated.Count;
        if (maxDraftLen <= 0 || totalLen < ngramSize) return [];

        int searchFloor = Math.Max(0, totalLen - SpecMaxLookback);
        int windowLen = totalLen - searchFloor;
        int[] context = new int[windowLen];
        for (int i = 0; i < windowLen; i++)
        {
            int idx = searchFloor + i;
            context[i] = idx < promptIds.Length ? promptIds[idx] : generated[idx - promptIds.Length];
        }

        int needleStart = windowLen - ngramSize;
        for (int start = 0; start < needleStart; start++)
        {
            bool match = true;
            for (int k = 0; k < ngramSize; k++)
            {
                if (context[start + k] != context[needleStart + k]) { match = false; break; }
            }
            if (!match) continue;

            int matchEnd = start + ngramSize;
            int draftLen = Math.Min(maxDraftLen, windowLen - matchEnd);
            if (draftLen <= 0) continue;
            int[] draft = new int[draftLen];
            Array.Copy(context, matchEnd, draft, 0, draftLen);
            return draft;
        }
        return [];
    }

    private int[] BuildPromptIds(GenerationRequest request) => PromptBuilder.BuildPromptIds(request, _tokenizer, _template);

    private static unsafe Span<float> RowAt(Tensor logits, int row, int vocab)
    {
        float* p = (float*)logits.DataPointer;
        return new Span<float>(p + (long)row * vocab, vocab);
    }

    private static Span<float> LastRow(Tensor logits, int t, int vocab) => RowAt(logits, t - 1, vocab);
}
