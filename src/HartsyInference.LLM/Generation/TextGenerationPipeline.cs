using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.Tokenizers;

namespace HartsyInference.LLM.Generation;

/// <summary>End-to-end LLM text generation: chat template → tokenize → GPU-resident prefill → autoregressive
/// decode (per-token sampler chain) → stop on EOS/limit → detokenize. Composes a <see cref="GenericTransformer"/>,
/// an <see cref="ILlmTokenizer"/> (any model family), an <see cref="IChatTemplate"/>, and a backend-selected
/// <see cref="IBackend"/>. Stop tokens come from the tokenizer. The pipeline does not own the model/tokenizer.</summary>
public sealed class TextGenerationPipeline
{
    private readonly GenericTransformer _model;
    private readonly ILlmTokenizer _tokenizer;
    private readonly IChatTemplate _template;
    private readonly IBackend _backend;
    private readonly HashSet<int> _stopIds;

    /// <summary>Creates the pipeline. <paramref name="template"/> defaults to ChatML when not supplied.</summary>
    public TextGenerationPipeline(GenericTransformer model, ILlmTokenizer tokenizer, IBackend backend,
        IChatTemplate? template = null)
    {
        _model = model;
        _tokenizer = tokenizer;
        _backend = backend;
        _template = template ?? new ChatMlTemplate();
        _stopIds = [.. tokenizer.StopIds];
    }

    /// <summary>Generates text for <paramref name="request"/>. <paramref name="onToken"/> is invoked with each
    /// new token id as it is produced (for streaming). <paramref name="ct"/> is checked once per generated
    /// token (both the eager loop and the graph-decode replay loop) — cancelling stops generation between
    /// tokens and throws <see cref="OperationCanceledException"/>; already-produced tokens are lost (callers
    /// that want partial output on cancellation should rely on <paramref name="onToken"/>, which still fires
    /// for every token generated before the cancellation is observed).</summary>
    public GenerationResult Generate(GenerationRequest request, Action<int>? onToken = null, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        int[] promptIds = BuildPromptIds(request);
        if (promptIds.Length == 0) throw new ArgumentException("Prompt produced zero tokens.", nameof(request));

        TransformerConfig cfg = _model.Config;
        SamplerChain sampler = SamplerChain.FromOptions(request.Sampling, _tokenizer, cfg.VocabSize);
        List<int> generated = new(request.MaxTokens);
        HashSet<int> stops = _stopIds;
        if (request.StopTokenIds is not null) { stops = [.. _stopIds]; foreach (int s in request.StopTokenIds) stops.Add(s); }

        // Fixed-capacity KV (O(n) appends, bounded VRAM) sized for the prompt + the requested generation.
        int maxSeq = promptIds.Length + request.MaxTokens + 1;
        // Gemma-4: local/SWA layers use a narrower head dim than global layers (HeadDimFor); every other
        // architecture's HeadDimFor is just the uniform HeadDim, so this is a no-op for them.
        int[] headDimPerLayer = new int[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++) headDimPerLayer[i] = cfg.HeadDimFor(i);
        using FixedKvCache cache = new(cfg.NumLayers, 1, cfg.NumKvHeads, headDimPerLayer, maxSeq);

        bool stopped = false;
        int next;
        using (Tensor hidden = _model.Forward(_backend, promptIds, 0, cache))
        using (Tensor logits = _model.ProjectLogits(_backend, hidden, promptIds.Length))
        {
            Span<float> lastRow = LastRow(logits, promptIds.Length, cfg.VocabSize);
            next = sampler.Next(lastRow, generated);
        }

        // CUDA-graph decode: collapses the ~600-700 kernel launches/token the plain loop below issues into one
        // cuGraphLaunch/step, removing the CPU launch-issuance bottleneck the perf grind identified as the
        // biggest remaining gap to llama.cpp (docs/Checklists/LLM_DECODE_PERF_GRIND.md Phase 6). Opt-in
        // (env-gated) and scoped to what's actually graph-safe: greedy only (the on-device argmax has no
        // sampler chain yet) and the plain dense GQA/RoPE decoder shape (SupportsGraphDecode) — MoE/MLA/
        // cross-attention/sliding-window models fall through to the verified default loop unchanged.
        bool graphDecodeRequested = request.GraphDecode ?? (Environment.GetEnvironmentVariable("HARTSY_GRAPH_DECODE") == "1");
        bool useGraphDecode = request.Sampling.Greedy
            && graphDecodeRequested
            && _model.SupportsGraphDecode(_backend);

        // Prompt-lookup speculative decoding: batches a verify pass across several drafted tokens instead of
        // one plain decode step apiece. Mutually exclusive with graph decode (graph decode wins when both are
        // eligible — it's the more mature, unconditionally-faster path). See GenerateSpeculative's doc for why
        // this is restricted to greedy, non-JSON-mode requests.
        bool specDecodeRequested = request.SpeculativeDecode ?? (Environment.GetEnvironmentVariable("HARTSY_SPEC_DECODE") == "1");
        bool useSpecDecode = !useGraphDecode
            && request.Sampling.Greedy
            && !request.Sampling.JsonMode
            && specDecodeRequested;

        if (useGraphDecode)
        {
            stopped = GenerateGraphDecode(request, cache, promptIds.Length, next, generated, stops, onToken, ct);
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

                using Tensor hidden = _model.Forward(_backend, [next], cache.CurrentLength, cache);
                using Tensor logits = _model.ProjectLogits(_backend, hidden, 1);
                Span<float> row = LastRow(logits, 1, cfg.VocabSize);
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

    /// <summary>Greedy decode via one captured CUDA graph, replayed once per token. <paramref name="firstToken"/>
    /// is the token already sampled from the prefill's last position (mirrors the eager loop's starting state).
    /// Device state (position, current token id, the RoPE table, and — when a repetition penalty is requested —
    /// the token history) is refreshed OUTSIDE the graph before each replay — see IBackend's "Device-side decode
    /// position" docs for why that's what makes one capture valid for every step. Repetition penalty is the only
    /// sampler stage graph decode replicates (see <see cref="GenericTransformer.ForwardGraphDecodeStep"/> for why
    /// temperature/top-k/top-p/min-p are no-ops for a greedy pick); when the request's penalty is 1.0 the history
    /// buffers are still allocated (cheap, fixed-size) but the backend skips the append/penalty kernels entirely.
    /// Falls back silently to nothing extra on failure: if capture throws (an eligible-looking model hits
    /// something the graphed path doesn't actually support), the exception propagates — this path is opt-in
    /// (env-gated), so a user who turns it on and hits a gap gets a clear error, not silent mis-generation.</summary>
    private bool GenerateGraphDecode(GenerationRequest request, FixedKvCache cache, int promptLen, int firstToken,
        List<int> generated, HashSet<int> stops, Action<int>? onToken, CancellationToken ct)
    {
        TransformerConfig cfg = _model.Config;
        Tensor embedTable = _model.EnsureEmbedResidentForGraphDecode(_backend);
        (Tensor cosTable, Tensor sinTable) = _model.EnsureRopeTableForGraphDecode(_backend, cache.MaxSequenceLength);
        ulong devicePos = _backend.AllocDevicePos();
        ulong deviceTokenId = _backend.AllocDeviceTokenId();
        float repetitionPenalty = request.Sampling.RepetitionPenalty;
        ulong history = _backend.AllocDeviceHistory(cache.MaxSequenceLength);
        ulong historyCount = _backend.AllocDeviceCounter();
        object? graph = null;
        try
        {
            int pos = promptLen;   // absolute position of the token this step is about to generate
            _backend.WriteDeviceTokenId(deviceTokenId, firstToken);
            _backend.WriteDevicePos(devicePos, pos + 1, pos);
            _backend.WriteDeviceCounter(historyCount, 0);
            graph = _backend.CaptureGraph(() =>
                _model.ForwardGraphDecodeStep(_backend, embedTable, cache, cosTable, sinTable, devicePos, deviceTokenId,
                    history, historyCount, repetitionPenalty));

            int next = firstToken;
            for (int step = 0; step < request.MaxTokens; step++)
            {
                ct.ThrowIfCancellationRequested();
                if (stops.Contains(next)) return true;
                generated.Add(next);
                onToken?.Invoke(next);

                _backend.LaunchGraph(graph!);
                next = _backend.ReadDeviceTokenId(deviceTokenId);
                pos++;
                _backend.WriteDevicePos(devicePos, pos + 1, pos);   // prep for the NEXT replay
            }
            return false;
        }
        finally
        {
            if (graph is not null) _backend.DisposeGraph(graph);
            _backend.FreeDevicePos(devicePos);
            _backend.FreeDeviceTokenId(deviceTokenId);
            _backend.FreeDeviceHistory(history);
            _backend.FreeDeviceCounter(historyCount);
        }
    }

    // Prompt-lookup speculative decoding tuning: small, fixed constants rather than request-level knobs — the
    // technique either pays off on repetitive content or costs nothing (see GenerateSpeculative's doc), so
    // there's no per-request tradeoff worth exposing yet.
    private const int SpecNgramSize = 3;
    private const int SpecMaxDraftTokens = 8;
    private const int SpecMaxLookback = 4096;

    /// <summary>Prompt-lookup speculative decoding: greedy-only, draft-model-free. Each round drafts up to
    /// <see cref="SpecMaxDraftTokens"/> tokens via <see cref="FindDraftContinuation"/> (n-gram match against
    /// the prompt + generated-so-far) and verifies the whole draft plus one bonus position in ONE batched
    /// forward pass — reusing the exact same prefill-shaped <see cref="GenericTransformer.Forward"/> call the
    /// prompt itself goes through, just with a short token span at an arbitrary <c>posStart</c> against an
    /// already-partially-filled cache. The longest correct prefix (verified against this same model's own
    /// greedy pick, row by row) is accepted; a rejected or never-drafted token still costs exactly one forward
    /// call, same as the eager loop, so this is a pure speedup on repetitive content (regenerating similar
    /// JSON, quoting earlier text) and a no-op tax otherwise. Every accepted token's history-dependent sampler
    /// state (repetition penalty) is computed in the same left-to-right order the eager loop uses, so output is
    /// byte-identical to plain greedy decode — this is what makes the technique a pure speedup rather than an
    /// approximation. Rejected draft tokens' KV entries were already physically written by the verification
    /// forward pass (unavoidable — verification needs every candidate present in the batch before any is
    /// judged), so <see cref="IKvCache.Truncate"/> rolls them back on partial/zero acceptance.
    ///
    /// <para>Requires <see cref="SamplingOptions.Greedy"/> (there is no order-independent way to reproduce a
    /// non-greedy multinomial draw out of sequence) and excludes JSON grammar mode (its incremental state
    /// walker isn't designed to be rolled back mid-token) — both are enforced by the caller's dispatch gate,
    /// not re-checked here.</para></summary>
    private bool GenerateSpeculative(GenerationRequest request, FixedKvCache cache, int[] promptIds, SamplerChain sampler,
        int firstToken, List<int> generated, HashSet<int> stops, Action<int>? onToken, CancellationToken ct)
    {
        TransformerConfig cfg = _model.Config;
        int next = firstToken;

        while (generated.Count < request.MaxTokens)
        {
            ct.ThrowIfCancellationRequested();
            if (stops.Contains(next)) return true;

            generated.Add(next);
            onToken?.Invoke(next);
            if (generated.Count >= request.MaxTokens) return false;

            int maxDraft = Math.Min(SpecMaxDraftTokens, request.MaxTokens - generated.Count);
            int[] draft = FindDraftContinuation(promptIds, generated, SpecNgramSize, maxDraft);
            int k = draft.Length;

            int cachePos = cache.CurrentLength;
            int[] input = new int[k + 1];
            input[0] = next;
            Array.Copy(draft, 0, input, 1, k);

            using Tensor hidden = _model.Forward(_backend, input, cachePos, cache);
            using Tensor logits = _model.ProjectLogits(_backend, hidden, k + 1);

            int accepted = 0;
            int? mismatchNext = null;
            bool sawStop = false;
            for (int i = 0; i < k; i++)
            {
                Span<float> row = RowAt(logits, i, cfg.VocabSize);
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
            Span<float> bonusRow = RowAt(logits, k, cfg.VocabSize);
            next = sampler.Next(bonusRow, generated);
        }
        return false;
    }

    /// <summary>Searches <paramref name="promptIds"/> + <paramref name="generated"/> (context so far) for a
    /// prior occurrence of the last <paramref name="ngramSize"/> tokens and, if found, returns the
    /// up-to-<paramref name="maxDraftLen"/> tokens that followed it as a draft continuation guess. Deliberately
    /// searches OLDEST-match-first rather than nearest-match-first: the nearer a match is to the current
    /// position, the less context trails it (a match found 1 token back only has 1 token of continuation to
    /// draw from before running into the present), so nearest-first pathologically degenerates to
    /// single-token drafts on short-period repeats (e.g. a stuck "the the the the..." loop, a known failure
    /// mode of weak/small models under pure greedy decoding — the exact case this technique should help most).
    /// Oldest-first costs nothing extra: every draft is verified against the real model regardless of which
    /// historical match produced it, so there's no reliability tradeoff, only more usable length. No draft
    /// model — this only pays off when the n-gram genuinely recurs (repetitive structure); returns an empty
    /// array whenever no match exists (the common case for free-form prose), which is always safe since an
    /// empty draft just costs one plain decode step. The search window is capped at
    /// <see cref="SpecMaxLookback"/> tokens so a very long generation can't turn this into an O(n²) scan —
    /// missing a distant match only forgoes a speedup, it never affects correctness.</summary>
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
