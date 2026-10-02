using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Correctness gate for <see cref="TextGenerationPipeline"/>'s opt-in prefix-KV reuse overload: across a
/// growing multi-turn conversation (each turn's prompt is the previous turn's full committed sequence plus a few
/// new tokens — exactly the shape <c>VoiceConversation</c> produces turn over turn), generating WITH a
/// <see cref="RetainedSequence"/> carried across turns must be byte-for-byte identical to generating the SAME
/// prompt fresh every turn (reuse off) — including when the retained cache is grown by copy at check-out, shrunk by
/// copy at check-in, or freed for exceeding its byte cap. A tiny random-weight CPU model under the session's actual
/// default sampling (temperature, top-p, fixed seed — see <see cref="SamplingOptions.Seed"/>'s "0 is a fixed
/// reproducible constant", not random) is deterministic, so this covers the non-greedy path the voice session uses,
/// not just greedy. <see cref="PrefixCacheReuseGpuTests"/> runs the same conversation on CUDA and Vulkan.</summary>
public sealed class PrefixCacheReuseTests
{
    private static GenericTransformer Load(TransformerConfig cfg, Dictionary<string, Tensor> weights)
    {
        GenericTransformer model = new(cfg);
        model.LoadWeights(weights, "model");
        return model;
    }

    private static SamplingOptions Sampling(bool greedy, float temperature) =>
        SamplingOptions.Default with { Greedy = greedy, Temperature = temperature, TopP = 0.95f, Seed = 0 };

    private static string Ids(IReadOnlyList<int> ids) => string.Join(",", ids);

    [Theory]
    [InlineData(true, 0f)]            // greedy: the simplest, strongest determinism guarantee.
    [InlineData(false, 0.7f)]         // the voice session's actual default: non-greedy, fixed seed.
    public void GrowingConversation_RetainedPrefix_MatchesFreshPrefillEveryTurn(bool greedy, float temperature)
    {
        PrefixCacheTestModel rig = new(0x51ED270Bu);
        TransformerConfig cfg = PrefixCacheTestModel.Config();
        Dictionary<string, Tensor> w = rig.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = Load(cfg, w);
        PrefixCacheTestModel.StubTokenizer tokenizer = new();
        SamplingOptions sampling = Sampling(greedy, temperature);

        TextGenerationPipeline pipeline = new(model, tokenizer, backend);   // the ONE pipeline carried across turns
        List<int> history = [];
        using RetainedSequence retained = new();

        for (int turn = 0; turn < 6; turn++)
        {
            for (int i = 0; i < 5; i++) history.Add(rig.NextToken(cfg.VocabSize));
            int[] promptIds = [.. history];
            // A capacity hint sized for the whole conversation and the default headroom: the retained cache is
            // reused in place every turn, never resized — the baseline the resizing tests below compare against.
            GenerationRequest request = new() { RawTokenIds = promptIds, MaxTokens = 8, Sampling = sampling, PrefixCacheCapacityHint = 200 };

            // Reference: a brand-new pipeline and a brand-new cache every turn.
            GenerationResult reference = new TextGenerationPipeline(model, tokenizer, backend).Generate(request);

            // Actual: the SAME retained sequence carried from the previous turn.
            GenerationResult actual = pipeline.Generate(request, retained);

            Assert.Equal(reference.StoppedOnStopToken, actual.StoppedOnStopToken);
            Assert.Equal(Ids(reference.TokenIds), Ids(actual.TokenIds));
            Assert.Equal(reference.PromptTokens, actual.PromptTokens);
            if (turn > 0)
            {
                // Each turn's prompt extends the previous turn's full committed sequence by exactly 5 tokens, so
                // everything up to (but not including) the final new token must have been served from the cache.
                Assert.True(actual.ReusedPromptTokens >= promptIds.Length - 5 - 1,
                    $"turn {turn}: expected most of the prefix reused, got {actual.ReusedPromptTokens} of {promptIds.Length}");
            }

            history.AddRange(reference.TokenIds);   // next turn's prompt is a strict extension of this one
        }

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Theory]
    [InlineData(true, 0f)]
    [InlineData(false, 0.7f)]
    public void GrowAndShrinkEveryTurn_MatchesFreshPrefill_AndKeepsOnlyLengthPlusHeadroom(bool greedy, float temp)
    {
        // A headroom far below MaxTokens + 1: turn 0's hint-sized allocation is shrunk at check-in, and every later
        // turn outgrows what was kept, so each one grows by copy at check-out and shrinks again at check-in.
        const int headroom = 3;
        PrefixCacheTestModel rig = new(0x6A09E667u);
        TransformerConfig cfg = PrefixCacheTestModel.Config();
        Dictionary<string, Tensor> w = rig.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = Load(cfg, w);
        using GenericTransformerModel sizing = new(model, backend);
        PrefixCacheTestModel.StubTokenizer tokenizer = new();
        SamplingOptions sampling = Sampling(greedy, temp);

        TextGenerationPipeline pipeline = new(model, tokenizer, backend);
        List<int> history = [];
        using RetainedSequence retained = new();
        int previousLength = 0;

        for (int turn = 0; turn < 6; turn++)
        {
            for (int i = 0; i < 5; i++) history.Add(rig.NextToken(cfg.VocabSize));
            int[] promptIds = [.. history];
            GenerationRequest request = new()
            {
                RawTokenIds = promptIds, MaxTokens = 8, Sampling = sampling, PrefixCacheCapacityHint = 200,
                PrefixCacheHeadroomTokens = headroom,
            };

            GenerationResult reference = new TextGenerationPipeline(model, tokenizer, backend).Generate(request);
            GenerationResult actual = pipeline.Generate(request, retained);

            Assert.Equal(reference.StoppedOnStopToken, actual.StoppedOnStopToken);
            Assert.Equal(Ids(reference.TokenIds), Ids(actual.TokenIds));
            // Every later turn needed a grow (what was kept had at most `headroom` spare tokens, the turn needs
            // 5 + MaxTokens + 1 more), and the WHOLE previous sequence was still reused: grown by copy, not
            // re-prefilled the way an outgrown cache used to be.
            Assert.Equal(turn == 0 ? 0 : previousLength, actual.ReusedPromptTokens);

            ISequenceState kept = Assert.IsAssignableFrom<ISequenceState>(retained.Cache);
            Assert.Equal(kept.Length, retained.TokenIds.Length);
            Assert.InRange(kept.Capacity - kept.Length, 0, headroom);   // turn 0's 200-token hint included
            Assert.Equal(sizing.EstimateSequenceBytes(kept.Capacity), retained.Bytes);
            previousLength = kept.Length;

            history.AddRange(reference.TokenIds);
        }

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void ToolRoundOutgrowingTheFirstRoundsHint_GrowsByCopy_AndReusesTheWholeRound()
    {
        // ToolLoop carries round 1's request — hint included — into every round, so a large tool result makes round
        // 2's prompt outgrow round 1's hint-sized cache. That used to re-prefill the whole conversation.
        PrefixCacheTestModel rig = new(0xBB67AE85u);
        TransformerConfig cfg = PrefixCacheTestModel.Config();
        Dictionary<string, Tensor> w = rig.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = Load(cfg, w);
        PrefixCacheTestModel.StubTokenizer tokenizer = new();
        SamplingOptions sampling = Sampling(greedy: true, temperature: 0f);

        TextGenerationPipeline pipeline = new(model, tokenizer, backend);
        using RetainedSequence retained = new();
        int[] round1 = [.. Enumerable.Range(0, 6).Select(_ => rig.NextToken(cfg.VocabSize))];
        GenerationRequest first = new()
        {
            RawTokenIds = round1, MaxTokens = 4, Sampling = sampling, PrefixCacheCapacityHint = 20,
        };
        pipeline.Generate(first, retained);
        int roundOneLength = retained.TokenIds.Length;

        int[] round2 = [.. retained.TokenIds, .. Enumerable.Range(0, 30).Select(_ => rig.NextToken(cfg.VocabSize))];
        GenerationRequest second = first with { RawTokenIds = round2 };
        Assert.True(round2.Length + second.MaxTokens + 1 > 20, "round 2 must outgrow round 1's hint");

        GenerationResult reference = new TextGenerationPipeline(model, tokenizer, backend).Generate(second);
        GenerationResult actual = pipeline.Generate(second, retained);

        Assert.Equal(Ids(reference.TokenIds), Ids(actual.TokenIds));
        Assert.Equal(roundOneLength, actual.ReusedPromptTokens);

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void ASequenceOverTheByteCap_IsFreedWithoutBeingCopied_AndTheNextTurnStillMatches()
    {
        PrefixCacheTestModel rig = new(0x3C6EF372u);
        TransformerConfig cfg = PrefixCacheTestModel.Config();
        Dictionary<string, Tensor> w = rig.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = Load(cfg, w);
        using GenericTransformerModel real = new(model, backend);
        InstrumentedModel observed = new(real);
        PrefixCacheTestModel.StubTokenizer tokenizer = new();
        SamplingOptions sampling = Sampling(greedy: true, temperature: 0f);
        long perToken = real.EstimateSequenceBytes(1);

        TextGenerationPipeline pipeline = new(observed, tokenizer);
        using RetainedSequence retained = new();

        // Turn 1 fits: retained, unshrunk (a 64-token headroom covers the whole 64-token hint).
        int[] prompt1 = [.. Enumerable.Range(0, 6).Select(_ => rig.NextToken(cfg.VocabSize))];
        GenerationRequest request1 = new()
        {
            RawTokenIds = prompt1, MaxTokens = 6, Sampling = sampling, PrefixCacheCapacityHint = 64,
            PrefixCacheHeadroomTokens = 64,
        };
        pipeline.Generate(request1, retained);
        Assert.NotNull(retained.Cache);

        // Turn 2 still fits in that cache (no grow), but keeping it would cost more than the cap — even its own
        // prompt alone does. It must be freed outright, never first shrunk into a buffer that is then thrown away.
        int[] prompt2 = [.. retained.TokenIds, .. Enumerable.Range(0, 3).Select(_ => rig.NextToken(cfg.VocabSize))];
        GenerationRequest request2 = request1 with
        {
            RawTokenIds = prompt2, PrefixCacheHeadroomTokens = 4, PrefixCacheMaxBytes = perToken * prompt2.Length,
        };
        int resizesBefore = observed.ResizeCalls;
        GenerationResult reference2 = new TextGenerationPipeline(model, tokenizer, backend).Generate(request2);
        GenerationResult actual2 = pipeline.Generate(request2, retained);

        Assert.Equal(Ids(reference2.TokenIds), Ids(actual2.TokenIds));
        Assert.Equal(prompt2.Length - 3, actual2.ReusedPromptTokens);
        Assert.Equal(resizesBefore, observed.ResizeCalls);
        Assert.Null(retained.Cache);
        Assert.Empty(retained.TokenIds);
        Assert.Equal(0, retained.Bytes);

        // Turn 3: nothing was retained, so it prefills in full — and still matches.
        int[] prompt3 =
            [.. prompt2, .. actual2.TokenIds, .. Enumerable.Range(0, 2).Select(_ => rig.NextToken(cfg.VocabSize))];
        GenerationRequest request3 = request1 with { RawTokenIds = prompt3 };
        GenerationResult reference3 = new TextGenerationPipeline(model, tokenizer, backend).Generate(request3);
        GenerationResult actual3 = pipeline.Generate(request3, retained);

        Assert.Equal(Ids(reference3.TokenIds), Ids(actual3.TokenIds));
        Assert.Equal(0, actual3.ReusedPromptTokens);
        Assert.NotNull(retained.Cache);

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void ExactRepeatPrompt_StillPrefillsAtLeastOneToken_AndMatchesFreshDecode()
    {
        // A request whose prompt is identical to (or a prefix of) what is already retained must not degenerate
        // to a zero-length prefill -- AcquireCache always leaves at least the final prompt token to be prefilled
        // so sampling has a real logits row.
        PrefixCacheTestModel rig = new(0xC0FFEEu);
        TransformerConfig cfg = PrefixCacheTestModel.Config();
        Dictionary<string, Tensor> w = rig.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = Load(cfg, w);
        PrefixCacheTestModel.StubTokenizer tokenizer = new();
        SamplingOptions sampling = Sampling(greedy: true, temperature: 0f);

        int[] promptIds = [.. Enumerable.Range(0, 6).Select(_ => rig.NextToken(cfg.VocabSize))];
        GenerationRequest request = new() { RawTokenIds = promptIds, MaxTokens = 6, Sampling = sampling };

        TextGenerationPipeline pipeline = new(model, tokenizer, backend);
        using RetainedSequence retained = new();

        pipeline.Generate(request, retained);
        GenerationResult reference = new TextGenerationPipeline(model, tokenizer, backend).Generate(request);
        GenerationResult second = pipeline.Generate(request, retained);   // exact same prompt again, same key

        Assert.Equal(Ids(reference.TokenIds), Ids(second.TokenIds));
        Assert.True(second.ReusedPromptTokens < promptIds.Length, "must always prefill at least the final prompt token.");

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void OnTokenCancellation_BetweenAddAndItsOwnPrefill_RetainsExactlyWhatTheCacheCommitted()
    {
        // Regression: onToken runs right after generated.Add(next) but BEFORE that token's own Prefill commits
        // it to the cache -- exactly where TextService's filter/tool-call-stop wrapper throws
        // OperationCanceledException from inside onToken once it sees a completed tool call. Without clamping to
        // cache.Length, the retained sequence would claim one more token than the cache actually holds, and the
        // NEXT turn's AcquireCache (whenever the new prompt's common prefix reaches that same length -- exactly
        // what happens when ToolLoop's next request is built from this turn's own text) would call
        // FixedKvCache.Truncate with a length past the cache's current length, which throws.
        PrefixCacheTestModel rig = new(0xFACADEu);
        TransformerConfig cfg = PrefixCacheTestModel.Config();
        Dictionary<string, Tensor> w = rig.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = Load(cfg, w);
        PrefixCacheTestModel.StubTokenizer tokenizer = new();
        SamplingOptions sampling = Sampling(greedy: true, temperature: 0f);
        int[] promptIds = [.. Enumerable.Range(0, 6).Select(_ => rig.NextToken(cfg.VocabSize))];

        TextGenerationPipeline pipeline = new(model, tokenizer, backend);
        using RetainedSequence retained = new();

        GenerationRequest cancelling = new() { RawTokenIds = promptIds, MaxTokens = 8, Sampling = sampling, PrefixCacheCapacityHint = 64 };
        int seen = 0;
        Assert.Throws<OperationCanceledException>(() =>
            pipeline.Generate(cancelling, retained, onToken: _ => { if (++seen == 2) throw new OperationCanceledException(); }));

        Assert.NotNull(retained.Cache);
        // The invariant the fix guarantees: never more ids retained than the cache actually committed.
        Assert.Equal(retained.Cache!.Length, retained.TokenIds.Length);

        // The next turn's prompt is built from this turn's own (possibly-truncated) text, same as ToolLoop does
        // -- a strict extension of exactly what got retained, which used to be able to exceed old.Length.
        int[] nextPromptIds =
            [.. retained.TokenIds, .. Enumerable.Range(0, 4).Select(_ => rig.NextToken(cfg.VocabSize))];
        GenerationRequest next = new() { RawTokenIds = nextPromptIds, MaxTokens = 6, Sampling = sampling, PrefixCacheCapacityHint = 64 };

        GenerationResult actual = pipeline.Generate(next, retained);   // must not throw
        GenerationResult reference = new TextGenerationPipeline(model, tokenizer, backend).Generate(next);

        Assert.Equal(Ids(reference.TokenIds), Ids(actual.TokenIds));
        Assert.True(actual.ReusedPromptTokens > 0, "the retained prefix from turn 1 should still have been reused.");

        foreach (Tensor t in w.Values) t.Dispose();
    }

    /// <summary>Delegates every <see cref="IGenerationModel"/> member to a real <see cref="GenericTransformerModel"/>,
    /// counting resize and prefill calls, and throws <see cref="OperationCanceledException"/> from the
    /// <see cref="ThrowOnPrefillCall"/>-th prefill instead of running it — a prefill that observes a cancellation
    /// mid-operation, on a call index the test picks. The real path, <see cref="GenericTransformerModel"/> stopping
    /// between layers on the request's token, is driven by <c>PrefillCancellationTests</c>.</summary>
    private sealed class InstrumentedModel(GenericTransformerModel inner) : IGenerationModel
    {
        public int PrefillCalls { get; private set; }
        public int ResizeCalls { get; private set; }
        public int ThrowOnPrefillCall { get; set; } = -1;
        public GenerationModelInfo Info => inner.Info;
        public GenerationCapabilities Capabilities => inner.Capabilities;
        public IBackend OutputBackend => inner.OutputBackend;
        public ISequenceState CreateSequenceState(SequenceStateOptions options) => inner.CreateSequenceState(options);
        public Tensor DecodeBatch(ReadOnlySpan<int> tokenIds, ISequenceState[] states) => inner.DecodeBatch(tokenIds, states);
        public Tensor ProjectLogits(Tensor hidden, int rows) => inner.ProjectLogits(hidden, rows);
        public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits) => inner.EnumerateWeights(includeRedundantSplits);
        public long EstimateSequenceBytes(int contextTokens) => inner.EstimateSequenceBytes(contextTokens);
        public CapacitySnapshot Capacity() => inner.Capacity();
        public void Dispose() => inner.Dispose();

        public ISequenceState? ResizeSequenceState(ISequenceState state, int capacity)
        {
            ResizeCalls++;
            return inner.ResizeSequenceState(state, capacity);
        }

        public Tensor Prefill(in PrefillChunk chunk, ISequenceState state) => Prefill(chunk, state, CancellationToken.None);

        public Tensor Prefill(in PrefillChunk chunk, ISequenceState state, CancellationToken cancel)
        {
            if (PrefillCalls++ == ThrowOnPrefillCall)
            {
                throw new OperationCanceledException("injected: backend observed cancellation mid-prefill");
            }
            return inner.Prefill(chunk, state, cancel);
        }
    }

    [Fact]
    public void OnCancellationDuringTheFirstPrefill_DiscardsCleanly_NoCorruptedReuse()
    {
        // Regression: if Prefill throws OperationCanceledException during the very FIRST prefill (cache.Length
        // still short of promptIds.Length — nothing has been committed yet), treating that as "committed" would
        // store the full prompt in RetainedSequence.TokenIds while the cache itself only holds part of it: the same
        // class of mismatch as the eager-loop bug above, but for the call index BEFORE firstPrefillDone is set. The
        // request's token now reaches that prefill (PrefillCancellationTests drives the real path); the
        // fault-injecting model pins the pipeline's side of the guarantee for any model that throws there.
        PrefixCacheTestModel rig = new(0xB0BACAFEu);
        TransformerConfig cfg = PrefixCacheTestModel.Config();
        Dictionary<string, Tensor> w = rig.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = Load(cfg, w);
        PrefixCacheTestModel.StubTokenizer tokenizer = new();
        SamplingOptions sampling = Sampling(greedy: true, temperature: 0f);
        int[] promptIds = [.. Enumerable.Range(0, 6).Select(_ => rig.NextToken(cfg.VocabSize))];

        using GenericTransformerModel real = new(model, backend);
        InstrumentedModel faulty = new(real) { ThrowOnPrefillCall = 0 };   // throws on the very first Prefill call
        TextGenerationPipeline pipeline = new(faulty, tokenizer);
        using RetainedSequence retained = new();

        GenerationRequest request = new() { RawTokenIds = promptIds, MaxTokens = 8, Sampling = sampling, PrefixCacheCapacityHint = 64 };
        Assert.Throws<OperationCanceledException>(() => pipeline.Generate(request, retained));

        // The genuine-fault branch must have run: nothing retained at all, not a cache claiming a partial prompt.
        Assert.Null(retained.Cache);
        Assert.Empty(retained.TokenIds);

        // A later call under the same (now-empty) key must run uncached, not throw from a stale/partial cache.
        TextGenerationPipeline freshPipeline = new(model, tokenizer, backend);
        GenerationResult actual = freshPipeline.Generate(request, retained);
        GenerationResult reference = new TextGenerationPipeline(model, tokenizer, backend).Generate(request);
        Assert.Equal(Ids(reference.TokenIds), Ids(actual.TokenIds));
        Assert.Equal(0, actual.ReusedPromptTokens);

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void OnCancellationDuringTheFirstPrefill_AfterAGrow_DiscardsBothCachesCleanly()
    {
        // The grow at check-out already replaced the retained cache with a copy before the first prefill runs, so
        // a cancellation there leaves two caches in play: the outgrown one (already freed by the grow) and the copy
        // (owned by this call alone). Both must be released exactly once and nothing retained.
        PrefixCacheTestModel rig = new(0x510E527Fu);
        TransformerConfig cfg = PrefixCacheTestModel.Config();
        Dictionary<string, Tensor> w = rig.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = Load(cfg, w);
        using GenericTransformerModel real = new(model, backend);
        InstrumentedModel faulty = new(real);
        PrefixCacheTestModel.StubTokenizer tokenizer = new();
        SamplingOptions sampling = Sampling(greedy: true, temperature: 0f);

        TextGenerationPipeline pipeline = new(faulty, tokenizer);
        using RetainedSequence retained = new();
        int[] prompt1 = [.. Enumerable.Range(0, 6).Select(_ => rig.NextToken(cfg.VocabSize))];
        GenerationRequest request1 = new()
        {
            RawTokenIds = prompt1, MaxTokens = 6, Sampling = sampling, PrefixCacheHeadroomTokens = 1,
        };
        pipeline.Generate(request1, retained);

        int[] prompt2 = [.. retained.TokenIds, .. Enumerable.Range(0, 4).Select(_ => rig.NextToken(cfg.VocabSize))];
        GenerationRequest request2 = request1 with { RawTokenIds = prompt2 };
        Assert.True(prompt2.Length + request2.MaxTokens + 1 > retained.Cache!.Capacity, "turn 2 must need a grow");
        int resizesBefore = faulty.ResizeCalls;
        faulty.ThrowOnPrefillCall = faulty.PrefillCalls;   // the very next Prefill: turn 2's first
        Assert.Throws<OperationCanceledException>(() => pipeline.Generate(request2, retained));

        Assert.Equal(resizesBefore + 1, faulty.ResizeCalls);   // the grow ran before the prefill was cancelled
        Assert.Null(retained.Cache);
        Assert.Empty(retained.TokenIds);

        faulty.ThrowOnPrefillCall = -1;
        GenerationResult actual = pipeline.Generate(request2, retained);
        GenerationResult reference = new TextGenerationPipeline(model, tokenizer, backend).Generate(request2);
        Assert.Equal(Ids(reference.TokenIds), Ids(actual.TokenIds));
        Assert.Equal(0, actual.ReusedPromptTokens);
        Assert.NotNull(retained.Cache);

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void CapacityOutgrown_GrowsByCopy_ReusingTheWholePreviousTurn()
    {
        // No hint: each cache is sized for just its own prompt + MaxTokens, so every turn outgrows the last. The
        // retained prefix must be carried into the bigger cache rather than dropped and prefilled again.
        PrefixCacheTestModel rig = new(0xFEEDFACEu);
        TransformerConfig cfg = PrefixCacheTestModel.Config();
        Dictionary<string, Tensor> w = rig.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = Load(cfg, w);
        PrefixCacheTestModel.StubTokenizer tokenizer = new();
        SamplingOptions sampling = Sampling(greedy: true, temperature: 0f);

        TextGenerationPipeline pipeline = new(model, tokenizer, backend);
        List<int> history = [];
        using RetainedSequence retained = new();

        for (int turn = 0; turn < 5; turn++)
        {
            for (int i = 0; i < 6; i++) history.Add(rig.NextToken(cfg.VocabSize));
            int[] promptIds = [.. history];
            int previousLength = retained.TokenIds.Length;
            GenerationRequest request = new() { RawTokenIds = promptIds, MaxTokens = 4, Sampling = sampling };

            GenerationResult reference = new TextGenerationPipeline(model, tokenizer, backend).Generate(request);
            GenerationResult actual = pipeline.Generate(request, retained);

            Assert.Equal(Ids(reference.TokenIds), Ids(actual.TokenIds));
            Assert.Equal(previousLength, actual.ReusedPromptTokens);
            history.AddRange(reference.TokenIds);
        }

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void ReuseIsOptIn_DefaultOverloadBehavesExactlyAsBefore()
    {
        PrefixCacheTestModel rig = new(0x1337u);
        TransformerConfig cfg = PrefixCacheTestModel.Config();
        Dictionary<string, Tensor> w = rig.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = Load(cfg, w);
        PrefixCacheTestModel.StubTokenizer tokenizer = new();
        SamplingOptions sampling = Sampling(greedy: true, temperature: 0f);
        int[] promptIds = [.. Enumerable.Range(0, 7).Select(_ => rig.NextToken(cfg.VocabSize))];
        GenerationRequest request = new() { RawTokenIds = promptIds, MaxTokens = 5, Sampling = sampling };

        GenerationResult viaOldOverload = new TextGenerationPipeline(model, tokenizer, backend).Generate(request);
        GenerationResult viaNewOverloadNoReuse = new TextGenerationPipeline(model, tokenizer, backend).Generate(request, reuse: null);

        Assert.Equal(Ids(viaOldOverload.TokenIds), Ids(viaNewOverloadNoReuse.TokenIds));
        Assert.Equal(0, viaNewOverloadNoReuse.ReusedPromptTokens);

        foreach (Tensor t in w.Values) t.Dispose();
    }
}
