using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Correctness gate for <see cref="TextGenerationPipeline"/>'s opt-in prefix-KV reuse overload: across a
/// growing multi-turn conversation (each turn's prompt is the previous turn's full committed sequence plus a few
/// new tokens — exactly the shape <c>VoiceConversation</c> produces turn over turn), generating WITH a
/// <see cref="RetainedSequence"/> carried across turns must be byte-for-byte identical to generating the SAME
/// prompt fresh every turn (today's unchanged behavior, reuse off) — the "output tokens identical with the cache
/// on vs off" requirement. A tiny random-weight CPU model under the session's actual default sampling (temperature,
/// top-p, fixed seed — see <see cref="SamplingOptions.Seed"/>'s "0 is a fixed reproducible constant", not random)
/// is deterministic, so this also covers the non-greedy path the voice session actually uses, not just greedy.</summary>
public sealed class PrefixCacheReuseTests
{
    private static uint _rng = 0x9E3779B9u;
    private static uint NextRaw() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return _rng; }
    private static float Rand() { return ((NextRaw() & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static unsafe Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static unsafe Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    private static TransformerConfig Cfg() => new()
    {
        HiddenSize = 16, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
        IntermediateSize = 32, VocabSize = 37, MaxPositionEmbeddings = 512, AttentionBias = true, QkNorm = false,
    };

    private static Dictionary<string, Tensor> Weights(TransformerConfig c)
    {
        int h = c.HiddenSize, qDim = c.QDim, kvDim = c.KvDim;
        Dictionary<string, Tensor> w = new() { ["model.embed_tokens.weight"] = F2(c.VocabSize, h), ["model.norm.weight"] = Ones(h) };
        for (int i = 0; i < c.NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            w[$"{p}.input_layernorm.weight"] = Ones(h);
            w[$"{p}.post_attention_layernorm.weight"] = Ones(h);
            w[$"{p}.self_attn.q_proj.weight"] = F2(qDim, h);
            w[$"{p}.self_attn.k_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.v_proj.weight"] = F2(kvDim, h);
            w[$"{p}.self_attn.o_proj.weight"] = F2(h, qDim);
            w[$"{p}.self_attn.q_proj.bias"] = F1(qDim);
            w[$"{p}.self_attn.k_proj.bias"] = F1(kvDim);
            w[$"{p}.self_attn.v_proj.bias"] = F1(kvDim);
            w[$"{p}.mlp.gate_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.up_proj.weight"] = F2(c.IntermediateSize, h);
            w[$"{p}.mlp.down_proj.weight"] = F2(h, c.IntermediateSize);
        }
        return w;
    }

    /// <summary>Bypasses Encode/chat-template entirely: every test drives prompts via <see cref="GenerationRequest.RawTokenIds"/>.</summary>
    private sealed class StubTokenizer : ILlmTokenizer
    {
        public int[] Encode(string text, bool addSpecial) => throw new NotSupportedException();
        public int[] EncodeOrdinary(string text) => throw new NotSupportedException();
        public string Decode(IReadOnlyList<int> ids) => string.Join(",", ids);
        public int? SpecialId(string token) => null;
        public int? BosId => null;
        public int? EosId => 36;
        public IReadOnlyList<int> StopIds => [36];
        public string? BosToken => null;
        public string? EosToken => null;
    }

    private static int NextToken(int vocab) => (int)(NextRaw() % (uint)(vocab - 1));

    [Theory]
    [InlineData(true, 0f, 0u)]            // greedy: the simplest, strongest determinism guarantee.
    [InlineData(false, 0.7f, 0u)]         // the voice session's actual default: non-greedy, fixed seed.
    public void GrowingConversation_RetainedPrefix_MatchesFreshPrefillEveryTurn(bool greedy, float temperature, ulong seed)
    {
        _rng = 0x51ED270Bu;
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        SamplingOptions sampling = SamplingOptions.Default with { Greedy = greedy, Temperature = temperature, TopP = 0.95f, Seed = seed };

        TextGenerationPipeline pipeline = new(model, tokenizer, backend);   // the ONE pipeline carried across turns
        List<int> history = [];
        using RetainedSequence retained = new();

        for (int turn = 0; turn < 6; turn++)
        {
            for (int i = 0; i < 5; i++) history.Add(NextToken(cfg.VocabSize));
            int[] promptIds = [.. history];
            // A generous capacity hint, as a real caller (e.g. the voice session) would pass: sized for the
            // whole conversation's eventual growth, not just this turn -- without it, every turn's own tight
            // sizing is outgrown by the next and AcquireCache correctly (see CapacityOutgrown_* below) falls back
            // to a fresh, unreused cache every time.
            GenerationRequest request = new() { RawTokenIds = promptIds, MaxTokens = 8, Sampling = sampling, PrefixCacheCapacityHint = 200 };

            // Reference: today's unchanged behavior -- a brand-new pipeline and a brand-new cache every turn.
            GenerationResult reference = new TextGenerationPipeline(model, tokenizer, backend).Generate(request);

            // Actual: the SAME retained sequence carried from the previous turn.
            GenerationResult actual = pipeline.Generate(request, retained);

            Assert.Equal(reference.StoppedOnStopToken, actual.StoppedOnStopToken);
            Assert.Equal(string.Join(",", reference.TokenIds), string.Join(",", actual.TokenIds));
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

    [Fact]
    public void ExactRepeatPrompt_StillPrefillsAtLeastOneToken_AndMatchesFreshDecode()
    {
        // A request whose prompt is identical to (or a prefix of) what is already retained must not degenerate
        // to a zero-length prefill -- AcquireCache always leaves at least the final prompt token to be prefilled
        // so sampling has a real logits row.
        _rng = 0xC0FFEEu;
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        SamplingOptions sampling = SamplingOptions.Default with { Greedy = true };

        int[] promptIds = [.. Enumerable.Range(0, 6).Select(_ => NextToken(cfg.VocabSize))];
        GenerationRequest request = new() { RawTokenIds = promptIds, MaxTokens = 6, Sampling = sampling };

        TextGenerationPipeline pipeline = new(model, tokenizer, backend);
        using RetainedSequence retained = new();

        GenerationResult first = pipeline.Generate(request, retained);
        GenerationResult reference = new TextGenerationPipeline(model, tokenizer, backend).Generate(request);
        GenerationResult second = pipeline.Generate(request, retained);   // exact same prompt again, same key

        Assert.Equal(string.Join(",", reference.TokenIds), string.Join(",", second.TokenIds));
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
        _rng = 0xFACADEu;
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        SamplingOptions sampling = SamplingOptions.Default with { Greedy = true };
        int[] promptIds = [.. Enumerable.Range(0, 6).Select(_ => NextToken(cfg.VocabSize))];

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
        int[] nextPromptIds = [.. retained.TokenIds, .. Enumerable.Range(0, 4).Select(_ => NextToken(cfg.VocabSize))];
        GenerationRequest next = new() { RawTokenIds = nextPromptIds, MaxTokens = 6, Sampling = sampling, PrefixCacheCapacityHint = 64 };

        GenerationResult actual = pipeline.Generate(next, retained);   // must not throw
        GenerationResult reference = new TextGenerationPipeline(model, tokenizer, backend).Generate(next);

        Assert.Equal(string.Join(",", reference.TokenIds), string.Join(",", actual.TokenIds));
        Assert.True(actual.ReusedPromptTokens > 0, "the retained prefix from turn 1 should still have been reused.");

        foreach (Tensor t in w.Values) t.Dispose();
    }

    /// <summary>Delegates every <see cref="IGenerationModel"/> member to a real <see cref="GenericTransformerModel"/>
    /// except <see cref="Prefill"/>, which throws <see cref="OperationCanceledException"/> on a caller-chosen call
    /// index instead of running it — a <c>Prefill</c> that observes a cancellation mid-operation, on a call index the
    /// test picks. The real path, <see cref="GenericTransformerModel"/> stopping between layers on the request's token,
    /// is driven by <c>PrefillCancellationTests</c>.</summary>
    private sealed class FaultInjectingModel(GenericTransformerModel inner, int throwOnCallIndex) : IGenerationModel
    {
        private int _calls;
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

        public Tensor Prefill(in PrefillChunk chunk, ISequenceState state)
        {
            if (_calls++ == throwOnCallIndex) throw new OperationCanceledException("injected: backend observed cancellation mid-prefill");
            return inner.Prefill(chunk, state);
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
        _rng = 0xB0BACAFEu;
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        SamplingOptions sampling = SamplingOptions.Default with { Greedy = true };
        int[] promptIds = [.. Enumerable.Range(0, 6).Select(_ => NextToken(cfg.VocabSize))];

        using GenericTransformerModel real = new(model, backend);
        FaultInjectingModel faulty = new(real, throwOnCallIndex: 0);   // throws on the very first Prefill call
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
        Assert.Equal(string.Join(",", reference.TokenIds), string.Join(",", actual.TokenIds));
        Assert.Equal(0, actual.ReusedPromptTokens);

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void CapacityOutgrown_ReallocatesAndStaysCorrect()
    {
        // A tiny initial cache (sized for turn 1 only) must be transparently replaced once the conversation grows
        // past it, without corrupting output or leaking the old buffer.
        _rng = 0xFEEDFACEu;
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        SamplingOptions sampling = SamplingOptions.Default with { Greedy = true };

        TextGenerationPipeline pipeline = new(model, tokenizer, backend);
        List<int> history = [];
        using RetainedSequence retained = new();

        for (int turn = 0; turn < 5; turn++)
        {
            for (int i = 0; i < 6; i++) history.Add(NextToken(cfg.VocabSize));
            int[] promptIds = [.. history];
            // No PrefixCacheCapacityHint: each new cache is sized just for ITS OWN prompt + MaxTokens, so growth
            // past a few turns forces AcquireCache's reallocation branch repeatedly.
            GenerationRequest request = new() { RawTokenIds = promptIds, MaxTokens = 4, Sampling = sampling };

            GenerationResult reference = new TextGenerationPipeline(model, tokenizer, backend).Generate(request);
            GenerationResult actual = pipeline.Generate(request, retained);

            Assert.Equal(string.Join(",", reference.TokenIds), string.Join(",", actual.TokenIds));
            history.AddRange(reference.TokenIds);
        }

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void ReuseIsOptIn_DefaultOverloadBehavesExactlyAsBefore()
    {
        _rng = 0x1337u;
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        SamplingOptions sampling = SamplingOptions.Default with { Greedy = true };
        int[] promptIds = [.. Enumerable.Range(0, 7).Select(_ => NextToken(cfg.VocabSize))];
        GenerationRequest request = new() { RawTokenIds = promptIds, MaxTokens = 5, Sampling = sampling };

        GenerationResult viaOldOverload = new TextGenerationPipeline(model, tokenizer, backend).Generate(request);
        GenerationResult viaNewOverloadNoReuse = new TextGenerationPipeline(model, tokenizer, backend).Generate(request, reuse: null);

        Assert.Equal(string.Join(",", viaOldOverload.TokenIds), string.Join(",", viaNewOverloadNoReuse.TokenIds));
        Assert.Equal(0, viaNewOverloadNoReuse.ReusedPromptTokens);

        foreach (Tensor t in w.Values) t.Dispose();
    }
}
