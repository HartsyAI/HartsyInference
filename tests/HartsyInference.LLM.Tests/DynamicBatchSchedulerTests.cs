using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Correctness + concurrency gates for <see cref="DynamicBatchScheduler"/> (the Phase 4 replacement
/// for the old static-batch scheduler): (1) concurrently-submitted requests must produce byte-identical
/// output to running each one alone through <see cref="TextGenerationPipeline"/> — batching changes
/// throughput, not results, on the CPU backend's batch-invariant GEMM (same guarantee the design this
/// replaces documented); (2) requests submitted AFTER the loop is already running must still be admitted
/// (proving admission is dynamic, not a fixed up-front list); (3) cancelling one request mid-batch must not
/// affect the others; (4) a KV pool too small for the workload must fail admission with
/// <see cref="KvPoolExhaustedException"/> rather than hang or corrupt other sequences.</summary>
public sealed class DynamicBatchSchedulerTests
{
    private static uint _rng = 0x9E3779B9u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.2f; }
    private static unsafe Tensor Fill(Tensor t) { float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor F2(int a, int b) => Fill(new Tensor(new TensorShape(a, b), DType.F32));
    private static Tensor F1(int a) => Fill(new Tensor(new TensorShape(a), DType.F32));
    private static unsafe Tensor Ones(int n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (int i = 0; i < n; i++) p[i] = 1f; return t; }

    private static TransformerConfig Cfg() => new()
    {
        HiddenSize = 16, NumLayers = 2, NumHeads = 4, NumKvHeads = 2, HeadDim = 4,
        IntermediateSize = 32, VocabSize = 32, MaxPositionEmbeddings = 64, AttentionBias = true, QkNorm = false,
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

    /// <summary>Minimal tokenizer stub: tests drive prompts via <see cref="GenerationRequest.RawTokenIds"/>
    /// (bypassing Encode/chat-template entirely), so only Decode/StopIds need real behavior.</summary>
    private sealed class StubTokenizer : ILlmTokenizer
    {
        public int[] Encode(string text, bool addSpecial) => throw new NotSupportedException();
        public int[] EncodeOrdinary(string text) => throw new NotSupportedException();
        public string Decode(IReadOnlyList<int> ids) => string.Join(",", ids);
        public int? SpecialId(string token) => null;
        public int? BosId => null;
        public int? EosId => 31;
        public IReadOnlyList<int> StopIds => [31];
        public string? BosToken => null;
        public string? EosToken => null;
    }

    private static GenerationRequest Req(int[] promptIds, int maxTokens, ulong seed) => new()
    {
        RawTokenIds = promptIds,
        MaxTokens = maxTokens,
        Sampling = SamplingOptions.Default with { Greedy = true, Seed = seed },
    };

    [Fact]
    public async Task ConcurrentRequests_MatchSingleSequenceReference()
    {
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();

        int[][] prompts = [[1, 2, 3], [4], [5, 6, 7, 8, 9]];
        const int maxTokens = 6;

        // Reference: each prompt decoded ALONE through TextGenerationPipeline, fresh model/cache each time —
        // this is the "batching must not change the answer" oracle.
        string[] reference = new string[prompts.Length];
        for (int i = 0; i < prompts.Length; i++)
        {
            TextGenerationPipeline refPipeline = new(model, tokenizer, backend);
            GenerationResult r = refPipeline.Generate(Req(prompts[i], maxTokens, seed: 0));
            reference[i] = string.Join(",", r.TokenIds);
        }

        // Batched: all three submitted concurrently to ONE scheduler sharing ONE pool.
        using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 64);
        using DynamicBatchScheduler scheduler = new(model, tokenizer, backend, pool);
        Task<GenerationResult>[] tasks = new Task<GenerationResult>[prompts.Length];
        for (int i = 0; i < prompts.Length; i++)
            tasks[i] = scheduler.SubmitAsync(Req(prompts[i], maxTokens, seed: 0), onToken: null, CancellationToken.None);
        GenerationResult[] results = await Task.WhenAll(tasks);

        for (int i = 0; i < prompts.Length; i++)
        {
            string got = string.Join(",", results[i].TokenIds);
            Assert.True(got == reference[i], $"prompt {i}: batched=[{got}] reference=[{reference[i]}]");
        }
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public async Task GpuGate_SerializesRoundsAndStillProducesCorrectResults()
    {
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();

        int[][] prompts = [[1, 2, 3], [4], [5, 6, 7]];
        const int maxTokens = 5;
        string[] reference = new string[prompts.Length];
        for (int i = 0; i < prompts.Length; i++)
        {
            TextGenerationPipeline refPipeline = new(model, tokenizer, backend);
            reference[i] = string.Join(",", refPipeline.Generate(Req(prompts[i], maxTokens, seed: 0)).TokenIds);
        }

        // A gate that (a) proves every GPU-touching call actually goes through it, and (b) would itself
        // throw if re-entered concurrently — simulating the exclusivity a shared InferenceQueue enforces.
        int concurrentCalls = 0, totalGateCalls = 0;
        async Task Gate(Action work)
        {
            if (Interlocked.Increment(ref concurrentCalls) != 1)
                throw new InvalidOperationException("gate was re-entered concurrently — exclusivity violated");
            Interlocked.Increment(ref totalGateCalls);
            try
            {
                await Task.Yield(); // force a real async hop so a race would actually be observable
                work();
            }
            finally
            {
                Interlocked.Decrement(ref concurrentCalls);
            }
        }

        using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 64);
        using DynamicBatchScheduler scheduler = new(model, tokenizer, backend, pool, gpuGate: Gate);
        Task<GenerationResult>[] tasks = new Task<GenerationResult>[prompts.Length];
        for (int i = 0; i < prompts.Length; i++)
            tasks[i] = scheduler.SubmitAsync(Req(prompts[i], maxTokens, seed: 0), onToken: null, CancellationToken.None);
        GenerationResult[] results = await Task.WhenAll(tasks);

        for (int i = 0; i < prompts.Length; i++)
            Assert.Equal(reference[i], string.Join(",", results[i].TokenIds));
        Assert.True(totalGateCalls > 0, "gate should have been invoked at least once (admission + decode rounds)");
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public async Task RequestsSubmittedAfterLoopIsRunning_AreAdmitted()
    {
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 64);
        using DynamicBatchScheduler scheduler = new(model, tokenizer, backend, pool);

        // First wave, then a real delay (the loop is actively decoding), then a SECOND wave — proves
        // admission isn't a fixed up-front list read once at construction.
        Task<GenerationResult> first = scheduler.SubmitAsync(Req([1, 2], 8, seed: 0), null, CancellationToken.None);
        await Task.Delay(20);
        Task<GenerationResult> second = scheduler.SubmitAsync(Req([3, 4], 8, seed: 0), null, CancellationToken.None);

        GenerationResult[] results = await Task.WhenAll(first, second);
        Assert.Equal(8, results[0].TokenIds.Count);
        Assert.Equal(8, results[1].TokenIds.Count);
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public async Task CancellingOneRequest_DoesNotAffectOthers()
    {
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 64);
        using DynamicBatchScheduler scheduler = new(model, tokenizer, backend, pool);

        using CancellationTokenSource cts = new();
        Task<GenerationResult> cancelled = scheduler.SubmitAsync(Req([1, 2], 500, seed: 0), null, cts.Token);
        Task<GenerationResult> survivor = scheduler.SubmitAsync(Req([3, 4], 6, seed: 0), null, CancellationToken.None);

        cts.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => cancelled);
        GenerationResult survivorResult = await survivor;
        Assert.Equal(6, survivorResult.TokenIds.Count);
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public async Task PoolExhaustion_FailsAdmissionWithoutCorruptingOthers()
    {
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();

        // Deliberately tiny pool: pageSize=4, maxPages=2 -> only 8 tokens of total KV capacity, shared.
        using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 2);
        using DynamicBatchScheduler scheduler = new(model, tokenizer, backend, pool);

        // This one alone fits (prompt=4 tokens, maxTokens small) — reference for "should still succeed".
        Task<GenerationResult> fits = scheduler.SubmitAsync(Req([1, 2, 3, 4], 2, seed: 0), null, CancellationToken.None);
        // This one's prompt alone (10 tokens) cannot possibly fit in an 8-token pool -> must fail admission.
        Task<GenerationResult> tooBig = scheduler.SubmitAsync(Req([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2, seed: 0), null, CancellationToken.None);

        await Assert.ThrowsAsync<KvPoolExhaustedException>(() => tooBig);
        GenerationResult fitsResult = await fits;
        Assert.Equal(2, fitsResult.TokenIds.Count);
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
