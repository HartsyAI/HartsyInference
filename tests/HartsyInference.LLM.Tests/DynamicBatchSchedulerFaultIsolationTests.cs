using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Regression gate for the 2026-07-11 crash-containment fix: an exception escaping a decode round
/// (native kernel bug, corrupted state, anything) used to kill <see cref="DynamicBatchScheduler"/>'s entire
/// background loop silently — no crash, no log, every future request to that model just hung forever since
/// nothing was left reading the admission channel. This uses the internal
/// <see cref="DynamicBatchScheduler.TestFaultInjector"/> hook (real backend crashes like the
/// AccessViolationException that motivated this can't be triggered on demand in a fast, safe unit test) to
/// prove: (1) a failed round faults only the sequences in it, not unrelated ones already active; (2) the
/// loop survives and keeps serving new requests after a fault; (3) <see cref="DynamicBatchScheduler.IsLoopAlive"/>
/// correctly reflects loop health in both the "kept running" and "genuinely died" cases.</summary>
public sealed class DynamicBatchSchedulerFaultIsolationTests
{
    private static uint _rng = 0x1234ABCDu;
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

    private static GenerationRequest Req(int[] promptIds, int maxTokens) => new()
    {
        RawTokenIds = promptIds,
        MaxTokens = maxTokens,
        Sampling = SamplingOptions.Default with { Greedy = true },
    };

    [Fact]
    public async Task RoundFailure_FaultsOnlyThatRoundsSequences_LoopSurvives()
    {
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 64);
        using DynamicBatchScheduler scheduler = new(model, tokenizer, backend, pool);

        // Armed BEFORE either request is submitted, so every decode round fails from the start — avoids any
        // dependency on exact admission/round timing (admission itself isn't gated by this injector, so
        // both sequences reach "active" normally; the very first decode round attempted against them fails).
        InvalidOperationException fault = new("synthetic decode-round failure (test)");
        scheduler.TestFaultInjector = _ => fault;

        Task<GenerationResult> victimA = scheduler.SubmitAsync(Req([1, 2, 3], 20), onToken: null, CancellationToken.None);
        Task<GenerationResult> victimB = scheduler.SubmitAsync(Req([4, 5], 20), onToken: null, CancellationToken.None);

        await Assert.ThrowsAsync<InvalidOperationException>(() => victimA);
        await Assert.ThrowsAsync<InvalidOperationException>(() => victimB);

        // The loop must still be alive after the fault — this is the actual regression this test guards:
        // before the fix, an uncaught round exception killed RunLoopAsync entirely.
        Assert.True(scheduler.IsLoopAlive);

        // Disarm the injector and prove the scheduler still serves NEW requests correctly post-fault.
        scheduler.TestFaultInjector = null;
        GenerationResult recovered = await scheduler.SubmitAsync(Req([6, 7, 8], 10), onToken: null, CancellationToken.None);
        Assert.NotEmpty(recovered.TokenIds);

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public async Task UnrelatedModel_UnaffectedByAnotherModels_RoundFailure()
    {
        // Two independent schedulers (as ModelManager would hold, one per loaded model) sharing nothing —
        // proves fault containment is per-scheduler-instance, not some shared static state that could leak
        // a failure from one loaded model into another's traffic.
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w1 = Weights(cfg), w2 = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer modelA = new(cfg);
        using GenericTransformer modelB = new(cfg);
        modelA.LoadWeights(w1, "model");
        modelB.LoadWeights(w2, "model");
        StubTokenizer tokenizer = new();
        using PagedKvPool poolA = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 64);
        using PagedKvPool poolB = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 64);
        using DynamicBatchScheduler schedulerA = new(modelA, tokenizer, backend, poolA);
        using DynamicBatchScheduler schedulerB = new(modelB, tokenizer, backend, poolB);

        schedulerA.TestFaultInjector = _ => new InvalidOperationException("model A is broken (test)");

        await Assert.ThrowsAsync<InvalidOperationException>(
            () => schedulerA.SubmitAsync(Req([1, 2, 3], 10), onToken: null, CancellationToken.None));

        // Model B never had a fault injected — must complete normally regardless of model A's failure.
        GenerationResult resultB = await schedulerB.SubmitAsync(Req([1, 2, 3], 10), onToken: null, CancellationToken.None);
        Assert.NotEmpty(resultB.TokenIds);
        Assert.True(schedulerB.IsLoopAlive);

        foreach (Tensor t in w1.Values) t.Dispose();
        foreach (Tensor t in w2.Values) t.Dispose();
    }
}
