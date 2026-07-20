using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Phase 2 gate for the graph-decode-into-scheduler retrofit: proves the admission-time
/// capture-once + circuit-breaker logic in <see cref="DynamicBatchScheduler.AdmitAndPrefill"/> without
/// needing a real CUDA capture failure (the CPU backend never satisfies graph-decode eligibility for real,
/// so this uses the internal <see cref="DynamicBatchScheduler.TestForceSupportsGraphDecode"/> and
/// <see cref="DynamicBatchScheduler.TestGraphCaptureFailureInjector"/> hooks — same pattern as
/// <see cref="DynamicBatchSchedulerFaultIsolationTests"/>'s <c>TestFaultInjector</c>). A real capture failure
/// is architecture/backend-determined, not request-specific, so the breaker exists to stop repeating a
/// failure that will always recur — this test proves that behavior deterministically.</summary>
public sealed class GraphDecodeAdmissionTests
{
    private static uint _rng = 0x9E1AC03u;
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
        GraphDecode = true,
    };

    [Fact]
    public async Task CaptureFailure_TripsCircuitBreaker_LoopSurvives_SecondRequestUsesPagedPathSuccessfully()
    {
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 64);
        using DynamicBatchScheduler scheduler = new(model, tokenizer, backend, pool);

        int captureAttempts = 0;
        scheduler.TestForceSupportsGraphDecode = true;
        scheduler.TestGraphCaptureFailureInjector = () =>
        {
            captureAttempts++;
            return new InvalidOperationException("synthetic capture failure (test)");
        };

        // First solo request: capture is attempted and fails — the request itself must fail loudly with the
        // injected exception (not silently degrade to something else), and the loop must survive it.
        InvalidOperationException thrown = await Assert.ThrowsAsync<InvalidOperationException>(
            () => scheduler.SubmitAsync(Req([1, 2, 3], 10), onToken: null, CancellationToken.None));
        Assert.Equal("synthetic capture failure (test)", thrown.Message);
        Assert.True(scheduler.IsLoopAlive);
        Assert.Equal(1, captureAttempts);

        // Second solo request (same eligibility on paper): the breaker must skip capture entirely — no second
        // attempt — and the request must still complete correctly via the ordinary PagedKvCache path.
        GenerationResult result = await scheduler.SubmitAsync(Req([4, 5, 6], 10), onToken: null, CancellationToken.None);
        Assert.Equal(1, captureAttempts);   // unchanged — breaker skipped the attempt
        Assert.NotEmpty(result.TokenIds);

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public async Task SoloEligibleRequest_WithNoInjectedFailure_CompletesNormallyOnCpuBackend()
    {
        // CaptureGraph's DEFAULT (no CUDA backend) returns null and LaunchGraph throws NotSupportedException —
        // this test doesn't force eligibility, so on a real CpuBackend (GraphDecodeSupported = false by
        // default) graphEligible is false and the request just takes the ordinary PagedKvCache path. Sanity
        // check that a "looks graph-eligible" request (greedy, GraphDecode=true) still works end-to-end when
        // the backend genuinely doesn't support it — i.e. the new eligibility check correctly gates on the
        // real backend capability, not just the request's own opt-in flag.
        TransformerConfig cfg = Cfg();
        Dictionary<string, Tensor> w = Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        StubTokenizer tokenizer = new();
        using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 64);
        using DynamicBatchScheduler scheduler = new(model, tokenizer, backend, pool);

        GenerationResult result = await scheduler.SubmitAsync(Req([1, 2, 3], 10), onToken: null, CancellationToken.None);
        Assert.NotEmpty(result.TokenIds);

        foreach (Tensor t in w.Values) t.Dispose();
    }
}
