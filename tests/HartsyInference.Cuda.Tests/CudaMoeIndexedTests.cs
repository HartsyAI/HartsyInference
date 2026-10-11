using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The device-resident routed stage of <see cref="MoeFeedForward"/> (router, <c>MoeRoute</c>, then expert-indexed GEMVs for up to 16 tokens or dispatch plus grouped GEMMs above that)
/// against the same block on the CPU backend, which routes on the host and dequantizes the same quantized experts. The only difference
/// the GPU path may add is the int8 rounding of the activation.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaMoeIndexedTests
{
    private readonly ITestOutputHelper _output;
    public CudaMoeIndexedTests(ITestOutputHelper output) => _output = output;

    private static uint _rng = 0x7A11C0DEu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.5f; }
    private static Tensor F2(int a, int b) { Tensor t = new(new TensorShape(a, b), DType.F32); float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor X(int n, int h) { Tensor t = new(new TensorShape(1, n, h), DType.F32); float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand() * 4f; return t; }
    private static Tensor Q(int a, int b, DType type) { using Tensor f = F2(a, b); return GgufQuantizer.Quantize(f, type); }

    public static IEnumerable<object[]> Cases()
    {
        foreach (int n in new[] { 1, 3, 16, 40, 200 })
        {
            yield return [n, "Q4_K", "Q4_K", false];
            yield return [n, "Q4_K", "Q6_K", false];   // the Q4_K_M mix: gate/up Q4_K, down Q6_K
            yield return [n, "Q8_0", "Q8_0", false];
            yield return [n, "Q4_K", "Q6_K", true];    // shared expert with its sigmoid gate (Qwen2-MoE)
        }
        // About 500 rows per expert: past the point where a dense GEMM per expert replaces the grouped call.
        yield return [2000, "Q4_K", "Q6_K", false];
        yield return [2000, "Q8_0", "Q8_0", false];
    }

    /// <summary>The same comparison with every down projection split across 4 warps per row (the long-K path Mixtral's
    /// ffn_down takes), forced through <see cref="EngineKnobs.GemvKsplit"/> because the test's K is short.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(1, "Q4_K", "Q4_K")]
    [InlineData(3, "Q4_K", "Q6_K")]
    [InlineData(16, "Q8_0", "Q8_0")]
    public void IndexedMoe_KsplitDown_MatchesHostRoutedMoe(int n, string gateUpType, string downType)
    {
        KnobStore.Set(EngineKnobs.GemvKsplit, 4);
        try { IndexedMoe_MatchesHostRoutedMoe(n, gateUpType, downType, false); }
        finally { KnobStore.Clear(EngineKnobs.GemvKsplit); }
    }

    [Fact]
    public void DownKsplit_DefaultHeuristic_SplitsOnlyLongKFewRowLaunches()
    {
        try
        {
            KnobStore.Clear(EngineKnobs.GemvKsplit);
            Assert.Equal(4u, CudaKernels.MoeDownKsplitWarps(4096, 14336, 2));     // Mixtral ffn_down, decode
            Assert.Equal(4u, CudaKernels.MoeDownKsplitWarps(4096, 14336, 16));    // a speculative-verify batch still fits the cap
            Assert.Equal(1u, CudaKernels.MoeDownKsplitWarps(4096, 14336, 64));    // many rows: plenty of blocks without splitting
            Assert.Equal(1u, CudaKernels.MoeDownKsplitWarps(2048, 768, 8));       // Qwen3-30B-A3B ffn_down: short K stays unsplit
            Assert.Equal(1u, CudaKernels.MoeDownKsplitWarps(1, 14336, 70000));    // gridDim.y limit
            KnobStore.Set(EngineKnobs.GemvKsplit, 0);
            Assert.Equal(1u, CudaKernels.MoeDownKsplitWarps(4096, 14336, 2));     // the knob turns it off
            KnobStore.Set(EngineKnobs.GemvKsplit, 8);
            Assert.Equal(8u, CudaKernels.MoeDownKsplitWarps(2048, 768, 8));       // and forces it
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.GemvKsplit);
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [MemberData(nameof(Cases))]
    public void IndexedMoe_MatchesHostRoutedMoe(int n, string gateUpType, string downType, bool shared)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        DType gu = gateUpType == "Q8_0" ? DType.Q8_0 : DType.Q4_K;
        DType down = downType == "Q6_K" ? DType.Q6_K : downType == "Q8_0" ? DType.Q8_0 : DType.Q4_K;

        const int hidden = 512, inter = 256, shInter = 256, e = 16, topK = 4;
        MoeConfig moe = new()
        {
            NumExperts = e, NumExpertsPerTok = topK, MoeIntermediateSize = inter,
            NormTopKProb = true, Scoring = MoeScoring.Softmax,
            SharedExpertIntermediateSize = shared ? shInter : 0,
        };
        const string p = "model.layers.0";
        Dictionary<string, Tensor> w = new() { [$"{p}.mlp.gate.weight"] = F2(e, hidden) };
        if (shared)
        {
            w[$"{p}.mlp.shared_expert.gate_proj.weight"] = F2(shInter, hidden);
            w[$"{p}.mlp.shared_expert.up_proj.weight"] = F2(shInter, hidden);
            w[$"{p}.mlp.shared_expert.down_proj.weight"] = F2(hidden, shInter);
            w[$"{p}.mlp.shared_expert_gate.weight"] = F2(1, hidden);
        }
        for (int i = 0; i < e; i++)
        {
            w[$"{p}.mlp.experts.{i}.gate_proj.weight"] = Q(inter, hidden, gu);
            w[$"{p}.mlp.experts.{i}.up_proj.weight"] = Q(inter, hidden, gu);
            w[$"{p}.mlp.experts.{i}.down_proj.weight"] = Q(hidden, inter, down);
        }
        MoeFeedForward moeFf = new(moe, hidden, lowVram: false);
        moeFf.LoadWeights(w, p);

        // The reference is the same block with every expert dequantized to F32, run on the CPU backend (host routing, F32 math).
        Dictionary<string, Tensor> wRef = new();
        foreach ((string key, Tensor t) in w) wRef[key] = t.DType.IsQuantized ? GgufDequantizer.Dequantize(t, DType.F32) : t;
        MoeFeedForward refFf = new(moe, hidden, lowVram: false);
        refFf.LoadWeights(wRef, p);

        using Tensor x = X(n, hidden);
        float[] reference;
        using (CpuBackend cpu = new())
        using (Tensor oc = refFf.Forward(cpu, x, n))
        {
            reference = new float[oc.ElementCount];
            float* pc = (float*)oc.DataPointer;
            for (long i = 0; i < reference.Length; i++) reference[i] = pc[i];
        }

        using CudaBackend cuda = new(0, ptxDir);
        cuda.PreloadWeightGroups(moeFf.EnumerateExpertGroups());
        cuda.PreloadWeights([.. moeFf.EnumerateWeights()]);
        Assert.True(n <= MoeFeedForward.IndexedMaxTokens ? moeFf.CanRunIndexed(cuda) : moeFf.CanRunGrouped(cuda),
            "the device-routed path must be eligible for this layer");

        using Tensor og = moeFf.Forward(cuda, x, n);
        cuda.Sync();
        float* pg = (float*)og.DataPointer;
        double diff = 0, norm = 0;
        float peak = 0;
        for (long i = 0; i < reference.Length; i++)
        {
            double d = pg[i] - reference[i];
            diff += d * d;
            norm += (double)reference[i] * reference[i];
            peak = MathF.Max(peak, MathF.Abs(reference[i]));
        }
        double rel = Math.Sqrt(diff / Math.Max(norm, 1e-30));
        _output.WriteLine($"n={n} {gateUpType}/{downType} shared={shared}: relative L2 error {rel:E3}, peak |ref| {peak:F4}");
        Assert.True(rel <= 3e-2, $"indexed MoE diverges from the host-routed MoE by {rel:E3}");

        foreach ((string key, Tensor t) in wRef) if (!ReferenceEquals(t, w[key])) t.Dispose();
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
