using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary><see cref="MoeFeedForward"/> on the Vulkan backend against the same block on the CPU backend, which routes on the host and uses the
/// same experts dequantized to F32: a decode-sized batch takes the device-routed stage (router GEMV, <c>MoeRoute</c>, expert-indexed
/// GEMVs, slot combine) with no host read in the layer; a larger batch keeps the host-routed loop over the same grouped weights.
/// Skips without a Vulkan device.</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class VulkanMoeLayerTests(ITestOutputHelper output)
{
    private static uint _rng = 0x7A11C0DEu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.5f; }
    private static Tensor F2(int a, int b) { Tensor t = new(new TensorShape(a, b), DType.F32); float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }
    private static Tensor X(int n, int h) { Tensor t = new(new TensorShape(1, n, h), DType.F32); float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand() * 4f; return t; }
    private static Tensor Q(int a, int b, DType type) { using Tensor f = F2(a, b); return GgufQuantizer.Quantize(f, type); }

    public static IEnumerable<object[]> Cases()
    {
        foreach (int n in new[] { 1, 3, 16, 40 })
        {
            yield return [n, "Q4_K", "Q4_K", false];
            yield return [n, "Q4_K", "Q6_K", false];   // the Q4_K_M mix: gate/up Q4_K, down Q6_K
            yield return [n, "Q8_0", "Q8_0", false];
            yield return [n, "Q4_K", "Q6_K", true];    // shared expert with its sigmoid gate (Qwen2-MoE)
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MoeLayer_MatchesTheHostRoutedCpuLayer(int n, string gateUpType, string downType, bool shared)
    {
        using VulkanBackend? vk = Dsv41TestData.TryCreateBackend(out string? skip);
        if (vk is null) { output.WriteLine($"SKIPPED: {skip}"); return; }
        output.WriteLine($"device: {vk.Vk.DeviceName}");
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

        vk.PreloadWeightGroups(moeFf.EnumerateExpertGroups());
        vk.PreloadWeights([.. moeFf.EnumerateWeights()]);
        bool indexed = n <= MoeFeedForward.IndexedMaxTokens;
        Assert.True(moeFf.CanRunIndexed(vk), "the layer must be eligible for the device-routed stage");
        long before = vk.GetD2hSyncCount();
        using Tensor og = moeFf.Forward(vk, x, n);
        long hostReads = vk.GetD2hSyncCount() - before;
        if (indexed) Assert.Equal(0, hostReads);   // the device-routed layer reads nothing back
        else Assert.True(hostReads > 0);           // the host-routed loop reads the router and each expert's rows back
        vk.Sync();
        float* pg = (float*)og.DataPointer;
        double diff = 0, norm = 0;
        for (long i = 0; i < reference.Length; i++)
        {
            double d = pg[i] - reference[i];
            diff += d * d;
            norm += (double)reference[i] * reference[i];
        }
        double rel = Math.Sqrt(diff / Math.Max(norm, 1e-30));
        output.WriteLine($"n={n} {gateUpType}/{downType} shared={shared} deviceRouted={indexed}: relative L2 error {rel:E3}, host reads in Forward {hostReads}");
        // The device-routed stage is F32 end to end; the host-routed loop above 16 tokens runs its per-expert GEMMs in F16 (numerics.vkF16Gemm).
        Assert.True(rel <= (indexed ? 1e-5 : 2e-3), $"MoE layer diverges from the host-routed CPU layer by {rel:E3}");

        foreach ((string key, Tensor t) in wRef) if (!ReferenceEquals(t, w[key])) t.Dispose();
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
