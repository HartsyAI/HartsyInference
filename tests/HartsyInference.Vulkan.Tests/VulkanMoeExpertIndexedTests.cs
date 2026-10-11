using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Vulkan.Tests.Dsv41TestData;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The expert-indexed MoE decode ops (<c>MoeExpertGateUp</c>, <c>MoeExpertDown</c>, <c>MoeCombineSlots</c>) and the grouped expert
/// upload they read, against a double-precision reference built from the host dequantizer. Skips without a Vulkan device.</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class VulkanMoeExpertIndexedTests(ITestOutputHelper log)
{
    private static Tensor[] RandomExperts(DType type, int experts, int n, int k, int seed)
    {
        Tensor[] list = new Tensor[experts];
        for (int e = 0; e < experts; e++) list[e] = VulkanQuantGemvTests.RandomQuant(type, n, k, seed + 131 * e);
        return list;
    }

    /// <summary>Host-dequantized experts as flat float arrays [n * k].</summary>
    private static float[][] Dequantized(Tensor[] experts)
    {
        float[][] result = new float[experts.Length][];
        for (int e = 0; e < experts.Length; e++)
        {
            using Tensor f = GgufDequantizer.Dequantize(experts[e], DType.F32);
            result[e] = new ReadOnlySpan<float>((void*)f.DataPointer, (int)f.ElementCount).ToArray();
        }
        return result;
    }

    private static (double Value, double Magnitude) Dot(float[] w, int row, int k, float[] x, int xOffset)
    {
        double s = 0, a = 0;
        for (int i = 0; i < k; i++)
        {
            double prod = (double)w[(long)row * k + i] * x[xOffset + i];
            s += prod;
            a += Math.Abs(prod);
        }
        return (s, a);
    }

    private static double Silu(double x) => x / (1 + Math.Exp(-x));

    private static double GeluTanh(double x) => 0.5 * x * (1 + Math.Tanh(0.7978845608 * (x + 0.044715 * x * x * x)));

    private static int[] RandomIds(int pairs, int experts, int seed)
    {
        Random rng = new(seed);
        int[] ids = new int[pairs];
        for (int i = 0; i < pairs; i++) ids[i] = rng.Next(experts);
        return ids;
    }

    private static void Preload(VulkanBackend vk, params Tensor[][] groups) => vk.PreloadWeightGroups(groups.Select(g => (IReadOnlyList<Tensor>)g));

    public static IEnumerable<object[]> Types()
    {
        foreach (string t in new[] { "Q8_0", "Q4_K", "Q6_K", "Q4_0", "Q5_0", "Q5_K" }) yield return new object[] { t };
    }

    [Theory]
    [MemberData(nameof(Types))]
    public void GateUp_MatchesTheReference_SiluAndGelu(string typeName)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        log.WriteLine($"device: {vk.Vk.DeviceName}");
        DType type = VulkanQuantGemvTests.Named(typeName);
        Assert.True(vk.SupportsMoeExpertIndexed(type));
        const int Experts = 8, N = 37, K = 512, Tokens = 5, TopK = 3;
        Tensor[] gate = RandomExperts(type, Experts, N, K, 1), up = RandomExperts(type, Experts, N, K, 1000);
        try
        {
            float[][] gateF = Dequantized(gate), upF = Dequantized(up);
            Preload(vk, gate, up);
            Tensor[] dummyDown = RandomExperts(type, Experts, K, 256, 7);
            try
            {
                Preload(vk, dummyDown);
                Assert.True(vk.MoeExpertsResident(gate, up, dummyDown));
                float[] x = Random(Tokens * K, seed: 5, scale: 1f);
                int[] ids = RandomIds(Tokens * TopK, Experts, 11);
                foreach (bool gelu in new[] { false, true })
                {
                    using Tensor xt = F32(x, 1, Tokens, K);
                    using Tensor idx = I32(ids, Tokens, TopK);
                    using Tensor act = EmptyF32(1, Tokens * TopK, N);
                    vk.MoeExpertGateUp(act, xt, gate, up, idx, TopK, gelu);
                    float[] got = ReadF32(act);
                    double worst = 0;
                    for (int r = 0; r < Tokens * TopK; r++)
                    {
                        int e = ids[r], token = r / TopK;
                        for (int n = 0; n < N; n++)
                        {
                            (double g, double gm) = Dot(gateF[e], n, K, x, token * K);
                            (double u, double um) = Dot(upF[e], n, K, x, token * K);
                            double a = gelu ? GeluTanh(g) : Silu(g);
                            double expected = a * u;
                            double bound = 5e-5 * (1.2 * gm * Math.Abs(u) + Math.Abs(a) * um) + 1e-6;
                            double err = Math.Abs(expected - got[r * N + n]);
                            worst = Math.Max(worst, err / bound);
                            Assert.True(err <= bound, $"{typeName} gelu={gelu} pair {r} row {n}: ref {expected:R} vk {got[r * N + n]:R} bound {bound:R}");
                        }
                    }
                    log.WriteLine($"{typeName} gate/up gelu={gelu}: worst error {worst:F3} of the bound");
                }
            }
            finally { foreach (Tensor t in dummyDown) t.Dispose(); }
        }
        finally
        {
            foreach (Tensor t in gate.Concat(up)) t.Dispose();
        }
    }

    [Theory]
    [MemberData(nameof(Types))]
    public void Down_MatchesTheReference(string typeName)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        DType type = VulkanQuantGemvTests.Named(typeName);
        const int Experts = 8, N = 41, K = 512, Tokens = 4, TopK = 2;
        Tensor[] down = RandomExperts(type, Experts, N, K, 21);
        try
        {
            float[][] downF = Dequantized(down);
            Preload(vk, down);
            float[] act = Random(Tokens * TopK * K, seed: 6, scale: 1f);
            int[] ids = RandomIds(Tokens * TopK, Experts, 12);
            using Tensor actT = F32(act, 1, Tokens * TopK, K);
            using Tensor idx = I32(ids, Tokens, TopK);
            using Tensor slot = EmptyF32(1, Tokens * TopK, N);
            vk.MoeExpertDown(slot, actT, down, idx, TopK);
            float[] got = ReadF32(slot);
            double worst = 0;
            for (int r = 0; r < Tokens * TopK; r++)
            {
                for (int n = 0; n < N; n++)
                {
                    (double v, double m) = Dot(downF[ids[r]], n, K, act, r * K);
                    double bound = 2e-5 * m + 1e-6;
                    double err = Math.Abs(v - got[r * N + n]);
                    worst = Math.Max(worst, err / bound);
                    Assert.True(err <= bound, $"{typeName} down pair {r} row {n}: ref {v:R} vk {got[r * N + n]:R}");
                }
            }
            log.WriteLine($"{typeName} down: worst error {worst:F3} of the bound");
        }
        finally { foreach (Tensor t in down) t.Dispose(); }
    }

    /// <summary>OLMoE's layout: Q4_K gate/up and a Q6_K down in one layer, each projection its own group and dtype.</summary>
    [Fact]
    public void GateUpDown_MixedTypesInOneLayer_ChainThroughTheDevice()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int Experts = 16, Inter = 256, Hidden = 512, Tokens = 6, TopK = 4;
        Tensor[] gate = RandomExperts(DType.Q4_K, Experts, Inter, Hidden, 31), up = RandomExperts(DType.Q4_K, Experts, Inter, Hidden, 32);
        Tensor[] down = RandomExperts(DType.Q6_K, Experts, Hidden, Inter, 33);
        try
        {
            float[][] gateF = Dequantized(gate), upF = Dequantized(up), downF = Dequantized(down);
            Preload(vk, gate, up, down);
            Assert.True(vk.MoeExpertsResident(gate, up, down));
            float[] x = Random(Tokens * Hidden, seed: 7, scale: 1f);
            int[] ids = RandomIds(Tokens * TopK, Experts, 13);
            using Tensor xt = F32(x, 1, Tokens, Hidden);
            using Tensor idx = I32(ids, Tokens, TopK);
            using Tensor act = EmptyF32(1, Tokens * TopK, Inter);
            using Tensor slot = EmptyF32(1, Tokens * TopK, Hidden);
            long syncBefore = vk.GetD2hSyncCount();
            vk.MoeExpertGateUp(act, xt, gate, up, idx, TopK, gelu: false);
            vk.MoeExpertDown(slot, act, down, idx, TopK);
            Assert.Equal(syncBefore, vk.GetD2hSyncCount());   // nothing was read back between the two ops
            float[] got = ReadF32(slot);
            for (int r = 0; r < Tokens * TopK; r++)
            {
                int e = ids[r];
                double[] a = new double[Inter];
                for (int n = 0; n < Inter; n++)
                {
                    a[n] = Silu(Dot(gateF[e], n, Hidden, x, r / TopK * Hidden).Value) * Dot(upF[e], n, Hidden, x, r / TopK * Hidden).Value;
                }
                for (int n = 0; n < Hidden; n++)
                {
                    double s = 0, m = 0;
                    for (int i = 0; i < Inter; i++)
                    {
                        double prod = (double)downF[e][n * Inter + i] * (float)a[i];
                        s += prod;
                        m += Math.Abs(prod);
                    }
                    Assert.True(Math.Abs(s - got[r * Hidden + n]) <= 1e-4 * m + 1e-5, $"pair {r} row {n}: ref {s:R} vk {got[r * Hidden + n]:R}");
                }
            }
        }
        finally { foreach (Tensor t in gate.Concat(up).Concat(down)) t.Dispose(); }
    }

    /// <summary>More pairs than one dispatch's 65535 workgroups along y: the launcher chunks and offsets the pair index.</summary>
    [Fact]
    public void GateUpDown_MorePairsThanOneDispatch_ChunkTheLaunch()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int Experts = 4, N = 4, K = 32, Tokens = 70001, TopK = 1;
        Tensor[] gate = RandomExperts(DType.Q8_0, Experts, N, K, 41), up = RandomExperts(DType.Q8_0, Experts, N, K, 42);
        Tensor[] down = RandomExperts(DType.Q8_0, Experts, K, N * 8, 43);
        try
        {
            float[][] gateF = Dequantized(gate), upF = Dequantized(up);
            Preload(vk, gate, up);
            float[] x = Random(Tokens * K, seed: 8, scale: 1f);
            int[] ids = RandomIds(Tokens, Experts, 14);
            using Tensor xt = F32(x, 1, Tokens, K);
            using Tensor idx = I32(ids, Tokens, TopK);
            using Tensor act = EmptyF32(1, Tokens, N);
            vk.MoeExpertGateUp(act, xt, gate, up, idx, TopK, gelu: false);
            float[] got = ReadF32(act);
            foreach (int r in new[] { 0, 1, 65534, 65535, 65536, 70000 })
            {
                for (int n = 0; n < N; n++)
                {
                    (double g, double gm) = Dot(gateF[ids[r]], n, K, x, r * K);
                    (double u, double um) = Dot(upF[ids[r]], n, K, x, r * K);
                    double expected = Silu(g) * u;
                    double bound = 5e-5 * (1.2 * gm * Math.Abs(u) + Math.Abs(Silu(g)) * um) + 1e-6;
                    Assert.True(Math.Abs(expected - got[r * N + n]) <= bound, $"pair {r} row {n}: ref {expected:R} vk {got[r * N + n]:R}");
                }
            }
        }
        finally { foreach (Tensor t in gate.Concat(up).Concat(down)) t.Dispose(); }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void CombineSlots_SumsInSlotOrder_WithTheGatedSharedRow(bool withShared, bool withGate)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int Tokens = 7, TopK = 5, Hidden = 300;
        float[] slots = Random(Tokens * TopK * Hidden, seed: 51, scale: 1f);
        float[] weights = Random(Tokens * TopK, seed: 52, scale: 1f);
        float[] shared = Random(Tokens * Hidden, seed: 53, scale: 1f);
        float[] gate = Random(Tokens, seed: 54, scale: 3f);
        using Tensor slotT = F32(slots, 1, Tokens * TopK, Hidden);
        using Tensor weightT = F32(weights, Tokens, TopK);
        using Tensor? sharedT = withShared ? F32(shared, 1, Tokens, Hidden) : null;
        using Tensor? gateT = withGate ? F32(gate, 1, Tokens, 1) : null;
        using Tensor output = EmptyF32(1, Tokens, Hidden);
        vk.MoeCombineSlots(output, slotT, weightT, sharedT, gateT, TopK);
        float[] got = ReadF32(output);
        for (int t = 0; t < Tokens; t++)
        {
            for (int i = 0; i < Hidden; i++)
            {
                float acc = 0;
                if (withShared)
                {
                    float s = shared[t * Hidden + i];
                    if (withGate) s *= 1.0f / (1.0f + MathF.Exp(-gate[t]));
                    acc = s;
                }
                for (int j = 0; j < TopK; j++) acc += weights[t * TopK + j] * slots[(t * TopK + j) * Hidden + i];
                Assert.True(Math.Abs(acc - got[t * Hidden + i]) <= 1e-5 * (1 + Math.Abs(acc)), $"token {t} column {i}: ref {acc:R} vk {got[t * Hidden + i]:R}");
            }
        }
    }

    /// <summary>The members of a group stay ordinary resident weights: an op that binds one whole buffer reads that expert, not the first.</summary>
    [Fact]
    public void GroupedMembers_AreOrdinaryWeights_ForEveryOtherOp()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int Experts = 6, N = 33, K = 256;
        Tensor[] experts = RandomExperts(DType.Q4_K, Experts, N, K, 61);
        try
        {
            float[][] f = Dequantized(experts);
            Preload(vk, experts);
            float[] x = Random(K, seed: 9, scale: 1f);
            foreach (int e in new[] { 0, 1, 3, Experts - 1 })
            {
                using Tensor xt = F32(x, 1, 1, K);
                using Tensor o = EmptyF32(1, 1, N);
                vk.Linear(o, xt, experts[e], null);
                float[] got = ReadF32(o);
                for (int n = 0; n < N; n++)
                {
                    (double v, double m) = Dot(f[e], n, K, x, 0);
                    Assert.True(Math.Abs(v - got[n]) <= 2e-5 * m + 1e-6, $"expert {e} row {n}: ref {v:R} vk {got[n]:R}");
                }
            }
        }
        finally { foreach (Tensor t in experts) t.Dispose(); }
    }

    [Fact]
    public void Resident_IsRefused_ForUngroupedFreedAndMixedGroups()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int Experts = 6, N = 16, K = 256;
        Tensor[] a = RandomExperts(DType.Q4_K, Experts, N, K, 71), b = RandomExperts(DType.Q4_K, Experts, N, K, 72), c = RandomExperts(DType.Q4_K, Experts, K, N * 16, 73);
        try
        {
            vk.PreloadWeights(a);   // member by member: resident, but not a group
            Preload(vk, b, c);
            Assert.False(vk.MoeExpertsResident(a, b, c));
            Assert.False(vk.MoeExpertsResident(b, a, c));
            Assert.True(vk.MoeExpertsResident(b, b, c));
            vk.FreeWeights([b[2]]);
            Assert.False(vk.MoeExpertsResident(b, b, c));
            vk.FreeWeights(b);
            vk.FreeWeights(c);
            Assert.False(vk.MoeExpertsResident(c, c, c));
            // Freeing returns the allocation: a second grouped upload of the same size still fits and works.
            Preload(vk, c);
            Assert.True(vk.MoeExpertsResident(c, c, c));
            Assert.False(vk.MoeExpertsResident(c.Take(3).ToArray(), c.Take(3).ToArray(), c.Take(3).ToArray()));   // a prefix is not the group
        }
        finally { foreach (Tensor t in a.Concat(b).Concat(c)) t.Dispose(); }
    }

    [Fact]
    public void Unsupported_Types_AreDeclined()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        Assert.False(vk.SupportsMoeExpertIndexed(DType.F32));
        Assert.False(vk.SupportsMoeExpertIndexed(DType.Q2_K));
        Assert.False(((HartsyInference.Core.Backends.IBackend)vk).SupportsMoeExpertsGrouped(DType.Q4_K));
    }
}
