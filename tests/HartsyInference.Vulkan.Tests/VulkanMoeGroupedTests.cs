using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Vulkan.Tests.Dsv41TestData;
using static HartsyInference.Vulkan.Tests.VulkanMoeExpertIndexedTests;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The device-routed MoE prefill ops (<c>MoeExpertsGrouped</c>, <c>MoeCombinePairs</c>) and the guards the decode ops carry, against a
/// double-precision reference built from the host dequantizer. Skips without a Vulkan device (or, for the grouped GEMMs, without
/// cooperative matrices).</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class VulkanMoeGroupedTests(ITestOutputHelper log)
{
    /// <summary>An expert id outside [0, experts) (a router bug, or the padding of a masked pair) writes zeros instead of reading past the stack.</summary>
    [Fact]
    public void OutOfRangeExpertIds_WriteZeros_InGateUpAndDown()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int Experts = 4, Inter = 32, Hidden = 256, Tokens = 3, TopK = 2;
        Tensor[] gate = RandomExperts(DType.Q4_K, Experts, Inter, Hidden, 81), up = RandomExperts(DType.Q4_K, Experts, Inter, Hidden, 82);
        Tensor[] down = RandomExperts(DType.Q4_K, Experts, Hidden, 256, 83);
        try
        {
            Preload(vk, gate, up, down);
            int[] ids = { 1, -1, Experts, 2, Experts + 1000, 0 };   // pairs 1, 2 and 4 are out of range
            using Tensor xt = F32(Random(Tokens * Hidden, seed: 3, scale: 1f), 1, Tokens, Hidden);
            using Tensor idx = I32(ids, Tokens, TopK);
            using Tensor act = EmptyF32(1, Tokens * TopK, Inter);
            vk.MoeExpertGateUp(act, xt, gate, up, idx, TopK, gelu: false);
            float[] a = ReadF32(act);
            using Tensor actIn = F32(Random(Tokens * TopK * 256, seed: 4, scale: 1f), 1, Tokens * TopK, 256);
            using Tensor slot = EmptyF32(1, Tokens * TopK, Hidden);
            vk.MoeExpertDown(slot, actIn, down, idx, TopK);
            float[] d = ReadF32(slot);
            for (int r = 0; r < ids.Length; r++)
            {
                bool valid = ids[r] >= 0 && ids[r] < Experts;
                bool aZero = a.Skip(r * Inter).Take(Inter).All(v => v == 0f), dZero = d.Skip(r * Hidden).Take(Hidden).All(v => v == 0f);
                Assert.Equal(!valid, aZero);
                Assert.Equal(!valid, dZero);
            }
        }
        finally { foreach (Tensor t in gate.Concat(up).Concat(down)) t.Dispose(); }
    }

    [Fact]
    public void GateUp_RejectsANonPositiveTopK()
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        Tensor[] gate = RandomExperts(DType.Q4_K, 2, 16, 256, 91);
        try
        {
            Preload(vk, gate);
            using Tensor xt = F32(Random(256, seed: 1), 1, 1, 256);
            using Tensor idx = I32(new[] { 0 }, 1, 1);
            using Tensor act = EmptyF32(1, 1, 16);
            Assert.Throws<ArgumentOutOfRangeException>(() => vk.MoeExpertGateUp(act, xt, gate, gate, idx, 0, gelu: false));
        }
        finally { foreach (Tensor t in gate) t.Dispose(); }
    }

    public static IEnumerable<object[]> GroupedCases()
    {
        foreach ((string gu, string dn) in new[] { ("Q4_K", "Q6_K"), ("Q8_0", "Q8_0"), ("Q4_0", "Q5_0"), ("Q5_K", "Q4_K") })
            foreach (bool gelu in new[] { false, true })
                yield return new object[] { gu, dn, gelu };
    }

    /// <summary>The grouped prefill GEMMs: per active expert, F16 weights and activations, F32 accumulation, against the double-precision reference.
    /// Some experts get no rows, and an expert's rows end inside a 16-row fragment.</summary>
    [Theory]
    [MemberData(nameof(GroupedCases))]
    public void ExpertsGrouped_MatchesTheReference(string guName, string dnName, bool gelu)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        IBackend ib = vk;
        DType gu = VulkanQuantGemvTests.Named(guName), dn = VulkanQuantGemvTests.Named(dnName);
        if (!ib.SupportsMoeExpertsGrouped(gu) || !ib.SupportsMoeExpertsGrouped(dn)) { log.WriteLine("SKIPPED: no cooperative-matrix GEMM on this device"); return; }
        const int Experts = 9, Inter = 256, Hidden = 512, Tokens = 37, TopK = 3;
        Tensor[] gate = RandomExperts(gu, Experts, Inter, Hidden, 101), up = RandomExperts(gu, Experts, Inter, Hidden, 202);
        Tensor[] down = RandomExperts(dn, Experts, Hidden, Inter, 303);
        try
        {
            float[][] gateF = Dequantized(gate), upF = Dequantized(up), downF = Dequantized(down);
            Preload(vk, gate, up, down);
            // Pairs go to experts 0..6 except 3, so experts 3, 7 and 8 are empty; the dispatch is built by hand like MoeBuildDispatch.
            int[] pool = { 0, 1, 2, 4, 5, 6 };
            Random rng = new(5);
            int[] ids = new int[Tokens * TopK];
            for (int i = 0; i < ids.Length; i++) ids[i] = pool[rng.Next(pool.Length)];
            int[] offsets = new int[Experts + 1];
            foreach (int id in ids) offsets[id + 1]++;
            for (int e = 0; e < Experts; e++) offsets[e + 1] += offsets[e];
            int[] cursor = (int[])offsets.Clone();
            int[] perm = new int[ids.Length];
            for (int p = 0; p < ids.Length; p++) perm[cursor[ids[p]]++] = p / TopK;
            float[] x = Random(Tokens * Hidden, seed: 17, scale: 1f);
            using Tensor xt = F32(x, 1, Tokens, Hidden);
            using Tensor permT = I32(perm, ids.Length);
            using Tensor outT = EmptyF32(1, ids.Length, Hidden);
            long syncBefore = vk.GetD2hSyncCount();
            ib.MoeExpertsGrouped(outT, xt, permT, offsets, gate, up, down, gelu);
            Assert.Equal(syncBefore, vk.GetD2hSyncCount());   // the layer's one read is the caller's, of the offsets
            float[] got = ReadF32(outT);
            double diff = 0, norm = 0;
            for (int row = 0; row < ids.Length; row++)
            {
                int e = 0;
                while (row >= offsets[e + 1]) e++;
                int token = perm[row];
                double[] a = new double[Inter];
                for (int n = 0; n < Inter; n++)
                {
                    double g = Dot(gateF[e], n, Hidden, x, token * Hidden).Value, u = Dot(upF[e], n, Hidden, x, token * Hidden).Value;
                    a[n] = (gelu ? GeluTanh(g) : Silu(g)) * u;
                }
                for (int n = 0; n < Hidden; n++)
                {
                    double s = 0;
                    for (int i = 0; i < Inter; i++) s += (double)downF[e][n * Inter + i] * a[i];
                    double d = got[row * Hidden + n] - s;
                    diff += d * d;
                    norm += s * s;
                }
            }
            double rel = Math.Sqrt(diff / norm);
            log.WriteLine($"{guName}/{dnName} gelu={gelu}: relative L2 error {rel:E3}");
            Assert.True(rel <= 4e-3, $"grouped experts diverge from the reference by {rel:E3}");
        }
        finally { foreach (Tensor t in gate.Concat(up).Concat(down)) t.Dispose(); }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void CombinePairs_FollowsThePairMap_WithTheGatedSharedRow(bool withShared, bool withGate)
    {
        using VulkanBackend? vk = TryCreateBackend(out string? skip);
        if (vk is null) { log.WriteLine($"SKIPPED: {skip}"); return; }
        const int Tokens = 7, TopK = 4, Hidden = 300, Rows = Tokens * TopK;
        float[] expertOut = Random(Rows * Hidden, seed: 61, scale: 1f);
        float[] weights = Random(Tokens * TopK, seed: 62, scale: 1f);
        float[] shared = Random(Tokens * Hidden, seed: 63, scale: 1f);
        float[] gate = Random(Tokens, seed: 64, scale: 3f);
        int[] slots = Enumerable.Range(0, Rows).OrderBy(i => (i * 7919) % 101).ToArray();   // a permutation of the rows
        slots[3] = -1;                                                                    // a masked pair adds nothing
        using Tensor expertT = F32(expertOut, 1, Rows, Hidden);
        using Tensor slotT = I32(slots, Tokens, TopK);
        using Tensor weightT = F32(weights, Tokens, TopK);
        using Tensor? sharedT = withShared ? F32(shared, 1, Tokens, Hidden) : null;
        using Tensor? gateT = withGate ? F32(gate, 1, Tokens, 1) : null;
        using Tensor output = EmptyF32(1, Tokens, Hidden);
        ((IBackend)vk).MoeCombinePairs(output, expertT, slotT, weightT, sharedT, gateT, TopK);
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
                for (int j = 0; j < TopK; j++)
                {
                    int slot = slots[t * TopK + j];
                    if (slot >= 0) acc += weights[t * TopK + j] * expertOut[slot * Hidden + i];
                }
                Assert.True(Math.Abs(acc - got[t * Hidden + i]) <= 1e-5 * (1 + Math.Abs(acc)), $"token {t} column {i}: ref {acc:R} vk {got[t * Hidden + i]:R}");
            }
        }
    }
}
