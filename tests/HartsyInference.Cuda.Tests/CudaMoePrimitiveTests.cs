using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Cuda.Tests.MoePrimitiveTestData;

namespace HartsyInference.Cuda.Tests;

/// <summary>Every MoE primitive on the CUDA backend against the CPU reference on the same inputs. Integer outputs
/// (ids, permutations, top-k indices) must match exactly; the router's float weights differ from libm only by exp ulps,
/// and combine, top-k values and dispatch are bit-identical by construction. Skips without CUDA.</summary>
[Collection("CudaSerial")]
public sealed class CudaMoePrimitiveTests(ITestOutputHelper output)
{
    // exp is the only libm-dependent step (a few ulps), so weights agree to 5e-6 relative.
    private const double WeightRelTol = 5e-6;

    private static void AssertWeightsClose(float[] cpu, float[] cuda, string what)
    {
        Assert.Equal(cpu.Length, cuda.Length);
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(Math.Abs(cpu[i] - cuda[i]) <= WeightRelTol * Math.Abs(cpu[i]) + 1e-9,
                $"{what} weight {i}: cpu {cpu[i]:R} cuda {cuda[i]:R}");
    }

    private static void AssertBitEqual(float[] cpu, float[] cuda, string what)
    {
        Assert.Equal(cpu.Length, cuda.Length);
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(cpu[i]) == BitConverter.SingleToInt32Bits(cuda[i]),
                $"{what} element {i}: cpu {cpu[i]:R} cuda {cuda[i]:R}");
    }

    private static (int[] Idx, float[] W) Route(IBackend be, float[] logits, in MoeRouteArgs args, float[]? bias = null,
        float[]? alt = null, int[]? kinds = null)
    {
        int tokens = logits.Length / args.NumExperts;
        using Tensor l = F32(logits, tokens, args.NumExperts);
        using Tensor idx = EmptyI32(tokens, args.TopK);
        using Tensor w = EmptyF32(tokens, args.TopK);
        using Tensor? b = bias is null ? null : F32(bias, args.NumExperts);
        using Tensor? a = alt is null ? null : F32(alt, args.NumExperts);
        using Tensor? k = kinds is null ? null : I32(kinds, tokens);
        be.MoeRoute(idx, w, l, args, b, a, k);
        return (ReadI32(idx), ReadF32(w));
    }

    public static IEnumerable<object[]> RouteCases()
    {
        yield return new object[] { "softmax", new MoeRouteArgs(8, 2, MoeRouteScoring.Softmax), 8, false };
        yield return new object[] { "softmax-renorm", new MoeRouteArgs(64, 6, MoeRouteScoring.Softmax, Renormalize: true), 64, false };
        yield return new object[]
        {
            "sigmoid-bias-8of4-groups",
            new MoeRouteArgs(32, 4, MoeRouteScoring.Sigmoid, 8, 4, 0f, true, 1e-20f, 2.5f), 32, true,
        };
        yield return new object[]
        {
            "sqrtsoftplus-bias-temp-scale",
            new MoeRouteArgs(256, 8, MoeRouteScoring.SqrtSoftplus, Renormalize: true, RenormEpsilon: 1e-20f, Scale: 1.5f,
                LogitDivisor: 1.7f), 256, true,
        };
        yield return new object[] { "single-expert", new MoeRouteArgs(1, 1, MoeRouteScoring.Softmax, Renormalize: true), 1, false };
        yield return new object[] { "k-equals-e", new MoeRouteArgs(16, 16, MoeRouteScoring.Sigmoid, Renormalize: true), 16, false };
        yield return new object[] { "max-experts", new MoeRouteArgs(1024, 8, MoeRouteScoring.Softmax, Renormalize: true), 1024, true };
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [MemberData(nameof(RouteCases))]
    public void MoeRoute_MatchesCpu(string name, MoeRouteArgs args, int experts, bool useBias)
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: no CUDA device"); return; }
        const int Tokens = 37;
        float[] logits = Random(Tokens * experts, experts * 7 + 1, 3f);
        float[] bias = Random(experts, experts + 2, 0.4f);
        float[] alt = Random(experts, experts + 3, 0.4f);
        int[] kinds = Enumerable.Range(0, Tokens).Select(i => i % 3 == 0 ? 1 : 0).ToArray();
        float[]? b = useBias ? bias : null;
        float[]? a = useBias ? alt : null;
        int[]? k = useBias ? kinds : null;

        (int[] cpuIdx, float[] cpuW) = Route(new CpuBackend(), logits, args, b, a, k);
        using CudaBackend cuda = new(0, PtxDir());
        (int[] gpuIdx, float[] gpuW) = Route(cuda, logits, args, b, a, k);

        output.WriteLine($"{name}: {Tokens} tokens compared");
        Assert.Equal(cpuIdx, gpuIdx);
        AssertWeightsClose(cpuW, gpuW, name);
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void MoeRoute_ExactTies_TakeTheLowestIndexLikeCpu()
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: no CUDA device"); return; }
        float[] logits = new float[5 * 40];
        MoeRouteArgs flat = new(40, 6, MoeRouteScoring.Softmax, Renormalize: true);
        float[] bias = new float[40];
        bias[7] = bias[31] = bias[33] = 1f;
        MoeRouteArgs biased = new(40, 4, MoeRouteScoring.Sigmoid);
        MoeRouteArgs grouped = new(40, 4, MoeRouteScoring.Sigmoid, 8, 2, 0f, true, 1e-20f);

        using CudaBackend cuda = new(0, PtxDir());
        CpuBackend cpu = new();
        foreach ((MoeRouteArgs args, float[]? bs) in new[] { (flat, (float[]?)null), (biased, bias), (grouped, bias) })
        {
            (int[] ci, float[] cw) = Route(cpu, logits, args, bs);
            (int[] gi, float[] gw) = Route(cuda, logits, args, bs);
            Assert.Equal(ci, gi);
            AssertWeightsClose(cw, gw, "tie");
        }
        (int[] first, _) = Route(cuda, logits, flat);
        Assert.Equal(new[] { 0, 1, 2, 3, 4, 5 }, first.Take(6));
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void MoeBuildDispatch_IsByteIdenticalToCpu_IncludingEmptyExpertsAndDroppedPairs()
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: no CUDA device"); return; }
        foreach ((int tokens, int experts, int k) in new[] { (1, 1, 1), (7, 5, 2), (4096, 64, 6), (300, 300, 3) })
        {
            Random rng = new(tokens + experts);
            int[] ids = new int[tokens * k];
            for (int i = 0; i < ids.Length; i++)
            {
                int v = rng.Next(-1, experts + 2);
                ids[i] = experts > 4 && v == 3 ? 0 : v;
            }
            (int[] Counts, int[] Offsets, int[] Perm, int[] Slot) cpu = Dispatch(new CpuBackend(), ids, tokens, k, experts);
            using CudaBackend cuda = new(0, PtxDir());
            (int[] Counts, int[] Offsets, int[] Perm, int[] Slot) gpu = Dispatch(cuda, ids, tokens, k, experts);
            Assert.Equal(cpu.Counts, gpu.Counts);
            Assert.Equal(cpu.Offsets, gpu.Offsets);
            Assert.Equal(cpu.Perm, gpu.Perm);
            Assert.Equal(cpu.Slot, gpu.Slot);
            if (experts > 4) Assert.Contains(0, cpu.Counts);
            output.WriteLine($"dispatch T={tokens} E={experts} k={k}: {cpu.Offsets[experts]} live pairs");
        }
    }

    private static (int[], int[], int[], int[]) Dispatch(IBackend be, int[] ids, int tokens, int k, int experts)
    {
        using Tensor topk = I32(ids, tokens, k);
        using Tensor counts = EmptyI32(experts), offsets = EmptyI32(experts + 1);
        using Tensor perm = EmptyI32(tokens * k), slot = EmptyI32(tokens * k);
        be.MoeBuildDispatch(counts, offsets, perm, slot, topk, experts);
        return (ReadI32(counts), ReadI32(offsets), ReadI32(perm), ReadI32(slot));
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MoeCombine_IsBitIdenticalToCpu(bool accumulate)
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: no CUDA device"); return; }
        const int Tokens = 129, K = 6, H = 300, Rows = Tokens * K;
        Random rng = new(5);
        int[] slots = Enumerable.Range(0, Tokens * K).Select(i => i % 11 == 0 ? -1 : rng.Next(0, Rows)).ToArray();
        float[] weights = Random(Tokens * K, 6, 2f), rows = Random(Rows * H, 7), seed = Random(Tokens * H, 8);

        float[] Run(IBackend be)
        {
            using Tensor x = F32(rows, Rows, H), w = F32(weights, Tokens, K), o = F32(seed, Tokens, H);
            using Tensor s = I32(slots, Tokens, K);
            be.MoeCombine(o, x, s, w, K, accumulate);
            return ReadF32(o);
        }

        float[] cpu = Run(new CpuBackend());
        using CudaBackend cuda = new(0, PtxDir());
        AssertBitEqual(cpu, Run(cuda), "combine");
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void MoeCombine_SkipsSlotsBeyondTheExpertRowsInsteadOfReadingOutOfBounds()
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: no CUDA device"); return; }
        using Tensor rows = F32(new float[] { 1, 2, 10, 20 }, 2, 2);
        using Tensor slots = I32(new[] { 1, 2, 999999, 0 }, 2, 2);
        using Tensor weights = F32(new float[] { 2, 3, 4, 5 }, 2, 2);
        using Tensor result = EmptyF32(2, 2);
        using CudaBackend cuda = new(0, PtxDir());
        cuda.MoeCombine(result, rows, slots, weights, 2, accumulate: false);
        Assert.Equal(new float[] { 20, 40, 5, 10 }, ReadF32(result));
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Route_Dispatch_Combine_Chain_Stays_On_Device_And_Matches_Cpu()
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: no CUDA device"); return; }
        const int Tokens = 200, E = 32, K = 4, H = 96;
        MoeRouteArgs args = new(E, K, MoeRouteScoring.SqrtSoftplus, Renormalize: true, RenormEpsilon: 1e-20f, Scale: 1.5f);
        float[] logits = Random(Tokens * E, 41, 2f), bias = Random(E, 42, 0.3f), x = Random(Tokens * H, 43);

        float[] Run(IBackend be)
        {
            using Tensor l = F32(logits, Tokens, E), b = F32(bias, E), tokens = F32(x, Tokens, H);
            using Tensor idx = EmptyI32(Tokens, K), w = EmptyF32(Tokens, K);
            using Tensor counts = EmptyI32(E), offsets = EmptyI32(E + 1), perm = EmptyI32(Tokens * K), slot = EmptyI32(Tokens * K);
            be.MoeRoute(idx, w, l, args, b);
            be.MoeBuildDispatch(counts, offsets, perm, slot, idx, E);
            using Tensor rows = EmptyF32(Tokens * K, H);
            be.RowGather(rows, tokens, perm, Tokens * K, H);
            using Tensor result = EmptyF32(Tokens, H);
            be.MoeCombine(result, rows, slot, w, K, accumulate: false);
            return ReadF32(result);
        }

        float[] cpu = Run(new CpuBackend());
        using CudaBackend cuda = new(0, PtxDir());
        float[] gpu = Run(cuda);
        double maxDiff = 0;
        for (int i = 0; i < cpu.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(cpu[i] - gpu[i]));
        output.WriteLine($"chain max |cpu - cuda| = {maxDiff:E3}");
        Assert.True(maxDiff <= 1e-5, $"route->dispatch->combine chain diverges by {maxDiff:E3}");
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(5, 300, 8, false, false)]
    [InlineData(3, 4096, 64, true, false)]
    [InlineData(2, 129280, 1024, false, true)]
    [InlineData(2, 5000, 2048, true, true)]
    [InlineData(4, 17, 17, false, false)]
    [InlineData(4, 9, 1, true, false)]
    public void TopKLastDim_MatchesCpu_Exactly(int rows, int n, int k, bool sortByIndex, bool heavyTies)
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: no CUDA device"); return; }
        Random rng = new(n + k);
        float[] data = new float[rows * n];
        for (int i = 0; i < data.Length; i++)
            data[i] = heavyTies ? rng.Next(0, 40) - 20f : (float)(rng.NextDouble() * 20 - 10);
        int[] lens = Enumerable.Range(0, rows).Select(r => r == 0 ? n : Math.Max(1, n - 1 - r * n / 7)).ToArray();

        foreach (bool useLens in new[] { false, true })
        {
            (int[] ci, float[] cv) = TopK(new CpuBackend(), data, rows, n, k, useLens ? lens : null, sortByIndex);
            using CudaBackend cuda = new(0, PtxDir());
            (int[] gi, float[] gv) = TopK(cuda, data, rows, n, k, useLens ? lens : null, sortByIndex);
            Assert.Equal(ci, gi);
            AssertBitEqual(cv, gv, "topk value");
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void TopKLastDim_HandlesNanNegativeZeroAndInfinities_LikeCpu()
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: no CUDA device"); return; }
        float nan = float.NaN, inf = float.PositiveInfinity;
        float[] row = { 1f, nan, -0f, 0f, -inf, inf, 2f, nan, 0f, -0f, 2f, -inf };
        float[] data = row.Concat(row.Reverse()).ToArray();
        foreach (int k in new[] { 1, 5, 12 })
        foreach (bool byIndex in new[] { false, true })
        {
            (int[] ci, float[] cv) = TopK(new CpuBackend(), data, 2, 12, k, null, byIndex);
            using CudaBackend cuda = new(0, PtxDir());
            (int[] gi, float[] gv) = TopK(cuda, data, 2, 12, k, null, byIndex);
            Assert.Equal(ci, gi);
            for (int i = 0; i < cv.Length; i++)
                Assert.True(cv[i].Equals(gv[i]) || (float.IsNaN(cv[i]) && float.IsNaN(gv[i])) || cv[i] == gv[i],
                    $"k={k} value {i}: cpu {cv[i]} cuda {gv[i]}");
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void TopKLastDim_Rejects_K_Beyond_The_Kernel_Limit()
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: no CUDA device"); return; }
        using CudaBackend cuda = new(0, PtxDir());
        using Tensor input = EmptyF32(1, 4096), values = EmptyF32(1, 2049), indices = EmptyI32(1, 2049);
        Assert.Throws<NotSupportedException>(() => cuda.TopKLastDim(values, indices, input, 2049));
    }

    private static (int[], float[]) TopK(IBackend be, float[] data, int rows, int n, int k, int[]? lens, bool byIndex)
    {
        using Tensor input = F32(data, rows, n);
        using Tensor values = EmptyF32(rows, k), indices = EmptyI32(rows, k);
        using Tensor? len = lens is null ? null : I32(lens, rows);
        be.TopKLastDim(values, indices, input, k, len, byIndex);
        return (ReadI32(indices), ReadF32(values));
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Softplus_MatchesCpu_IncludingInPlace()
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: no CUDA device"); return; }
        float[] x = Random(10007, 9, 40f).Concat(new[] { 0f, 20f, 20.001f, -100f, 88f }).ToArray();
        float[] Run(IBackend be, bool inPlace)
        {
            using Tensor input = F32(x, x.Length);
            if (inPlace) { be.Softplus(input, input); return ReadF32(input); }
            using Tensor o = EmptyF32(x.Length);
            be.Softplus(o, input);
            return ReadF32(o);
        }

        float[] cpu = Run(new CpuBackend(), false);
        using CudaBackend cuda = new(0, PtxDir());
        foreach (bool inPlace in new[] { false, true })
        {
            float[] gpu = Run(cuda, inPlace);
            for (int i = 0; i < cpu.Length; i++)
                Assert.True(Math.Abs(cpu[i] - gpu[i]) <= 1e-6 * Math.Abs(cpu[i]) + 1e-30, $"softplus {i}: x={x[i]} cpu {cpu[i]:R} cuda {gpu[i]:R}");
        }
    }
}
