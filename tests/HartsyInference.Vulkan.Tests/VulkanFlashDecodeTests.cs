using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Flash-decoding for one query row (the decode step) on the RTX-class Vulkan device: the split kernel pair against
/// the CPU reference, against the scalar sdpa_flash kernel, and the device-position path against the eager path.</summary>
/// <remarks>Tolerance: every operand is F32. The split pair sums keys in a different order than the scalar kernel, so the
/// bound is an absolute <see cref="Tolerance"/> on outputs whose magnitude is at most about 1 (V is uniform in [-1, 1]).
/// The worst error seen is written to the test output. The eager and device paths run the same arithmetic, so they are
/// compared exactly, not within a tolerance.</remarks>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanFlashDecodeTests(ITestOutputHelper output)
{
    private const float Tolerance = 1e-4f;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    /// <summary>The five models: (query heads, KV heads, head dim). GQA ratios 4, 7, 2, 1 and 4.</summary>
    public static IEnumerable<object[]> ModelCases()
    {
        (int Hq, int Hkv, int D)[] shapes = [(32, 8, 64), (14, 2, 64), (16, 8, 64), (16, 16, 128), (32, 8, 128)];
        int[] contexts = [1, 17, 750, 4096, 8192];
        int[] windows = [0, 256];
        foreach ((int hq, int hkv, int d) in shapes)
            foreach (int ctx in contexts)
                foreach (int window in windows)
                    yield return new object[] { hq, hkv, d, ctx, window };
    }

    /// <summary>Q is one row; K/V are allocated at capacity (ctx + 37) with garbage past ctx, as a KV cache is.</summary>
    private sealed class DecodeInputs
    {
        public required float[] Q { get; init; }
        public required float[] KCap { get; init; }
        public required float[] VCap { get; init; }
        public required float[] KCompact { get; init; }
        public required float[] VCompact { get; init; }
    }

    private static DecodeInputs MakeInputs(int hq, int hkv, int d, int ctx, int capacity, int seed)
    {
        float[] q = VulkanBackendSmokeTests.FillRandom(hq * d, seed);
        float[] kCap = VulkanBackendSmokeTests.FillRandom(hkv * capacity * d, seed + 1);
        float[] vCap = VulkanBackendSmokeTests.FillRandom(hkv * capacity * d, seed + 2);
        float[] kC = new float[hkv * ctx * d];
        float[] vC = new float[hkv * ctx * d];
        for (int h = 0; h < hkv; h++)
        {
            Array.Copy(kCap, h * capacity * d, kC, h * ctx * d, ctx * d);
            Array.Copy(vCap, h * capacity * d, vC, h * ctx * d, ctx * d);
        }
        return new DecodeInputs { Q = q, KCap = kCap, VCap = vCap, KCompact = kC, VCompact = vC };
    }

    private static Tensor Tensor4(int a, int b, int c, int d, float[] data)
    {
        Tensor t = new(new TensorShape(a, b, c, d), DType.F32);
        data.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static float MaxAbsDiff(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float worst = 0f;
        for (int i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        return worst;
    }

    /// <summary>Eager flash-decoding (split pair) against the CPU reference, at every model shape, context length and window.</summary>
    [Theory]
    [MemberData(nameof(ModelCases))]
    public void Eager_MatchesCpuReference(int hq, int hkv, int d, int ctx, int window)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        backend.EnableFlashDecodeSplit = true;
        int capacity = ctx + 37;
        float scale = 1f / MathF.Sqrt(d);
        DecodeInputs inp = MakeInputs(hq, hkv, d, ctx, capacity, seed: 100 + ctx + window);
        float[] expected = VulkanBackendSmokeTests.CpuFlashReference(inp.Q, inp.KCompact, inp.VCompact, null,
            1, hq, hkv, 1, ctx, d, scale, causal: true, qOffset: ctx - 1, slidingWindow: window);

        using Tensor qT = Tensor4(1, hq, 1, d, inp.Q);
        using Tensor kT = Tensor4(1, hkv, capacity, d, inp.KCap);
        using Tensor vT = Tensor4(1, hkv, capacity, d, inp.VCap);
        using Tensor oT = new(new TensorShape(1, hq, 1, d), DType.F32);
        backend.FlashAttention(oT, qT, kT, vT, kvLen: ctx, kvGroup: hq / hkv, causal: true, qOffset: ctx - 1, scale,
            softcap: 0f, sink: null, slidingWindow: window, alibiSlopes: null);

        float err = MaxAbsDiff(oT.AsReadOnlySpan<float>(), expected);
        output.WriteLine($"hq={hq} hkv={hkv} d={d} ctx={ctx} window={window}: max |split - cpu| = {err:E2}");
        Assert.True(err < Tolerance, $"max abs error {err} exceeds {Tolerance}");
    }

    /// <summary>The split pair against the scalar sdpa_flash kernel on the same device and inputs (the A/B the switch provides).</summary>
    [Theory]
    [MemberData(nameof(ModelCases))]
    public void Eager_MatchesScalarKernel(int hq, int hkv, int d, int ctx, int window)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        int capacity = ctx + 37;
        float scale = 1f / MathF.Sqrt(d);
        DecodeInputs inp = MakeInputs(hq, hkv, d, ctx, capacity, seed: 300 + ctx + window);
        using Tensor qT = Tensor4(1, hq, 1, d, inp.Q);
        using Tensor kT = Tensor4(1, hkv, capacity, d, inp.KCap);
        using Tensor vT = Tensor4(1, hkv, capacity, d, inp.VCap);
        using Tensor split = new(new TensorShape(1, hq, 1, d), DType.F32);
        using Tensor scalar = new(new TensorShape(1, hq, 1, d), DType.F32);

        backend.EnableFlashDecodeSplit = true;
        backend.FlashAttention(split, qT, kT, vT, kvLen: ctx, kvGroup: hq / hkv, causal: true, qOffset: ctx - 1, scale, 0f, null, window, null);
        backend.EnableFlashDecodeSplit = false;
        backend.FlashAttention(scalar, qT, kT, vT, kvLen: ctx, kvGroup: hq / hkv, causal: true, qOffset: ctx - 1, scale, 0f, null, window, null);
        backend.EnableFlashDecodeSplit = true;

        float err = MaxAbsDiff(split.AsReadOnlySpan<float>(), scalar.AsReadOnlySpan<float>());
        output.WriteLine($"hq={hq} hkv={hkv} d={d} ctx={ctx} window={window}: max |split - scalar| = {err:E2}");
        Assert.True(err < Tolerance, $"max abs error {err} exceeds {Tolerance}");
    }

    /// <summary>Graph decode's device-position path runs the same arithmetic as eager: exactly equal, placeholders ignored.</summary>
    [Theory]
    [MemberData(nameof(ModelCases))]
    public void Device_IsBitIdenticalToEager(int hq, int hkv, int d, int ctx, int window)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        backend.EnableFlashDecodeSplit = true;
        int capacity = ctx + 37;
        float scale = 1f / MathF.Sqrt(d);
        DecodeInputs inp = MakeInputs(hq, hkv, d, ctx, capacity, seed: 500 + ctx + window);
        using Tensor qT = Tensor4(1, hq, 1, d, inp.Q);
        using Tensor kT = Tensor4(1, hkv, capacity, d, inp.KCap);
        using Tensor vT = Tensor4(1, hkv, capacity, d, inp.VCap);
        using Tensor eager = new(new TensorShape(1, hq, 1, d), DType.F32);
        using Tensor dev = new(new TensorShape(1, hq, 1, d), DType.F32);

        backend.FlashAttention(eager, qT, kT, vT, kvLen: ctx, kvGroup: hq / hkv, causal: true, qOffset: ctx - 1, scale, 0f, null, window, null);
        ulong pos = backend.AllocDevicePos();
        try
        {
            backend.WriteDevicePos(pos, kvLen: ctx, qOffset: ctx - 1);
            // Deliberately wrong host placeholders: devicePos, not these arguments, must drive the kernel.
            backend.FlashAttentionDev(dev, qT, kT, vT, kvLen: 0, kvGroup: hq / hkv, causal: true, qOffset: 0, scale, pos, 0f, window);
        }
        finally { backend.FreeDevicePos(pos); }

        Assert.True(eager.AsReadOnlySpan<float>().SequenceEqual(dev.AsReadOnlySpan<float>()),
            $"device-position output differs from eager at hq={hq} hkv={hkv} d={d} ctx={ctx} window={window}");
    }

    /// <summary>A causal decode row whose KV prefix runs past the query: keys above the query position are masked.</summary>
    [Theory]
    [InlineData(32, 8, 64, 750, 500)]
    [InlineData(16, 16, 128, 4096, 1000)]
    public void Eager_CausalMasksKeysAboveTheQuery(int hq, int hkv, int d, int kvLen, int qOffset)
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        backend.EnableFlashDecodeSplit = true;
        int capacity = kvLen + 37;
        float scale = 1f / MathF.Sqrt(d);
        DecodeInputs inp = MakeInputs(hq, hkv, d, kvLen, capacity, seed: 700 + qOffset);
        float[] expected = VulkanBackendSmokeTests.CpuFlashReference(inp.Q, inp.KCompact, inp.VCompact, null,
            1, hq, hkv, 1, kvLen, d, scale, causal: true, qOffset: qOffset, slidingWindow: 0);
        using Tensor qT = Tensor4(1, hq, 1, d, inp.Q);
        using Tensor kT = Tensor4(1, hkv, capacity, d, inp.KCap);
        using Tensor vT = Tensor4(1, hkv, capacity, d, inp.VCap);
        using Tensor oT = new(new TensorShape(1, hq, 1, d), DType.F32);
        backend.FlashAttention(oT, qT, kT, vT, kvLen: kvLen, kvGroup: hq / hkv, causal: true, qOffset: qOffset, scale, 0f, null, 0, null);
        float err = MaxAbsDiff(oT.AsReadOnlySpan<float>(), expected);
        output.WriteLine($"causal kvLen={kvLen} qOffset={qOffset}: max |split - cpu| = {err:E2}");
        Assert.True(err < Tolerance, $"max abs error {err} exceeds {Tolerance}");
    }

    /// <summary>A query length above one is prefill: it keeps the scalar kernel, so the switch changes nothing.</summary>
    [Fact]
    public void PrefillRows_AreBitIdenticalWithTheSwitchOn()
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        const int hq = 14, hkv = 2, d = 64, sq = 7, kvLen = 100, qOffset = 93;
        float scale = 1f / MathF.Sqrt(d);
        float[] q = VulkanBackendSmokeTests.FillRandom(hq * sq * d, 901);
        float[] k = VulkanBackendSmokeTests.FillRandom(hkv * (kvLen + 5) * d, 902);
        float[] v = VulkanBackendSmokeTests.FillRandom(hkv * (kvLen + 5) * d, 903);
        using Tensor qT = Tensor4(1, hq, sq, d, q);
        using Tensor kT = Tensor4(1, hkv, kvLen + 5, d, k);
        using Tensor vT = Tensor4(1, hkv, kvLen + 5, d, v);
        using Tensor on = new(new TensorShape(1, hq, sq, d), DType.F32);
        using Tensor off = new(new TensorShape(1, hq, sq, d), DType.F32);

        backend.EnableFlashDecodeSplit = true;
        backend.FlashAttention(on, qT, kT, vT, kvLen, hq / hkv, causal: true, qOffset, scale, 0f, null, 0, null);
        backend.EnableFlashDecodeSplit = false;
        backend.FlashAttention(off, qT, kT, vT, kvLen, hq / hkv, causal: true, qOffset, scale, 0f, null, 0, null);

        Assert.True(on.AsReadOnlySpan<float>().SequenceEqual(off.AsReadOnlySpan<float>()));
    }

    /// <summary>A head dim the subgroup does not divide keeps the scalar kernel, and its result still matches the reference.</summary>
    [Fact]
    public void UnsupportedHeadDim_FallsBackToTheScalarKernel()
    {
        if (!VulkanAvailable()) { output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = VulkanTestDevice.Create();
        backend.EnableFlashDecodeSplit = true;
        const int hq = 8, hkv = 4, d = 48, ctx = 300, capacity = ctx + 37;
        float scale = 1f / MathF.Sqrt(d);
        DecodeInputs inp = MakeInputs(hq, hkv, d, ctx, capacity, seed: 911);
        float[] expected = VulkanBackendSmokeTests.CpuFlashReference(inp.Q, inp.KCompact, inp.VCompact, null,
            1, hq, hkv, 1, ctx, d, scale, causal: true, qOffset: ctx - 1, slidingWindow: 0);
        using Tensor qT = Tensor4(1, hq, 1, d, inp.Q);
        using Tensor kT = Tensor4(1, hkv, capacity, d, inp.KCap);
        using Tensor vT = Tensor4(1, hkv, capacity, d, inp.VCap);
        using Tensor oT = new(new TensorShape(1, hq, 1, d), DType.F32);
        backend.FlashAttention(oT, qT, kT, vT, kvLen: ctx, kvGroup: hq / hkv, causal: true, qOffset: ctx - 1, scale, 0f, null, 0, null);
        float err = MaxAbsDiff(oT.AsReadOnlySpan<float>(), expected);
        Assert.True(err < Tolerance, $"max abs error {err} exceeds {Tolerance}");
    }
}

/// <summary>The split-count geometry is pure host arithmetic; these invariants hold for every cache size and device.</summary>
public sealed class VulkanFlashDecodeGeometryTests
{
    [Theory]
    [InlineData(879, 8, 1, 28)]      // the bench shape: 750 prompt + 128 new, Llama-class KV heads, RTX 3060
    [InlineData(879, 2, 1, 28)]      // Qwen2.5-class: two KV heads
    [InlineData(8229, 8, 1, 28)]     // 8192 context
    [InlineData(16384, 16, 1, 0)]    // unknown compute units: the fallback applies
    public void Geometry_CoversTheCacheWithoutEmptyTrailingSplits(int capacity, int kvHeads, int batch, uint units)
    {
        (int chunk, int splits) = VulkanBackend.FlashDecodeGeometry(capacity, kvHeads, batch, units);
        Assert.True(chunk >= 1 && splits >= 1);
        Assert.True((long)splits * chunk >= capacity, "the splits must cover the whole cache");
        Assert.True((long)(splits - 1) * chunk < capacity, "the last split must not be empty");
        Assert.True(splits <= VulkanBackend.FlashDecodeMaxSplits);
        Assert.True(chunk <= Math.Max(VulkanBackend.FlashDecodeMaxChunk, (capacity + VulkanBackend.FlashDecodeMaxSplits - 1) / VulkanBackend.FlashDecodeMaxSplits));
    }

    [Fact]
    public void Geometry_BenchShapeOnTheRtx3060()
    {
        // 28 SMs x 16 workgroups per SM over 8 KV heads wants 56 splits; 879 keys at 16 per split is 55 splits.
        Assert.Equal((16, 55), VulkanBackend.FlashDecodeGeometry(879, 8, 1, 28));
    }

    [Fact]
    public void Geometry_IsAPureFunctionOfTheCapacityAndDevice()
    {
        Assert.Equal(VulkanBackend.FlashDecodeGeometry(4096, 2, 1, 28), VulkanBackend.FlashDecodeGeometry(4096, 2, 1, 28));
        Assert.True(VulkanBackend.FlashDecodeGeometry(4096, 2, 1, 128).Splits >= VulkanBackend.FlashDecodeGeometry(4096, 2, 1, 28).Splits);
    }
}
