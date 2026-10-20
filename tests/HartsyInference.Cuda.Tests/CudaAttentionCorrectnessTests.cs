using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>CUDA attention defects pinned by cases that fail without their fixes: per-batch key-mask rows on the tiled path,
/// saturating F16 casts and mask adds on the materialized path, device positions past the cache, and the launch guards.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudaAttentionCorrectnessTests
{
    private readonly ITestOutputHelper _output;
    public CudaAttentionCorrectnessTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir) ? dir : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private static Tensor Make(DType dtype, long[] dims, Func<long, float> value)
    {
        Tensor t = new(new TensorShape(dims), dtype);
        for (long i = 0; i < t.ElementCount; i++)
        {
            float x = value(i);
            if (dtype == DType.F32) ((float*)t.DataPointer)[i] = x;
            else ((Half*)t.DataPointer)[i] = (Half)x;
        }
        return t;
    }

    private static Func<long, float> Noise(int seed, float amplitude)
    {
        Random rng = new(seed);
        return _ => (float)((rng.NextDouble() - 0.5) * 2.0 * amplitude);
    }

    /// <summary>The contract reference on <paramref name="call"/>, written into a fresh F32 output of the call's shape.</summary>
    private static float[] Reference(Func<Tensor, AttentionCall> call, long[] outputDims)
    {
        using Tensor expected = new(new TensorShape(outputDims), DType.F32);
        AttentionHostReference.Run(call(expected));
        return AttnInvoker.ReadF32(expected);
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(2, 1)]   // [B,1,1,Skv]: the shape the old (b*H+h) % B row got wrong
    [InlineData(1, 3)]   // [1,H,1,Skv]
    [InlineData(2, 3)]   // [B,H,1,Skv]
    public void TiledKeyMask_EachBatchItemAndHeadReadsItsOwnRow(int maskBatch, int maskHeads)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int b = 2, h = 3, sq = 8, skv = 32, d = 64;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Make(DType.F32, [b, h, sq, d], Noise(1, 1f));
        using Tensor k = Make(DType.F32, [b, h, skv, d], Noise(2, 1f));
        using Tensor v = Make(DType.F32, [b, h, skv, d], Noise(3, 1f));
        // Each mask row hides a different half of the keys, so a row read for the wrong batch item or head is visible.
        using Tensor mask = Make(DType.F32, [maskBatch, maskHeads, 1, skv], i => (i % skv + i / skv * 7) % skv < skv / 2 ? -1e4f : 0f);
        KnobStore.Set(EngineKnobs.SdpaCudnn, false);
        KnobStore.Set(EngineKnobs.SdpaForceTiled, true);
        try
        {
            using CudaBackend cuda = new(0, PtxDir());
            using Tensor o = new(new TensorShape(b, h, sq, d), DType.F32);
            cuda.ScaledDotProductAttention(o, q, k, v, mask, scale, allowF16: false);
            cuda.Sync();
            Assert.Equal("S.10", CudaBackend.LastAttentionRoute);
            float[] expected = Reference(e => AttentionCall.FromScaledDotProductAttention(e, q, k, v, mask, scale, false), [b, h, sq, d]);
            (double rel, double cos, double max) = AttnAssert.Agrees(AttnInvoker.ReadF32(o), expected, d,
                AttnTolerance.For(AttnAchieved.Tf32), "on the tiled path with a per-batch key mask");
            _output.WriteLine($"tiled key mask [{maskBatch},{maskHeads},1,{skv}]: relL2 {rel:E2}, min cos {cos:F7}, max |Δ| {max:E2}");
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SdpaCudnn);
            KnobStore.Clear(EngineKnobs.SdpaForceTiled);
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void MaterializedF16Cast_SaturatesInsteadOfOverflowing()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int h = 2, s = 16, d = 64;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Make(DType.F32, [1, h, s, d], Noise(4, 1f));
        using Tensor k = Make(DType.F32, [1, h, s, d], Noise(5, 1f));
        Func<long, float> vNoise = Noise(6, 1f);
        const long hot = 3 * d + 5;   // head 0, key 3, dim 5: past F16's 65504
        using Tensor v = Make(DType.F32, [1, h, s, d], i => i == hot ? 1e5f : vNoise(i));
        using Tensor vClamped = Make(DType.F32, [1, h, s, d], i => Math.Clamp(((float*)v.DataPointer)[i], -65504f, 65504f));
        KnobStore.Set(EngineKnobs.SdpaCudnn, false);
        try
        {
            using CudaBackend cuda = new(0, PtxDir());
            using Tensor o = new(new TensorShape(1, h, s, d), DType.F32);
            cuda.ScaledDotProductAttention(o, q, k, v, null, scale, allowF16: true);
            cuda.Sync();
            Assert.Equal("S.13", CudaBackend.LastAttentionRoute);
            float[] actual = AttnInvoker.ReadF32(o);
            Assert.All(actual, x => Assert.True(float.IsFinite(x), "a V value past the F16 range must clip, not become infinity"));
            float[] expected = Reference(e => AttentionCall.FromScaledDotProductAttention(e, q, k, vClamped, null, scale, false), [1, h, s, d]);
            (double rel, _, _) = AttnAssert.Agrees(actual, expected, d, AttnTolerance.For(AttnAchieved.F16), "against V clipped to the F16 range");
            _output.WriteLine($"materialized F16 cast saturates: relL2 {rel:E2} vs V clipped at 65504");
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SdpaCudnn);
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void NativeF16Mask_FullyMaskedRowStaysFinite()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int h = 2, sq = 4, skv = 16, d = 64, hidden = 2;
        float scale = 1f / MathF.Sqrt(d);
        Func<long, float> qNoise = Noise(7, 0.5f), kNoise = Noise(8, 0.05f);
        // The hidden row scores about -32 against every key, so the old -65504 clamp plus the score left F16's range.
        using Tensor q = Make(DType.F16, [1, h, sq, d], i => (i / d) % sq == hidden ? 2f : qNoise(i));
        using Tensor k = Make(DType.F16, [1, h, skv, d], i => -2f + kNoise(i));
        using Tensor v = Make(DType.F16, [1, h, skv, d], Noise(9, 1f));
        using Tensor mask = Make(DType.F32, [1, 1, sq, skv], i => i / skv == hidden ? -1e30f : 0f);
        KnobStore.Set(EngineKnobs.SdpaCudnn, false);
        try
        {
            using CudaBackend cuda = new(0, PtxDir());
            using Tensor o = new(new TensorShape(1, h, sq, d), DType.F16);
            cuda.ScaledDotProductAttention(o, q, k, v, mask, scale, allowF16: true);
            cuda.Sync();
            Assert.Equal("S.15", CudaBackend.LastAttentionRoute);
            float[] actual = AttnInvoker.ReadF32(o), values = AttnInvoker.ReadF32(v);
            float[] expected = Reference(e => AttentionCall.FromScaledDotProductAttention(e, q, k, v, mask, scale, true), [1, h, sq, d]);
            for (int head = 0; head < h; head++)
                for (int row = 0; row < sq; row++)
                {
                    int at = (head * sq + row) * d;
                    if (row == hidden)
                    {
                        // A fully masked row keeps the finite-sentinel convention: some average of the values, never NaN.
                        for (int c = 0; c < d; c++)
                        {
                            float lo = float.MaxValue, hi = float.MinValue;
                            for (int j = 0; j < skv; j++) { float x = values[(head * skv + j) * d + c]; lo = Math.Min(lo, x); hi = Math.Max(hi, x); }
                            float y = actual[at + c];
                            Assert.True(float.IsFinite(y) && y >= lo - 1e-2f && y <= hi + 1e-2f,
                                $"head {head} masked row dim {c}: {y} is not an average of the values [{lo}, {hi}]");
                        }
                        continue;
                    }
                    AttnAssert.Agrees(actual[at..(at + d)], expected[at..(at + d)], d, AttnTolerance.For(AttnAchieved.F16),
                        $"on unmasked row {row} of head {head}");
                }
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SdpaCudnn);
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    // Capacities whose split chunks overshoot the cache (3 x 67 > 200, 3 x 17 > 50), so the last chunk reaches past it.
    [InlineData("V.2", 200, false, true)]   // grouped-query decode kernel
    [InlineData("V.2", 200, true, true)]
    [InlineData("V.3", 50, false, false)]   // split-K
    [InlineData("V.5", 24, false, false)]   // monolithic, below the split floor
    [InlineData("V.4", 24, true, false)]
    public void DevicePositionPastTheCache_ReadsNoFurtherThanTheCache(string route, int capacity, bool f16Kv, bool gqa)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int hq = 4, hkv = 2, d = 64;
        float scale = 1f / MathF.Sqrt(d);
        DType kvType = f16Kv ? DType.F16 : DType.F32;
        using Tensor q = Make(DType.F32, [1, hq, 1, d], Noise(10, 1f));
        using Tensor k = Make(kvType, [1, hkv, capacity, d], Noise(11, 1f));
        using Tensor v = Make(kvType, [1, hkv, capacity, d], Noise(12, 1f));
        KnobStore.Set(EngineKnobs.FlashDecodeGqa, gqa);
        using CudaBackend cuda = new(0, PtxDir());
        ulong pos = cuda.AllocDevicePos();
        try
        {
            // A position 16 keys past the capacity: the old kernels read the next head's rows (or past the allocation).
            cuda.WriteDevicePos(pos, capacity + 16, capacity + 15);
            using Tensor o = new(new TensorShape(1, hq, 1, d), DType.F32);
            cuda.FlashAttentionDev(o, q, k, v, 0, hq / hkv, causal: true, 0, scale, pos);
            cuda.Sync();
            Assert.Equal(route, CudaBackend.LastAttentionRoute);
            float[] expected = Reference(e => AttentionCall.FromFlashAttention(e, q, k, v, capacity, true, capacity - 1, scale, 0f, null, 0, null),
                [1, hq, 1, d]);
            (double rel, _, _) = AttnAssert.Agrees(AttnInvoker.ReadF32(o), expected, d, AttnTolerance.For(AttnAchieved.F32),
                "with a device position past the cache");
            _output.WriteLine($"{route} capacity {capacity} f16Kv {f16Kv}: relL2 {rel:E2} against the whole cache");
        }
        finally
        {
            cuda.FreeDevicePos(pos);
            KnobStore.Clear(EngineKnobs.FlashDecodeGqa);
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("append", false, 16)]
    [InlineData("append", true, 16)]
    [InlineData("append2", false, -1)]   // two new rows at -1: the unsigned slot used to wrap to row 0
    [InlineData("append2", true, -1)]
    [InlineData("qkv", false, 16)]
    [InlineData("qkv", true, 16)]
    [InlineData("qknorm", false, 16)]
    [InlineData("qknorm", true, 16)]
    [InlineData("qknormfull", false, 16)]
    [InlineData("qknormfull", true, 16)]
    [InlineData("qknormfull-composed", false, 16)]
    public void DevicePositionOutsideTheCache_WritesNoKv(string op, bool f16Cache, int slot)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int hq = 4, hkv = 2, d = 64, maxSeq = 16;
        DType cacheType = f16Cache ? DType.F16 : DType.F32;
        int elem = f16Cache ? 2 : 4;
        // Head h's slot maxSeq is head h+1's row 0, so a write past the capacity shows up in the next head.
        using Tensor kc = Make(cacheType, [1, hkv, maxSeq, d], i => 1000f + i % 997);
        using Tensor vc = Make(cacheType, [1, hkv, maxSeq, d], i => -1000f - i % 991);
        byte[] kBefore = new ReadOnlySpan<byte>((void*)kc.DataPointer, (int)kc.ElementCount * elem).ToArray();
        byte[] vBefore = new ReadOnlySpan<byte>((void*)vc.DataPointer, (int)vc.ElementCount * elem).ToArray();
        using Tensor cos = Make(DType.F32, [maxSeq + 8, d], Noise(13, 1f)), sin = Make(DType.F32, [maxSeq + 8, d], Noise(14, 1f));
        using Tensor qkv = Make(DType.F32, [1, 1, (hq + 2 * hkv) * d], Noise(15, 1f));
        using Tensor qk = Make(DType.F32, [1, 1, (hq + hkv) * d], Noise(16, 1f));
        using Tensor vIn = Make(DType.F32, [1, hkv, 1, d], Noise(17, 1f)), vIn2 = Make(DType.F32, [1, hkv, 2, d], Noise(18, 1f));
        using Tensor qNorm = Make(DType.F32, [d], _ => 1f), kNorm = Make(DType.F32, [d], _ => 1f);
        using Tensor qNormFull = Make(DType.F32, [hq * d], _ => 1f), kNormFull = Make(DType.F32, [hkv * d], _ => 1f);
        KnobStore.Set(EngineKnobs.QknormFullScatter, op == "qknormfull");
        using CudaBackend cuda = new(0, PtxDir());
        ulong pos = cuda.AllocDevicePos();
        try
        {
            cuda.WriteDevicePos(pos, slot + 1, slot);
            using Tensor qOut = new(new TensorShape(1, hq, 1, d), DType.F32);
            switch (op)
            {
                case "append":
                    cuda.KvCacheAppendDev(kc, vIn, 0, pos);
                    break;
                case "append2":
                    cuda.KvCacheAppendDev(kc, vIn2, 0, pos);
                    break;
                case "qkv":
                    cuda.QkvRopeScatterDecodeStep(qOut, kc, vc, qkv, cos, sin, hq, hkv, d, d, false, pos);
                    break;
                case "qknorm":
                    cuda.QkvNormRopeScatterDecodeStep(qOut, kc, vc, qkv, qNorm, kNorm, 1e-6f, cos, sin, hq, hkv, d, d, false, pos);
                    break;
                default:
                    cuda.QkNormFullRopeScatterDecodeStep(qOut, kc, vc, qk, vIn, qNormFull, kNormFull, 1e-6f, cos, sin, hq, hkv, d, d, false, pos);
                    break;
            }
            cuda.Sync();
            Assert.Equal(kBefore, new ReadOnlySpan<byte>((void*)kc.DataPointer, kBefore.Length).ToArray());
            Assert.Equal(vBefore, new ReadOnlySpan<byte>((void*)vc.DataPointer, vBefore.Length).ToArray());
            _output.WriteLine($"{op} f16 {f16Cache} slot {slot}: both caches untouched");
        }
        finally
        {
            cuda.FreeDevicePos(pos);
            KnobStore.Clear(EngineKnobs.QknormFullScatter);
        }
    }

    [Fact]
    public void FlashF16RowsAligned_RefusesRowsOffSixteenBytes()
    {
        CudaKernels.FlashStrides headMajor = new(4 * 256 * 256, 256 * 256, 256), odd = new(4 * 256 * 260, 256 * 260, 260);
        Assert.True(CudaKernels.FlashF16RowsAligned(0x1000, 0x2000, 0x3000, 0x4000, headMajor, headMajor, headMajor, headMajor));
        Assert.False(CudaKernels.FlashF16RowsAligned(0x1000, 0x2008, 0x3000, 0x4000, headMajor, headMajor, headMajor, headMajor));
        Assert.False(CudaKernels.FlashF16RowsAligned(0x1000, 0x2000, 0x3000, 0x4000, headMajor, headMajor, odd, headMajor));
        Assert.True(CudaKernels.FlashF16RowsAligned(0x1000, 0x2000, 0x3000, 0x4000, new(0, 256, 24 * 256), new(0, 256, 24 * 256),
            new(0, 256, 24 * 256), new(0, 256, 24 * 256)));
    }

    [Fact]
    public void GridLimits_AreAdmissions()
    {
        Assert.True(CudaKernels.GridYzFits(65_535, 65_535));
        Assert.False(CudaKernels.GridYzFits(65_536, 1));
        Assert.False(CudaKernels.GridYzFits(1, 65_536));
        Assert.True(CudaKernels.KvToF16GridFits(524_280, 128));
        Assert.False(CudaKernels.KvToF16GridFits(524_288, 128));
    }
}
