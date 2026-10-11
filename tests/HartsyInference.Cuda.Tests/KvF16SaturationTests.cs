using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Every write into an F16 KV cache saturates at the F16 maximum (a NaN stays a NaN), and the F16 default resolves per backend and model type.
/// An F16 cache is the default for text models on CUDA, so one extreme key or value must not become infinity and poison every later softmax.</summary>
[Collection("CudaSerial")]
public sealed unsafe class KvF16SaturationTests
{
    private readonly ITestOutputHelper _output;
    public KvF16SaturationTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    private static Tensor Filled(Func<long, float> value, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = value(i);
        return t;
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void EagerAppend_SaturatesOutOfRangeValues_AndKeepsNaNAndNormalValues()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDir());
        IBackend b = backend;
        float[] src = [1e6f, -1e6f, float.NaN, 70000f, 65504f, 65519f, 100.5f, -3.25f];   // 1 head, 1 token, headDim 8
        float[] want = [65504f, -65504f, float.NaN, 65504f, 65504f, 65504f, 100.5f, -3.25f];
        using Tensor buffer = new(new TensorShape(1, 1, 4, 8), DType.F16);
        b.ResidentAllocateKv(buffer);
        using Tensor step = Filled(i => src[i], 1, 1, 1, 8);
        b.KvCacheAppend(buffer, step, offset: 1);
        backend.Sync();
        Half* p = (Half*)buffer.DataPointer;
        for (int i = 0; i < 8; i++)
        {
            float got = (float)p[1 * 8 + i];
            if (float.IsNaN(want[i])) Assert.True(float.IsNaN(got), $"element {i}: a NaN must stay a NaN, got {got}");
            else Assert.Equal(want[i], got);
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void GraphScatter_AndFullWidthQkNormScatter_NeverStoreInfinity()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int hq = 8, hkv = 2, d = 128, maxSeq = 32, pos = 5;
        using CudaBackend cuda = new(0, PtxDir());
        ulong posBuf = CudaMemory.Allocate(2 * sizeof(int));
        int* hostPos = stackalloc int[2] { pos + 1, pos };
        CudaMemory.CopyHostToDevice(posBuf, hostPos, 2 * sizeof(int));
        try
        {
            using Tensor cos = Filled(_ => 0.8f, maxSeq, d), sin = Filled(_ => 0.6f, maxSeq, d);
            using Tensor q = Filled(i => 0.25f, 1, hq, 1, d);
            using Tensor kHuge = Filled(i => i % 2 == 0 ? 3e5f : -3e5f, 1, hkv, 1, d);
            using Tensor vHuge = Filled(i => i % 3 == 0 ? 1e6f : -1e6f, 1, hkv, 1, d);

            // Plain rope + scatter: K is rotated (3e5 * 0.8 + 3e5 * 0.6 is beyond the F16 range), V is copied.
            using Tensor kc = new(new TensorShape(1, hkv, maxSeq, d), DType.F16), vc = new(new TensorShape(1, hkv, maxSeq, d), DType.F16);
            using Tensor qo = new(new TensorShape(1, hq, 1, d), DType.F32);
            cuda.RopeScatterKvDecodeStep(qo, kc, vc, q, kHuge, vHuge, cos, sin, hq, hkv, d, d, false, posBuf);
            cuda.Sync();
            AssertSaturatedFinite((Half*)kc.DataPointer, hkv, maxSeq, d, pos, "rope scatter K", expectClamp: true);
            AssertSaturatedFinite((Half*)vc.DataPointer, hkv, maxSeq, d, pos, "rope scatter V", expectClamp: true);

            // Full-width QK-norm + rope + scatter (the fused OLMoE kernel): a huge norm weight pushes the normalised K far past the F16 range.
            using Tensor qk = Filled(i => (i % 5) - 2f, 1, 1, (hq + hkv) * d);
            using Tensor v = Filled(i => 2e5f, 1, hkv, 1, d);
            using Tensor qNorm = Filled(_ => 1f, hq * d), kNorm = Filled(_ => 1e6f, hkv * d);
            using Tensor kc2 = new(new TensorShape(1, hkv, maxSeq, d), DType.F16), vc2 = new(new TensorShape(1, hkv, maxSeq, d), DType.F16);
            using Tensor qo2 = new(new TensorShape(1, hq, 1, d), DType.F32);
            cuda.QkNormFullRopeScatterDecodeStep(qo2, kc2, vc2, qk, v, qNorm, kNorm, 1e-6f, cos, sin, hq, hkv, d, d, false, posBuf);
            cuda.Sync();
            AssertSaturatedFinite((Half*)kc2.DataPointer, hkv, maxSeq, d, pos, "fused QK-norm scatter K", expectClamp: true);
            AssertSaturatedFinite((Half*)vc2.DataPointer, hkv, maxSeq, d, pos, "fused QK-norm scatter V", expectClamp: true);
        }
        finally { CudaMemory.Free(posBuf); }
    }

    private void AssertSaturatedFinite(Half* cache, int heads, int maxSeq, int d, int pos, string what, bool expectClamp)
    {
        bool clamped = false;
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < d; i++)
            {
                float f = (float)cache[((long)h * maxSeq + pos) * d + i];
                Assert.False(float.IsInfinity(f) || float.IsNaN(f), $"{what}: head {h} dim {i} stored {f}");
                if (MathF.Abs(f) == 65504f) clamped = true;
            }
        if (expectClamp) Assert.True(clamped, $"{what}: the test input is past the F16 range, so some stored value must be at the limit");
        _output.WriteLine($"{what}: finite, clamped at the F16 maximum");
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void Fa2Prefill_WithKeysPastTheF16Range_StaysFinite()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        // A key stride larger than kvLen keeps the call off the cuDNN SDPA path (which wants a tight cache) and on the FA2 kernel, whose lm_kv_to_f16
        // and tile packing round the F32 keys to F16. Without saturation a key of 2e5 becomes infinity there and the whole output NaN.
        const int hq = 8, hkv = 2, d = 128, tq = 64, keyStride = 128;
        float scale = 1f / MathF.Sqrt(d);
        using CudaBackend cuda = new(0, PtxDir());
        using Tensor q = Filled(i => 0.01f * ((i % 7) - 3), 1, hq, tq, d);
        using Tensor k = Filled(i => (i % 11 == 0 ? 2e5f : 1f) * (((i / 3) % 2 == 0) ? 1f : -1f), 1, hkv, keyStride, d);
        using Tensor v = Filled(i => ((i % 13) - 6) * 0.1f, 1, hkv, keyStride, d);
        using Tensor o = new(new TensorShape(1, hq, tq, d), DType.F32);
        cuda.FlashAttention(o, q, k, v, tq, hq / hkv, true, 0, scale, 0f, null, 0, null);
        cuda.Sync();
        float* p = (float*)o.DataPointer;
        for (long i = 0; i < o.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"attention output {i} is {p[i]}");
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void TextKvF16Default_IsOnForCuda_AndOffWhenPinned()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend cuda = new(0, PtxDir());
        try
        {
            KnobStore.Clear(EngineKnobs.TextKvF16);
            KnobStore.Clear(EngineKnobs.KvF16);
            Assert.True(cuda.SupportsF16KvCache);
            Assert.True(KvCaches.F16For(cuda));
            Assert.False(KvCaches.F16Enabled, "the automatic text default must not force F16 onto audio decoders");
            KnobStore.Set(EngineKnobs.TextKvF16, false);
            Assert.False(KvCaches.F16For(cuda), "the reference profile's pin must win over the CUDA default");
        }
        finally { KnobStore.Clear(EngineKnobs.TextKvF16); KnobStore.Clear(EngineKnobs.KvF16); }
    }
}
