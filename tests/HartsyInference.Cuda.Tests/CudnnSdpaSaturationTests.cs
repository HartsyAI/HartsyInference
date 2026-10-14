using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The cuDNN fused SDPA casts an F32 Q/K/V to F16 before the graph runs. That cast saturates at +-65504 (as the F16 KV store does, #383),
/// so one extreme value must not become an infinity that the fused softmax spreads over every row, and a NaN must stay a NaN.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudnnSdpaSaturationTests
{
    private readonly ITestOutputHelper _output;
    public CudnnSdpaSaturationTests(ITestOutputHelper output) => _output = output;

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

    private static bool CudnnAvailable(ITestOutputHelper output)
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: CUDA unavailable"); return false; }
        CudnnRuntime.EnsureProbed();
        if (!CudnnRuntime.SupportsSdpa) { output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}"); return false; }
        return true;
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void CudnnSdpa_ValuesPastTheF16Range_StayFinite()
    {
        if (!CudnnAvailable(_output)) return;
        const int heads = 2, s = 128, d = 128;
        float scale = 1f / MathF.Sqrt(d);
        KnobStore.Set(EngineKnobs.SdpaCudnn, true);
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            IBackend b = backend;
            // Q and K are ordinary; V carries values far past F16's 65504 (1e6 and -3e5), so an unsaturated cast makes them infinite.
            using Tensor q = Filled(i => 0.01f * ((i % 7) - 3), 1, heads, s, d);
            using Tensor k = Filled(i => 0.01f * ((i % 5) - 2), 1, heads, s, d);
            using Tensor v = Filled(i => i % 9 == 0 ? 1e6f : (i % 9 == 4 ? -3e5f : 0.5f), 1, heads, s, d);
            using Tensor o = new(new TensorShape(1, heads, s, d), DType.F32);
            b.ScaledDotProductAttention(o, q, k, v, null, scale, allowF16: true);
            backend.Sync();
            Assert.True(backend.CudnnSdpaEngaged, "cuDNN SDPA did not engage, so this test would not exercise the saturating cast");
            float* op = (float*)o.DataPointer;
            for (long i = 0; i < o.ElementCount; i++)
                Assert.True(float.IsFinite(op[i]), $"cuDNN SDPA output {i} is {op[i]}: a V value past the F16 range became non-finite");
        }
        finally { KnobStore.Clear(EngineKnobs.SdpaCudnn); }
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void CudnnSdpa_NaNInValue_StaysNaN()
    {
        if (!CudnnAvailable(_output)) return;
        const int heads = 1, s = 128, d = 128;
        float scale = 1f / MathF.Sqrt(d);
        KnobStore.Set(EngineKnobs.SdpaCudnn, true);
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            IBackend b = backend;
            using Tensor q = Filled(i => 0.01f * ((i % 7) - 3), 1, heads, s, d);
            using Tensor k = Filled(i => 0.01f * ((i % 5) - 2), 1, heads, s, d);
            using Tensor v = Filled(i => i == 3 * d + 7 ? float.NaN : 0.5f, 1, heads, s, d);   // one NaN in V at key row 3
            using Tensor o = new(new TensorShape(1, heads, s, d), DType.F32);
            b.ScaledDotProductAttention(o, q, k, v, null, scale, allowF16: true);
            backend.Sync();
            Assert.True(backend.CudnnSdpaEngaged, "cuDNN SDPA did not engage, so this test would not exercise the saturating cast");
            // The NaN must reach the output. A clamp built on max/min would have replaced it with -65504 and hidden it.
            float* op = (float*)o.DataPointer;
            int nan = 0;
            for (long i = 0; i < o.ElementCount; i++) if (float.IsNaN(op[i])) nan++;
            _output.WriteLine($"NaN outputs: {nan} of {o.ElementCount}");
            Assert.True(nan > 0, "a NaN in V did not propagate to the cuDNN SDPA output: the cast replaced it with a finite value");
        }
        finally { KnobStore.Clear(EngineKnobs.SdpaCudnn); }
    }
}
