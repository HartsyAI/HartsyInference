using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary><c>numerics.sdpaNoTf32</c> on every F32 SDPA row whose GEMMs it reaches: by default the row agrees with the oracle
/// only at TF32's bar, and with the knob the same row meets F32's.</summary>
[Collection("CudaSerial")]
public sealed unsafe class SdpaExactKnobTests
{
    private readonly ITestOutputHelper _output;
    public SdpaExactKnobTests(ITestOutputHelper output) => _output = output;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir) ? dir : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private static Tensor Uniform(long[] dims, int seed)
    {
        Random rng = new(seed);
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("materialized", "S.14")]
    [InlineData("materialized-perhead", "S.14")]
    [InlineData("tiled", "S.9")]
    [InlineData("tiled-key", "S.10")]
    public void TheExactKnob_RunsTheRowsF32GemmsInFp32(string name, string route)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int h = 4, s = 256;
        int d = name.EndsWith("perhead", StringComparison.Ordinal) || name.EndsWith("key", StringComparison.Ordinal) ? 64 : 128;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Uniform([1, h, s, d], 1), k = Uniform([1, h, s, d], 2), v = Uniform([1, h, s, d], 3);
        using Tensor? mask = name switch
        {
            "materialized-perhead" => Uniform([1, h, s, s], 4),
            "tiled-key" => Uniform([1, 1, 1, s], 4),
            _ => null,
        };
        using Tensor expectedT = new(new TensorShape(1, h, s, d), DType.F32);
        AttentionHostReference.Run(AttentionCall.FromScaledDotProductAttention(expectedT, q, k, v, mask, scale, false));
        float[] expected = AttnInvoker.ReadF32(expectedT);
        int smMajor = 0;

        float[] Run(bool exact)
        {
            if (exact) KnobStore.Set(EngineKnobs.SdpaNoTf32, true);
            if (name.StartsWith("tiled", StringComparison.Ordinal)) KnobStore.Set(EngineKnobs.SdpaForceTiled, true);
            try
            {
                using CudaBackend cuda = new(0, PtxDir());
                using Tensor o = new(new TensorShape(1, h, s, d), DType.F32);
                cuda.ScaledDotProductAttention(o, q, k, v, mask, scale, allowF16: false);
                cuda.Sync();
                Assert.Equal(route, CudaBackend.LastAttentionRoute);
                smMajor = cuda.Context.ComputeCapabilityMajor;
                return AttnInvoker.ReadF32(o);
            }
            finally
            {
                KnobStore.Clear(EngineKnobs.SdpaNoTf32);
                KnobStore.Clear(EngineKnobs.SdpaForceTiled);
            }
        }

        float[] tf32 = Run(exact: false), fp32 = Run(exact: true);
        (double relExact, _, double maxExact) = AttnAssert.Agrees(fp32, expected, d, AttnTolerance.For(AttnAchieved.F32),
            $"on {route} with the exact knob");
        (double relDefault, _, double maxDefault) = AttnAssert.Agrees(tf32, expected, d, AttnTolerance.For(AttnAchieved.Tf32),
            $"on {route} by default");
        _output.WriteLine($"{name} ({route}): default relL2 {relDefault:E2} max {maxDefault:E2}; exact relL2 {relExact:E2} max {maxExact:E2}");
        if (smMajor >= 8)
            Assert.True(relDefault > AttnTolerance.For(AttnAchieved.F32).RelL2,
                $"the default arm met F32's bar on {route} (relL2 {relDefault:E2}), so it did not run TF32 and the knob proves nothing");
    }
}
