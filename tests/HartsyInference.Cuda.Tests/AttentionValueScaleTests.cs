using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Attention is linear in V, and on every row an unmasked F32 call can take a power-of-two scale of V passes through
/// exactly: the output for V·2⁻⁸, scaled back by 2⁸, is the output for V bit for bit while no value or partial product is
/// subnormal. So a caller's power-of-two V damp, as Lens carried for F16's range, changes nothing on these rows.</summary>
[Collection("CudaSerial")]
public sealed unsafe class AttentionValueScaleTests
{
    private readonly ITestOutputHelper _output;
    public AttentionValueScaleTests(ITestOutputHelper output) => _output = output;

    private const int Heads = 4, Tokens = 300, HeadDim = 64;
    private const float Damp = 1f / 256f;

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir) ? dir : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private static Tensor Filled(Func<long, float> value)
    {
        Tensor t = new(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = value(i);
        return t;
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("S.14", false, false)]   // materialized, TF32
    [InlineData("S.14", false, true)]    // materialized, FP32 (numerics.sdpaNoTf32)
    [InlineData("S.9", false, false)]    // query-tiled
    [InlineData("S.16", true, false)]    // the BF16 cast row
    public void APowerOfTwoValueScale_PassesThroughTheRowExactly(string route, bool bf16Grant, bool exact)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        if (bf16Grant)
        {
            CudnnRuntime.EnsureProbed();
            if (!CudnnRuntime.SupportsSdpa) { _output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}"); return; }
        }
        float scale = 1f / MathF.Sqrt(HeadDim);
        Random rng = new(7);
        using Tensor q = Filled(_ => (float)(rng.NextDouble() * 2 - 1)), k = Filled(_ => (float)(rng.NextDouble() * 2 - 1));
        // Every 13th element sits past F16's range, as a few of Lens's late-block V columns do.
        using Tensor v = Filled(i => (float)(rng.NextDouble() * 2 - 1) * (i % 13 == 0 ? 1.4e5f : 1f));
        using Tensor damped = Filled(i => ((float*)v.DataPointer)[i] * Damp);

        if (exact) KnobStore.Set(EngineKnobs.SdpaNoTf32, true);
        if (route == "S.9") KnobStore.Set(EngineKnobs.SdpaForceTiled, true);
        KnobStore.Set(EngineKnobs.SdpaCudnn, true);
        try
        {
            using CudaBackend cuda = new(0, PtxDir());
            float[] Run(Tensor value)
            {
                using Tensor o = new(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32);
                AttentionCall call = AttentionCall.FromScaledDotProductAttention(o, q, k, value, null, scale, allowF16: false);
                ((IBackend)cuda).Attention(bf16Grant ? call with { Numerics = AttentionNumerics.Bf16Storage } : call);
                cuda.Sync();
                Assert.Equal(route, CudaBackend.LastAttentionRoute);
                return AttnInvoker.ReadF32(o);
            }

            float[] plain = Run(v), viaDamp = Run(damped);
            using Tensor expected = new(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32);
            AttentionHostReference.Run(AttentionCall.FromScaledDotProductAttention(expected, q, k, v, null, scale, false));
            // A row that ignored V would pass the comparison below vacuously.
            double rel = AttnAssert.Measure(plain, AttnInvoker.ReadF32(expected), HeadDim).RelL2;
            Assert.True(rel <= 8e-3, $"the output is {rel:E2} from the contract reference");
            int differ = 0;
            for (int i = 0; i < plain.Length; i++)
                if (BitConverter.SingleToUInt32Bits(plain[i]) != BitConverter.SingleToUInt32Bits(viaDamp[i] / Damp)) differ++;
            _output.WriteLine($"{route}{(exact ? " exact" : "")}: {differ} of {plain.Length} outputs differ; relL2 {rel:E2}");
            Assert.True(differ == 0, $"{differ} of {plain.Length} outputs of {route} moved when V was scaled by 2^-8 and the output by 2^8");
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SdpaNoTf32);
            KnobStore.Clear(EngineKnobs.SdpaForceTiled);
            KnobStore.Clear(EngineKnobs.SdpaCudnn);
        }
    }
}
