using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>A BF16 attention whose score matrix would not fit runs the query-tiled F32 core on its F32 casts instead of
/// materializing every score (a BF16 AutoencoderKL mid-block at 2048² asked for one 16 GB allocation). The tiled route
/// must give what the materialized route gives: same casts and kernels, with only the GEMM tiling different.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("CudaSerial")]
public sealed unsafe class Bf16TiledSdpaTests
{
    private readonly ITestOutputHelper _output;

    public Bf16TiledSdpaTests(ITestOutputHelper output) => _output = output;

    /// <summary>The VAE mid-block's shape (one head, head dim = channels), shortened; a forced query tile that does not
    /// divide the sequence covers the ragged last tile.</summary>
    [Fact]
    public void Bf16TiledRoute_MatchesTheMaterializedRoute()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }
        const int seq = 1000, dim = 512;
        TensorShape shape = new TensorShape(1, 1, seq, dim);
        using Tensor q = Bf16Uniform(shape, seed: 1);
        using Tensor k = Bf16Uniform(shape, seed: 2);
        using Tensor v = Bf16Uniform(shape, seed: 3);
        // Sharper than 1/sqrt(dim), so the output follows individual V rows rather than averaging toward zero.
        const float scale = 0.25f;

        using CudaBackend cuda = new CudaBackend(0, PtxDir());
        long before = cuda.SdpaTiledExecutionCount;
        using Tensor materialized = Attend(cuda, q, k, v, scale);
        Assert.Equal(before, cuda.SdpaTiledExecutionCount);
        Tensor tiledBf16;
        using (KnobProfile.Create("bf16-tiled").With(EngineKnobs.SdpaForceTiled, true).With(EngineKnobs.SdpaTile, (int?)96).Push())
        {
            tiledBf16 = Attend(cuda, q, k, v, scale);
        }
        using Tensor tiled = tiledBf16;
        Assert.Equal(before + 1, cuda.SdpaTiledExecutionCount);

        float* m = (float*)materialized.DataPointer, t = (float*)tiled.DataPointer;
        float peak = 0f, worst = 0f;
        double sumDiff = 0;
        long differing = 0, outOfBound = 0;
        for (long i = 0; i < materialized.ElementCount; i++)
        {
            float diff = MathF.Abs(m[i] - t[i]);
            peak = MathF.Max(peak, MathF.Abs(m[i]));
            worst = MathF.Max(worst, diff);
            sumDiff += diff;
            if (diff > 0f) differing++;
            // One BF16 output ulp (at most 2^-7 of the value), plus one TF32 rounding of a softmax weight (2^-11 against
            // |V| <= 1): the two routes tile QK^T differently, so a weight near a TF32 boundary can round either way.
            if (diff > MathF.Abs(m[i]) / 128f + 1f / 2048f) outOfBound++;
        }
        double meanDiff = sumDiff / materialized.ElementCount;
        _output.WriteLine($"max|out| {peak:F4}, max|diff| {worst:E3}, differing {differing} of {materialized.ElementCount}, "
            + $"mean|diff| {meanDiff:E3}, out of bound {outOfBound}");
        Assert.True(peak > 0.05f, $"the attention output is near zero (max {peak}), so agreement would prove nothing");
        Assert.Equal(0, outOfBound);
        Assert.True(meanDiff < 1e-6, $"mean |materialized - tiled| is {meanDiff:E3}: a systematic difference, not rounding");
    }

    /// <summary>Runs one BF16 attention and returns its output widened to F32 on the host.</summary>
    private static Tensor Attend(CudaBackend cuda, Tensor q, Tensor k, Tensor v, float scale)
    {
        using Tensor output = new Tensor(q.Shape, DType.BF16);
        ((IBackend)cuda).ScaledDotProductAttention(output, q, k, v, null, scale);
        return output.CastTo(DType.F32);
    }

    private static Tensor Bf16Uniform(TensorShape shape, int seed)
    {
        using Tensor f32 = new Tensor(shape, DType.F32);
        float* p = (float*)f32.DataPointer;
        Random rng = new Random(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return f32.CastTo(DType.BF16);
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir)
            ? dir
            : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }
}
