using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Pins two query-tile heights on one attention call and compares the outputs. The tile only changes which rows a GEMM computes at once, so
/// with plain FP32 GEMMs the two outputs must agree to rounding. A gap of 1e-3 or more there is a defect in the tiled path. Under TF32 the GEMM
/// rounding is larger, and a gap that appears only there is recorded as a TF32 effect.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("SdpaKnobs")]
public sealed class SdpaTileHeightPinTests
{
    private readonly ITestOutputHelper _output;

    public SdpaTileHeightPinTests(ITestOutputHelper output) => _output = output;

    // The production width of the LongCat avatar DiT (32 heads of 128) at about 640x368 of video tokens, and a tiny shape for the same checks.
    [Theory]
    [InlineData("production", 32, 128, 4600, 256, 1024)]
    [InlineData("production", 32, 128, 4600, 1024, 4600)]
    [InlineData("tiny", 4, 32, 300, 37, 128)]
    public void TileHeight_MovesTheOutputOnlyByRounding_WithExactGemms(string shape, int heads, int headDim, int tokens, int heightA, int heightB)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        double exact = PinnedGap(heads, headDim, tokens, heightA, heightB, exactGemms: true);
        double tf32 = PinnedGap(heads, headDim, tokens, heightA, heightB, exactGemms: false);
        _output.WriteLine($"{shape} heads={heads} d={headDim} tokens={tokens} tile {heightA} vs {heightB}: exact relative L2 {exact:E3}, TF32 relative L2 {tf32:E3}");
        Assert.True(exact < 1e-4, $"{shape}: exact-GEMM gap {exact:E3} between tile {heightA} and {heightB}");
    }

    // The same pair with an additive attention bias ([1, 1, tokens, tokens], shared by the heads), as conditioning-aware attention passes one.
    [Theory]
    [InlineData(464, 503)]
    public void TileHeight_WithBias_MovesTheOutputOnlyByRounding_WithExactGemms(int heightA, int heightB)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        double exact = PinnedGap(32, 128, 4600, heightA, heightB, exactGemms: true, biased: true);
        double tf32 = PinnedGap(32, 128, 4600, heightA, heightB, exactGemms: false, biased: true);
        _output.WriteLine($"biased production heads=32 d=128 tokens=4600 tile {heightA} vs {heightB}: exact relative L2 {exact:E3}, TF32 relative L2 {tf32:E3}");
        Assert.True(exact < 1e-4, $"biased: exact-GEMM gap {exact:E3} between tile {heightA} and {heightB}");
    }

    // Scores grow with the product of Q and K, so a gain on both sets how peaked the softmax is; a trained attention is far peakier than the
    // uniform [-0.5, 0.5] inputs above, and TF32 rounding in QK^T scales with the score magnitude.
    [Theory]
    [InlineData(8f, 464, 503)]
    public void TileHeight_PeakedScores_WithExactGemms(float gain, int heightA, int heightB)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        double exact = PinnedGap(32, 128, 4600, heightA, heightB, exactGemms: true, gain: gain);
        double tf32 = PinnedGap(32, 128, 4600, heightA, heightB, exactGemms: false, gain: gain);
        _output.WriteLine($"peaked gain={gain} production heads=32 d=128 tokens=4600 tile {heightA} vs {heightB}: exact relative L2 {exact:E3}, TF32 relative L2 {tf32:E3}");
        Assert.True(exact < 1e-4, $"peaked: exact-GEMM gap {exact:E3} between tile {heightA} and {heightB}");
    }

    private double PinnedGap(int heads, int headDim, int tokens, int heightA, int heightB, bool exactGemms, bool biased = false, float gain = 1f)
    {
        float[] a = RunAt(heads, headDim, tokens, heightA, exactGemms, biased, gain);
        float[] b = RunAt(heads, headDim, tokens, heightB, exactGemms, biased, gain);
        double error = 0, reference = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            error += d * d;
            reference += b[i] * (double)b[i];
        }
        return Math.Sqrt(error / reference);
    }

    private static float[] RunAt(int heads, int headDim, int tokens, int tileRows, bool exactGemms, bool biased, float gain = 1f)
    {
        KnobStore.Set(EngineKnobs.SdpaForceTiled, true);
        KnobStore.Set(EngineKnobs.SdpaTile, (int?)tileRows);
        if (exactGemms)
        {
            KnobStore.Set(EngineKnobs.HighPrecisionGemm, true);
            KnobStore.Set(EngineKnobs.SdpaNoTf32, true);
        }
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            Random rng = new(5);
            using Tensor q = Random(rng, heads, tokens, headDim, gain), k = Random(rng, heads, tokens, headDim, gain), v = Random(rng, heads, tokens, headDim);
            using Tensor output = new(new TensorShape(1, heads, tokens, headDim), DType.F32);
            Tensor? bias = null;
            if (biased)
            {
                bias = new Tensor(new TensorShape(1, 1, tokens, tokens), DType.F32);
                Span<float> values = bias.AsSpan<float>();
                for (int i = 0; i < values.Length; i++) values[i] = (float)(rng.NextDouble() - 0.5) * 2f;
            }
            try
            {
                backend.ScaledDotProductAttention(output, q, k, v, bias, 1f / MathF.Sqrt(headDim));
            }
            finally
            {
                bias?.Dispose();
            }
            backend.OffloadActivation(output);
            return output.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.SdpaForceTiled);
            KnobStore.Clear(EngineKnobs.SdpaTile);
            KnobStore.Clear(EngineKnobs.HighPrecisionGemm);
            KnobStore.Clear(EngineKnobs.SdpaNoTf32);
        }
    }

    private static Tensor Random(Random rng, int heads, int tokens, int headDim, float gain = 1f)
    {
        Tensor t = new(new TensorShape(1, heads, tokens, headDim), DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() - 0.5) * gain;
        return t;
    }

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }

        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx not found above the test binary.");
    }
}
