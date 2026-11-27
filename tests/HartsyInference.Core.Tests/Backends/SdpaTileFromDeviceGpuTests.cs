using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>The A/B for <c>numerics.tileFromDevice</c> on one attention call at production width (32 heads of 128, 4600 tokens). The free-memory reading
/// is forced to two values. With the knob on and a planned resident size, the gate path must not move with the reading, so the outputs are bit-identical.
/// With the knob off, the gate reads free memory and the path can move (tiled against materialized), which shows up as a rounding-level difference.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("SdpaKnobs")]
public sealed class SdpaTileFromDeviceGpuTests
{
    private const int Heads = 32, HeadDim = 128, Tokens = 4600;
    private const ulong Gib = 1UL << 30;
    private readonly ITestOutputHelper _output;

    public SdpaTileFromDeviceGpuTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void KnobOn_TheSameResidentPlan_GivesTheSamePathAtTwoFreeReadings()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        // 10 GiB planned resident on a 12 GiB card leaves a 1 GiB budget: the 2.7 GiB score matrix is tiled at both readings.
        float[] tight = Run(knobOn: true, freeBytes: 2 * Gib, plannedResident: 10 * Gib);
        float[] loose = Run(knobOn: true, freeBytes: 20 * Gib, plannedResident: 10 * Gib);
        double gap = RelativeL2(tight, loose);
        _output.WriteLine($"knob on: relative L2 between free readings 2 GiB and 20 GiB: {gap:E3}");
        Assert.True(tight.AsSpan().SequenceEqual(loose), $"knob on: outputs differ across free readings (relative L2 {gap:E3})");
    }

    [Fact]
    public void KnobOff_TheFreeReadingCanMoveThePath_ByRoundingOnly()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        float[] tight = Run(knobOn: false, freeBytes: 2 * Gib, plannedResident: 0);
        float[] loose = Run(knobOn: false, freeBytes: 20 * Gib, plannedResident: 0);
        double gap = RelativeL2(tight, loose);
        _output.WriteLine($"knob off: relative L2 between free readings 2 GiB and 20 GiB: {gap:E3}");
        Assert.True(gap < 1e-4, $"knob off: the path change is larger than rounding: relative L2 {gap:E3}");
    }

    private static float[] Run(bool knobOn, ulong freeBytes, ulong plannedResident)
    {
        if (knobOn) KnobStore.Set(EngineKnobs.TileFromDevice, true);
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            backend.TestFreeVramOverride = () => (nuint)freeBytes;
            backend.NoteAttentionPlannedResident((long)plannedResident);
            Random rng = new(5);
            using Tensor q = Random(rng), k = Random(rng), v = Random(rng);
            using Tensor output = new(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32);
            backend.ScaledDotProductAttention(output, q, k, v, null, 1f / MathF.Sqrt(HeadDim));
            backend.OffloadActivation(output);
            return output.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.TileFromDevice);
        }
    }

    private static Tensor Random(Random rng)
    {
        Tensor t = new(new TensorShape(1, Heads, Tokens, HeadDim), DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    private static double RelativeL2(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        double error = 0, reference = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            error += d * d;
            reference += expected[i] * (double)expected[i];
        }

        return Math.Sqrt(error / reference);
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
