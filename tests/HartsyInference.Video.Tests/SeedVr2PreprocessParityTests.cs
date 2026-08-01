using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Video.Pipelines;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Video.Tests;

/// <summary>Parity for the SeedVR2 preprocessing chain vs ByteDance's own transforms (Part A2 gate).
/// Run order: (1) <c>&lt;seedvr2-venv&gt; tests/python-reference/seedvr2_reference/dump_seedvr2_preprocess_reference.py
/// &lt;SeedVR-checkout&gt; &lt;out.safetensors&gt;</c>, (2) this test with <c>SEEDVR2_PRE_REF</c> pointing at the dump.
/// Crop/normalize/pad must be exact; the antialiased-bicubic resize is float-tolerance (maxAbs ≤ 1e-5 fp32).
/// Skips cleanly when the env var is unset.</summary>
public sealed class SeedVr2PreprocessParityTests
{
    private const long TargetArea = 1280L * 720L;
    private static readonly string[] Cases =
        ["up_360p_t5", "up_240p_t1", "up_odd_t7", "down_1080p_t5", "tiny_t3", "nondiv_t9", "exact_720p_t5"];

    private readonly ITestOutputHelper _output;

    public SeedVr2PreprocessParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Preprocess_MatchesReference_AllCases()
    {
        string? refPath = Environment.GetEnvironmentVariable("SEEDVR2_PRE_REF");
        if (string.IsNullOrWhiteSpace(refPath) || !File.Exists(refPath))
        {
            _output.WriteLine("SKIPPED: set SEEDVR2_PRE_REF to the preprocess reference dump.");
            return;
        }

        using SafeTensorsLoader loader = new();
        loader.Load(refPath);

        double worst = 0;
        foreach (string name in Cases)
        {
            Tensor input = loader.GetTensor($"{name}.input");     // (T,3,H,W) u8
            Tensor expected = loader.GetTensor($"{name}.output"); // (3,T',H',W') f32
            int t = (int)input.Shape[0], h = (int)input.Shape[2], w = (int)input.Shape[3];

            List<byte[]> frames = new List<byte[]>(t);
            ReadOnlySpan<byte> u8 = input.AsSpan<byte>();
            for (int f = 0; f < t; f++)
            {
                byte[] rgb = new byte[h * w * 3];
                for (int c = 0; c < 3; c++)
                {
                    int plane = (f * 3 + c) * h * w;
                    for (int i = 0; i < h * w; i++)
                        rgb[i * 3 + c] = u8[plane + i];
                }
                frames.Add(rgb);
            }

            SeedVr2Preprocess.Result actual = SeedVr2Preprocess.Run(frames, w, h, TargetArea);

            Assert.True(expected.Shape[1] == actual.Frames && expected.Shape[2] == actual.Height
                && expected.Shape[3] == actual.Width,
                $"{name}: shape (3,{actual.Frames},{actual.Height},{actual.Width}) vs reference " +
                $"(3,{expected.Shape[1]},{expected.Shape[2]},{expected.Shape[3]})");

            ReadOnlySpan<float> exp = expected.AsSpan<float>();
            double maxAbs = 0;
            for (int i = 0; i < exp.Length; i++)
            {
                double d = Math.Abs(exp[i] - actual.Data[i]);
                if (d > maxAbs)
                    maxAbs = d;
            }
            _output.WriteLine($"{name}: out (3,{actual.Frames},{actual.Height},{actual.Width}) maxAbs {maxAbs:e2}");
            worst = Math.Max(worst, maxAbs);
            Assert.True(maxAbs <= 1e-5, $"{name}: maxAbs {maxAbs:e2} exceeds 1e-5");
        }
        _output.WriteLine($"All {Cases.Length} cases within tolerance; worst maxAbs {worst:e2}.");
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 5)]
    [InlineData(4, 5)]
    [InlineData(5, 5)]
    [InlineData(6, 9)]
    [InlineData(7, 9)]
    [InlineData(9, 9)]
    [InlineData(24, 25)]
    [InlineData(25, 25)]
    public void PaddedFrameCount_MatchesCutVideos(int frames, int expected)
        => Assert.Equal(expected, SeedVr2Preprocess.PaddedFrameCount(frames));
}
