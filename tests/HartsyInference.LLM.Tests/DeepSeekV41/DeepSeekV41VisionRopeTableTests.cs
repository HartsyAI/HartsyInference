using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;
using Xunit.Abstractions;
using VisionFixture = HartsyInference.LLM.Tests.DeepSeekV41.DeepSeekV41VisionFixture;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The vision tower's 2D rotary tables and the half-split rotation they feed, against upstream <c>get_vision_cos_sin</c> and <c>apply_rotary</c>.</summary>
public sealed class DeepSeekV41VisionRopeTableTests(ITestOutputHelper output)
{
    // float32 angles from different pow/sin/cos implementations: a few ulps, well under a visible rotation error
    private const float TableTolerance = 2e-6f;

    [Theory]
    [InlineData(5, 7, 8)]
    [InlineData(3, 5, 32)]
    [InlineData(4, 1, 8)]
    public void Table_MatchesUpstreamAndRepeatsTheRotatedHalf(int gridHeight, int gridWidth, int ropeDim)
    {
        JsonElement c = VisionFixture.RopeCase(gridHeight, gridWidth, ropeDim);
        float[] cos = VisionFixture.Floats(c.GetProperty("cos")), sin = VisionFixture.Floats(c.GetProperty("sin"));

        DeepSeekV41VisionRopeTable table = DeepSeekV41VisionRopeTable.Build(gridHeight, gridWidth, 2 * ropeDim, c.GetProperty("theta").GetDouble());

        Assert.Equal(gridHeight * gridWidth, table.Patches);
        Assert.Equal(table.Patches * 2 * ropeDim, table.Cos.Length);
        double worst = 0;
        for (int patch = 0; patch < table.Patches; patch++)
            for (int j = 0; j < ropeDim; j++)
            {
                int row = patch * 2 * ropeDim;
                double cosDiff = Math.Abs(table.Cos[row + j] - cos[patch * ropeDim + j]);
                double sinDiff = Math.Abs(table.Sin[row + j] - sin[patch * ropeDim + j]);
                worst = Math.Max(worst, Math.Max(cosDiff, sinDiff));
                Assert.Equal(table.Cos[row + j], table.Cos[row + ropeDim + j]);
                Assert.Equal(table.Sin[row + j], table.Sin[row + ropeDim + j]);
            }
        output.WriteLine($"grid {gridHeight}x{gridWidth} ropeDim {ropeDim}: max |table - upstream| {worst:E2}");
        Assert.True(worst <= TableTolerance, $"table differs from upstream by {worst:E2}");
    }

    [Fact]
    public void Table_PutsTheRowAnglesFirstAndTheColumnAnglesSecond()
    {
        // head width 8: rotated half 4, so two frequencies per axis; patch (row 2, column 3) of a 4 x 5 grid
        DeepSeekV41VisionRopeTable table = DeepSeekV41VisionRopeTable.Build(4, 5, 8, 10000.0);

        int row = (2 * 5 + 3) * 8;
        float[] invFreq = [1f, 1f / MathF.Pow(10000f, 0.5f)];
        for (int j = 0; j < 2; j++)
        {
            Assert.Equal(MathF.Cos(2 * invFreq[j]), table.Cos[row + j], 5);
            Assert.Equal(MathF.Cos(3 * invFreq[j]), table.Cos[row + 2 + j], 5);
            Assert.Equal(MathF.Sin(2 * invFreq[j]), table.Sin[row + j], 5);
            Assert.Equal(MathF.Sin(3 * invFreq[j]), table.Sin[row + 2 + j], 5);
        }
        // the first patch is not rotated at all
        Assert.All(table.Cos.Take(8), static v => Assert.Equal(1f, v));
        Assert.All(table.Sin.Take(8), static v => Assert.Equal(0f, v));
    }

    [Theory]
    [InlineData(5, 7, 8)]
    [InlineData(3, 5, 32)]
    [InlineData(4, 1, 8)]
    public void HalfSplitRotation_MatchesUpstreamApplyRotary(int gridHeight, int gridWidth, int ropeDim)
    {
        JsonElement c = VisionFixture.RopeCase(gridHeight, gridWidth, ropeDim);
        int heads = c.GetProperty("heads").GetInt32(), headDim = 2 * ropeDim, patches = gridHeight * gridWidth;
        DeepSeekV41VisionRopeTable table = DeepSeekV41VisionRopeTable.Build(gridHeight, gridWidth, headDim, c.GetProperty("theta").GetDouble());
        using CpuBackend cpu = new();
        IBackend backend = cpu;
        using Tensor x = DeepSeekV41HostMath.Tensor(VisionFixture.Floats(c.GetProperty("x")), 1, patches, heads, headDim);
        using Tensor cos = DeepSeekV41HostMath.Tensor(table.Cos, 1, patches, headDim);
        using Tensor sin = DeepSeekV41HostMath.Tensor(table.Sin, 1, patches, headDim);

        backend.ApplyRopeSingle(x, cos, sin);

        VisionFixture.AssertClose(output, $"rotated {gridHeight}x{gridWidth} ropeDim {ropeDim}", x.AsReadOnlySpan<float>().ToArray(),
            VisionFixture.Floats(c.GetProperty("rotated")));
    }

    [Theory]
    [InlineData(0, 1, 8, 10000.0)]
    [InlineData(1, 0, 8, 10000.0)]
    [InlineData(2, 2, 6, 10000.0)]
    [InlineData(2, 2, 0, 10000.0)]
    [InlineData(2, 2, 8, 0.0)]
    [InlineData(2, 2, 8, double.NaN)]
    public void Build_RejectsAnUnusableGridWidthOrBase(int gridHeight, int gridWidth, int headDim, double theta) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => DeepSeekV41VisionRopeTable.Build(gridHeight, gridWidth, headDim, theta));
}
