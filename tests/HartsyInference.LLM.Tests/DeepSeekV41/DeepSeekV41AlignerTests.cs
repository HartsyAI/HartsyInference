using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;
using Xunit.Abstractions;
using VisionFixture = HartsyInference.LLM.Tests.DeepSeekV41.DeepSeekV41VisionFixture;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The aligner against the unmodified upstream <c>Aligner</c>: the zero-padded 3 x 3 fold, the hidden activation and the output, over grids that pad on one side, both sides or neither.</summary>
public sealed class DeepSeekV41AlignerTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(2, 5)]
    [InlineData(7, 2)]
    public void EveryStage_MatchesTheUpstreamAligner(int gridHeight, int gridWidth)
    {
        JsonElement testCase = VisionFixture.AlignerCase(gridHeight, gridWidth);
        using CpuBackend backend = new();
        using DeepSeekV41Aligner aligner = new(backend, VisionFixture.Config(), VisionFixture.OutputDim, VisionFixture.AlignerWeights());
        Dictionary<string, float[]> taps = [];
        aligner.Probe = (stage, values) => taps.Add(stage, values);

        float[] aligned = aligner.Forward(VisionFixture.Floats(testCase.GetProperty("features")), gridHeight, gridWidth);

        Assert.Equal(["unfold", "hidden"], taps.Keys);
        VisionFixture.AssertClose(output, $"{gridHeight}x{gridWidth} unfold", taps["unfold"], VisionFixture.Floats(testCase.GetProperty("unfold")));
        VisionFixture.AssertClose(output, $"{gridHeight}x{gridWidth} hidden", taps["hidden"], VisionFixture.Floats(testCase.GetProperty("hidden")));
        VisionFixture.AssertClose(output, $"{gridHeight}x{gridWidth} output", aligned, VisionFixture.Floats(testCase.GetProperty("output")));
        (int tokenHeight, int tokenWidth) = DeepSeekV41Aligner.TokenGrid(gridHeight, gridWidth, 3);
        Assert.Equal(tokenHeight * tokenWidth * VisionFixture.OutputDim, aligned.Length);
    }

    [Fact]
    public void Unfold_FoldsChannelMajorAndZeroFillsTheEdge()
    {
        // a 2 x 4 grid of 2-channel features where cell (y, x), channel c, holds 100 * y + 10 * x + c + 1
        DeepSeekV41VisionConfig config = new(1, 2, 1, 4, 1, 3);
        float[] features = new float[2 * 4 * 2];
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 4; x++)
                for (int c = 0; c < 2; c++) features[(y * 4 + x) * 2 + c] = 100 * y + 10 * x + c + 1;
        using CpuBackend backend = new();
        using DeepSeekV41Aligner aligner = new(backend, config, 3, ZeroWeights(config, 3));
        float[]? folded = null;
        aligner.Probe = (stage, values) => { if (stage == "unfold") folded = values; };

        aligner.Forward(features, 2, 4);

        // padded to 3 x 6: one block row, two block columns, and each folded row is 2 channels x 9 block slots
        Assert.NotNull(folded);
        Assert.Equal(2 * 18, folded!.Length);
        for (int block = 0; block < 2; block++)
            for (int c = 0; c < 2; c++)
                for (int ky = 0; ky < 3; ky++)
                    for (int kx = 0; kx < 3; kx++)
                    {
                        int y = ky, x = block * 3 + kx;
                        float expected = y < 2 && x < 4 ? 100 * y + 10 * x + c + 1 : 0f;
                        Assert.Equal(expected, folded[block * 18 + c * 9 + ky * 3 + kx]);
                    }
    }

    [Theory]
    [InlineData(4, 7, 2, 3)]
    [InlineData(17, 23, 6, 8)]
    public void TokenGrid_RoundsEachSideUp(int gridHeight, int gridWidth, int tokenHeight, int tokenWidth) =>
        Assert.Equal((tokenHeight, tokenWidth), DeepSeekV41Aligner.TokenGrid(gridHeight, gridWidth, 3));

    [Fact]
    public void Forward_RejectsAFeatureCountThatIsNotTheGrid()
    {
        using CpuBackend backend = new();
        DeepSeekV41VisionConfig config = VisionFixture.Config();
        using DeepSeekV41Aligner aligner = new(backend, config, VisionFixture.OutputDim, VisionFixture.AlignerWeights());

        Assert.Throws<ArgumentException>(() => aligner.Forward(new float[5 * config.HiddenSize], 2, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => aligner.Forward(new float[config.HiddenSize], 0, 1));
    }

    private static DeepSeekV41AlignerWeights ZeroWeights(DeepSeekV41VisionConfig config, int outDim) =>
        new(DeepSeekV41HostMath.Tensor(new float[outDim * config.AlignerInputDim], outDim, config.AlignerInputDim),
            DeepSeekV41HostMath.Tensor(new float[outDim], outDim), DeepSeekV41HostMath.Tensor(new float[outDim * outDim], outDim, outDim),
            DeepSeekV41HostMath.Tensor(new float[outDim], outDim));
}
