using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The image encoder (tower, then aligner) against the unmodified upstream <c>encode_image</c> on the small seeded fixture.</summary>
public sealed class DeepSeekV41VisionModelTests(ITestOutputHelper output)
{
    private static DeepSeekV41VisionModel Build(CpuBackend backend, float[]? imageEnd = null)
    {
        DeepSeekV41VisionConfig config = DeepSeekV41VisionFixture.Config();
        int outDim = DeepSeekV41VisionFixture.OutputDim;
        return new DeepSeekV41VisionModel(new DeepSeekV41VisionTower(backend, config, DeepSeekV41VisionFixture.TowerWeights()),
            new DeepSeekV41Aligner(backend, config, outDim, DeepSeekV41VisionFixture.AlignerWeights()),
            Enumerable.Range(0, outDim).Select(static i => 1f + i).ToArray(), imageEnd ?? new float[outDim], Enumerable.Range(0, outDim).Select(static i => -1f - i).ToArray());
    }

    [Theory]
    [InlineData(5, 7)]
    [InlineData(4, 3)]
    [InlineData(1, 4)]
    public void Encode_MatchesTheUpstreamImageEncoder(int gridHeight, int gridWidth)
    {
        JsonElement testCase = DeepSeekV41VisionFixture.TowerCase(gridHeight, gridWidth);
        using CpuBackend backend = new();
        using DeepSeekV41VisionModel model = Build(backend);

        float[] embeddings = model.Encode(DeepSeekV41VisionFixture.Floats(testCase.GetProperty("patches")), gridHeight, gridWidth);

        DeepSeekV41VisionFixture.AssertClose(output, $"{gridHeight}x{gridWidth} encode", embeddings, DeepSeekV41VisionFixture.Floats(testCase.GetProperty("output")));
        (int tokenHeight, int tokenWidth) = model.TokenGrid(gridHeight, gridWidth);
        Assert.Equal(tokenHeight * tokenWidth * model.OutputDim, embeddings.Length);
    }

    [Fact]
    public void TheImageSpanEmbeddingsAreExposedAsGiven()
    {
        using CpuBackend backend = new();
        using DeepSeekV41VisionModel model = Build(backend);

        Assert.Equal(DeepSeekV41VisionFixture.OutputDim, model.OutputDim);
        Assert.Equal(1f, model.ImageStart[0]);
        Assert.Equal(0f, model.ImageEnd[^1]);
        Assert.Equal(-1f - (model.OutputDim - 1), model.ImageNewline[^1]);
    }

    [Fact]
    public void Constructor_RejectsAnEmbeddingOfTheWrongWidth()
    {
        using CpuBackend backend = new();

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => Build(backend, imageEnd: new float[3]));

        Assert.Contains("image_end", error.Message);
    }

    [Fact]
    public void Dispose_ReleasesBothHalves()
    {
        using CpuBackend backend = new();
        DeepSeekV41VisionModel model = Build(backend);
        float[] patches = new float[DeepSeekV41VisionFixture.Config().PatchInputDim];

        model.Dispose();

        Assert.Throws<ObjectDisposedException>(() => model.Tower.Forward(patches, 1, 1));
        Assert.Throws<ObjectDisposedException>(() => model.Aligner.Forward(new float[DeepSeekV41VisionFixture.Config().HiddenSize], 1, 1));
    }
}
