using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.Models.MossVl;

namespace HartsyInference.LLM.Tests;

public sealed class MossVlConfigTests
{
    private static string FixtureText() => File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "moss_vl_config.json"));

    private static MossVlConfig LoadFixture()
    {
        using JsonDocument doc = JsonDocument.Parse(FixtureText());
        return MossVlConfig.FromJson(doc.RootElement);
    }

    [Fact]
    public void ReadsTextDecoderFromRealConfig()
    {
        MossVlConfig config = LoadFixture();
        Assert.Equal(4096, config.Text.HiddenSize);
        Assert.Equal(48, config.Text.NumLayers);
        Assert.Equal(32, config.Text.NumHeads);
        Assert.Equal(8, config.Text.NumKvHeads);
        Assert.Equal(128, config.Text.HeadDim);
        Assert.Equal(12288, config.Text.IntermediateSize);
        Assert.Equal(151936, config.Text.VocabSize);
        Assert.Equal(5_000_000f, config.Text.RopeTheta);
        Assert.True(config.Text.QkNorm);
        Assert.False(config.Text.TieWordEmbeddings);
    }

    [Fact]
    public void MarksEveryFourthLayerFromTwoAsCrossAttention()
    {
        MossVlConfig config = LoadFixture();
        int[] expected = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46];
        Assert.Equal(expected.OrderBy(i => i), config.Text.CrossAttnLayers.OrderBy(i => i));
        Assert.True(config.Text.IsCrossAttnLayer(2));
        Assert.False(config.Text.IsCrossAttnLayer(3));
    }

    [Fact]
    public void ReadsVisionTowerAndDeepstackIndexes()
    {
        MossVlVisionConfig vision = LoadFixture().Vision;
        Assert.Equal(27, vision.Depth);
        Assert.Equal(1152, vision.HiddenSize);
        Assert.Equal(4304, vision.IntermediateSize);
        Assert.Equal(16, vision.NumHeads);
        Assert.Equal(16, vision.PatchSize);
        Assert.Equal(2, vision.SpatialMergeSize);
        Assert.Equal(1, vision.TemporalPatchSize);
        Assert.Equal(2304, vision.NumPositionEmbeddings);
        Assert.Equal(4096, vision.OutHiddenSize);
        Assert.Equal(new[] { 8, 16, 24 }, vision.DeepstackIndexes);
    }

    [Fact]
    public void ReadsInterleavedMropeAndSpecialTokenIds()
    {
        MossVlConfig config = LoadFixture();
        Assert.Equal(new[] { 24, 20, 20 }, config.MropeSection);
        Assert.True(config.MropeInterleaved);
        Assert.Equal(151655, config.ImageTokenId);
        Assert.Equal(151656, config.VideoTokenId);
        Assert.Equal(151652, config.VisionStartTokenId);
        Assert.Equal(151653, config.VisionEndTokenId);
    }

    [Fact]
    public void RejectsAnotherModelType()
    {
        using JsonDocument doc = JsonDocument.Parse("{\"model_type\":\"qwen3\"}");
        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => MossVlConfig.FromJson(doc.RootElement));
        Assert.Contains("qwen3", ex.Message);
    }

    [Fact]
    public void NamesTheMissingField()
    {
        string text = FixtureText().Replace("\"num_hidden_layers\": 48,", string.Empty);
        using JsonDocument doc = JsonDocument.Parse(text);
        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => MossVlConfig.FromJson(doc.RootElement));
        Assert.Contains("num_hidden_layers", ex.Message);
    }
}
