using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Loading the vision tower, aligner and image-span embeddings from a V4.1 checkpoint directory written under the official key names: naming, shapes, BF16 widening and ownership.</summary>
public sealed class DeepSeekV41VisionLoaderTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("dsv41-vision-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void AnF32Checkpoint_EncodesLikeTheUpstreamModel()
    {
        DeepSeekV41VisionFixtureCheckpoint.Write(_directory);
        JsonElement testCase = DeepSeekV41VisionFixture.TowerCase(5, 7);
        using CpuBackend backend = new();
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);

        using DeepSeekV41VisionModel model = DeepSeekV41VisionLoader.Load(backend, checkpoint);

        DeepSeekV41VisionFixture.AssertClose(output, "loaded encode 5x7", model.Encode(DeepSeekV41VisionFixture.Floats(testCase.GetProperty("patches")), 5, 7),
            DeepSeekV41VisionFixture.Floats(testCase.GetProperty("output")));
        Assert.Equal(DeepSeekV41VisionFixture.OutputDim, model.OutputDim);
        Assert.Equal(Enumerable.Range(0, model.OutputDim).Select(static i => DeepSeekV41VisionFixtureCheckpoint.EmbeddingValue("image_start", i)), model.ImageStart.ToArray());
        Assert.Equal(Enumerable.Range(0, model.OutputDim).Select(static i => DeepSeekV41VisionFixtureCheckpoint.EmbeddingValue("image_end", i)), model.ImageEnd.ToArray());
        Assert.Equal(Enumerable.Range(0, model.OutputDim).Select(static i => DeepSeekV41VisionFixtureCheckpoint.EmbeddingValue("image_newline", i)), model.ImageNewline.ToArray());
    }

    [Fact]
    public void ABf16Checkpoint_WidensExactly_AndTheModelOutlivesTheCheckpoint()
    {
        DeepSeekV41VisionFixtureCheckpoint.Write(_directory, bf16: true);
        float[] patches = DeepSeekV41VisionFixture.Floats(DeepSeekV41VisionFixture.TowerCase(4, 3).GetProperty("patches"));
        using CpuBackend backend = new();
        float[] viaDisk;
        DeepSeekV41VisionModel loaded;
        using (DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory))
        {
            loaded = DeepSeekV41VisionLoader.Load(backend, checkpoint);
            viaDisk = loaded.Encode(patches, 4, 3);
        }

        // the same forward on weights that went through bf16 in memory: any load-path difference would show as a nonzero gap
        DeepSeekV41VisionConfig config = DeepSeekV41VisionFixture.Config();
        Func<string, float[]> bf16Values = key => DeepSeekV41VisionFixture.Param(key).Select(DeepSeekV41VisionFixtureCheckpoint.ThroughBf16).ToArray();
        using DeepSeekV41VisionTower tower = new(backend, config, DeepSeekV41VisionFixture.TowerWeights(bf16Values));
        using DeepSeekV41Aligner aligner = new(backend, config, DeepSeekV41VisionFixture.OutputDim, DeepSeekV41VisionFixture.AlignerWeights(bf16Values));
        float[] direct = aligner.Forward(tower.Forward(patches, 4, 3), 4, 3);

        Assert.Equal(direct, viaDisk);
        Assert.Equal(viaDisk, loaded.Encode(patches, 4, 3));
        Assert.Equal(DeepSeekV41VisionFixtureCheckpoint.ThroughBf16(DeepSeekV41VisionFixtureCheckpoint.EmbeddingValue("image_start", 3)), loaded.ImageStart[3]);
        loaded.Dispose();
    }

    [Fact]
    public void AMissingTensor_IsNamed()
    {
        DeepSeekV41VisionFixtureCheckpoint.Write(_directory, omit: "vision.blocks.1.attn.wo.bias");
        using CpuBackend backend = new();
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41VisionLoader.Load(backend, checkpoint));

        Assert.Contains("vision.blocks.1.attn.wo.bias", error.Message);
    }

    [Fact]
    public void AMissingImageEmbedding_IsNamed()
    {
        DeepSeekV41VisionFixtureCheckpoint.Write(_directory, omit: "image_newline");
        using CpuBackend backend = new();
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);

        Assert.Contains("image_newline", Assert.Throws<HartsyInferenceException>(() => DeepSeekV41VisionLoader.Load(backend, checkpoint)).Message);
    }

    [Fact]
    public void AMisshapenTensor_IsRefusedByName()
    {
        DeepSeekV41VisionFixtureCheckpoint.Write(_directory, reshape: "vision.blocks.0.mlp.w2.weight");
        using CpuBackend backend = new();
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41VisionLoader.Load(backend, checkpoint));

        Assert.Contains("vision.blocks.0.mlp.w2.weight", error.Message);
    }

    [Fact]
    public void AConfigWithoutAVisionTower_IsRefused()
    {
        DeepSeekV41VisionFixtureCheckpoint.Write(_directory, withVisionConfig: false);
        using CpuBackend backend = new();
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);

        Assert.Contains("no vision tower", Assert.Throws<HartsyInferenceException>(() => DeepSeekV41VisionLoader.Load(backend, checkpoint)).Message);
    }
}
