using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;
using Xunit.Abstractions;
using VisionFixture = HartsyInference.LLM.Tests.DeepSeekV41.DeepSeekV41VisionFixture;
using VisionCheckpoint = HartsyInference.LLM.Tests.DeepSeekV41.DeepSeekV41VisionFixtureCheckpoint;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Loading the vision tower, aligner and image-span embeddings from a V4.1 checkpoint directory written under the official key names: naming, shapes, BF16 widening and ownership.</summary>
public sealed class DeepSeekV41VisionLoaderTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("dsv41-vision-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    private static float[] Written(string key, int width) =>
        Enumerable.Range(0, width).Select(i => VisionCheckpoint.EmbeddingValue(key, i)).ToArray();

    [Fact]
    public void AnF32Checkpoint_EncodesLikeTheUpstreamModel()
    {
        VisionCheckpoint.Write(_directory);
        JsonElement testCase = VisionFixture.TowerCase(5, 7);
        using CpuBackend backend = new();
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);

        using DeepSeekV41VisionModel model = DeepSeekV41VisionLoader.Load(backend, checkpoint);

        VisionFixture.AssertClose(output, "loaded encode 5x7", model.Encode(VisionFixture.Floats(testCase.GetProperty("patches")), 5, 7),
            VisionFixture.Floats(testCase.GetProperty("output")));
        Assert.Equal(VisionFixture.OutputDim, model.OutputDim);
        Assert.Equal(Written("image_start", model.OutputDim), model.ImageStart.ToArray());
        Assert.Equal(Written("image_end", model.OutputDim), model.ImageEnd.ToArray());
        Assert.Equal(Written("image_newline", model.OutputDim), model.ImageNewline.ToArray());
    }

    [Fact]
    public void ABf16Checkpoint_WidensExactly_AndTheModelOutlivesTheCheckpoint()
    {
        VisionCheckpoint.Write(_directory, bf16: true);
        float[] patches = VisionFixture.Floats(VisionFixture.TowerCase(4, 3).GetProperty("patches"));
        using CpuBackend backend = new();
        float[] viaDisk;
        DeepSeekV41VisionModel loaded;
        using (DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory))
        {
            loaded = DeepSeekV41VisionLoader.Load(backend, checkpoint);
            viaDisk = loaded.Encode(patches, 4, 3);
        }

        // the same forward on weights that went through bf16 in memory: any load-path difference would show as a nonzero gap
        DeepSeekV41VisionConfig config = VisionFixture.Config();
        Func<string, float[]> bf16Values = key => VisionFixture.Param(key).Select(VisionCheckpoint.ThroughBf16).ToArray();
        using DeepSeekV41VisionTower tower = new(backend, config, VisionFixture.TowerWeights(bf16Values));
        using DeepSeekV41Aligner aligner = new(backend, config, VisionFixture.OutputDim, VisionFixture.AlignerWeights(bf16Values));
        float[] direct = aligner.Forward(tower.Forward(patches, 4, 3), 4, 3);

        Assert.Equal(direct, viaDisk);
        Assert.Equal(viaDisk, loaded.Encode(patches, 4, 3));
        Assert.Equal(VisionCheckpoint.ThroughBf16(VisionCheckpoint.EmbeddingValue("image_start", 3)), loaded.ImageStart[3]);
        loaded.Dispose();
    }

    [Fact]
    public void AMissingTensor_IsNamed()
    {
        VisionCheckpoint.Write(_directory, omit: "vision.blocks.1.attn.wo.bias");
        using CpuBackend backend = new();
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41VisionLoader.Load(backend, checkpoint));

        Assert.Contains("vision.blocks.1.attn.wo.bias", error.Message);
    }
}
