using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The vision tower against the unmodified upstream <c>ViT</c> on the small seeded fixture: patch embedding, each block, the final norm, and the weight and argument checks.</summary>
public sealed class DeepSeekV41VisionTowerTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(5, 7)]
    [InlineData(4, 3)]
    [InlineData(1, 4)]
    public void EveryStage_MatchesTheUpstreamTower(int gridHeight, int gridWidth)
    {
        JsonElement testCase = DeepSeekV41VisionFixture.TowerCase(gridHeight, gridWidth);
        using CpuBackend backend = new();
        DeepSeekV41VisionConfig config = DeepSeekV41VisionFixture.Config();
        using DeepSeekV41VisionTower tower = new(backend, config, DeepSeekV41VisionFixture.TowerWeights());
        Dictionary<string, float[]> taps = [];
        tower.Probe = (stage, values) => taps.Add(stage, values);

        float[] features = tower.Forward(DeepSeekV41VisionFixture.Floats(testCase.GetProperty("patches")), gridHeight, gridWidth);

        string[] expectedStages = ["patch_embed", .. Enumerable.Range(0, config.NumLayers).Select(static i => $"block.{i}"), "norm"];
        Assert.Equal(expectedStages, taps.Keys);
        JsonElement stages = testCase.GetProperty("stages");
        foreach (string stage in expectedStages)
            DeepSeekV41VisionFixture.AssertClose(output, $"{gridHeight}x{gridWidth} {stage}", taps[stage], DeepSeekV41VisionFixture.Floats(stages.GetProperty(stage)));
        Assert.Equal(taps["norm"], features);
    }

    [Fact]
    public void Forward_IsRepeatableAndTheProbeCostsNothingWhenOff()
    {
        JsonElement testCase = DeepSeekV41VisionFixture.TowerCase(4, 3);
        float[] patches = DeepSeekV41VisionFixture.Floats(testCase.GetProperty("patches"));
        using CpuBackend backend = new();
        using DeepSeekV41VisionTower tower = new(backend, DeepSeekV41VisionFixture.Config(), DeepSeekV41VisionFixture.TowerWeights());

        float[] first = tower.Forward(patches, 4, 3);
        float[] second = tower.Forward(patches, 4, 3);

        Assert.Null(tower.Probe);
        Assert.Equal(first, second);
    }

    [Fact]
    public void Forward_RejectsAPatchCountThatIsNotTheGrid()
    {
        using CpuBackend backend = new();
        DeepSeekV41VisionConfig config = DeepSeekV41VisionFixture.Config();
        using DeepSeekV41VisionTower tower = new(backend, config, DeepSeekV41VisionFixture.TowerWeights());

        Assert.Throws<ArgumentException>(() => tower.Forward(new float[5 * config.PatchInputDim], 2, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => tower.Forward(new float[config.PatchInputDim], 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => tower.Forward(new float[config.PatchInputDim], 1, 0));
    }

    [Fact]
    public void Forward_AfterDispose_Throws()
    {
        using CpuBackend backend = new();
        DeepSeekV41VisionConfig config = DeepSeekV41VisionFixture.Config();
        DeepSeekV41VisionTower tower = new(backend, config, DeepSeekV41VisionFixture.TowerWeights());

        tower.Dispose();
        tower.Dispose();

        Assert.Throws<ObjectDisposedException>(() => tower.Forward(new float[config.PatchInputDim], 1, 1));
    }

    [Fact]
    public void Constructor_NamesAMisshapenWeight()
    {
        using CpuBackend backend = new();
        DeepSeekV41VisionConfig config = DeepSeekV41VisionFixture.Config();
        DeepSeekV41VisionWeights good = DeepSeekV41VisionFixture.TowerWeights();
        DeepSeekV41VisionBlockWeights block = good.Blocks[1];
        using Tensor wrong = DeepSeekV41HostMath.Tensor(new float[config.HiddenSize * 2], config.HiddenSize, 2);
        DeepSeekV41VisionWeights bad = good with { Blocks = [good.Blocks[0], block with { Wo = wrong }] };

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => new DeepSeekV41VisionTower(backend, config, bad));

        Assert.Contains("vision.blocks.1.attn.wo.weight", error.Message);
    }

    [Fact]
    public void Constructor_RejectsAWrongBlockCountAndANonF32Weight()
    {
        using CpuBackend backend = new();
        DeepSeekV41VisionConfig config = DeepSeekV41VisionFixture.Config();
        DeepSeekV41VisionWeights good = DeepSeekV41VisionFixture.TowerWeights();

        DeepSeekV41VisionWeights fewer = good with { Blocks = [good.Blocks[0]] };
        Assert.Contains("1 blocks", Assert.Throws<HartsyInferenceException>(() => new DeepSeekV41VisionTower(backend, config, fewer)).Message);

        using Tensor half = new(good.FinalNorm.Shape, DType.F16);
        DeepSeekV41VisionWeights f16 = good with { FinalNorm = half };
        Assert.Contains("vision.norm.weight", Assert.Throws<HartsyInferenceException>(() => new DeepSeekV41VisionTower(backend, config, f16)).Message);
    }

    [Fact]
    public void Constructor_RejectsAConfigWithoutAWholeRotaryWidth()
    {
        using CpuBackend backend = new();
        DeepSeekV41VisionConfig config = DeepSeekV41VisionFixture.Config() with { NumHeads = 3 };
        DeepSeekV41VisionWeights weights = DeepSeekV41VisionFixture.TowerWeights();

        Assert.Throws<HartsyInferenceException>(() => new DeepSeekV41VisionTower(backend, config, weights));
    }
}
