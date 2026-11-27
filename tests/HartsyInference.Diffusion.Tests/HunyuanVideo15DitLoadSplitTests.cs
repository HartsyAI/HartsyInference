using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The DiT's weight load split: <c>LoadSharedWeights</c> must work with no <c>transformer_blocks</c> keys, and one
/// <c>LoadBlockWeights</c> call must load only that block. Uses the tiny synthetic DiT fixture; the full <c>LoadWeights</c> still loads every block.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class HunyuanVideo15DitLoadSplitTests
{
    private const int Blocks = 2;
    private const int Hidden = 32;

    private static HunyuanVideo15Config TinyConfig() => new()
    {
        NumBlocks = Blocks, HiddenSize = Hidden, NumHeads = 2, LatentChannels = 4, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
    };

    private static Dictionary<string, Tensor> ConvertedFixture(SafeTensorsLoader loader)
    {
        Dictionary<string, Tensor> source = loader.GetAllTensors()
            .Where(pair => !pair.Key.StartsWith("ref.", StringComparison.Ordinal) && !pair.Key.StartsWith("in.", StringComparison.Ordinal)
                && !pair.Key.StartsWith("win.", StringComparison.Ordinal) && !pair.Key.StartsWith("rec.", StringComparison.Ordinal))
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        return HunyuanVideo15CheckpointConverter.Convert(source, Blocks, Hidden);
    }

    [Fact]
    public void LoadSharedWeights_WorksWithNoBlockKeys()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors"));
        Dictionary<string, Tensor> converted = ConvertedFixture(loader);
        Dictionary<string, Tensor> shared = converted
            .Where(pair => !pair.Key.StartsWith("transformer_blocks.", StringComparison.Ordinal))
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        Assert.NotEmpty(shared);
        Assert.DoesNotContain(shared.Keys, key => key.StartsWith("transformer_blocks.", StringComparison.Ordinal));

        using HunyuanVideo15Dit dit = new(TinyConfig());
        dit.LoadSharedWeights(shared); // must not require any block key
        Assert.All(Enumerable.Range(0, Blocks), i => Assert.Equal(0, dit.GetBlock(i).EstimatedWeightBytes));
    }

    [Fact]
    public void LoadBlockWeights_LoadsOnlyThatBlock()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors"));
        Dictionary<string, Tensor> converted = ConvertedFixture(loader);

        using HunyuanVideo15Dit dit = new(TinyConfig());
        dit.LoadSharedWeights(converted);
        dit.LoadBlockWeights(0, converted);
        Assert.True(dit.GetBlock(0).EstimatedWeightBytes > 0, "block 0 should be loaded");
        Assert.Equal(0, dit.GetBlock(1).EstimatedWeightBytes);

        dit.LoadBlockWeights(1, converted);
        Assert.True(dit.GetBlock(1).EstimatedWeightBytes > 0, "block 1 should be loaded after its own call");
    }

    [Fact]
    public void LoadWeights_StillLoadsEveryBlock()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors"));
        Dictionary<string, Tensor> converted = ConvertedFixture(loader);

        using HunyuanVideo15Dit dit = new(TinyConfig());
        dit.LoadWeights(converted);
        Assert.All(Enumerable.Range(0, Blocks), i => Assert.True(dit.GetBlock(i).EstimatedWeightBytes > 0));
    }
}
