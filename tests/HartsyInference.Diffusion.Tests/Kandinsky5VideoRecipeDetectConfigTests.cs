using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Regression coverage for the Kandinsky-5 Video Lite-vs-Pro config-selection fix:
/// <see cref="Kandinsky5VideoRecipe.Construct"/> used to hardcode <see cref="Kandinsky5Config.VideoLite2B"/>
/// regardless of which checkpoint loaded, so a Pro-19B checkpoint would silently construct the wrong (Lite)
/// architecture. Weight-free — <see cref="Kandinsky5VideoRecipe.DetectConfig"/> only reads one tensor's shape,
/// no checkpoint or GPU needed, which is exactly the silent-misconfiguration class of bug a unit test should
/// catch (the real Lite/Pro checkpoints are large downloads not available in every dev environment).</summary>
public sealed class Kandinsky5VideoRecipeDetectConfigTests
{
    private static Tensor TextProjWeight(int modelDim) =>
        new Tensor(new TensorShape(modelDim, 3584), DType.F32);

    [Fact]
    public void DetectConfig_LiteModelDim_ReturnsLite()
    {
        Dictionary<string, Tensor> weights = new() { ["text_embeddings.in_layer.weight"] = TextProjWeight(Kandinsky5Config.VideoLite2B.ModelDim) };

        Kandinsky5Config config = Kandinsky5VideoRecipe.DetectConfig(weights);

        Assert.Equal(Kandinsky5Config.VideoLite2B.ModelDim, config.ModelDim);
        Assert.Equal(Kandinsky5Config.VideoLite2B.NumVisualBlocks, config.NumVisualBlocks);
    }

    [Fact]
    public void DetectConfig_ProModelDim_ReturnsPro()
    {
        Dictionary<string, Tensor> weights = new() { ["text_embeddings.in_layer.weight"] = TextProjWeight(Kandinsky5Config.VideoPro19B.ModelDim) };

        Kandinsky5Config config = Kandinsky5VideoRecipe.DetectConfig(weights);

        Assert.Equal(Kandinsky5Config.VideoPro19B.ModelDim, config.ModelDim);
        Assert.Equal(Kandinsky5Config.VideoPro19B.NumVisualBlocks, config.NumVisualBlocks);
    }

    [Fact]
    public void DetectConfig_UnrecognizedModelDim_FallsBackToLiteRatherThanThrowing()
    {
        Dictionary<string, Tensor> weights = new() { ["text_embeddings.in_layer.weight"] = TextProjWeight(999) };

        Kandinsky5Config config = Kandinsky5VideoRecipe.DetectConfig(weights);

        Assert.Equal(Kandinsky5Config.VideoLite2B.ModelDim, config.ModelDim);
    }

    [Fact]
    public void DetectConfig_MissingKey_FallsBackToLiteRatherThanThrowing()
    {
        Dictionary<string, Tensor> weights = new();

        Kandinsky5Config config = Kandinsky5VideoRecipe.DetectConfig(weights);

        Assert.Equal(Kandinsky5Config.VideoLite2B.ModelDim, config.ModelDim);
    }
}
