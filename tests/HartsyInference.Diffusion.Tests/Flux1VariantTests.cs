using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>FLUX.1 Dev and Schnell are told apart by the guidance embedder alone. The catalog must answer what the
/// converter's own detection answers, and the defaults it names must be the numbers the removed pipeline defaults gave.</summary>
public sealed class Flux1VariantTests
{
    private const string Transformer = "double_blocks.0.img_mod.lin.weight";

    [Fact]
    public void Dev_ResolvesStructurally_FromGuidanceEmbedder()
    {
        ResolvedModelVariant resolved = Classify(Transformer, "guidance_in.in_layer.weight");
        Assert.True(resolved.Is(Flux1Variants.Dev));
        Assert.Equal(ModelVariantSource.Structure, resolved.Source);
    }

    [Fact]
    public void Schnell_ResolvesStructurally_WithoutGuidanceEmbedder()
    {
        ResolvedModelVariant resolved = Classify(Transformer);
        Assert.True(resolved.Is(Flux1Variants.Schnell));
        Assert.Equal(ModelVariantSource.Structure, resolved.Source);
    }

    [Fact]
    public void DefaultsFor_ReturnsTheResolvedVariantsDefaults()
    {
        Flux1Recipe recipe = new Flux1Recipe();
        Assert.Equal((28, 3.5f, 1024, 1024), Numbers(recipe.DefaultsFor(Classify(Transformer, "guidance_in.in_layer.weight"))));
        Assert.Equal((4, 1.0f, 1024, 1024), Numbers(recipe.DefaultsFor(Classify(Transformer))));
    }

    [Theory]
    [InlineData(new[] { "double_blocks.0.img_mod.lin.weight", "guidance_in.in_layer.weight" }, true)]
    [InlineData(new[] { "double_blocks.0.img_mod.lin.weight", "guidance_in.out_layer.weight" }, true)]
    [InlineData(new[] { "model.diffusion_model.double_blocks.0.img_mod.lin.weight", "model.diffusion_model.guidance_in.in_layer.weight" }, true)]
    [InlineData(new[] { "double_blocks.0.img_mod.lin.weight" }, false)]
    [InlineData(new[] { "double_blocks.0.img_mod.lin.weight", "guidance_in.weight" }, false)]
    [InlineData(new[] { "double_blocks.0.img_mod.lin.weight", "time_text_embed.guidance_embedder.linear_1.weight" }, false)]
    [InlineData(new[] { "transformer_blocks.0.norm1.linear.weight", "time_text_embed.guidance_embedder.linear_1.weight" }, true)]
    [InlineData(new[] { "transformer_blocks.0.norm1.linear.weight" }, false)]
    public void Structure_AgreesWithTheConverterGuidanceDetection(string[] keys, bool hasGuidance)
    {
        Assert.Equal(hasGuidance, ConverterHasGuidance(keys));
        ResolvedModelVariant resolved = Classify(keys);
        Assert.Equal(hasGuidance, resolved.Is(Flux1Variants.Dev));
        Assert.Equal(ModelVariantSource.Structure, resolved.Source);
    }

    private static ResolvedModelVariant Classify(params string[] keys) =>
        ModelVariantResolver.Classify(Flux1Variants.Catalog, CheckpointProbe.Empty with { Keys = new HashSet<string>(keys) }, []);

    private static (int Steps, float Cfg, int Width, int Height) Numbers(ImageDefaults defaults) =>
        (defaults.Steps, defaults.CfgScale, defaults.Width, defaults.Height);

    /// <summary>The converter's own answer: convert the names with placeholder tensors and read what it emits.</summary>
    private static bool ConverterHasGuidance(string[] keys)
    {
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        try
        {
            foreach (string key in keys)
            {
                weights[key] = new Tensor(new TensorShape(1), DType.F32);
            }
            FluxCheckpointConverter.ConvertedWeights converted = FluxCheckpointConverter.Convert(weights);
            return FluxCheckpointConverter.DetectArchitecture(converted.Transformer).hasGuidance;
        }
        finally
        {
            foreach (Tensor tensor in weights.Values)
            {
                tensor.Dispose();
            }
        }
    }
}
