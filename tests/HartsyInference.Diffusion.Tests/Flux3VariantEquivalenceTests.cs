using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Variants;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The FLUX.3 distilled variant must resolve exactly when the converter's flow-map check passes, so the
/// variant the catalog names and the sampling path the transformer runs cannot disagree.</summary>
public sealed class Flux3VariantEquivalenceTests
{
    [Theory]
    [InlineData(new[] { "ref_time_in.in_layer.weight" }, true)]
    [InlineData(new[] { "model.diffusion_model.ref_time_in.in_layer.weight" }, true)]
    [InlineData(new[] { "diffusion_model.ref_time_in.in_layer.weight", "emb_in.image.weight" }, true)]
    [InlineData(new[] { "ref_time_in.out_layer.weight" }, false)]
    [InlineData(new[] { "time_in.in_layer.weight" }, false)]
    [InlineData(new[] { "emb_in.image.weight" }, false)]
    public void Distilled_ResolvesExactlyWhenTheConverterSetsFlowMap(string[] keys, bool flowMap)
    {
        Assert.Equal(flowMap, ConvertedHasFlowMap(keys));
        ResolvedModelVariant resolved = ModelVariantResolver.Classify(Flux3Variants.Catalog,
            CheckpointProbe.Empty with { Keys = new HashSet<string>(keys) }, []);
        Assert.Equal(flowMap, resolved.Is(Flux3Variants.Distilled));
    }

    private static bool ConvertedHasFlowMap(string[] keys)
    {
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        try
        {
            foreach (string key in keys)
            {
                weights[key] = new Tensor(new TensorShape(1), DType.F32);
            }
            return Flux3CheckpointConverter.ConvertTransformer(weights).ContainsKey("ref_time_in.in_layer.weight");
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
