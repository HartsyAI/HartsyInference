using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>A quantized FastFrames generator is refused by name; a plain float generator passes.</summary>
public sealed class FastFramesQuantRefusalTests
{
    [Fact]
    public void AQuantizedWeightIsRefusedByNameAndFormat()
    {
        using Tensor plain = new Tensor(new TensorShape(2, 3), DType.F32);
        using Tensor quant = new Tensor(new TensorShape(2, 3), DType.F32) { QuantInfo = new QuantWeightInfo { Format = "int8_tensorwise" } };
        Dictionary<string, Tensor> weights = new() { ["blocks.0.ffn.w"] = plain, ["blocks.1.ffn.w"] = quant };

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => FastFramesRecipe.RefuseQuantised(weights));
        Assert.Contains("blocks.1.ffn.w", ex.Message);
        Assert.Contains("int8_tensorwise", ex.Message);
    }

    [Fact]
    public void APlainFloatGeneratorPasses()
    {
        using Tensor plain = new Tensor(new TensorShape(2, 3), DType.F32);

        FastFramesRecipe.RefuseQuantised(new Dictionary<string, Tensor> { ["blocks.0.ffn.w"] = plain });
    }
}
