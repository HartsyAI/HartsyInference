using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using Xunit;

namespace HartsyInference.Cuda.Tests;

/// <summary>A recipe's Format string must never select the Blackwell cuBLASLt block-scaled path meant for Comfy formats.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class QuantRecipeFormatRoutingTests
{
    [Fact]
    public void EveryRecipeFormat_IsInvisibleToBlockScaleFormats()
    {
        foreach (QuantEncoding encoding in Enum.GetValues<QuantEncoding>())
        {
            QuantRecipe recipe = new QuantRecipe
            {
                Encoding = encoding, Geometry = new BlockGeometry(1, 32), ScaleDType = DType.U8,
                LogicalRows = 1, LogicalCols = 32,
            };

            Assert.Null(BlockScaleFormats.FromQuantFormat(recipe.FormatName));
        }
    }
}
