using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>The execution policy dequantizes every derivative format to BF16 and labels its activation precision honestly.</summary>
public sealed class QuantExecutionPolicyTests
{
    private static QuantRecipe Recipe(QuantEncoding encoding, long rows = 64, long cols = 128) => new()
    {
        Encoding = encoding, Geometry = new BlockGeometry(1, 16), ScaleDType = DType.F8E4M3, LogicalRows = rows, LogicalCols = cols,
    };

    [Theory]
    [InlineData(QuantEncoding.Nvfp4, "W4A16")]
    [InlineData(QuantEncoding.Mxfp4E8M0, "W4A16")]
    [InlineData(QuantEncoding.AffineInt4, "W4A16")]
    [InlineData(QuantEncoding.AffineInt8, "W8A16")]
    [InlineData(QuantEncoding.Fp8E4M3BlockE8M0, "W8A16")]
    [InlineData(QuantEncoding.Exl3Trellis, "W2A16")]
    public void Plan_DequantizesToBf16AndNeverClaimsFourBitActivations(QuantEncoding encoding, string label)
    {
        QuantRecipe recipe = Recipe(encoding);

        QuantExecutionPlan plan = QuantExecutionPolicy.Plan(recipe);

        Assert.Equal(QuantExecutionKind.DequantBf16, plan.Kind);
        Assert.Equal(QuantActivationPrecision.Bf16, plan.Activation);
        Assert.Equal(64 * 128 * 2, plan.WorkspaceBytes);
        Assert.Equal(label, plan.Label(recipe));
        Assert.True(QuantExecutionPolicy.SupportsBf16Dequant(encoding));
    }

    [Theory]
    [InlineData(QuantEncoding.Gguf)]
    public void Plan_RefusesEncodingsWithoutADequantPath(QuantEncoding encoding)
    {
        Assert.False(QuantExecutionPolicy.SupportsBf16Dequant(encoding));
        Assert.Throws<NotSupportedException>(() => QuantExecutionPolicy.Plan(Recipe(encoding)));
    }

    [Fact]
    public void Label_ReportsW4A4OnlyForAFourBitActivationPlan()
    {
        QuantRecipe recipe = Recipe(QuantEncoding.Nvfp4);
        QuantExecutionPlan blackwell = new(QuantExecutionKind.DequantBf16, 0, QuantActivationPrecision.Fp4);

        Assert.Equal("W4A4", blackwell.Label(recipe));
    }
}
