using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The GGUF presets keep Iris-Z's <c>modality_in.weight</c> dense. It is a 4-row table the FLUX.2 transformer reads
/// as F32 (a block-quantized copy made the transformer throw at load), and its row length divides every block size, so
/// nothing else would stop a preset from quantizing it.</summary>
public sealed class GgufModalityRowsPolicyTests
{
    [Theory]
    [InlineData("modality_in.weight")]
    [InlineData("double_blocks.0.img_attn.proj.modality_in.weight")]
    public void ModalityRowsStayDenseUnderEveryPreset(string name)
    {
        using Tensor rows = new(new TensorShape(4, 4096), DType.BF16);
        GgufQuantPolicy[] presets = [GgufQuantPolicy.Q8_0, GgufQuantPolicy.Q6_K, GgufQuantPolicy.Q5_K_M, GgufQuantPolicy.Q4_K_M, GgufQuantPolicy.Q4_K_S];

        foreach (GgufQuantPolicy preset in presets)
        {
            DType target = preset.ResolveTargetDType(name, rows);
            Assert.False(target.IsQuantized, $"{name} would be {target.Name} under {preset.BackboneDType.Name}");
        }
    }

    [Fact]
    public void ProjectionWeightsOfTheSameShapeAreStillQuantized()
    {
        using Tensor projection = new(new TensorShape(4096, 4096), DType.BF16);

        Assert.True(GgufQuantPolicy.Q8_0.ResolveTargetDType("double_blocks.0.img_attn.proj.weight", projection).IsQuantized);
    }
}
