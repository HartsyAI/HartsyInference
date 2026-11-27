using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The `_M` presets give the output side of the BFL-named FLUX.2 and Iris-Z transformer the high-fidelity type, as
/// llama.cpp's Q4_K_M gives output.weight and ffn_down. The fused qkv carries V and cannot be split, so it keeps the backbone
/// quant, and so do the attention output projections, which llama.cpp's mix leaves at the backbone too.</summary>
public sealed class GgufOutputSideMixTests
{
    [Theory]
    [InlineData("final_layer.linear.weight")]
    [InlineData("double_blocks.3.img_mlp.2.weight")]
    [InlineData("double_blocks.3.txt_mlp.2.weight")]
    [InlineData("single_blocks.7.linear2.weight")]
    public void OutputSideProjectionsGetTheHighFidelityType(string name)
    {
        using Tensor weight = new(new TensorShape(256, 512), DType.BF16);

        Assert.Equal(DType.Q6_K, GgufQuantPolicy.Q4_K_M.ResolveTargetDType(name, weight));
        Assert.Equal(DType.Q6_K, GgufQuantPolicy.Q5_K_M.ResolveTargetDType(name, weight));
    }

    [Theory]
    [InlineData("double_blocks.3.img_attn.qkv.weight")]
    [InlineData("single_blocks.7.linear1.weight")]
    [InlineData("double_blocks.3.img_attn.proj.weight")]
    public void FusedAndAttentionOutputProjectionsKeepTheBackboneQuant(string name)
    {
        using Tensor weight = new(new TensorShape(256, 512), DType.BF16);

        Assert.Equal(DType.Q4_K, GgufQuantPolicy.Q4_K_M.ResolveTargetDType(name, weight));
    }
}
