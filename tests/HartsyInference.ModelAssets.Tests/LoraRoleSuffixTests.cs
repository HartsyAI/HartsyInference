using HartsyInference.ModelAssets.Lora;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Pins the one suffix table every LoRA mapper classifies through. A miss here is silent: an unrecognized
/// suffix does not fail, it makes the key invisible, so the file loads with fewer layers than it has and the LoRA
/// comes out weak. The near-collisions (<c>.lora_A.default.weight</c> vs <c>.lora_A.weight</c>, <c>.lokr_w1_a</c> vs
/// <c>.lokr_w1</c>, <c>.diff_b</c> vs <c>.diff</c>) are the reason the table is ordered.</summary>
public sealed class LoraRoleSuffixTests
{
    [Theory]
    [InlineData("transformer.blocks.0.attn.to_q.lora_A.weight", "transformer.blocks.0.attn.to_q", LoraRole.Down)]
    [InlineData("blocks.0.attn.qkv_proj.lora_A.default.weight", "blocks.0.attn.qkv_proj", LoraRole.Down)]
    [InlineData("lora_unet_input_blocks_1_0_op.lora_down.weight", "lora_unet_input_blocks_1_0_op", LoraRole.Down)]
    [InlineData("lora_unet_input_blocks_1_0_op.alpha", "lora_unet_input_blocks_1_0_op", LoraRole.Alpha)]
    [InlineData("lora_te_text_model_encoder.hada_w1_a", "lora_te_text_model_encoder", LoraRole.HadaW1A)]
    [InlineData("lora_unet_conv.hada_t1", "lora_unet_conv", LoraRole.HadaT1)]
    [InlineData("lora_unet_proj.lokr_w1", "lora_unet_proj", LoraRole.LokrW1)]
    [InlineData("lora_unet_proj.lokr_w2", "lora_unet_proj", LoraRole.LokrW2)]
    [InlineData("lora_unet_proj.lokr_t2", "lora_unet_proj", LoraRole.LokrT2)]
    [InlineData("transformer.blocks.0.attn.to_q.dora_scale", "transformer.blocks.0.attn.to_q", LoraRole.DoraScale)]
    [InlineData("diffusion_model.blocks.0.norm3.diff", "diffusion_model.blocks.0.norm3", LoraRole.Diff)]
    [InlineData("diffusion_model.blocks.0.norm3.diff_b", "diffusion_model.blocks.0.norm3", LoraRole.BiasDiff)]
    public void TryStrip_RecognizesEverySuffixAndReturnsTheRoot(string key, string expectedRoot, LoraRole expectedRole)
    {
        Assert.True(LoraRoleSuffix.TryStrip(key, out string root, out LoraRole role));
        Assert.Equal(expectedRoot, root);
        Assert.Equal(expectedRole, role);
    }

    [Theory]
    [InlineData("transformer.blocks.0.attn.to_q.weight")]
    [InlineData("lora_unet_proj.lokr_w3")]
    [InlineData("")]
    public void TryStrip_RejectsKeysWithNoRoleSuffix(string key)
    {
        Assert.False(LoraRoleSuffix.TryStrip(key, out string root, out LoraRole role));
        Assert.Equal(string.Empty, root);
        Assert.Equal(default, role);
    }

    [Theory]
    [InlineData(LoraRole.Down, true)]
    [InlineData(LoraRole.Alpha, false)]
    [InlineData(LoraRole.DoraScale, false)]
    [InlineData(LoraRole.Diff, false)]
    public void IsDecompositionMatrix_SeparatesMatricesFromScalarsAndFullWeightDiffs(LoraRole role, bool expected) =>
        Assert.Equal(expected, LoraRoleSuffix.IsDecompositionMatrix(role));
}
