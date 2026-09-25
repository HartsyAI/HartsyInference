using Xunit;
using HartsyInference.ModelAssets.CheckpointConverters;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Tests the Comfy-Org → original-Tencent key normalization that lets the Comfy-Org HunyuanImage 2.1 repack reach <c>HunyuanImageTransformer</c>. Pure string mapping — no checkpoint files needed. The repack loaded as-is throws <c>KeyNotFoundException: x_embedder.proj.weight</c>, because it nests the sub-modules Tencent flattens and hides them under a doubled <c>model.model.</c> prefix.</summary>
public class HunyuanImageCheckpointConverterTests
{
    [Theory]
    // Double blocks: fused qkv and the output projection lose their stream-level nesting.
    [InlineData("double_blocks.0.img_attn.qkv.weight", "double_blocks.0.img_attn_qkv.weight")]
    [InlineData("double_blocks.3.txt_attn.qkv.bias", "double_blocks.3.txt_attn_qkv.bias")]
    [InlineData("double_blocks.0.img_attn.proj.weight", "double_blocks.0.img_attn_proj.weight")]
    [InlineData("double_blocks.0.txt_attn.proj.bias", "double_blocks.0.txt_attn_proj.bias")]
    // The QK RMSNorms are ".scale" here and ".weight" in the Tencent layout.
    [InlineData("double_blocks.1.img_attn.norm.query_norm.scale", "double_blocks.1.img_attn_q_norm.weight")]
    [InlineData("double_blocks.1.img_attn.norm.key_norm.scale", "double_blocks.1.img_attn_k_norm.weight")]
    [InlineData("double_blocks.1.txt_attn.norm.query_norm.scale", "double_blocks.1.txt_attn_q_norm.weight")]
    [InlineData("double_blocks.1.txt_attn.norm.key_norm.scale", "double_blocks.1.txt_attn_k_norm.weight")]
    // Modulation and the two-layer MLPs are positional here, named there.
    [InlineData("double_blocks.2.img_mod.lin.weight", "double_blocks.2.img_mod.linear.weight")]
    [InlineData("double_blocks.2.txt_mod.lin.bias", "double_blocks.2.txt_mod.linear.bias")]
    [InlineData("double_blocks.4.img_mlp.0.weight", "double_blocks.4.img_mlp.fc1.weight")]
    [InlineData("double_blocks.4.img_mlp.2.bias", "double_blocks.4.img_mlp.fc2.bias")]
    [InlineData("double_blocks.4.txt_mlp.0.weight", "double_blocks.4.txt_mlp.fc1.weight")]
    [InlineData("double_blocks.4.txt_mlp.2.bias", "double_blocks.4.txt_mlp.fc2.bias")]
    // Single blocks carry the same norms without a stream prefix — the reason the rules are scoped per section.
    [InlineData("single_blocks.7.norm.query_norm.scale", "single_blocks.7.q_norm.weight")]
    [InlineData("single_blocks.7.norm.key_norm.scale", "single_blocks.7.k_norm.weight")]
    [InlineData("single_blocks.7.modulation.lin.weight", "single_blocks.7.modulation.linear.weight")]
    // Token refiner.
    [InlineData("txt_in.individual_token_refiner.blocks.0.self_attn.qkv.weight", "txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight")]
    [InlineData("txt_in.individual_token_refiner.blocks.1.self_attn.proj.bias", "txt_in.individual_token_refiner.blocks.1.self_attn_proj.bias")]
    [InlineData("txt_in.individual_token_refiner.blocks.0.mlp.0.weight", "txt_in.individual_token_refiner.blocks.0.mlp.fc1.weight")]
    [InlineData("txt_in.individual_token_refiner.blocks.0.mlp.2.bias", "txt_in.individual_token_refiner.blocks.0.mlp.fc2.bias")]
    // MLP heads: time/t_embedder land on Tencent's indexed mlp, c_embedder feeds a diffusers text_embedder directly.
    [InlineData("time_in.in_layer.weight", "time_in.mlp.0.weight")]
    [InlineData("time_in.out_layer.bias", "time_in.mlp.2.bias")]
    [InlineData("txt_in.t_embedder.in_layer.weight", "txt_in.t_embedder.mlp.0.weight")]
    [InlineData("txt_in.c_embedder.in_layer.weight", "txt_in.c_embedder.linear_1.weight")]
    [InlineData("txt_in.c_embedder.out_layer.bias", "txt_in.c_embedder.linear_2.bias")]
    public void NormalizeComfyOrgNames_MapsComfyRepackToTencent(string comfy, string tencent)
    {
        Assert.Equal(tencent, HunyuanImageCheckpointConverter.NormalizeComfyOrgNames(comfy));
    }

    [Theory]
    // Already-Tencent keys must survive the pass untouched, or the GGUF repacks that used to work break.
    [InlineData("double_blocks.0.img_attn_qkv.weight")]
    [InlineData("double_blocks.0.img_mod.linear.weight")]
    [InlineData("double_blocks.0.img_mlp.fc1.weight")]
    [InlineData("double_blocks.0.img_attn_q_norm.weight")]
    [InlineData("single_blocks.0.linear1.weight")]
    [InlineData("single_blocks.0.linear2.bias")]
    [InlineData("single_blocks.0.modulation.linear.weight")]
    [InlineData("txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight")]
    [InlineData("time_in.mlp.0.weight")]
    [InlineData("img_in.proj.weight")]
    [InlineData("byt5_in.fc1.weight")]
    [InlineData("final_layer.adaLN_modulation.1.weight")]
    [InlineData("final_layer.linear.bias")]
    public void NormalizeComfyOrgNames_LeavesTencentKeysUnchanged(string key)
    {
        Assert.Equal(key, HunyuanImageCheckpointConverter.NormalizeComfyOrgNames(key));
    }
}
