using Xunit;
using HartsyInference.ModelHandler.CheckpointConverters;

namespace HartsyInference.ModelHandler.Tests;

/// <summary>Tests the pure key mapping of <see cref="WanVideoCheckpointConverter.MapKey"/> — the single-file prefix
/// strip, the ordered original→diffusers renames (including the norm2/norm3 swap: original <c>norm3</c> is the
/// cross-attn norm = diffusers <c>norm2</c>), and diffusers-named pass-through. No checkpoint files needed.</summary>
public class WanVideoCheckpointConverterTests
{
    [Theory]
    // Original Wan naming → the diffusers names WanVideoTransformer.LoadWeights expects.
    [InlineData("patch_embedding.weight", "patch_embedding.weight")]
    [InlineData("time_embedding.0.weight", "condition_embedder.time_embedder.linear_1.weight")]
    [InlineData("time_embedding.2.bias", "condition_embedder.time_embedder.linear_2.bias")]
    [InlineData("text_embedding.0.weight", "condition_embedder.text_embedder.linear_1.weight")]
    [InlineData("text_embedding.2.weight", "condition_embedder.text_embedder.linear_2.weight")]
    [InlineData("time_projection.1.weight", "condition_embedder.time_proj.weight")]
    [InlineData("head.modulation", "scale_shift_table")]
    [InlineData("head.head.weight", "proj_out.weight")]
    [InlineData("blocks.0.modulation", "blocks.0.scale_shift_table")]
    [InlineData("blocks.0.self_attn.q.weight", "blocks.0.attn1.to_q.weight")]
    [InlineData("blocks.0.self_attn.k.bias", "blocks.0.attn1.to_k.bias")]
    [InlineData("blocks.0.self_attn.v.weight", "blocks.0.attn1.to_v.weight")]
    [InlineData("blocks.0.self_attn.o.weight", "blocks.0.attn1.to_out.0.weight")]
    [InlineData("blocks.0.self_attn.norm_q.weight", "blocks.0.attn1.norm_q.weight")]
    [InlineData("blocks.0.self_attn.norm_k.weight", "blocks.0.attn1.norm_k.weight")]
    [InlineData("blocks.29.cross_attn.q.weight", "blocks.29.attn2.to_q.weight")]
    [InlineData("blocks.29.cross_attn.o.bias", "blocks.29.attn2.to_out.0.bias")]
    [InlineData("blocks.29.cross_attn.norm_k.weight", "blocks.29.attn2.norm_k.weight")]
    // norm swap: original norm3 (cross-attn affine norm) → diffusers norm2; original norm2 (ffn) → norm3.
    [InlineData("blocks.5.norm3.weight", "blocks.5.norm2.weight")]
    [InlineData("blocks.5.norm3.bias", "blocks.5.norm2.bias")]
    [InlineData("blocks.5.norm2.weight", "blocks.5.norm3.weight")]
    [InlineData("blocks.0.ffn.0.weight", "blocks.0.ffn.net.0.proj.weight")]
    [InlineData("blocks.0.ffn.2.weight", "blocks.0.ffn.net.2.weight")]
    // I2V-14B image-embedder keys carried for a future variant.
    [InlineData("img_emb.proj.0.weight", "condition_embedder.image_embedder.norm1.weight")]
    [InlineData("blocks.0.cross_attn.k_img.weight", "blocks.0.attn2.add_k_proj.weight")]
    [InlineData("blocks.0.cross_attn.v_img.weight", "blocks.0.attn2.add_v_proj.weight")]
    [InlineData("blocks.0.cross_attn.norm_k_img.weight", "blocks.0.attn2.norm_added_k.weight")]
    // ComfyUI repackage prefix.
    [InlineData("model.diffusion_model.blocks.0.self_attn.q.weight", "blocks.0.attn1.to_q.weight")]
    public void MapKey_OriginalNaming_MapsToDiffusers(string key, string expected)
    {
        Assert.Equal(expected, WanVideoCheckpointConverter.MapKey(key, fromOriginalNaming: true));
    }

    [Theory]
    // Diffusers-named checkpoints pass through untouched — especially norm2/norm3, which the rename
    // table would otherwise swap.
    [InlineData("blocks.0.attn1.to_q.weight")]
    [InlineData("blocks.0.norm2.weight")]
    [InlineData("blocks.0.ffn.net.0.proj.weight")]
    [InlineData("condition_embedder.time_embedder.linear_1.weight")]
    [InlineData("scale_shift_table")]
    [InlineData("proj_out.weight")]
    public void MapKey_DiffusersNaming_PassesThrough(string key)
    {
        Assert.Equal(key, WanVideoCheckpointConverter.MapKey(key, fromOriginalNaming: false));
    }

    [Theory]
    // Wan2.2-Animate keys must survive the original→diffusers rename chain UNCHANGED — WanAnimateTransformer
    // loads them under their original module-tree names. Pins them against future rule additions (e.g. a rule
    // touching "norm2"/"ffn.0"/"modulation" substrings must not hit these).
    [InlineData("pose_patch_embedding.weight")]
    [InlineData("pose_patch_embedding.bias")]
    [InlineData("motion_encoder.enc.net_app.convs.0.0.weight")]
    [InlineData("motion_encoder.enc.net_app.convs.0.1.bias")]
    [InlineData("motion_encoder.enc.net_app.convs.1.conv1.0.weight")]
    [InlineData("motion_encoder.enc.net_app.convs.1.conv2.0.kernel")]
    [InlineData("motion_encoder.enc.net_app.convs.1.conv2.1.weight")]
    [InlineData("motion_encoder.enc.net_app.convs.1.conv2.2.bias")]
    [InlineData("motion_encoder.enc.net_app.convs.1.skip.1.weight")]
    [InlineData("motion_encoder.enc.net_app.convs.8.weight")]
    [InlineData("motion_encoder.enc.fc.0.weight")]
    [InlineData("motion_encoder.enc.fc.4.bias")]
    [InlineData("motion_encoder.dec.direction.weight")]
    [InlineData("face_encoder.conv1_local.conv.weight")]
    [InlineData("face_encoder.conv2.conv.weight")]
    [InlineData("face_encoder.conv3.conv.bias")]
    [InlineData("face_encoder.out_proj.weight")]
    [InlineData("face_encoder.padding_tokens")]
    [InlineData("face_adapter.fuser_blocks.0.linear1_kv.weight")]
    [InlineData("face_adapter.fuser_blocks.7.linear1_q.bias")]
    [InlineData("face_adapter.fuser_blocks.3.linear2.weight")]
    [InlineData("face_adapter.fuser_blocks.0.q_norm.weight")]
    [InlineData("face_adapter.fuser_blocks.0.k_norm.weight")]
    [InlineData("ref_conv.weight")]
    [InlineData("ref_conv.bias")]
    public void MapKey_AnimateKeys_PassThroughUnchanged(string key)
    {
        Assert.Equal(key, WanVideoCheckpointConverter.MapKey(key, fromOriginalNaming: true));
        Assert.Equal(key, WanVideoCheckpointConverter.MapKey($"model.diffusion_model.{key}", fromOriginalNaming: true));
    }

    [Fact]
    public void MapKey_DropsFp8Metadata()
    {
        Assert.Null(WanVideoCheckpointConverter.MapKey("scaled_fp8", fromOriginalNaming: true));
        Assert.Null(WanVideoCheckpointConverter.MapKey("blocks.0.self_attn.q.scaled_fp8", fromOriginalNaming: true));
    }

    [Fact]
    public void IsOriginalNaming_DetectsBothFormats()
    {
        Assert.True(WanVideoCheckpointConverter.IsOriginalNaming(["blocks.0.self_attn.q.weight"]));
        Assert.True(WanVideoCheckpointConverter.IsOriginalNaming(["model.diffusion_model.time_embedding.0.weight"]));
        Assert.False(WanVideoCheckpointConverter.IsOriginalNaming(["blocks.0.attn1.to_q.weight"]));
        Assert.False(WanVideoCheckpointConverter.IsOriginalNaming(["condition_embedder.time_proj.weight"]));
    }
}
