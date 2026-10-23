using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Comfy-Org FLUX.2 VAE repack uses ComfyUI's ldm names; the loader renames them to the diffusers layout.</summary>
public sealed class Flux2VaeLayoutTests
{
    [Fact]
    public void LdmLayout_IsRenamedAndAttentionProjectionsFlattened()
    {
        Dictionary<string, Tensor> ldm = new()
        {
            ["bn.running_mean"] = new Tensor(new TensorShape(4), DType.F32),
            ["decoder.mid.attn_1.q.weight"] = new Tensor(new TensorShape(3, 3, 1, 1), DType.F32),
            ["decoder.mid.attn_1.norm.weight"] = new Tensor(new TensorShape(3), DType.F32),
            ["decoder.post_quant_conv.weight"] = new Tensor(new TensorShape(2, 2, 1, 1), DType.F32),
            ["encoder.quant_conv.weight"] = new Tensor(new TensorShape(4, 4, 1, 1), DType.F32),
            ["decoder.up.0.block.0.nin_shortcut.weight"] = new Tensor(new TensorShape(2, 4, 1, 1), DType.F32),
            ["decoder.norm_out.weight"] = new Tensor(new TensorShape(2), DType.F32),
        };

        Dictionary<string, Tensor> converted = Flux2VaeComponents.ToDiffusersLayout(ldm);

        Assert.Equal(new TensorShape(3, 3), converted["decoder.mid_block.attentions.0.to_q.weight"].Shape);
        Assert.Contains("decoder.mid_block.attentions.0.group_norm.weight", converted.Keys);
        Assert.Contains("post_quant_conv.weight", converted.Keys);
        Assert.Contains("quant_conv.weight", converted.Keys);
        Assert.Contains("bn.running_mean", converted.Keys);
        Assert.Contains("decoder.up_blocks.3.resnets.0.conv_shortcut.weight", converted.Keys);
        Assert.Contains("decoder.conv_norm_out.weight", converted.Keys);
    }

    [Fact]
    public void DiffusersLayout_PassesThroughUnchanged()
    {
        Dictionary<string, Tensor> weights = new() { ["post_quant_conv.weight"] = new Tensor(new TensorShape(2, 2), DType.F32) };

        Assert.Same(weights, Flux2VaeComponents.ToDiffusersLayout(weights));
    }
}
