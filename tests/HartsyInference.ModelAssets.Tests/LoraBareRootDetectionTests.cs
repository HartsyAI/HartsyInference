using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Pins the bare-root arm of <see cref="LoraFormatDetector"/>: a file whose keys carry a PEFT or kohya role
/// suffix and no recognized wrapper prefix is <see cref="LoraFormat.DiffusersBareDit"/> — roots are canonical.
/// <para>This arm used to be an allow-list of block roots, which silently rejected every family naming its blocks
/// something other than <c>blocks.</c>/<c>transformer_blocks.</c>. The failure was quiet in the worst way: the LoRA was
/// refused at load as an unrecognized format, which reads as "bad LoRA file" rather than "unwired architecture". The
/// per-family cases below are the actual roots each transformer's <c>LoadWeights</c> uses, so a future narrowing of the
/// rule fails here instead of in a user's generation.</para></summary>
public sealed class LoraBareRootDetectionTests
{
    private static SafeTensorDescriptor Desc(string name) => new()
    {
        Name = name,
        DType = DType.F16,
        Shape = new TensorShape(16, 2048),
        DataOffset = 0,
        ByteLength = 16 * 2048 * 2,
    };

    private static Dictionary<string, SafeTensorDescriptor> Descriptors(params string[] keys)
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = [];
        foreach (string key in keys)
        {
            descriptors[key] = Desc(key);
        }
        return descriptors;
    }

    /// <summary>Each family's canonical block root, taken from its transformer's own <c>LoadWeights</c> prefix.</summary>
    [Theory]
    // Ideogram 4 / ERNIE-Image / Lance-Image: Ideogram4Transformer.cs, ErnieImageTransformer.cs, LanceTransformer.cs.
    [InlineData("layers.0.attn.to_q")]
    // Kandinsky 5: Kandinsky5Transformer.cs builds two block stacks under distinct roots.
    [InlineData("text_transformer_blocks.0.self_attn.to_q")]
    // AuraFlow: AuraFlowTransformer.cs.
    // Boogu-Image: BooguImageTransformer.cs.
    [InlineData("double_stream_layers.0.attn.qkv")]
    // The roots the old allow-list already covered — kept so widening the rule can't drop them.
    [InlineData("transformer_blocks.0.attn.to_q")]
    public void BareRoot_WithPeftSuffix_IsBareDit(string root)
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = Descriptors(
            $"{root}.lora_A.weight",
            $"{root}.lora_B.weight");
        Assert.Equal(LoraFormat.DiffusersBareDit, LoraFormatDetector.Detect(descriptors));
    }

    /// <summary>The kohya spelling of the same two roles is accepted on the same roots (lightx2v Lightning LoRAs).</summary>
    [Fact]
    public void BareRoot_WithKohyaSuffix_IsBareDit()
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = Descriptors(
            "layers.0.attn.to_q.lora_down.weight",
            "layers.0.attn.to_q.lora_up.weight");
        Assert.Equal(LoraFormat.DiffusersBareDit, LoraFormatDetector.Detect(descriptors));
    }

    /// <summary>Bare-root stays LAST in precedence. Each of these files has keys that would satisfy the bare-root rule
    /// on their own if the wrapper exclusion were dropped, so a precedence regression shows up here rather than as a
    /// mis-mapped merge.</summary>
    [Theory]
    [InlineData("transformer.transformer_blocks.0.attn.to_q", LoraFormat.DiffusersFlux)]
    [InlineData("diffusion_model.double_blocks.0.img_attn.qkv", LoraFormat.ComfyBflDit)]
    [InlineData("diffusion_model.blocks.0.self_attn.q", LoraFormat.DiffusersWan)]
    [InlineData("lora_transformer_single_transformer_blocks_0_attn_to_q", LoraFormat.AiToolkitFlux)]
    [InlineData("lora_unet_double_blocks_0_img_attn_qkv", LoraFormat.KohyaFlux)]
    public void WrapperPrefix_WinsOverBareRoot(string root, LoraFormat expected)
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = Descriptors(
            $"{root}.lora_A.weight",
            $"{root}.lora_B.weight");
        Assert.Equal(expected, LoraFormatDetector.Detect(descriptors));
    }

    /// <summary>Bare original-Wan naming keeps winning over the generic fallback: its <c>self_attn</c>/<c>cross_attn</c>
    /// segments mean the roots are Wan module names, which need translating, not passing through as canonical.</summary>
    [Fact]
    public void BareWanNaming_StaysWan_NotBareDit()
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = Descriptors(
            "blocks.0.self_attn.q.lora_A.weight",
            "blocks.0.self_attn.q.lora_B.weight",
            "blocks.0.cross_attn.k.lora_A.weight",
            "blocks.0.cross_attn.k.lora_B.weight");
        Assert.Equal(LoraFormat.DiffusersWan, LoraFormatDetector.Detect(descriptors));
    }

    /// <summary>A LoRA role suffix is required — the fallback must not claim an arbitrary safetensors file. Without this
    /// the rule would degrade from "weakest marker" to "no marker".</summary>
    [Fact]
    public void NoLoraSuffix_IsStillUnknown()
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = Descriptors(
            "layers.0.attn.to_q.weight",
            "layers.0.attn.to_k.weight");
        Assert.Equal(LoraFormat.Unknown, LoraFormatDetector.Detect(descriptors));
    }
}
