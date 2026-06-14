using HartsyInference.ModelHandler.Gguf;
using Xunit;

namespace HartsyInference.ModelHandler.Tests;

/// <summary>Tests the registry + per-architecture key-detection heuristics. End-to-end <see cref="GgufModelLoader.Load"/> integration against a synthetic GGUF file is covered by <see cref="GgufLoaderTests"/> + a separate integration test once a fixture is added.</summary>
public sealed class GgufKeyMapperTests
{
    [Fact]
    public void Registry_AllArchitecturesPresent()
    {
        IReadOnlyCollection<string> archs = GgufKeyMapperRegistry.Architectures;
        Assert.Contains("flux", archs);
        Assert.Contains("sdxl", archs);
        Assert.Contains("sd3", archs);
        Assert.Contains("sd15", archs);
        Assert.Contains("flite", archs);
        Assert.Contains("chroma", archs);
        Assert.Contains("auraflow", archs);
        Assert.Contains("zimage", archs);
        Assert.Contains("passthrough", archs);
    }

    [Fact]
    public void GetByArchitecture_FluxReturnsFluxMapper()
    {
        IGgufKeyMapper? m = GgufKeyMapperRegistry.GetByArchitecture("flux");
        Assert.NotNull(m);
        Assert.Equal("flux", m.Architecture);
    }

    [Fact]
    public void GetByArchitecture_CaseInsensitive()
    {
        Assert.NotNull(GgufKeyMapperRegistry.GetByArchitecture("FLUX"));
        Assert.NotNull(GgufKeyMapperRegistry.GetByArchitecture("Sdxl"));
        Assert.NotNull(GgufKeyMapperRegistry.GetByArchitecture("ZIMAGE"));
    }

    [Fact]
    public void GetByArchitecture_UnknownReturnsNull()
    {
        Assert.Null(GgufKeyMapperRegistry.GetByArchitecture("nonexistent_arch_2099"));
    }

    [Fact]
    public void DetectByKeys_FluxFromBlockNames()
    {
        string[] keys =
        [
            "model.diffusion_model.double_blocks.0.img_attn.qkv.weight",
            "model.diffusion_model.single_blocks.0.linear1.weight",
            "model.diffusion_model.img_in.weight",
        ];
        IGgufKeyMapper m = GgufKeyMapperRegistry.DetectByKeys(keys);
        Assert.Equal("flux", m.Architecture);
    }

    [Fact]
    public void DetectByKeys_SdxlFromLabelEmb()
    {
        string[] keys =
        [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.label_emb.0.0.weight",
        ];
        IGgufKeyMapper m = GgufKeyMapperRegistry.DetectByKeys(keys);
        Assert.Equal("sdxl", m.Architecture);
    }

    [Fact]
    public void DetectByKeys_Sd15WhenNoLabelEmb()
    {
        string[] keys =
        [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.middle_block.0.in_layers.0.weight",
        ];
        IGgufKeyMapper m = GgufKeyMapperRegistry.DetectByKeys(keys);
        Assert.Equal("sd15", m.Architecture);
    }

    [Fact]
    public void DetectByKeys_Sd3FromJointBlocks()
    {
        string[] keys =
        [
            "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight",
        ];
        IGgufKeyMapper m = GgufKeyMapperRegistry.DetectByKeys(keys);
        Assert.Equal("sd3", m.Architecture);
    }

    [Fact]
    public void DetectByKeys_ChromaFromDistilledGuidance()
    {
        string[] keys =
        [
            "model.diffusion_model.double_blocks.0.img_attn.qkv.weight",
            "model.diffusion_model.distilled_guidance_layer.in_proj.weight",
        ];
        IGgufKeyMapper m = GgufKeyMapperRegistry.DetectByKeys(keys);
        Assert.Equal("chroma", m.Architecture);
    }

    [Fact]
    public void DetectByKeys_AuraFlowFromDoubleLayers()
    {
        string[] keys =
        [
            "double_layers.0.attn.w2q.weight",
            "modF.1.weight",
        ];
        IGgufKeyMapper m = GgufKeyMapperRegistry.DetectByKeys(keys);
        Assert.Equal("auraflow", m.Architecture);
    }

    [Fact]
    public void DetectByKeys_ZImageFromRefiners()
    {
        string[] keys =
        [
            "model.diffusion_model.noise_refiner.0.attention.qkv.weight",
            "model.diffusion_model.context_refiner.0.attention.qkv.weight",
        ];
        IGgufKeyMapper m = GgufKeyMapperRegistry.DetectByKeys(keys);
        Assert.Equal("zimage", m.Architecture);
    }

    [Fact]
    public void DetectByKeys_FLiteFromRegisterTokens()
    {
        string[] keys =
        [
            "register_tokens",
            "blocks.0.self_attn.qkv.weight",
            "patch_embed.patch_proj.weight",
        ];
        IGgufKeyMapper m = GgufKeyMapperRegistry.DetectByKeys(keys);
        Assert.Equal("flite", m.Architecture);
    }

    [Fact]
    public void DetectByKeys_UnknownFallsBackToPassthrough()
    {
        string[] keys =
        [
            "totally.fictional.key",
            "another.weird.tensor",
        ];
        IGgufKeyMapper m = GgufKeyMapperRegistry.DetectByKeys(keys);
        Assert.Equal("passthrough", m.Architecture);
    }

    [Fact]
    public void Passthrough_PreservesKeyVerbatim()
    {
        IGgufKeyMapper m = GgufKeyMapperRegistry.GetByArchitecture("passthrough")!;
        Assert.Equal("model.diffusion_model.foo.bar", m.MapKey("model.diffusion_model.foo.bar"));
    }
}
