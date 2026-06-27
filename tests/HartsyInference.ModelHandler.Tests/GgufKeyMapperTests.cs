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
    public void GetByArchitecture_Qwen2AndQwen3ResolveToLlamaFamilyMapper()
    {
        // llama.cpp emits an identical tensor dialect for llama/qwen2/qwen3 dense decoders, so the one mapper
        // declares all three. These must resolve by NAME (no key-heuristic fallback, no warning) and map the
        // QKV bias that Qwen2 carries.
        IGgufKeyMapper? llama = GgufKeyMapperRegistry.GetByArchitecture("llama");
        IGgufKeyMapper? qwen2 = GgufKeyMapperRegistry.GetByArchitecture("qwen2");
        IGgufKeyMapper? qwen3 = GgufKeyMapperRegistry.GetByArchitecture("qwen3");
        Assert.NotNull(llama);
        Assert.Same(llama, qwen2);
        Assert.Same(llama, qwen3);
        Assert.Equal("model.layers.0.self_attn.q_proj.bias", qwen2!.MapKey("blk.0.attn_q.bias"));
    }

    [Fact]
    public void GetByArchitecture_GemmaFamilyResolvesToGemmaMapper_WithSandwichNormKeys()
    {
        IGgufKeyMapper? gemma = GgufKeyMapperRegistry.GetByArchitecture("gemma");
        IGgufKeyMapper? gemma2 = GgufKeyMapperRegistry.GetByArchitecture("gemma2");
        IGgufKeyMapper? gemma3 = GgufKeyMapperRegistry.GetByArchitecture("gemma3");
        Assert.NotNull(gemma);
        Assert.Same(gemma, gemma2);
        Assert.Same(gemma, gemma3);
        // Gemma's sandwich + Q/K norms map to the HF names the transformer loader expects.
        Assert.Equal("model.layers.0.post_attention_layernorm.weight", gemma3!.MapKey("blk.0.post_attention_norm.weight"));
        Assert.Equal("model.layers.0.pre_feedforward_layernorm.weight", gemma3.MapKey("blk.0.ffn_norm.weight"));
        Assert.Equal("model.layers.0.post_feedforward_layernorm.weight", gemma3.MapKey("blk.0.post_ffw_norm.weight"));
        Assert.Equal("model.layers.0.self_attn.q_norm.weight", gemma3.MapKey("blk.0.attn_q_norm.weight"));
    }

    [Fact]
    public void GetByArchitecture_Phi3ResolvesToPhiMapper_WithFusedKeys()
    {
        IGgufKeyMapper? phi = GgufKeyMapperRegistry.GetByArchitecture("phi3");
        Assert.NotNull(phi);
        Assert.Equal("phi3", phi!.Architecture);
        // Phi-3 fuses qkv and gate+up; the mapper routes them to fused names that the loader splits downstream.
        Assert.Equal("model.layers.0.self_attn.qkv_proj.weight", phi.MapKey("blk.0.attn_qkv.weight"));
        Assert.Equal("model.layers.0.mlp.gate_up_proj.weight", phi.MapKey("blk.0.ffn_up.weight"));
        Assert.Equal("model.rope_factors_long.weight", phi.MapKey("rope_factors_long.weight"));
    }

    [Fact]
    public void GetByArchitecture_MoeArchesResolveToLlamaMapper_WithExpertKeys()
    {
        IGgufKeyMapper? llama = GgufKeyMapperRegistry.GetByArchitecture("llama");
        foreach (string moeArch in new[] { "olmoe", "qwen2moe", "qwen3moe" })
            Assert.Same(llama, GgufKeyMapperRegistry.GetByArchitecture(moeArch));
        // Router + stacked-expert + shared-expert tensors map to the names the MoE block / split expect.
        Assert.Equal("model.layers.0.mlp.gate.weight", llama!.MapKey("blk.0.ffn_gate_inp.weight"));
        Assert.Equal("model.layers.0.mlp.gate_exps.weight", llama.MapKey("blk.0.ffn_gate_exps.weight"));
        Assert.Equal("model.layers.0.mlp.down_exps.weight", llama.MapKey("blk.0.ffn_down_exps.weight"));
        Assert.Equal("model.layers.0.mlp.shared_expert.up_proj.weight", llama.MapKey("blk.0.ffn_up_shexp.weight"));
    }

    [Fact]
    public void GetByArchitecture_Deepseek2ResolvesToDeepSeekMapper_WithMlaKeys()
    {
        IGgufKeyMapper? ds = GgufKeyMapperRegistry.GetByArchitecture("deepseek2");
        Assert.NotNull(ds);
        Assert.Equal("deepseek2", ds!.Architecture);
        // MLA + DeepSeek-MoE tensors map to the names the transformer's MLA + MoE paths expect.
        Assert.Equal("model.layers.0.self_attn.kv_a_proj.weight", ds.MapKey("blk.0.attn_kv_a_mqa.weight"));
        Assert.Equal("model.layers.0.self_attn.kv_a_norm.weight", ds.MapKey("blk.0.attn_kv_a_norm.weight"));
        Assert.Equal("model.layers.0.self_attn.kv_b_proj.weight", ds.MapKey("blk.0.attn_kv_b.weight"));
        Assert.Equal("model.layers.0.self_attn.q_proj.weight", ds.MapKey("blk.0.attn_q.weight"));
        Assert.Equal("model.layers.0.mlp.shared_expert.down_proj.weight", ds.MapKey("blk.0.ffn_down_shexp.weight"));
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
