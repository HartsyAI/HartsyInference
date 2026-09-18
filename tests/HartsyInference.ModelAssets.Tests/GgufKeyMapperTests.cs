using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Gguf.KeyMappers;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

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

    /// <summary>The precedence rules that the family table's ordering exists for. Each of these checkpoints also
    /// satisfies a broader family's signature, and losing the ordering does not throw — it loads the file as another
    /// architecture's weights, which renders as a black image.</summary>
    [Theory]
    // Radiance is classic Chroma plus the pixel-space NeRF head, so it matches Chroma's signature too.
    [InlineData("chroma-radiance", "distilled_guidance_layer.0.weight", "nerf_blocks.0.weight", "double_blocks.0.img_attn.qkv.weight")]
    // Zeta is the Z-Image DiT with a dec_net head, so it matches Z-Image's refiner signature too.
    [InlineData("zeta-chroma", "noise_refiner.0.weight", "context_refiner.0.weight", "dec_net.0.weight")]
    // The Tencent Hunyuan Image repack carries Flux's double+single block naming; byt5_in is what separates it,
    // and from HunyuanVideo, which shares the same block naming again.
    [InlineData("hunyuan_image", "double_blocks.0.img_attn_qkv.weight", "single_blocks.0.linear1.weight", "byt5_in.proj.weight")]
    [InlineData("hunyuan_image", "transformer_blocks.0.dual_attention.to_q.weight", "transformer_blocks.0.ff.net.0.weight", "x_embedder.weight")]
    [InlineData("flux2", "double_stream_modulation_img.weight", "double_blocks.0.mlp.linear_in.weight", "single_blocks.0.linear1.weight")]
    [InlineData("ernie_image", "shared_adaLN_modulation.1.weight", "transformer_blocks.0.mlp.gate_proj.weight", "patch_embed.proj.weight")]
    [InlineData("qwen_image", "transformer_blocks.0.attn.add_q_proj.weight", "transformer_blocks.0.ff.net.0.proj.weight", "img_in.weight")]
    // Wan shares blocks.N.self_attn.* with F-Lite, which is asked first and claimed by its register_tokens.
    [InlineData("wan", "blocks.0.self_attn.q.weight", "blocks.0.cross_attn.k.weight", "patch_embedding.weight")]
    public void DetectByKeys_PrefersTheSpecificFamilyOverTheBroaderOneItAlsoMatches(string expected, params string[] keys)
    {
        Assert.Equal(expected, GgufKeyMapperRegistry.DetectByKeys(keys).Architecture);
    }

    /// <summary>Wan's published GGUFs declare <c>general.architecture = "wan"</c>, so the direct lookup is the path
    /// they actually take; the key heuristic only covers a repack that declared nothing.</summary>
    [Fact]
    public void GetByArchitecture_ResolvesTheNameWansGgufsDeclare()
    {
        Assert.Equal("wan", GgufKeyMapperRegistry.GetByArchitecture("wan")!.Architecture);
    }

    [Theory]
    // The shipped repacks are of the ComfyUI single file, in original-Wan naming, with or without its prefix.
    [InlineData("blocks.0.self_attn.q.weight", "blocks.0.cross_attn.v.weight", "patch_embedding.weight")]
    [InlineData("model.diffusion_model.blocks.39.self_attn.o.weight", "model.diffusion_model.blocks.39.cross_attn.k_img.weight",
        "model.diffusion_model.patch_embedding.weight")]
    // The conditioning builds keep the backbone's naming and add their own embeds.
    [InlineData("blocks.0.self_attn.q.weight", "blocks.0.cross_attn.k.weight", "vace_patch_embedding.weight", "patch_embedding.weight")]
    [InlineData("blocks.0.self_attn.q.weight", "blocks.0.cross_attn.k.weight", "pose_patch_embedding.weight", "patch_embedding.weight")]
    // The Wan-AI diffusers export renames the attentions but keeps patch_embedding and the condition embedder.
    [InlineData("blocks.0.attn1.to_q.weight", "condition_embedder.time_embedder.linear_1.weight", "patch_embedding.weight")]
    public void DetectByKeys_WanFromItsBackboneNaming(params string[] keys)
    {
        Assert.Equal("wan", GgufKeyMapperRegistry.DetectByKeys(keys).Architecture);
    }

    /// <summary>The other half of the Wan/F-Lite pairing: Wan's row must not claim an F-Lite file either. It cannot,
    /// because F-Lite has no cross-attention and spells its patch embed <c>patch_embed.</c> — but the two rows sit
    /// next to each other and a later edit to one is exactly how that stops being true.</summary>
    [Fact]
    public void DetectByKeys_WanDoesNotClaimAnFLiteFile()
    {
        string[] fLite = ["register_tokens", "blocks.0.self_attn.qkv.weight", "patch_embed.patch_proj.weight"];
        Assert.Equal("flite", GgufKeyMapperRegistry.DetectByKeys(fLite).Architecture);
        // And without the register_tokens that claims it, it still falls through Wan rather than into it.
        Assert.NotEqual("wan", GgufKeyMapperRegistry.DetectByKeys(fLite[1..]).Architecture);
    }

    /// <summary>HunyuanVideo's token refiner has <c>blocks.N.self_attn_qkv</c> — no trailing dot — which is why the
    /// Wan row's markers carry both dots. Without them a HunyuanVideo GGUF that also happened to satisfy the other
    /// two markers would load as Wan weights.</summary>
    [Fact]
    public void DetectByKeys_WanIgnoresHunyuanVideosUnderscoredSelfAttention()
    {
        string[] hunyuanVideo =
        [
            "txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight",
            "double_blocks.0.img_attn_qkv.weight",
            "single_blocks.0.linear1.weight",
        ];
        Assert.NotEqual("wan", GgufKeyMapperRegistry.DetectByKeys(hunyuanVideo).Architecture);
    }

    [Fact]
    public void DiffusionFamilies_MapEveryKeyToItself()
    {
        // A diffusion GGUF is a repack of the safetensors build, so the whole point of these mappers is that they do
        // nothing to the keys; a family that started rewriting them would break the converter that reads them.
        foreach (IGgufKeyMapper family in DiffusionGgufFamilies.InDetectionOrder)
        {
            Assert.Equal("blocks.0.attn.to_q.weight", family.MapKey("blocks.0.attn.to_q.weight"));
        }
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
