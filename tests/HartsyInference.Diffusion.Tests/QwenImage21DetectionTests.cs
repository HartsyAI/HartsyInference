using HartsyInference.Engine.Recipes;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Registry;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Qwen-Image 2.1 and Qwen-Image v1 share the <c>transformer_blocks</c> naming but not the architecture,
/// so each family has to refuse the other's checkpoint by name rather than load it into the wrong denoiser. These
/// pin the signature both directions, and that the architecture the detector returns actually resolves to a
/// registered recipe — a detector rule whose family id nothing matches is dead code that reads as support.</summary>
public sealed class QwenImage21DetectionTests
{
    /// <summary>The released Comfy-Org 2.1 diffusion file's distinguishing keys.</summary>
    private static string[] Keys21 =>
    [
        "img_in.weight", "modulation.1.weight", "norm_out.linear.weight", "proj_out.weight",
        "time_text_embed.timestep_embedder.linear_1.weight", "time_text_embed.timestep_embedder.linear_2.weight",
        "transformer_blocks.0.attn.norm_k.weight", "transformer_blocks.0.attn.norm_q.weight",
        "transformer_blocks.0.attn.to_k.weight", "transformer_blocks.0.attn.to_out.0.weight",
        "transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.attn.to_v.weight",
        "transformer_blocks.0.img_mlp.gate_up.weight", "transformer_blocks.0.img_mlp.out.weight",
        "txt_in.in_layer.weight", "txt_in.out_layer.weight", "txt_in.text_norm.weight",
    ];

    /// <summary>Qwen-Image v1: dual-stream, biased, with a plain <c>txt_norm</c> and no shared modulation.</summary>
    private static string[] KeysV1 =>
    [
        "img_in.weight", "img_in.bias", "txt_norm.weight", "txt_in.weight", "txt_in.bias",
        "time_text_embed.timestep_embedder.linear_1.weight", "time_text_embed.timestep_embedder.linear_1.bias",
        "transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.attn.to_q.bias",
        "transformer_blocks.0.attn.add_k_proj.weight", "transformer_blocks.0.attn.add_k_proj.bias",
        "transformer_blocks.0.img_mlp.net.0.proj.weight", "transformer_blocks.0.img_mod.1.weight",
    ];

    [Fact]
    public void TheSignatureMatchesOnlyTheTwoPointOneCheckpoint()
    {
        Assert.True(QwenImage21CheckpointConverter.MatchesByKeys(Keys21));
        Assert.False(QwenImage21CheckpointConverter.MatchesByKeys(KeysV1));
    }

    /// <summary>A community repack may wrap the bare keys; the signature has to survive that or the file reads as
    /// an unrelated architecture.</summary>
    [Fact]
    public void TheSignatureSurvivesADiffusionModelWrapperPrefix()
    {
        string[] wrapped = [.. Keys21.Select(k => "model.diffusion_model." + k)];
        Assert.True(QwenImage21CheckpointConverter.MatchesByKeys(wrapped));
    }

    /// <summary>Detection must land on a family id some recipe answers to. The engine's enum-to-family fallback
    /// lowercases the enum name, which would give "qwenimage21" — a string nothing matches, so the checkpoint
    /// would be reported unsupported after being correctly identified.</summary>
    [Fact]
    public void TheDetectedArchitectureResolvesToARegisteredRecipe()
    {
        Assert.Equal(ModelArchitecture.QwenImage21, ModelArchitectureDetector.Detect(Keys21));
        Assert.NotEqual(ModelArchitecture.QwenImage21, ModelArchitectureDetector.Detect(KeysV1));

        IArchitectureRecipe? recipe = RecipeRegistry.Resolve("qwen-image-2.1");
        Assert.NotNull(recipe);
        Assert.Equal("qwen-image-2.1", recipe!.Name);
        Assert.Null(RecipeRegistry.Resolve(ModelArchitecture.QwenImage21.ToString().ToLowerInvariant()));
    }
}
