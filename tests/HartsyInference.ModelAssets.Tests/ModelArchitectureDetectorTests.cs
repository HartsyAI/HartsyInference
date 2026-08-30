using HartsyInference.ModelAssets.Registry;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Architecture detection from representative tensor-key sets. Uses the distinctive prefix
/// signatures that each <c>*CheckpointConverter</c> keys off, so no real checkpoint is required.</summary>
public sealed class ModelArchitectureDetectorTests
{
    [Fact]
    public void Detects_Sdxl_FromDualClipEmbedders()
    {
        string[] keys =
        [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.label_emb.0.0.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "conditioner.embedders.1.model.token_embedding.weight",
            "first_stage_model.decoder.conv_in.weight",
        ];
        Assert.Equal(ModelArchitecture.Sdxl, ModelArchitectureDetector.Detect(keys));
    }

    [Fact]
    public void Detects_SdxlRefiner_FromSingleEmbedderAndLabelEmb()
    {
        string[] keys =
        [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.label_emb.0.0.weight",
            "conditioner.embedders.0.model.token_embedding.weight",
            "first_stage_model.decoder.conv_in.weight",
        ];
        // No conditioner.embedders.1 → refiner, not base.
        Assert.Equal(ModelArchitecture.SdxlRefiner, ModelArchitectureDetector.Detect(keys));
    }

    [Fact]
    public void Detects_Sd15_FromCondStageModel()
    {
        string[] keys =
        [
            "model.diffusion_model.input_blocks.0.0.weight",
            "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight",
            "first_stage_model.decoder.conv_in.weight",
        ];
        Assert.Equal(ModelArchitecture.StableDiffusion15, ModelArchitectureDetector.Detect(keys));
    }

    [Fact]
    public void Detects_Sd3_FromTextEncodersClipG()
    {
        string[] keys =
        [
            "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight",
            "model.diffusion_model.x_embedder.proj.weight",
            "text_encoders.clip_g.transformer.text_model.embeddings.token_embedding.weight",
        ];
        Assert.Equal(ModelArchitecture.StableDiffusion3, ModelArchitectureDetector.Detect(keys));
    }

    [Fact]
    public void Detects_Flux1_FromDoubleAndSingleBlocks()
    {
        string[] keys =
        [
            "double_blocks.0.img_attn.qkv.weight",
            "single_blocks.0.linear1.weight",
            "img_in.weight",
        ];
        Assert.Equal(ModelArchitecture.Flux1, ModelArchitectureDetector.Detect(keys));
    }

    [Fact]
    public void Detects_Flux1_WithDiffusionModelPrefix()
    {
        string[] keys =
        [
            "model.diffusion_model.double_blocks.0.img_attn.qkv.weight",
            "model.diffusion_model.single_blocks.0.linear1.weight",
        ];
        Assert.Equal(ModelArchitecture.Flux1, ModelArchitectureDetector.Detect(keys));
    }

    [Fact]
    public void Detects_Flux2_BeforeFlux1()
    {
        string[] keys =
        [
            "model.diffusion_model.double_blocks.0.img_attn.qkv.weight",
            "model.diffusion_model.single_blocks.0.linear1.weight",
            "model.diffusion_model.double_stream_modulation_img.lin.weight",
        ];
        // The Flux.2 modulation key must win even though double/single blocks are present.
        Assert.Equal(ModelArchitecture.Flux2, ModelArchitectureDetector.Detect(keys));
    }

    [Fact]
    public void Detects_AuraFlow_FromDoubleSingleLayers()
    {
        string[] keys =
        [
            "model.double_layers.0.attn.w1q.weight",
            "model.single_layers.0.attn.w1q.weight",
        ];
        Assert.Equal(ModelArchitecture.AuraFlow, ModelArchitectureDetector.Detect(keys));
    }

    [Fact]
    public void ReturnsUnknown_ForUnrecognizedKeys()
    {
        string[] keys = ["some.random.tensor", "another.weight"];
        Assert.Equal(ModelArchitecture.Unknown, ModelArchitectureDetector.Detect(keys));
    }

    [Theory]
    [InlineData("")]
    [InlineData("diffusion_model.")]
    [InlineData("model.diffusion_model.")]
    public void DetectsMiniMaxH3FromPairedPatchProjections(string prefix)
    {
        string[] keys =
        [
            prefix + "video_patch_proj.weight",
            prefix + "audio_patch_proj.weight",
            prefix + "blocks.0.attn.qkv_proj.weight",
        ];

        Assert.Equal(ModelArchitecture.MiniMaxH3, ModelArchitectureDetector.Detect(keys));
    }

    [Fact]
    public void MiniMaxH3RequiresBothPatchProjections()
    {
        Assert.Equal(ModelArchitecture.Unknown,
            ModelArchitectureDetector.Detect(["video_patch_proj.weight", "blocks.0.attn.qkv_proj.weight"]));
    }
}
