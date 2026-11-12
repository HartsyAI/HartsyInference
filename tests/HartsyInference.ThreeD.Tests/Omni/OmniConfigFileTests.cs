using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.ThreeD.Models.Hunyuan3D.Omni;

namespace HartsyInference.ThreeD.Tests.Omni;

public sealed class OmniConfigFileTests
{
    // The three configs of tencent/Hunyuan3D-Omni at the pinned revision, trimmed to the keys the reader looks at.
    private const string Pipeline = """
        { "target": "hy3dshape.pipelines.Hunyuan3DOmniSiTFlowMatchingPipeline",
          "params": { "box_v": 1.01, "octree_resolution": 384, "num_chunks": 50000, "mc_level": 0.0, "output_type": "mesh", "num_inference_steps": 50, "scaling_factor": 1.0079887561230991 } }
        """;

    private const string Model = """
        { "target": "hy3dshape.models.denoisers.hunyuandit.HunYuanDiTPlain",
          "params": { "input_size": 4096, "in_channels": 64, "hidden_size": 2048, "context_dim": 1024, "depth": 21, "num_heads": 16, "qk_norm": true, "text_len": 1370,
                      "with_decoupled_ca": false, "use_attention_pooling": false, "qk_norm_type": "rms", "qkv_bias": false, "use_pos_emb": false,
                      "num_moe_layers": 6, "num_experts": 8, "moe_top_k": 2 } }
        """;

    private const string Vae = """
        { "target": "hy3dshape.models.autoencoders.shape_vae_pl.ShapeVAEPL",
          "params": { "module_cfg": { "target": "hy3dshape.models.autoencoders.shape_vae.ShapeVAE",
            "params": { "num_latents": 4096, "embed_dim": 64, "point_feats": 3, "sharpedge_label": true, "num_freqs": 8, "include_pi": false, "heads": 16, "width": 1024,
                        "num_encoder_layers": 8, "num_decoder_layers": 16, "use_ln_post": true, "qkv_bias": false, "qk_norm": true } } },
          "scale_factor": 1.0079887561230991 }
        """;

    [Fact]
    public void Parse_GivesThe21ArchitectureWithOmniDefaultsAndItsOwnVaeScale()
    {
        Hunyuan3DConfig config = OmniConfigFile.Parse(Pipeline, Model, Vae);

        Assert.Equal(4096, config.LatentTokens);
        Assert.Equal(64, config.LatentChannels);
        Assert.Equal(2048, config.Width);
        Assert.Equal(8192, config.MlpDim);
        Assert.Equal(16, config.NumHeads);
        Assert.Equal(1024, config.CondDim);
        Assert.Equal(new Hunyuan3DPlainDitConfig(21, 6, 8, 2), config.PlainDit);
        Assert.Equal(1024, config.VaeWidth);
        Assert.Equal(16, config.VaeDepth);
        Assert.Equal(1.0079887561230991f, config.VaeScaleFactor);
        Assert.Equal(50, config.NumInferenceSteps);
        Assert.Equal(7.5f, config.GuidanceScale);
        Assert.Equal(384, config.GridResolution);
        Assert.Equal(1.01f, config.BoundingBox);
        Assert.Equal(0f, config.IsoLevel);
        Assert.False(config.GuidanceEmbed);
    }

    [Theory]
    [InlineData("\"qk_norm_type\": \"rms\"", "\"qk_norm_type\": \"layer\"")]
    [InlineData("\"use_pos_emb\": false", "\"use_pos_emb\": true")]
    [InlineData("\"qkv_bias\": false", "\"qkv_bias\": true")]
    [InlineData("\"text_len\": 1370", "\"text_len\": 257")]
    [InlineData("\"input_size\": 4096", "\"input_size\": 3072")]
    [InlineData("HunYuanDiTPlain", "HunYuanDiT")]
    public void Parse_RefusesAModelConfigTheEngineDoesNotImplement(string from, string to)
    {
        Assert.Throws<UnsupportedModelException>(() => OmniConfigFile.Parse(Pipeline, Model.Replace(from, to), Vae));
    }

    [Fact]
    public void Parse_RefusesAScaleThePipelineDoesNotShareAndAVaeWithPiOrBias()
    {
        Assert.Throws<UnsupportedModelException>(() => OmniConfigFile.Parse(Pipeline.Replace("1.0079887561230991", "1.0"), Model, Vae));
        Assert.Throws<UnsupportedModelException>(() => OmniConfigFile.Parse(Pipeline, Model, Vae.Replace("\"include_pi\": false", "\"include_pi\": true")));
        Assert.Throws<UnsupportedModelException>(() => OmniConfigFile.Parse(Pipeline, Model, Vae.Replace("\"qkv_bias\": false", "\"qkv_bias\": true")));
        Assert.Throws<UnsupportedModelException>(() => OmniConfigFile.Parse(Pipeline.Replace("Hunyuan3DOmniSiTFlowMatchingPipeline", "Hunyuan3DDiTFlowMatchingPipeline"), Model, Vae));
    }

    [Fact]
    public void Parse_NamesAMissingKey()
    {
        UnsupportedModelException error = Assert.Throws<UnsupportedModelException>(() => OmniConfigFile.Parse(Pipeline, Model.Replace("\"depth\": 21,", string.Empty), Vae));
        Assert.Contains("depth", error.Message);
    }

    [Fact]
    public void Read_NamesTheMissingFile()
    {
        string folder = Path.Combine(Path.GetTempPath(), $"omni-{Guid.NewGuid():N}");
        Directory.CreateDirectory(folder);
        try
        {
            File.WriteAllText(Path.Combine(folder, "config.json"), Pipeline);
            FileNotFoundException error = Assert.Throws<FileNotFoundException>(() => OmniConfigFile.Read(folder));
            Assert.Contains("model/config.json", error.Message);
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
