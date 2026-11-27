using HartsyInference.Core.Exceptions;
using HartsyInference.ThreeD.Models.Hunyuan3D;

namespace HartsyInference.ThreeD.Tests.Models.Hunyuan3D;

public sealed class Hunyuan3DConfigFileTests
{
    [Fact]
    public void Parse_ReadsTheBaseCheckpointConfig()
    {
        Hunyuan3DConfig config = Hunyuan3DConfigFile.Parse(Yaml());

        Hunyuan3DConfig expected = Hunyuan3DConfig.Hunyuan3D2;
        Assert.Equal(expected.DepthDouble, config.DepthDouble);
        Assert.Equal(expected.DepthSingle, config.DepthSingle);
        Assert.Equal(expected.LatentTokens, config.LatentTokens);
        Assert.Equal(expected.MlpDim, config.MlpDim);
        Assert.Equal(expected.VaeScaleFactor, config.VaeScaleFactor);
        Assert.False(config.GuidanceEmbed);
        Assert.Equal(Hunyuan3DScheduler.FlowMatchEuler, config.Scheduler);
        Assert.Equal(50, config.NumInferenceSteps);
        Assert.Equal(1, config.ViewCount);
    }

    [Fact]
    public void Parse_ReadsTheMiniDimensions()
    {
        Hunyuan3DConfig config = Hunyuan3DConfigFile.Parse(Yaml(depth: 8, single: 16, latents: 512, scale: "1.0188137142395404"));

        Assert.Equal(8, config.DepthDouble);
        Assert.Equal(16, config.DepthSingle);
        Assert.Equal(512, config.LatentTokens);
        Assert.Equal(1.0188137142395404f, config.VaeScaleFactor);
    }

    [Fact]
    public void Parse_ReadsGuidanceEmbedAndTheConsistencyScheduler()
    {
        Hunyuan3DConfig fast = Hunyuan3DConfigFile.Parse(Yaml(extraModel: "    guidance_embed: true\n"));
        Hunyuan3DConfig turbo = Hunyuan3DConfigFile.Parse(Yaml(extraModel: "    guidance_embed: true\n", consistency: true));

        Assert.True(fast.GuidanceEmbed);
        Assert.Equal(Hunyuan3DScheduler.FlowMatchEuler, fast.Scheduler);
        Assert.Equal(10, fast.NumInferenceSteps);
        Assert.True(turbo.GuidanceEmbed);
        Assert.Equal(Hunyuan3DScheduler.ConsistencyFlowMatchEuler, turbo.Scheduler);
        Assert.Equal(100, turbo.PcmTimesteps);
        Assert.Equal(5, turbo.NumInferenceSteps);
    }

    [Fact]
    public void Parse_MultiviewEncoderGivesFourViews()
    {
        Hunyuan3DConfig config = Hunyuan3DConfigFile.Parse(Yaml(encoder: "DinoImageEncoderMV"));

        Assert.Equal(4, config.ViewCount);
    }

    [Fact]
    public void Parse_RefusesADinoInputSizeItCannotInterpolateTo()
    {
        Assert.Throws<UnsupportedModelException>(() => Hunyuan3DConfigFile.Parse(Yaml(dinoSize: "1022", processorSize: "1022")));
    }

    [Fact]
    public void Parse_RefusesAPreprocessorSizeOtherThan512()
    {
        Assert.Throws<UnsupportedModelException>(() => Hunyuan3DConfigFile.Parse(Yaml(processorSize: "1022")));
    }

    [Fact]
    public void Parse_NamesAMissingKey()
    {
        UnsupportedModelException error = Assert.Throws<UnsupportedModelException>(
            () => Hunyuan3DConfigFile.Parse(Yaml().Replace("depth_single_blocks:", "other_key:")));

        Assert.Contains("depth_single_blocks", error.Message);
    }

    [Fact]
    public void ParseVae_ReadsTheTurboGeoDecoder()
    {
        const string turbo = "target: hy3dgen.shapegen.models.ShapeVAE\nparams:\n  num_latents: 3072\n  embed_dim: 64\n  num_freqs: 8\n"
            + "  include_pi: false\n  heads: 16\n  width: 1024\n  num_decoder_layers: 16\n  qkv_bias: false\n  qk_norm: true\n"
            + "  scale_factor: 0.9990943042622529\n  geo_decoder_mlp_expand_ratio: 1\n  geo_decoder_downsample_ratio: 2\n  geo_decoder_ln_post: false";

        Hunyuan3DConfig config = Hunyuan3DConfigFile.ParseVae(turbo, Hunyuan3DConfig.Hunyuan3D2);

        Assert.Equal(2, config.GeoDownsample);
        Assert.Equal(1, config.GeoMlpRatio);
        Assert.False(config.GeoLnPost);
        Assert.Equal(1024, config.VaeWidth);
    }

    [Fact]
    public void Parse_ReadsThePlainDitOfHunyuan3D21()
    {
        Hunyuan3DConfig config = Hunyuan3DConfigFile.Parse(PlainYaml);

        Assert.Equal(new Hunyuan3DPlainDitConfig(21, 6, 8, 2), config.PlainDit);
        Assert.Equal(2048, config.Width);
        Assert.Equal(16, config.NumHeads);
        Assert.Equal(64, config.LatentChannels);
        Assert.Equal(1024, config.CondDim);
        Assert.Equal(4096, config.LatentTokens);
        Assert.Equal(1.0039506158752403f, config.VaeScaleFactor);
        Assert.False(config.GuidanceEmbed);
    }

    [Fact]
    public void Parse_ResolvesTheYamlAnchorThatSharesTheLatentCount()
    {
        string anchored = PlainYaml.Replace("input_size: 4096", "input_size: &num_latents 4096").Replace("num_latents: 4096", "num_latents: *num_latents");

        Assert.Equal(4096, Hunyuan3DConfigFile.Parse(anchored).LatentTokens);
    }

    [Fact]
    public void Parse_RefusesAPlainDitWithDecoupledCrossAttention()
    {
        Assert.Throws<UnsupportedModelException>(
            () => Hunyuan3DConfigFile.Parse(PlainYaml.Replace("with_decoupled_ca: false", "with_decoupled_ca: true")));
    }

    [Fact]
    public void Parse_AcceptsADoubleQuotedRmsNormAndReadsTheMlpRatio()
    {
        string yaml = PlainYaml.Replace("qk_norm_type: 'rms'", "qk_norm_type: \"rms\"").Replace("moe_top_k: 2", "moe_top_k: 2\n    mlp_ratio: 2.0");

        Hunyuan3DConfig config = Hunyuan3DConfigFile.Parse(yaml);

        Assert.Equal(4096, config.MlpDim);
    }

    [Fact]
    public void Parse_RefusesAPlainDitWithAnotherNormOrAGuidanceProjection()
    {
        Assert.Throws<UnsupportedModelException>(() => Hunyuan3DConfigFile.Parse(PlainYaml.Replace("'rms'", "'layer'")));
        Assert.Throws<UnsupportedModelException>(
            () => Hunyuan3DConfigFile.Parse(PlainYaml.Replace("moe_top_k: 2", "moe_top_k: 2\n    guidance_cond_proj_dim: 256")));
    }

    private const string PlainYaml =
        """
        model:
          target: hy3dshape.models.denoisers.hunyuandit.HunYuanDiTPlain
          params:
            input_size: 4096
            in_channels: 64
            hidden_size: 2048
            context_dim: 1024
            depth: 21
            num_heads: 16
            qk_norm: true
            text_len: 1370
            with_decoupled_ca: false
            use_attention_pooling: false
            qk_norm_type: 'rms'
            qkv_bias: false
            use_pos_emb: false
            num_moe_layers: 6
            num_experts: 8
            moe_top_k: 2
        vae:
          target: hy3dshape.models.autoencoders.ShapeVAE
          params:
            num_latents: 4096
            embed_dim: 64
            num_freqs: 8
            include_pi: false
            heads: 16
            width: 1024
            num_decoder_layers: 16
            qkv_bias: false
            qk_norm: true
            scale_factor: 1.0039506158752403
            geo_decoder_mlp_expand_ratio: 4
            geo_decoder_downsample_ratio: 1
            geo_decoder_ln_post: true
        conditioner:
          target: hy3dshape.models.conditioner.SingleImageEncoder
          params:
            main_image_encoder:
              type: DinoImageEncoder
              kwargs:
                config:
                  hidden_size: 1024
                  image_size: 518
                  num_hidden_layers: 24
                  use_swiglu_ffn: false
                image_size: 518
        scheduler:
          target: hy3dshape.schedulers.FlowMatchEulerDiscreteScheduler
          params:
            num_train_timesteps: 1000
        image_processor:
          target: hy3dshape.preprocessors.ImageProcessorV2
          params:
            size: 512
            border_ratio: 0.15
        """;

    private static string Yaml(int depth = 16, int single = 32, int latents = 3072, string scale = "0.9990943042622529",
        string extraModel = "", bool consistency = false, string encoder = "DinoImageEncoder", string dinoSize = "518", string processorSize = "512") =>
        $"""
        model:
          target: hy3dgen.shapegen.models.Hunyuan3DDiT
          params:
            in_channels: 64
            context_in_dim: 1536
            hidden_size: 1024
            mlp_ratio: 4.0
            num_heads: 16
            depth: {depth}
            depth_single_blocks: {single}
            axes_dim: [ 64 ]
            theta: 10000
            qkv_bias: True
        {extraModel}
        vae:
          target: hy3dgen.shapegen.models.ShapeVAE
          params:
            num_latents: {latents}
            embed_dim: 64
            num_freqs: 8
            include_pi: false
            heads: 16
            width: 1024
            num_decoder_layers: 16
            qkv_bias: false
            qk_norm: true
            scale_factor: {scale}

        conditioner:
          target: hy3dgen.shapegen.models.SingleImageEncoder
          params:
            main_image_encoder:
              type: {encoder} # dino giant
              kwargs:
                config:
                  hidden_size: 1536
                  num_hidden_layers: 40
                  image_size: 518
                image_size: {dinoSize}

        scheduler:
          target: hy3dgen.shapegen.schedulers.{(consistency ? "ConsistencyFlowMatchEulerDiscreteScheduler" : "FlowMatchEulerDiscreteScheduler")}
          params:
            num_train_timesteps: 1000
        {(consistency ? "    pcm_timesteps: 100" : "")}

        image_processor:
          target: hy3dgen.shapegen.preprocessors.ImageProcessorV2
          params:
            size: {processorSize}
            border_ratio: 0.15
        """;
}
