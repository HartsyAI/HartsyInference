using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The avatar fields of the LongCat config: the Avatar 1.6 base_model/config.json (pin acfc628) parses into the audio geometry, and a true
/// audio_prenorm is refused. The JSON is the shipped file's content for these keys, verbatim.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatAvatarConfigTests
{
    private const string Avatar16 = """
        {
          "_class_name": "LongCatVideoAvatarTransformer3DModel",
          "in_channels": 16,
          "out_channels": 16,
          "hidden_size": 4096,
          "depth": 48,
          "num_heads": 32,
          "caption_channels": 4096,
          "model_max_length": 512,
          "mlp_ratio": 4,
          "adaln_tembed_dim": 512,
          "frequency_embedding_size": 256,
          "patch_size": [1, 2, 2],
          "enable_flashattn2": true,
          "text_tokens_zero_pad": true,
          "audio_window": 5,
          "audio_block": 5,
          "audio_channel": 1280,
          "intermediate_dim": 512,
          "output_dim": 768,
          "context_tokens": 32,
          "vae_scale": 4,
          "audio_prenorm": false,
          "class_range": 24,
          "class_interval": 4
        }
        """;

    [Fact]
    public void TheAvatar16AudioGeometryParses()
    {
        LongCatConfig config = LongCatConfig.FromJson(Avatar16);

        Assert.Equal(5, config.AudioWindow);
        Assert.Equal(5, config.AudioBlock);
        Assert.Equal(1280, config.AudioChannel);
        Assert.Equal(512, config.AudioIntermediateDim);
        Assert.Equal(768, config.AudioOutputDim);
        Assert.Equal(32, config.AudioContextTokens);
        Assert.Equal(4, config.AudioVaeScale);
        Assert.Equal(48, config.Depth);
    }

    [Fact]
    public void AnAudioPrenormTrueIsRefused()
    {
        string json = Avatar16.Replace("\"audio_prenorm\": false", "\"audio_prenorm\": true", StringComparison.Ordinal);

        Assert.Throws<InvalidDataException>(() => LongCatConfig.FromJson(json));
    }
}
