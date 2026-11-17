using HartsyInference.Audio.Models.Haru;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Audio.Tests.Haru;

/// <summary>Haru's released <c>tokenizer.json</c> (<c>higurashi-ai/Haru-4B</c> @ 8d6e83bc, sha256 06b95093...) read by the engine,
/// against ids from the <c>tokenizers</c> library on the same text (<c>add_special_tokens=False</c>, special literals parsed).
/// <c>HARU_TOKENIZER_JSON</c> names the file.</summary>
[Trait("Category", "Integration")]
public sealed class HaruTokenizerTests
{
    // Released config: pad_token_id 248076, eop_token_id 248071, lm.eos_token_id 248044, vocab 248320.
    private const string ReleasedConfig = """
        {"model_type":"haru","sample_rate":24000,"frame_samples":3200,"latent_dim":64,"pad_token_id":248076,"eop_token_id":248071,
         "lm":{"model_type":"qwen3_5_text","vocab_size":248320,"hidden_size":2560,"intermediate_size":9216,"num_hidden_layers":32,
          "num_attention_heads":16,"num_key_value_heads":4,"head_dim":256,"attn_output_gate":true,"full_attention_interval":4,
          "linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_num_key_heads":16,
          "linear_num_value_heads":32,"tie_word_embeddings":true,"eos_token_id":248044,"max_position_embeddings":262144,
          "rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"partial_rotary_factor":0.25,"rope_theta":10000000}},
         "codec":{"latent_dim":64,"base_channels":32,"ratios":[8,5,5,4,2,2],"encoder_depths":[3,3,3,3,3,3,8],
          "decoder_depths":[8,3,3,3,3,3,3],"norm_eps":1e-05,"latent_std":0.5},
         "head":{"hidden_size":2560,"latent_dim":64,"num_layers":4,"ffn_ratio":3.0,"norm_eps":1e-05,"interval_embed":7.0,
          "num_steps":1,"guidance_scale":1.5,"guidance_range":[1.0,2.0]}}
        """;

    private static (string Text, int[] Ids)[] Cases =>
    [
        ("<|im_start|>system\nYou are Haru, a friendly assistant. 今日は元気ですか？<|im_end|>\n",
            [248045, 8678, 198, 2523, 513, 5061, 84, 11, 264, 11321, 17313, 13, 182365, 247411, 173399, 10992, 248046, 198]),
    ];

    private static Stream? Open()
    {
        string? path = Environment.GetEnvironmentVariable("HARU_TOKENIZER_JSON");
        if (!string.IsNullOrEmpty(path) && File.Exists(path))
        {
            return File.OpenRead(path);
        }
        Assert.False(Environment.GetEnvironmentVariable("HARTSY_REQUIRE_REAL_WEIGHTS") == "1", "HARU_TOKENIZER_JSON does not name the released tokenizer.json.");
        return null;
    }

    [Fact]
    public void ReleasedTokenizer_PutsPadAndEopOnTheConfigIdsAndEncodesLikeTheReferenceLibrary()
    {
        using Stream? json = Open();
        if (json is null)
        {
            return;
        }
        HaruConfig config = HaruConfig.FromJson(ReleasedConfig);
        GgufTokenizer tokenizer = HaruTokenizer.Load(json, config);
        Assert.Equal(248076, tokenizer.SpecialId(HaruTokenizer.PadLiteral));
        Assert.Equal(248071, tokenizer.SpecialId(HaruTokenizer.EopLiteral));
        foreach ((string text, int[] ids) in Cases)
        {
            Assert.Equal(ids, tokenizer.Encode(text, addSpecial: true));
        }
    }

    [Fact]
    public void ReleasedConfig_ParsesAndPassesItsOwnChecks()
    {
        HaruConfig config = HaruConfig.FromJson(ReleasedConfig);
        Assert.Equal(7.5, config.FrameRate);
        Assert.Equal(32, config.Lm.NumLayers);
        Assert.Equal(8, Enumerable.Range(0, 32).Count(i => !config.Lm.IsLinearLayer(i)));
        Assert.Equal([11, 11, 10], config.Lm.MropeSection);
        Assert.True(config.Lm.TieWordEmbeddings);
    }

    [Theory]
    [InlineData("\"attn_output_gate\":true", "\"attn_output_gate\":false")]
    [InlineData("\"pad_token_id\":248076", "\"pad_token_id\":248071")]
    [InlineData("\"pad_token_id\":248076", "\"pad_token_id\":999999")]
    [InlineData("\"frame_samples\":3200", "\"frame_samples\":3000")]
    [InlineData("\"tie_word_embeddings\":true", "\"tie_word_embeddings\":false")]
    [InlineData("\"model_type\":\"haru\"", "\"model_type\":\"other\"")]
    public void FromJson_RefusesAConfigThatDoesNotFit(string from, string to) =>
        Assert.Throws<InvalidDataException>(() => HaruConfig.FromJson(ReleasedConfig.Replace(from, to)));
}
