using System.Text.Json;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Exercises <see cref="Qwen3HfConfigReader.FromHuggingFace"/> against a real Qwen3 <c>config.json</c>
/// shape (IndexTTS-2's bundled <c>qwen0.6bemo4-merge</c> emotion classifier's own config, inlined here so the
/// test has no external file dependency) and confirms the result matches the hand-written
/// <see cref="TransformerConfig.Qwen3_0_6B"/> preset on every field this real checkpoint shares with it.</summary>
public sealed class Qwen3HfConfigReaderTests
{
    private const string RealQwenEmotionConfigJson = """
        {
          "architectures": ["Qwen3ForCausalLM"],
          "attention_bias": false,
          "attention_dropout": 0.0,
          "bos_token_id": 151643,
          "eos_token_id": 151643,
          "head_dim": 128,
          "hidden_act": "silu",
          "hidden_size": 1024,
          "initializer_range": 0.02,
          "intermediate_size": 3072,
          "max_position_embeddings": 32768,
          "max_window_layers": 28,
          "model_type": "qwen3",
          "num_attention_heads": 16,
          "num_hidden_layers": 28,
          "num_key_value_heads": 8,
          "rms_norm_eps": 1e-06,
          "rope_scaling": null,
          "rope_theta": 1000000,
          "sliding_window": null,
          "tie_word_embeddings": true,
          "torch_dtype": "bfloat16",
          "transformers_version": "4.52.1",
          "use_cache": true,
          "use_sliding_window": false,
          "vocab_size": 151936
        }
        """;

    private static TransformerConfig Parse(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        return Qwen3HfConfigReader.FromHuggingFace(doc.RootElement);
    }

    [Fact]
    public void FromHuggingFace_RealQwenEmotionConfig_MatchesEveryFieldOfTheHandWrittenPreset()
    {
        TransformerConfig cfg = Parse(RealQwenEmotionConfigJson);
        TransformerConfig preset = TransformerConfig.Qwen3_0_6B;

        Assert.Equal(preset.HiddenSize, cfg.HiddenSize);
        Assert.Equal(preset.NumLayers, cfg.NumLayers);
        Assert.Equal(preset.NumHeads, cfg.NumHeads);
        Assert.Equal(preset.NumKvHeads, cfg.NumKvHeads);
        Assert.Equal(preset.HeadDim, cfg.HeadDim);
        Assert.Equal(preset.IntermediateSize, cfg.IntermediateSize);
        Assert.Equal(preset.VocabSize, cfg.VocabSize);
        Assert.Equal(preset.AttentionBias, cfg.AttentionBias);
        Assert.Equal(preset.TieWordEmbeddings, cfg.TieWordEmbeddings);
        Assert.True(cfg.QkNorm);
        // The real checkpoint's own max_position_embeddings (32768) differs from the hand-written preset's
        // (40960, the base Qwen3-0.6B's value) — this is the one field a fine-tune can legitimately diverge
        // on, which is exactly why a generic reader exists instead of always reusing the preset.
        Assert.Equal(32_768, cfg.MaxPositionEmbeddings);
        Assert.Equal(1_000_000f, cfg.RopeTheta);
    }

    [Fact]
    public void FromHuggingFace_HeadDimFallsBackToHiddenOverHeads_WhenFieldIsAbsent()
    {
        string json = """
            {"model_type":"qwen3","hidden_size":512,"num_hidden_layers":4,"num_attention_heads":8,
             "num_key_value_heads":8,"intermediate_size":1024,"vocab_size":1000}
            """;
        TransformerConfig cfg = Parse(json);
        Assert.Equal(64, cfg.HeadDim);   // 512 / 8
    }

    [Fact]
    public void FromHuggingFace_WrongModelType_Throws()
    {
        string json = """{"model_type":"llama","hidden_size":512}""";
        Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => Parse(json));
    }
}
