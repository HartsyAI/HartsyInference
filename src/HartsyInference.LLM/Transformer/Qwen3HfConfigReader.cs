using System.Text.Json;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.LLM.Transformer;

/// <summary>Reads a real HuggingFace Qwen3 <c>config.json</c> into a <see cref="TransformerConfig"/> — the
/// <c>config.json</c>-sourced sibling of <see cref="GgufConfigFactory"/>'s GGUF-metadata-sourced loader, for
/// any model shipped as a raw HF <c>config.json</c> + <c>safetensors</c> Qwen3 checkpoint (not scoped to any
/// one caller — modeled on <c>HartsyInference.LLM.DeepSeekV41.DeepSeekV41ConfigReader</c>'s typed-field-read
/// pattern). <see cref="TransformerConfig.Qwen3_0_6B"/> already covers the hand-written preset for the base
/// model; this is for checkpoints (like IndexTTS-2's bundled emotion classifier) that ship their own
/// <c>config.json</c> rather than matching a known preset exactly.</summary>
public static class Qwen3HfConfigReader
{
    /// <summary>Builds a <see cref="TransformerConfig"/> from a parsed Qwen3 <c>config.json</c>. Fields this
    /// reader doesn't recognize (MoE, sliding window, etc.) are not read — Qwen3's dense text models don't use
    /// them; a MoE Qwen3 checkpoint needs its own reader.</summary>
    public static TransformerConfig FromHuggingFace(JsonElement config)
    {
        HfConfigFields r = new(config, "config", "Qwen3");
        string? modelType = r.OptionalString("model_type");
        if (modelType is not null && !string.Equals(modelType, "qwen3", StringComparison.OrdinalIgnoreCase))
            throw new HartsyInferenceException($"Qwen3HfConfigReader: config.model_type is '{modelType}', not 'qwen3'.");

        int hiddenSize = r.Int("hidden_size");
        int numHeads = r.Int("num_attention_heads");
        int headDim = r.Int("head_dim", hiddenSize / numHeads);

        return new TransformerConfig
        {
            HiddenSize = hiddenSize,
            NumLayers = r.Int("num_hidden_layers"),
            NumHeads = numHeads,
            NumKvHeads = r.Int("num_key_value_heads", numHeads),
            HeadDim = headDim,
            IntermediateSize = r.Int("intermediate_size"),
            VocabSize = r.Int("vocab_size"),
            MaxPositionEmbeddings = r.Int("max_position_embeddings", 32_768),
            RopeTheta = (float)r.Double("rope_theta", 1_000_000d),
            RmsNormEps = (float)r.Double("rms_norm_eps", 1e-6d),
            AttentionBias = r.Bool("attention_bias", false),
            TieWordEmbeddings = r.Bool("tie_word_embeddings", true),
            // Every real Qwen3 text checkpoint applies per-head Q/K RMSNorm — it is not a config.json field
            // (there is no "qk_norm" key); it's part of the architecture itself, confirmed by the family's
            // own modeling code always registering q_norm/k_norm. Qwen3-MoE's extra fields (num_experts, …)
            // are deliberately not read here — a MoE checkpoint needs its own reader.
            QkNorm = true,
        };
    }
}
