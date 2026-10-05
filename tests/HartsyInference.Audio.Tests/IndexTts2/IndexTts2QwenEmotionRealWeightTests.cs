using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Cpu;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using System.Text.Json;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>Loads the real bundled <c>qwen0.6bemo4-merge/</c> emotion classifier (IndexTTS-2's own Qwen3-0.6B
/// fine-tune; ~1.2 GB — too large to bundle, point <c>INDEXTTS2_QWEN_EMOTION_DIR</c> at a local copy of the
/// folder, containing <c>config.json</c>, <c>model.safetensors</c>, <c>tokenizer.json</c> and
/// <c>chat_template.jinja</c>) through the real stack: <see cref="Qwen3HfConfigReader"/> →
/// <see cref="GenericTransformer.LoadWeights"/> (confirmed zero key remapping — stock HF
/// <c>model.embed_tokens.weight</c>/<c>model.layers.N.*</c>/<c>model.norm.weight</c>, no separate
/// <c>lm_head.weight</c> since <c>tie_word_embeddings: true</c>) → <see cref="HfTokenizerJson.LoadByteLevelBpe"/>
/// → <see cref="JinjaChatTemplate"/> → <see cref="TextGenerationPipeline"/> → <see cref="IndexTts2QwenEmotion"/>.
/// One real classification call confirms the whole chain produces a finite, correctly-shaped 8-dim vector.</summary>
[Trait("Category", "Integration")]
public sealed class IndexTts2QwenEmotionRealWeightTests
{
    [Fact]
    public void Infer_SucceedsAgainstTheRealBundledQwenEmotionClassifier()
    {
        string? dir = Environment.GetEnvironmentVariable("INDEXTTS2_QWEN_EMOTION_DIR");
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return;   // resource-gated: skip without the real checkpoint folder

        string configPath = Path.Combine(dir, "config.json");
        string weightsPath = Path.Combine(dir, "model.safetensors");
        string tokenizerPath = Path.Combine(dir, "tokenizer.json");
        string templatePath = Path.Combine(dir, "chat_template.jinja");
        if (!File.Exists(configPath) || !File.Exists(weightsPath) || !File.Exists(tokenizerPath) || !File.Exists(templatePath)) return;

        using JsonDocument configDoc = JsonDocument.Parse(File.ReadAllText(configPath));
        TransformerConfig cfg = Qwen3HfConfigReader.FromHuggingFace(configDoc.RootElement);

        using SafeTensorsLoader loader = new();
        loader.Load(weightsPath);
        Dictionary<string, HartsyInference.Core.Tensors.Tensor> weights = loader.GetAllTensors();
        try
        {
            using GenericTransformer transformer = new(cfg);
            transformer.LoadWeights(weights, "model");

            using FileStream tokenizerStream = File.OpenRead(tokenizerPath);
            GgufTokenizer tokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerStream);
            JinjaChatTemplate template = new(File.ReadAllText(templatePath));

            using CpuBackend backend = new();
            TextGenerationPipeline pipeline = new(transformer, tokenizer, backend, template);
            IndexTts2QwenEmotion classifier = new(pipeline);

            float[] result = classifier.Infer("I am so incredibly happy and excited today!");

            Assert.Equal(8, result.Length);
            foreach (float v in result) Assert.True(float.IsFinite(v) && v is >= 0f and <= 1.2f);
        }
        finally
        {
            foreach (HartsyInference.Core.Tensors.Tensor t in weights.Values) t.Dispose();
        }
    }
}
