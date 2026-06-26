using HartsyInference.Cpu;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>End-to-end generation smoke test for real GGUF checkpoints, gated on the
/// <c>HARTSY_TEST_GGUF_MODELS</c> env var (a ';'- or ','-separated list of .gguf paths). Runs on the F32
/// <see cref="CpuBackend"/> so it needs no GPU and works wherever the models are provided; it skips cleanly
/// when the var is unset or none of the paths exist. Each model is greedy-decoded on a factual prompt and the
/// well-known answer must appear — a coherence regression guard for the whole
/// load → tokenize → chat-template → forward → sample pipeline across architectures (it is exactly this kind
/// of check that caught the Llama RoPE pairing/scaling bugs during bring-up).</summary>
public sealed class GgufEndToEndTests
{
    private const string Prompt = "What is the capital of France? Answer in one short sentence.";
    private const string Expected = "paris";

    private static string[] ModelPaths()
    {
        string? env = Environment.GetEnvironmentVariable("HARTSY_TEST_GGUF_MODELS");
        if (string.IsNullOrWhiteSpace(env)) return [];
        return [.. env
            .Split([';', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(File.Exists)];
    }

    [Fact]
    public void Gguf_GreedyGeneration_ProducesKnownAnswer()
    {
        string[] models = ModelPaths();
        if (models.Length == 0)
        {
            // Gated test: no models provided → nothing to verify (passes as a no-op). Set
            // HARTSY_TEST_GGUF_MODELS to a ';'-separated list of existing .gguf paths to exercise it.
            return;
        }

        List<string> failures = [];
        foreach (string path in models)
        {
            using CpuBackend backend = new();
            using GgufLanguageModel model = GgufLanguageModel.Load(path, dequantizeToF32: true);
            TextGenerationPipeline pipeline = new(model.Transformer, model.Tokenizer, backend, model.Template);
            GenerationResult result = pipeline.Generate(new GenerationRequest
            {
                Prompt = Prompt,
                MaxTokens = 48,
                Sampling = SamplingOptions.Default with { Greedy = true },
            });

            string name = $"{Path.GetFileName(path)} (arch={model.Architecture})";
            if (!result.Text.Contains(Expected, StringComparison.OrdinalIgnoreCase))
                failures.Add($"{name}: expected '{Expected}' in output, got [{result.Text.Trim()}]");
        }

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }
}
