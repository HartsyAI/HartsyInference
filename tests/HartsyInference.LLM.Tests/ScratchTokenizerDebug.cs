using HartsyInference.Cpu;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Sampling;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

public sealed class ScratchTokenizerDebug
{
    private readonly ITestOutputHelper _output;
    public ScratchTokenizerDebug(ITestOutputHelper output) => _output = output;

    [Fact]
    public void CpuBackend_NumericPrompt()
    {
        string? env = Environment.GetEnvironmentVariable("HARTSY_TEST_GGUF_MODELS");
        if (string.IsNullOrWhiteSpace(env)) return;
        string path = env.Split([';', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)[0];
        using CpuBackend backend = new();
        using GgufLanguageModel model = GgufLanguageModel.Load(path, dequantizeToF32: true);
        TextGenerationPipeline pipeline = new(model.Transformer, model.Tokenizer, backend, model.Template);
        GenerationResult result = pipeline.Generate(new GenerationRequest
        {
            Prompt = "What is 9 times 9?",
            MaxTokens = 40,
            Sampling = SamplingOptions.Default with { Greedy = true },
        });
        _output.WriteLine($"CPU F32 output: {result.Text}");
    }
}
