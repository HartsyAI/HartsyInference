using HartsyInference.Core.Tensors;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Cpu;
using HartsyInference.LLM.Models.Lfm3;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Lfm3;

/// <summary>LiquidAI/LFM3-1.7B on the CPU backend against F32 transformers logits for one fixed prompt.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "Slow")]
public sealed class Lfm3RealWeightTests(ITestOutputHelper output)
{
    private static readonly int[] Prompt = [597, 5205, 302, 3980, 355, 4741, 22, 440, 5205, 302, 6398, 355];

    // F32 reference logits of the next token after Prompt, by token id.
    private static readonly (int Id, float Logit)[] ReferenceTop =
    [(6552, 19.133f), (51861, 13.275f), (21332, 12.4477f), (26732, 12.4471f), (2726, 12.2002f), (278, 12.1382f), (63990, 11.8903f), (6398, 11.8417f)];

    // Greedy continuation of Prompt.
    private static readonly int[] ReferenceGreedy = [6552, 22, 440, 5205, 302, 3888, 355, 18943, 22, 440, 5205, 302];

    [Fact]
    public void LogitsAndGreedyContinuationMatchTransformers()
    {
        string dir = TestPaths.Llm.Lfm3Dir;
        string weights = Path.Combine(dir, "model.safetensors");
        if (!RealWeightGate.Require(output.WriteLine, weights, Path.Combine(dir, "config.json"))) return;
        Lfm3Config config = Lfm3Config.FromJson(File.ReadAllText(Path.Combine(dir, "config.json")));
        using SafeTensorsLoader loader = new();
        loader.Load(weights);
        Dictionary<string, Tensor> tensors = loader.GetAllTensors();
        using CpuBackend backend = new();
        using Lfm3Model model = new(tensors, config, 64);

        float[] logits = model.ForwardLastLogits(backend, Prompt);
        foreach ((int id, float expected) in ReferenceTop)
        {
            output.WriteLine($"{id}: engine {logits[id]:F4} reference {expected:F4}");
            Assert.InRange(logits[id], expected - 0.05f, expected + 0.05f);
        }
        List<int> generated = [];
        int next = Array.IndexOf(logits, logits.Max());
        for (int i = 0; i < ReferenceGreedy.Length; i++)
        {
            generated.Add(next);
            logits = model.ForwardLastLogits(backend, [next]);
            next = Array.IndexOf(logits, logits.Max());
        }
        Assert.Equal(ReferenceGreedy, generated);
    }

    [Fact]
    public void TokenizerMatchesTransformersIds()
    {
        string dir = TestPaths.Llm.Lfm3Dir;
        if (!RealWeightGate.Require(output.WriteLine, Path.Combine(dir, "tokenizer.json"), Path.Combine(dir, "model.safetensors"))) return;
        using SsmLanguageModel model = SsmLanguageModel.LoadLfm3(dir, 64);
        Assert.Equal(Prompt, model.Tokenizer.EncodeOrdinary("The capital of France is Paris. The capital of Germany is"));
    }

    [Fact]
    public async Task EngineAnswersAChatPrompt()
    {
        string dir = TestPaths.Llm.Lfm3Dir;
        if (!RealWeightGate.Require(output.WriteLine, Path.Combine(dir, "tokenizer.json"), Path.Combine(dir, "model.safetensors"))) return;
        using InferenceEngine engine = new("cpu");
        ModelSpec spec = new() { Requested = "lfm3", Modality = Modality.Text, LocalPath = dir };
        TextRequest request = new()
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "What is the capital of France? Answer with one word." }],
            Temperature = 0, TopP = 1, TopK = 1, Greedy = true, EnableThinking = false, MaxTokens = 16, Device = "cpu",
        };
        TextResult result = await engine.Text.GenerateAsync(spec, request);
        output.WriteLine(result.Text);
        Assert.Contains("Paris", result.Text);
    }
}
