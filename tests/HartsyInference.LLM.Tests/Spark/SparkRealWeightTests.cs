using HartsyInference.Cpu;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.Models.Spark;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Spark;

/// <summary>XHToken/Spark-X2.6-4B on the CPU backend against F32 logits for one fixed prompt, from a PyTorch script written from the checkpoint's modeling_spark.py.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "Slow")]
public sealed class SparkRealWeightTests(ITestOutputHelper output)
{
    private static readonly int[] Prompt = [1046, 11862, 314, 17349, 390, 19500, 27, 614, 11862, 314, 19356, 390];

    // F32 reference logits of the next token after Prompt, by token id.
    private static readonly (int Id, float Logit)[] ReferenceTop =
    [(34987, -0.517f), (103912, -8.2455f), (10063, -8.5674f), (24410, -8.9648f), (26724, -9.6566f), (94923, -9.7562f), (27, -10.1892f), (31329, -10.2855f)];

    // Greedy continuation of Prompt.
    private static readonly int[] ReferenceGreedy = [34987, 27, 614, 11862, 314, 26046, 390, 29178, 27, 614, 11862, 314];

    private static string[] RequiredFiles(string dir) =>
        [Path.Combine(dir, "config.json"), Path.Combine(dir, "model.safetensors.index.json"), Path.Combine(dir, "tokenizer.json"), Path.Combine(dir, "chat_template.jinja")];

    [Fact]
    public void LogitsAndGreedyContinuationMatchReference()
    {
        string dir = TestPaths.Llm.SparkDir;
        if (!RealWeightGate.Require(output.WriteLine, RequiredFiles(dir))) return;
        using SparkModel model = SparkModel.Load(dir, 64);
        using CpuBackend backend = new();

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

    // The same checkpoint with sliding_window lowered to 4, so the 12-token prompt crosses the window.
    private static readonly (int Id, float Logit)[] ReferenceTopWindow4 =
    [(34987, 2.0142f), (103912, -4.7929f), (24410, -5.4664f), (94923, -5.5752f), (10063, -5.814f), (26724, -6.2597f), (31329, -7.7777f), (59806, -8.338f)];

    private static readonly int[] ReferenceGreedyWindow4 = [34987, 27, 614, 11862, 314, 17349, 390, 19500, 27, 614, 11862, 314];

    [Fact]
    public void ShortSlidingWindowMatchesReference()
    {
        string dir = TestPaths.Llm.SparkDir;
        if (!RealWeightGate.Require(output.WriteLine, RequiredFiles(dir))) return;
        SparkConfig config = SparkConfig.FromJson(File.ReadAllText(Path.Combine(dir, "config.json"))) with { SlidingWindow = 4 };
        using SafeTensorsShardLoader loader = new();
        loader.LoadDirectory(dir);
        using SparkModel model = new(loader.GetAllTensors(), config, 64);
        using CpuBackend backend = new();

        float[] logits = model.ForwardLastLogits(backend, Prompt);
        foreach ((int id, float expected) in ReferenceTopWindow4)
        {
            output.WriteLine($"{id}: engine {logits[id]:F4} reference {expected:F4}");
            Assert.InRange(logits[id], expected - 0.05f, expected + 0.05f);
        }
        List<int> generated = [];
        int next = Array.IndexOf(logits, logits.Max());
        for (int i = 0; i < ReferenceGreedyWindow4.Length; i++)
        {
            generated.Add(next);
            logits = model.ForwardLastLogits(backend, [next]);
            next = Array.IndexOf(logits, logits.Max());
        }
        Assert.Equal(ReferenceGreedyWindow4, generated);
    }

    [Fact]
    public void TokenizerMatchesReferenceIds()
    {
        string dir = TestPaths.Llm.SparkDir;
        if (!RealWeightGate.Require(output.WriteLine, RequiredFiles(dir))) return;
        using SsmLanguageModel model = SsmLanguageModel.LoadHuggingFace(dir, SparkConfig.ModelType, 64);
        Assert.Equal(Prompt, model.Tokenizer.EncodeOrdinary("The capital of France is Paris. The capital of Germany is"));
    }

    [Fact]
    public async Task EngineAnswersAChatPrompt()
    {
        string dir = TestPaths.Llm.SparkDir;
        if (!RealWeightGate.Require(output.WriteLine, RequiredFiles(dir))) return;
        using InferenceEngine engine = new("cpu");
        ModelSpec spec = new() { Requested = "spark", Modality = Modality.Text, LocalPath = dir };
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
