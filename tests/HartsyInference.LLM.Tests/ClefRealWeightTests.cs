using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>Opt-in real-weight smoke test for the released Clef GGUF. This exercises model resolution, the
/// Qwen3.5 hybrid loader, chat-template rendering, prefill, and autoregressive decode through the public engine.</summary>
[Trait("Category", "RealWeights")]
[Trait("Category", "Slow")]
public sealed class ClefRealWeightTests
{
    private readonly ITestOutputHelper _output;

    public ClefRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task TextGeneration_ProducesNonEmptyCompletion()
    {
        string checkpoint = TestPaths.Llm.ClefIq2Xxs;
        if (!RealWeightGate.Require(_output.WriteLine, checkpoint)) return;

        using InferenceEngine engine = new("cpu");
        ModelSpec spec = new() { Requested = "clef", Modality = Modality.Text, LocalPath = checkpoint };
        TextRequest request = new()
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "Reply with exactly the word READY." }],
            Temperature = 0,
            TopP = 1,
            TopK = 1,
            Greedy = true,
            EnableThinking = false,
            MaxTokens = 8,
            Device = "cpu",
        };

        TextResult result = await engine.Text.GenerateAsync(spec, request);

        _output.WriteLine(result.Text);
        Assert.False(string.IsNullOrWhiteSpace(result.Text));
        Assert.True(result.CompletionTokens > 0);
    }
}
