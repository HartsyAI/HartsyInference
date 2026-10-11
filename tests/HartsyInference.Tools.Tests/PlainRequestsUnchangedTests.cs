using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Tools.Tests;

/// <summary>A request that offers no tools must generate exactly what it did before the tool filter existed: installing the filter may not change plain chat.</summary>
public sealed class PlainRequestsUnchangedTests
{
    [Fact]
    [Trait("Category", "Integration")]
    [Trait("Category", "RealWeights")]
    public async Task PlainChatIsIdenticalWithTheToolFilterInstalled()
    {
        string path = TestPaths.Llm.Qwen3_06BQ4KM;
        if (!File.Exists(path)) return;
        string without = await Generate(path, install: false);
        string with = await Generate(path, install: true);
        Assert.Equal(without, with);
    }

    private static async Task<string> Generate(string path, bool install)
    {
        EngineOptions options = new();
        if (install) ToolCalling.Install(options);
        using InferenceEngine engine = new("cpu", options);
        ModelSpec spec = ModelResolver.Resolve("qwen3", path, Modality.Text);
        TextRequest request = new()
        {
            Messages = [new TextMessage { Role = TextRole.User, Content = "Name three primary colours." }],
            EnableThinking = false,
            Greedy = true,
            Temperature = 0,
            MaxTokens = 32,
            Seed = 42,
        };
        TextResult result = await engine.Text.GenerateAsync(spec, request);
        return result.Text;
    }
}
