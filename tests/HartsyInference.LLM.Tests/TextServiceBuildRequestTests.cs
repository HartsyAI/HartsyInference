using HartsyInference.Engine.Services;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.OutputParsing;
using HartsyInference.LLM.Tests.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>A tools request must not arm the <c>&lt;tool_call&gt;</c> grammar: it never fires on GGUF text, and it turns off graph and speculative decode for every tool turn.</summary>
public sealed class TextServiceBuildRequestTests
{
    private static TextRequest Request(bool withTools) => new()
    {
        Messages = [new TextMessage { Role = TextRole.User, Content = "What time is it?" }],
        Tools = withTools ? [new ToolDefinition { Name = "get_time", Description = "time", JsonSchema = "{}" }] : null,
        Temperature = 0,
    };

    [Fact]
    public void ToolsDoNotArmTheJsonSentinel()
    {
        GenerationRequest withTools = TextService.BuildRequest(Request(true), rawCompletion: false, new PieceTokenizer());
        Assert.Null(withTools.Sampling.JsonModeSentinel);
        Assert.False(withTools.Sampling.HasJsonConstraint);
    }

    [Fact]
    public void SamplingIsIdenticalWithAndWithoutTools()
    {
        PieceTokenizer tok = new();
        GenerationRequest plain = TextService.BuildRequest(Request(false), rawCompletion: false, tok);
        GenerationRequest tools = TextService.BuildRequest(Request(true), rawCompletion: false, tok);
        Assert.Equal(plain.Sampling, tools.Sampling);
        Assert.Equal(plain.GraphDecode, tools.GraphDecode);
    }
}
