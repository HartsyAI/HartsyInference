using HartsyInference.Engine.Services;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.ChatTemplates;
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

/// <summary>Tool-less templates get the Hermes prompt written into the conversation; a forced tool must be offered and is stated in the system prompt.</summary>
public sealed class TextServiceToolPromptTests
{
    private const string ToolLess = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}";
    private const string ToolAware = "{% if tools %}<tools>{{ tools }}</tools>{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}";

    private static TextRequest Request(string? forced = null) => new()
    {
        Messages = [new TextMessage { Role = TextRole.User, Content = "What time is it?" }],
        Tools = [new ToolDefinition { Name = "get_time", Description = "time", JsonSchema = "{}" }],
        ForceToolId = forced,
        Temperature = 0,
    };

    [Fact]
    public void ToolLessTemplateGetsTheInjectedPrompt()
    {
        Assert.True(TextService.ShouldInjectToolPrompt(Request(), rawCompletion: false, new JinjaChatTemplate(ToolLess)));
        GenerationRequest generation = TextService.BuildRequest(Request(), rawCompletion: false, new LiteralControlTokenizer(), injectToolPrompt: true);
        Assert.Null(generation.Tools);
        Assert.Null(generation.SystemPrompt);
        Assert.Equal("system", generation.Messages![0].Role);
        Assert.Contains("<tools>", generation.Messages[0].Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ToolAwareTemplateIsNotInjected()
        => Assert.False(TextService.ShouldInjectToolPrompt(Request(), rawCompletion: false, new JinjaChatTemplate(ToolAware)));

    [Fact]
    public void TemplateWithoutAJinjaSourceIsNotInjected()
        => Assert.False(TextService.ShouldInjectToolPrompt(Request(), rawCompletion: false, new ChatMlTemplate()));

    [Fact]
    public void NoToolsNeverInjects()
    {
        TextRequest plain = new() { Messages = [new TextMessage { Role = TextRole.User, Content = "hi" }] };
        Assert.False(TextService.ShouldInjectToolPrompt(plain, rawCompletion: false, new JinjaChatTemplate(ToolLess)));
    }

    [Fact]
    public void ForcedToolUnknownToIsRefusedBeforeModelWork()
    {
        HartsyInference.Core.Exceptions.HartsyInferenceException error = Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(
            () => TextService.ValidateForcedTool(Request("delete_all")));
        Assert.Contains("delete_all", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ForcedOfferedToolPassesValidation()
        => TextService.ValidateForcedTool(Request("get_time"));

    [Fact]
    public void ForcedToolIsStatedInTheSystemPromptOnTheToolAwarePath()
    {
        GenerationRequest generation = TextService.BuildRequest(Request("get_time"), rawCompletion: false, new LiteralControlTokenizer());
        Assert.Contains(HartsyInference.LLM.ChatTemplates.HermesToolPrompt.ForceDirective("get_time"), generation.SystemPrompt, StringComparison.Ordinal);
        Assert.NotNull(generation.Tools);
    }
}
