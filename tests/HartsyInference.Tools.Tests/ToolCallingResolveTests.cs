using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tools.Parsing;
using HartsyInference.Tools.Tests.Parsing;
using Xunit;

namespace HartsyInference.Tools.Tests;

/// <summary>Per-request format resolution: the model's own template first, then its family name, then Hermes; a structured parser or a tool-less prompt decides before either.</summary>
public sealed class ToolCallingResolveTests
{
    private static TextStreamFilterContext Context(string? template, string? architecture = null, string? path = null, string templateName = "jinja",
        bool structured = false, bool injected = false, string? forced = null) => new()
    {
        Request = new TextRequest { Messages = [new TextMessage { Role = TextRole.User, Content = "hi" }], Tools = [new ToolDefinition { Name = "hang_up" }] },
        RequestId = 1,
        TemplateName = templateName,
        ChatTemplateSource = template,
        Architecture = architecture,
        ModelPath = path,
        HasStructuredParser = structured,
        ToolPromptInjected = injected,
        ForcedToolName = forced,
    };

    [Fact]
    public void QwenTemplateResolvesToHermes()
        => Assert.Equal(ToolCallFormat.Hermes, ToolCalling.ResolveFormat(Context(ChatTemplateFixtures.Qwen3_4B, "qwen3")));

    [Fact]
    public void GemmaTemplateResolvesToGemma()
        => Assert.Equal(ToolCallFormat.Gemma, ToolCalling.ResolveFormat(Context(ChatTemplateFixtures.Gemma4_E2B_It, "gemma4")));

    [Fact]
    public void LlamaTemplateWithoutAnEnvelopeFallsBackToTheFamilyName()
        => Assert.Equal(ToolCallFormat.Llama3, ToolCalling.ResolveFormat(Context(ChatTemplateFixtures.Llama32_1B_Instruct, "llama")));

    [Fact]
    public void MistralTemplateWithoutToolsFallsBackToTheFamilyName()
        => Assert.Equal(ToolCallFormat.Mistral, ToolCalling.ResolveFormat(Context(ChatTemplateFixtures.Mistral7B_Instruct_v0_3, "llama", "/models/Mistral-7B-Instruct-v0.3.gguf")));

    [Fact]
    public void ChatMlTemplateNameIsHermes()
        => Assert.Equal(ToolCallFormat.Hermes, ToolCalling.ResolveFormat(Context(null, "gemma", null, templateName: "chatml")));

    [Fact]
    public void InjectedToolPromptIsHermesWhateverTheFamily()
        => Assert.Equal(ToolCallFormat.Hermes, ToolCalling.ResolveFormat(Context(ChatTemplateFixtures.Mistral7B_Instruct_v0_3, "mistral", injected: true)));

    [Fact]
    public void StructuredParserNeedsNoFilter()
        => Assert.Null(ToolCalling.ResolveFormat(Context(ChatTemplateFixtures.DeepSeekR1DistillQwen1_5B, "deepseek", structured: true)));

    [Fact]
    public void NoFamilyHintIsHermes()
        => Assert.Equal(ToolCallFormat.Hermes, ToolCalling.ResolveFormat(Context(null)));

    [Fact]
    public void InstalledFilterResolvesPerRequest()
    {
        EngineOptions options = new();
        ToolCalling.Install(options);
        ITextStreamFilter? filter = options.TextStreamFilterFactory!(Context(ChatTemplateFixtures.Gemma4_E2B_It, "gemma4"));
        Assert.IsType<ToolCallStreamFilter>(filter);
        Assert.Equal(ToolCallFormat.Gemma, ((ToolCallStreamFilter)filter!).Format);
    }

    [Fact]
    public void MarkerLiteralsListTheFormatsSpecialMarkers()
    {
        Assert.Contains("<tool_call>", ToolCallFormats.RulesFor(ToolCallFormat.Hermes).MarkerLiterals);
        Assert.Contains("<|tool_call>", ToolCallFormats.RulesFor(ToolCallFormat.Gemma).MarkerLiterals);
        Assert.Contains("<|\"|>", ToolCallFormats.RulesFor(ToolCallFormat.Gemma).MarkerLiterals);
        Assert.Contains("<|python_tag|>", ToolCallFormats.RulesFor(ToolCallFormat.Llama3).MarkerLiterals);
        Assert.Contains("[TOOL_CALLS]", ToolCallFormats.RulesFor(ToolCallFormat.Mistral).MarkerLiterals);
    }

    [Fact]
    public void ForcedToolRestrictsBareFormsAndStopsAfterItsCall()
    {
        TextStreamFilterContext context = Context(null, forced: "hang_up");
        ToolCallStreamFilter filter = Assert.IsType<ToolCallStreamFilter>(ToolCalling.CreateFilter(context));
        Assert.True(filter.StopAfterFirstCall);
        // A bare call to another offered tool is not a call once the forced name is the only one known.
        ToolCallParseResult other = new ToolCallParser(ToolCallFormat.Hermes, knownTools: ["hang_up"]).Push("{\"name\": \"get_time\", \"arguments\": {}}");
        Assert.Empty(other.Calls ?? []);
    }
}
