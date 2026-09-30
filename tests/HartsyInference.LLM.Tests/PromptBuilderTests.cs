using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Tests.OutputParsing;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>A request's <c>SystemPrompt</c> must reach the template exactly once: prepended when <c>Messages</c> has no leading system turn, ignored when a host already folded it into the first message.</summary>
public sealed class PromptBuilderTests
{
    [Fact]
    public void SystemPromptIsPrependedWhenMessagesLackALeadingSystemTurn()
    {
        RecordingTemplate template = new();
        GenerationRequest request = new()
        {
            Messages = [ChatMessage.User("hi")],
            SystemPrompt = "Be terse.",
        };
        PromptBuilder.BuildPromptIds(request, new WordTokenizer(), template);
        Assert.Equal(["system", "user"], template.Messages!.Select(m => m.Role));
        Assert.Equal("Be terse.", template.Messages![0].Content);
    }

    [Fact]
    public void SystemPromptIsIgnoredWhenTheFirstMessageIsAlreadySystem()
    {
        RecordingTemplate template = new();
        GenerationRequest request = new()
        {
            Messages = [ChatMessage.System("Folded by the host."), ChatMessage.User("hi")],
            SystemPrompt = "Be terse.",
        };
        PromptBuilder.BuildPromptIds(request, new WordTokenizer(), template);
        Assert.Equal(["system", "user"], template.Messages!.Select(m => m.Role));
        Assert.Equal("Folded by the host.", template.Messages![0].Content);
        Assert.DoesNotContain(template.Messages!, m => m.Content == "Be terse.");
    }

    [Fact]
    public void EmptySystemPromptAddsNothingAndMessagesPassThroughByReference()
    {
        List<ChatMessage> messages = [ChatMessage.User("hi")];
        Assert.Same(messages, PromptBuilder.WithSystemPrompt(messages, null));
        Assert.Same(messages, PromptBuilder.WithSystemPrompt(messages, ""));
    }

    [Fact]
    public void EffectiveMessagesIsTheSameViewTheTemplateRenders()
    {
        // The output parser resolves its initial state from EffectiveMessages(), so it must apply the one rule.
        GenerationRequest prepend = new() { Messages = [ChatMessage.User("hi")], SystemPrompt = "Be terse." };
        Assert.Equal(["system", "user"], prepend.EffectiveMessages()!.Select(m => m.Role));
        GenerationRequest folded = new() { Messages = [ChatMessage.System("Host."), ChatMessage.User("hi")], SystemPrompt = "Be terse." };
        Assert.Same(folded.Messages, folded.EffectiveMessages());
        Assert.Null(new GenerationRequest { Prompt = "hi", SystemPrompt = "Be terse." }.EffectiveMessages());
    }

    [Fact]
    public void ToolsReachTheTemplateOnBothMessageAndPromptPaths()
    {
        List<ToolSpec> tools = [ToolSpec.FromJson("{\"type\":\"function\",\"function\":{\"name\":\"f\"}}")];
        RecordingTemplate template = new();
        PromptBuilder.BuildPromptIds(new GenerationRequest { Messages = [ChatMessage.User("hi")], Tools = tools }, new WordTokenizer(), template);
        Assert.Same(tools, template.Tools);
        PromptBuilder.BuildPromptIds(new GenerationRequest { Prompt = "hi", Tools = tools }, new WordTokenizer(), template);
        Assert.Same(tools, template.Tools);
    }

    private sealed class WordTokenizer : NoBytesTokenizer
    {
        public override string Decode(IReadOnlyList<int> ids) => "";
    }

    /// <summary>Records what the builder handed to the template.</summary>
    private sealed class RecordingTemplate : IChatTemplate
    {
        public IReadOnlyList<ChatMessage>? Messages { get; private set; }

        public IReadOnlyList<ToolSpec>? Tools { get; private set; }

        public string Name => "recording";

        public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking = null)
            => Encode(tokenizer, messages, addGenerationPrompt, enableThinking, null);

        public int[] Encode(ILlmTokenizer tokenizer, IReadOnlyList<ChatMessage> messages, bool addGenerationPrompt, bool? enableThinking,
            IReadOnlyList<ToolSpec>? tools)
        {
            Messages = messages;
            Tools = tools;
            return [1];
        }
    }
}
