using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>An assistant turn's <see cref="TextMessage.ReasoningContent"/> reaches the chat template as <c>reasoning_content</c>.</summary>
public sealed class TextServiceReasoningContentTests
{
    private const string Template = "{% for m in messages %}{{ m.role }}:{% if m.reasoning_content %}[{{ m.reasoning_content }}]{% endif %}"
        + "{{ m.content }}\n{% endfor %}";

    [Fact]
    public void AssistantReasoningIsRenderedByTheTemplate()
        => Assert.Equal("user:hi\nassistant:[The user said hi.]Hello!\nuser:and?\n", Render("The user said hi."));

    [Fact]
    public void ATurnWithoutReasoningRendersAsBefore()
        => Assert.Equal("user:hi\nassistant:Hello!\nuser:and?\n", Render(null));

    [Fact]
    public void ReasoningIsCarriedOntoTheGenerationMessage()
    {
        GenerationRequest generation = TextService.BuildRequest(Request("plan"), rawCompletion: false, new CharTokenizer());
        Assert.Equal([null, "plan", null], generation.Messages!.Select(m => m.ReasoningContent));
    }

    private static string Render(string? reasoning)
    {
        CharTokenizer tokenizer = new();
        GenerationRequest generation = TextService.BuildRequest(Request(reasoning), rawCompletion: false, tokenizer);
        return CharTokenizer.Text(PromptBuilder.BuildPromptIds(generation, tokenizer, new JinjaChatTemplate(Template)));
    }

    private static TextRequest Request(string? reasoning) => new()
    {
        Messages =
        [
            new TextMessage { Role = TextRole.User, Content = "hi" },
            new TextMessage { Role = TextRole.Assistant, Content = "Hello!", ReasoningContent = reasoning },
            new TextMessage { Role = TextRole.User, Content = "and?" },
        ],
    };
}
