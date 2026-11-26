using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Multimodal;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The turn the Qwen3.5-VL prompt ends with, pinned to the shipped chat template: the hand-built string must equal what the template renders
/// after the user's content, for thinking unset, on, and off.</summary>
public sealed class Qwen35VlAssistantTurnTests
{
    private static readonly string Source = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "SenseNovaLoop", "chat_template.jinja"));

    private static string RenderAfterUser(bool? thinking)
    {
        Dictionary<string, object?> user = new() { ["role"] = "user", ["content"] = "Q" };
        Dictionary<string, object?> context = new()
        {
            ["messages"] = new List<object?> { user }, ["add_generation_prompt"] = true, ["bos_token"] = "", ["eos_token"] = "<|im_end|>",
            ["tools"] = null, ["documents"] = null,
        };
        if (thinking.HasValue) context["enable_thinking"] = thinking.Value;
        string text = new JinjaEngine(Source).Render(context);
        Assert.EndsWith("Q" + Qwen35VlGenerator.AssistantTurn(thinking), text);
        return text;
    }

    [Fact]
    public void ThinkingUnsetOpensAReasoningBlockLikeTheTemplate() => RenderAfterUser(null);

    [Fact]
    public void ThinkingOnOpensAReasoningBlockLikeTheTemplate() => RenderAfterUser(true);

    [Fact]
    public void ThinkingOffPrefillsAnEmptyThinkBlockLikeTheTemplate() => RenderAfterUser(false);
}
