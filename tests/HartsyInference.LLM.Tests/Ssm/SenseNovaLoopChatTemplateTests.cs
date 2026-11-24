using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The chat template shipped with SenseNova-Loop-9B (<c>chat_template.jinja</c> @ 3cc220e, copied unchanged) must compile and render in the
/// engine's Jinja interpreter. A construct the interpreter lacks degrades silently to ChatML (a <c>[WRN]</c> line), which would drop the thinking
/// switch, so the shapes below are rendered directly through the template. They use template-level dictionaries; the service's own message
/// encoding (string content only, image parts not forwarded) is not covered here, and the image case is for the template only.</summary>
public sealed class SenseNovaLoopChatTemplateTests
{
    private static readonly string Source = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "SenseNovaLoop", "chat_template.jinja"));

    private static Dictionary<string, object?> Message(string role, object? content) => new() { ["role"] = role, ["content"] = content };

    private static string Render(List<object?> messages, bool generationPrompt = true, bool? thinking = null, object? tools = null)
    {
        Dictionary<string, object?> context = new()
        {
            ["messages"] = messages, ["add_generation_prompt"] = generationPrompt, ["bos_token"] = "", ["eos_token"] = "<|im_end|>",
            ["tools"] = tools, ["documents"] = null,
        };
        if (thinking.HasValue) context["enable_thinking"] = thinking.Value;
        return new JinjaEngine(Source).Render(context);
    }

    [Fact]
    public void SystemAndUser_OpenAThinkingBlockByDefault()
    {
        string text = Render([Message("system", "Be brief."), Message("user", "Hi")]);
        Assert.Equal("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n", text);
    }

    [Fact]
    public void ThinkingOff_PrefillsAnEmptyThinkBlock()
    {
        string text = Render([Message("user", "Hi")], thinking: false);
        Assert.EndsWith("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", text);
    }

    [Fact]
    public void ToolHistory_RendersWithoutTheChatMlFallback()
    {
        // The second request of a tool-calling loop: user, assistant with tool_calls, tool, user. The template reads tool_call.arguments|items,
        // which the interpreter must support, or the whole render falls back to ChatML.
        Dictionary<string, object?> call = new()
        {
            ["id"] = "c1", ["type"] = "function",
            ["function"] = new Dictionary<string, object?> { ["name"] = "get_weather", ["arguments"] = new Dictionary<string, object?> { ["city"] = "Oslo" } },
        };
        Dictionary<string, object?> assistant = new() { ["role"] = "assistant", ["content"] = "", ["tool_calls"] = new List<object?> { call } };
        Dictionary<string, object?> tool = new() { ["role"] = "tool", ["content"] = "4C" };
        string text = Render([Message("user", "Weather in Oslo?"), assistant, tool, Message("user", "Thanks")]);
        Assert.Contains("<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>", text);
        Assert.Contains("<tool_response>\n4C\n</tool_response>", text);
        Assert.EndsWith("<|im_start|>assistant\n<think>\n", text);
    }

    [Fact]
    public void AssistantHistoryWithAThinkBlock_KeepsItsAnswer()
    {
        // Content keeps reasoning markers literal, so an earlier turn with a think block must not render blank: split('</think>')[-1] is the answer.
        string text = Render([Message("user", "Time?"), Message("assistant", "<think>\nhm\n</think>\n\nIt is 5C."), Message("user", "And now?")]);
        Assert.Contains("It is 5C.", text);
        Assert.DoesNotContain("<|im_start|>assistant\n<|im_end|>", text);
    }

    [Fact]
    public void ImageContent_BecomesAVisionPadSpan()
    {
        List<object?> content = [new Dictionary<string, object?> { ["type"] = "image" }, new Dictionary<string, object?> { ["type"] = "text", ["text"] = "What is this?" }];
        string text = Render([Message("user", content)]);
        Assert.Contains("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is this?<|im_end|>\n", text);
    }

    [Fact]
    public void ToolList_RendersTheFunctionBlock()
    {
        Dictionary<string, object?> function = new()
        {
            ["name"] = "get_weather", ["description"] = "Weather for a city",
            ["parameters"] = new Dictionary<string, object?> { ["type"] = "object", ["properties"] = new Dictionary<string, object?>() },
        };
        List<object?> tools = [new Dictionary<string, object?> { ["type"] = "function", ["function"] = function }];
        string text = Render([Message("user", "Weather in Oslo?")], tools: tools);
        Assert.Contains("# Tools", text);
        Assert.Contains("\"name\": \"get_weather\"", text);
    }
}
