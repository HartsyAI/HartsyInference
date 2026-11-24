using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The chat template shipped with SenseNova-Loop-9B (<c>chat_template.jinja</c> @ 3cc220e, copied unchanged) must compile and render in the
/// engine's Jinja interpreter. A construct the interpreter lacks degrades silently to ChatML (a <c>[WRN]</c> line), which would drop the thinking
/// switch, so each shape the service sends is rendered here.</summary>
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
