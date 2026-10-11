using HartsyInference.LLM.ChatTemplates;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>The real Qwen3.5-0.8B chat template (extracted from the GGUF, <c>Fixtures/templates/qwen35-0.8b.jinja</c>) renders. It defines and calls a
/// <c>render_content</c> macro, which the engine did not support, so every Qwen3.5 request fell back to ChatML and its tool-calling turn came out empty.</summary>
public sealed class Qwen35TemplateRenderTests
{
    private readonly ITestOutputHelper _output;
    public Qwen35TemplateRenderTests(ITestOutputHelper output) => _output = output;

    private static string Template() => File.ReadAllText(Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path,
        "tests", "HartsyInference.LLM.Tests", "Fixtures", "templates", "qwen35-0.8b.jinja"));

    private static Dictionary<string, object?> Msg(string role, object? content, params (string Key, object? Value)[] extra)
    {
        Dictionary<string, object?> m = new() { ["role"] = role, ["content"] = content };
        foreach ((string k, object? v) in extra) m[k] = v;
        return m;
    }

    private static Dictionary<string, object?> Tool() => new()
    {
        ["type"] = "function",
        ["function"] = new Dictionary<string, object?>
        {
            ["name"] = "get_time",
            ["description"] = "Current time in a city",
            ["parameters"] = new Dictionary<string, object?>
            {
                ["type"] = "object",
                ["properties"] = new Dictionary<string, object?> { ["city"] = new Dictionary<string, object?> { ["type"] = "string" } },
                ["required"] = new List<object?> { "city" },
            },
        },
    };

    private string Render(List<object?> messages, bool withTools, bool addGen = true)
    {
        Dictionary<string, object?> ctx = new()
        {
            ["messages"] = messages,
            ["add_generation_prompt"] = addGen,
            ["bos_token"] = "",
            ["eos_token"] = "<|im_end|>",
        };
        if (withTools) ctx["tools"] = new List<object?> { Tool() };
        string result = new JinjaEngine(Template()).Render(ctx);
        _output.WriteLine(result);
        return result;
    }

    [Fact]
    public void PlainChat_Renders()
    {
        string r = Render([Msg("user", "Hello")], withTools: false);
        Assert.Contains("<|im_start|>user\nHello<|im_end|>", r);
        Assert.EndsWith("<|im_start|>assistant\n", r.Substring(0, r.IndexOf("<|im_start|>assistant\n", StringComparison.Ordinal) + "<|im_start|>assistant\n".Length));
    }

    [Fact]
    public void ToolDefinitions_AreRenderedIntoTheSystemTurn()
    {
        string r = Render([Msg("user", "What time is it in Paris?")], withTools: true);
        Assert.Contains("# Tools", r);
        Assert.Contains("\"name\": \"get_time\"", r);
        Assert.Contains("<|im_start|>user\nWhat time is it in Paris?<|im_end|>", r);
    }

    [Fact]
    public void ToolCallRoundTrip_ReplaysTheCallAndTheResult()
    {
        Dictionary<string, object?> call = new()
        {
            ["type"] = "function",
            ["function"] = new Dictionary<string, object?> { ["name"] = "get_time", ["arguments"] = new Dictionary<string, object?> { ["city"] = "Paris" } },
        };
        string r = Render([
            Msg("user", "What time is it in Paris?"),
            Msg("assistant", "", ("tool_calls", new List<object?> { call })),
            Msg("tool", "14:05"),
        ], withTools: true);
        // The call replays with its argument by name and value, not as a key/value pair printed into the tag.
        Assert.Contains("<tool_call>\n<function=get_time>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>", r);
        // A run of tool results opens one user turn for the model to read them in.
        Assert.Contains("<|im_end|>\n<|im_start|>user\n<tool_response>\n14:05\n</tool_response><|im_end|>\n<|im_start|>assistant\n", r);
    }
}
