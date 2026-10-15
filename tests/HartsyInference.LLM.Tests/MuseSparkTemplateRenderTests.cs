using System.Text.Json;
using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Both Muse Spark Open chat template revisions must render byte-for-byte what transformers' own chat-template renderer produces
/// (<c>&lt;template&gt;.expected.json</c>, from <c>tests/python-reference/dump_muse_spark_template_reference.py</c>): plain chat, a system
/// prompt, each reasoning effort, earlier reasoning, tool definitions, a tool call with its result, and caller text holding control
/// sequences the template must defuse. <c>muse-spark-open.jinja</c> is the one llama.cpp wrote into the released GGUFs;
/// <c>muse-spark-open-rev2.jinja</c> adds namespace tools and deferred tool loading, with cases of their own.</summary>
public sealed class MuseSparkTemplateRenderTests
{
    private static readonly string Dir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "tests", "HartsyInference.LLM.Tests",
        "Fixtures", "templates");

    private static readonly string[] Templates = ["muse-spark-open", "muse-spark-open-rev2"];

    public static TheoryData<string, string> Cases()
    {
        TheoryData<string, string> cases = new();
        foreach (string template in Templates)
        {
            using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, template + ".expected.json")));
            foreach (JsonElement c in doc.RootElement.EnumerateArray()) cases.Add(template, c.GetProperty("name").GetString()!);
        }
        return cases;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Render_MatchesTransformers(string template, string name)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, template + ".expected.json")));
        JsonElement c = doc.RootElement.EnumerateArray().Single(e => e.GetProperty("name").GetString() == name);
        Dictionary<string, object?> context = new()
        {
            ["messages"] = Values.FromJson(c.GetProperty("messages")),
            ["add_generation_prompt"] = true,
            ["tools"] = c.TryGetProperty("tools", out JsonElement tools) ? Values.FromJson(tools) : null,
        };
        foreach (JsonProperty kw in c.GetProperty("kwargs").EnumerateObject()) context[kw.Name] = Values.FromJson(kw.Value);

        string rendered = new JinjaEngine(File.ReadAllText(Path.Combine(Dir, template + ".jinja"))).Render(context);

        Assert.Equal(c.GetProperty("rendered").GetString(), rendered);
    }

    [Fact]
    public void UnsupportedEffort_RaisesLikeTheReference()
    {
        Dictionary<string, object?> context = new()
        {
            ["messages"] = new List<object?> { new Dictionary<string, object?> { ["role"] = "user", ["content"] = "Hi" } },
            ["add_generation_prompt"] = true,
            ["reasoning_effort"] = "max",
        };
        JinjaEngine engine = new(File.ReadAllText(Path.Combine(Dir, "muse-spark-open.jinja")));
        ChatTemplateRaiseException ex = Assert.Throws<ChatTemplateRaiseException>(() => engine.Render(context));
        Assert.Contains("Unsupported reasoning_effort", ex.Message, StringComparison.Ordinal);
    }
}
