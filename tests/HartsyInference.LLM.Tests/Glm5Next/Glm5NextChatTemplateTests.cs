using System.Text.Json;
using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests.Glm5Next;

/// <summary>The engine's Jinja engine against the string transformers' Jinja environment renders from Sage S2.1 Flash's real <c>chat_template.jinja</c>
/// (<c>tools/parity/glm5_next_template_reference.py</c>, transformers 52bac5d0). The template uses macros, namespaces, <c>break</c>, <c>list.0</c>,
/// <c>capitalize</c> and <c>tojson</c>.</summary>
public sealed class Glm5NextChatTemplateTests
{
    private static readonly string Fixtures = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Glm5Next");

    public static TheoryData<string> CaseNames()
    {
        TheoryData<string> names = [];
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "template_cases.json")));
        foreach (JsonElement c in doc.RootElement.EnumerateArray()) names.Add(c.GetProperty("name").GetString()!);
        return names;
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Rendered_prompt_equals_the_transformers_render(string name)
    {
        JinjaEngine engine = new(File.ReadAllText(Path.Combine(Fixtures, "sage-s21-flash-chat_template.jinja")).Replace("\r\n", "\n"), hfWhitespace: true);
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Fixtures, "template_cases.json")));
        JsonElement c = doc.RootElement.EnumerateArray().Single(e => e.GetProperty("name").GetString() == name);
        Dictionary<string, object?> context = new()
        {
            ["messages"] = Plain(c.GetProperty("messages")),
            ["add_generation_prompt"] = c.GetProperty("add_generation_prompt").GetBoolean(),
            ["bos_token"] = string.Empty,
            ["eos_token"] = string.Empty,
            ["tools"] = c.TryGetProperty("tools", out JsonElement tools) ? Plain(tools) : null,
            ["documents"] = null,
        };

        // A Windows checkout may hand the template and the expected strings CRLF line endings; the reference was rendered from LF.
        Assert.Equal(c.GetProperty("rendered").GetString()!.Replace("\r\n", "\n"), engine.Render(context));
    }

    private static object? Plain(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.Object => e.EnumerateObject().ToDictionary(p => p.Name, p => Plain(p.Value)),
        JsonValueKind.Array => e.EnumerateArray().Select(Plain).ToList(),
        JsonValueKind.String => e.GetString(),
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Number => e.TryGetInt64(out long l) ? l : e.GetDouble(),
        _ => null,
    };
}
