using System.Text.Json;
using HartsyInference.Tests.Common;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>One completion from parser_reference.json with the result encoding.py's <c>parse_message_from_completion_text</c> gave it.</summary>
internal sealed record ParserCase(
    string Name,
    bool Thinking,
    string Text,
    bool Ok,
    string Reasoning,
    string Content,
    IReadOnlyList<ExpectedCall> Calls,
    string? Prompt)
{
    public override string ToString() => Name;

    public static IReadOnlyList<ParserCase> All { get; } = Load();

    public static IEnumerable<object[]> Names(bool? ok = null) =>
        All.Where(c => ok is null || c.Ok == ok).Select(c => new object[] { c.Name });

    public static ParserCase ByName(string name) => All.Single(c => c.Name == name);

    private static List<ParserCase> Load()
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(DeepSeekV41ReferenceFiles.ParserReferenceJson));
        List<ParserCase> cases = [];
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            bool ok = c.GetProperty("ok").GetBoolean();
            List<ExpectedCall> calls = [];
            if (ok)
            {
                foreach (JsonElement call in c.GetProperty("tool_calls").EnumerateArray())
                {
                    JsonElement ns = call.GetProperty("namespace");
                    calls.Add(new ExpectedCall(call.GetProperty("name").GetString()!,
                        ns.ValueKind == JsonValueKind.Null ? null : ns.GetString(), call.GetProperty("arguments").GetString()!));
                }
            }
            cases.Add(new ParserCase(c.GetProperty("name").GetString()!, c.GetProperty("mode").GetString() == "thinking",
                c.GetProperty("text").GetString()!, ok, ok ? c.GetProperty("reasoning").GetString()! : "",
                ok ? c.GetProperty("content").GetString()! : "", calls,
                c.TryGetProperty("prompt", out JsonElement p) ? p.GetString() : null));
        }
        return cases;
    }
}
