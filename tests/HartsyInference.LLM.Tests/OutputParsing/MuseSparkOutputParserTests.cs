using System.Text.Json;
using System.Text.Json.Nodes;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.OutputParsing;
using Xunit;

namespace HartsyInference.LLM.Tests.OutputParsing;

/// <summary>Muse Spark completions split into reasoning (<c>to=self</c>), answer text (<c>to=user</c>, <c>to=commentary</c>) and
/// <c>&lt;atem:invoke&gt;</c> tool calls, at any token split. The round trip renders an assistant turn through the real Muse Spark Open
/// template (whose renders match transformers, see MuseSparkTemplateRenderTests) and parses the completion part back.</summary>
public sealed class MuseSparkOutputParserTests
{
    private static readonly string[] MuseLiterals = ["<|start|>", "<|message|>", "<|eom|>", "<|eot|>", "<|end_of_text|>"];

    private static readonly string TemplatePath = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "tests", "HartsyInference.LLM.Tests",
        "Fixtures", "templates", "muse-spark-open.jinja");

    private const string WeatherTool = """
        {"type":"function","function":{"name":"get_weather","description":"Forecast","parameters":{"type":"object","properties":{
        "city":{"type":"string"},"days":{"type":"integer"},"zip":{"type":"string"},"units":{"type":["string","null"]},
        "detail":{"type":"boolean"},"spots":{"type":"array","items":{"type":"string"}},"opts":{"type":"object"},"note":{"type":"string"}},
        "required":["city"]}}}
        """;

    private const string TimeTool = """
        {"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"tz":{"type":"string"}}}}}
        """;

    private const string WeatherArgs = """
        {"city":"São Paulo","days":3,"zip":"75001","units":"metric","detail":true,"spots":["a","b"],"opts":{"k":1},
        "note":"line1\nline2 \"q\" < |eot|>"}
        """;

    private static (ParsedAssistant Result, ReplayedTurn Replay) Parse(string completion, int seed, IReadOnlyList<ToolSpec>? tools = null)
    {
        PieceTokenizer tok = new(MuseLiterals);
        MuseSparkOutputParser parser = new(tok, tools);
        ParsedAssistant result = ReplayedTurn.Run(parser, tok.RandomWithSpecials(completion, new Random(seed), 4), out ReplayedTurn replay);
        return (result, replay);
    }

    private static void AssertReplayMatches(ParsedAssistant result, ReplayedTurn replay)
    {
        Assert.Equal(result.Reasoning, replay.Reasoning.ToString());
        Assert.Equal(result.Content, replay.Content.ToString());
        Assert.Equal(result.ToolCalls.Select(c => (c.Name, c.ArgumentsJson)), replay.Calls.Select(c => (c.Name, c.Args.ToString())));
        Assert.Equal(1, replay.Stops);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void ReasoningThenAnswer_SplitsTheChannels(int seed)
    {
        (ParsedAssistant result, ReplayedTurn replay) =
            Parse(" to=self<|message|>The user greets.<|eom|><|start|>assistant to=user<|message|>Hello!<|eot|>", seed);

        Assert.Equal("The user greets.", result.Reasoning);
        Assert.Equal("Hello!", result.Content);
        Assert.Empty(result.ToolCalls);
        Assert.False(result.Malformed);
        Assert.True(result.Completed);
        AssertReplayMatches(result, replay);
    }

    [Fact]
    public void MessageWithoutRecipient_IsTheAnswer()
    {
        (ParsedAssistant result, _) = Parse("<|message|>Hi<|eot|>", 1);

        Assert.Equal("Hi", result.Content);
        Assert.False(result.Malformed);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(9)]
    public void TemplateRenderedTurn_RoundTrips(int seed)
    {
        ToolSpec[] tools = [ToolSpec.FromJson(WeatherTool), ToolSpec.FromJson(TimeTool)];
        string completion = RenderAssistantTurn(tools);

        (ParsedAssistant result, ReplayedTurn replay) = Parse(completion, seed, tools);

        Assert.Equal("Need the forecast first.", result.Reasoning);
        Assert.Equal("Checking the forecast.", result.Content);
        Assert.Equal(["get_weather", "get_time"], result.ToolCalls.Select(c => c.Name));
        string weather = result.ToolCalls[0].ArgumentsJson;
        Assert.True(JsonNode.DeepEquals(JsonNode.Parse(WeatherArgs), JsonNode.Parse(weather)), weather);
        Assert.Equal("""{"tz":"Europe/Paris"}""", result.ToolCalls[1].ArgumentsJson);
        Assert.Equal(["call_0", "call_1"], result.ToolCalls.Select(c => c.Id));
        Assert.False(result.Malformed);
        AssertReplayMatches(result, replay);
    }

    [Fact]
    public void WithoutASchema_ValuesThatParseAsJsonAreJson()
    {
        string completion = RenderAssistantTurn([ToolSpec.FromJson(WeatherTool), ToolSpec.FromJson(TimeTool)]);

        (ParsedAssistant result, _) = Parse(completion, 1);

        JsonNode args = JsonNode.Parse(result.ToolCalls[0].ArgumentsJson)!;
        Assert.Equal(JsonValueKind.Number, args["zip"]!.GetValueKind());
        Assert.Equal("São Paulo", args["city"]!.GetValue<string>());
        Assert.Equal(3, args["days"]!.GetValue<int>());
    }

    [Fact]
    public void NullableStringParameter_KeepsBareNullAsText()
    {
        // The template writes None and the string "null" alike; a schema that admits a string keeps the text, as llama.cpp does.
        ToolSpec tool = ToolSpec.FromJson("""
            {"type":"function","function":{"name":"f","parameters":{"type":"object",
            "properties":{"u":{"type":["string","null"]},"n":{"type":["integer","null"]}}}}}
            """);
        const string body = " to=f<|message|><atem:function_calls>\n<atem:invoke name=\"f\">\n<atem:parameter name=\"u\">null</atem:parameter>\n" +
            "<atem:parameter name=\"n\">null</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>";

        (ParsedAssistant result, _) = Parse(body, 1, [tool]);

        Assert.Equal("""{"u":"null","n":null}""", Assert.Single(result.ToolCalls).ArgumentsJson);
    }

    [Fact]
    public void TruncatedReasoning_IsCompletedNotMalformed()
    {
        (ParsedAssistant result, ReplayedTurn replay) = Parse(" to=self<|message|>still thinking <|e", 1);

        Assert.Equal("still thinking <|e", result.Reasoning);
        Assert.Equal("", result.Content);
        Assert.False(result.Malformed);
        Assert.True(result.Completed);
        Assert.Equal(1, replay.Stops);
    }

    [Fact]
    public void PartialMarker_IsHeldBackWhileStreaming()
    {
        PieceTokenizer tok = new(MuseLiterals);
        MuseSparkOutputParser parser = new(tok);
        List<ParsedEvent> events = [];
        foreach (int id in tok.SplitBytes(" to=user<|message|>Hello <|eo", 3, 9, 19, 22, 27)) parser.Push(id, events.Add);

        Assert.Equal("Hello ", string.Concat(events.Where(e => e.Kind == ParsedEventKind.ContentDelta).Select(e => e.Text)));
        Assert.DoesNotContain(events, e => e.Kind is ParsedEventKind.Stop or ParsedEventKind.Malformed);
    }

    [Fact]
    public void TruncatedToolCall_IsMalformedAndDropped()
    {
        (ParsedAssistant result, ReplayedTurn replay) = Parse(
            " to=get_time<|message|><atem:function_calls>\n<atem:invoke name=\"get_time\">\n<atem:parameter name=\"tz\">Eur", 1);

        Assert.Empty(result.ToolCalls);
        Assert.Equal("", result.Content);
        Assert.True(result.Malformed);
        Assert.Equal(1, replay.Stops);
    }

    [Fact]
    public void ToolMessageWithoutMarkup_IsShownAsContent()
    {
        (ParsedAssistant result, _) = Parse(" to=get_time<|message|>I cannot call that.<|eot|>", 1);

        Assert.Empty(result.ToolCalls);
        Assert.Equal("I cannot call that.", result.Content);
        Assert.True(result.Malformed);
    }

    [Fact]
    public void AnotherRolesMessage_EndsTheTurn()
    {
        (ParsedAssistant result, ReplayedTurn replay) = Parse(" to=user<|message|>Done.<|eom|><|start|>user<|message|>Thanks!<|eot|>", 1);

        Assert.Equal("Done.", result.Content);
        Assert.True(result.Malformed);
        Assert.Equal(1, replay.Stops);
    }

    [Fact]
    public void EscapedNamesAreDecoded()
    {
        const string body = " to=a<|message|><atem:function_calls>\n<atem:invoke name=\"a&amp;b\">\n" +
            "<atem:parameter name=\"x&quot;y\">v</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>";

        (ParsedAssistant result, _) = Parse(body, 1);

        ChatToolCall call = Assert.Single(result.ToolCalls);
        Assert.Equal("a&b", call.Name);
        Assert.Equal("""{"x\"y":"v"}""", call.ArgumentsJson);
    }

    [Fact]
    public void Handles_RecognizesTheMuseTemplateAndVocabulary()
    {
        JinjaChatTemplate muse = new(File.ReadAllText(TemplatePath));

        Assert.True(MuseSparkOutputParser.Handles(muse, new PieceTokenizer(MuseLiterals)));
        Assert.False(MuseSparkOutputParser.Handles(muse, new PieceTokenizer()));
        Assert.False(MuseSparkOutputParser.Handles(new JinjaChatTemplate("{{ messages[0]['content'] }}"), new PieceTokenizer(MuseLiterals)));
    }

    /// <summary>The template's render of one assistant turn (reasoning, a commentary aside, two tool calls) after the generation prompt
    /// <c>&lt;|start|&gt;assistant</c>, which is exactly what the model writes for that turn.</summary>
    private static string RenderAssistantTurn(IReadOnlyList<ToolSpec> tools)
    {
        JsonArray calls = new(
            new JsonObject { ["type"] = "function", ["id"] = "call_0", ["function"] = new JsonObject
                { ["name"] = "get_weather", ["arguments"] = JsonNode.Parse(WeatherArgs.Replace("< |eot|>", "<|eot|>", StringComparison.Ordinal)) } },
            new JsonObject { ["type"] = "function", ["id"] = "call_1", ["function"] = new JsonObject
                { ["name"] = "get_time", ["arguments"] = new JsonObject { ["tz"] = "Europe/Paris" } } });
        JsonArray messages = new(
            new JsonObject { ["role"] = "user", ["content"] = "Plan a trip" },
            new JsonObject
            {
                ["role"] = "assistant",
                ["reasoning_content"] = "Need the forecast first.",
                ["content"] = "Checking the forecast.",
                ["tool_calls"] = calls,
            });
        using JsonDocument doc = JsonDocument.Parse(messages.ToJsonString());
        using JsonDocument toolDoc = JsonDocument.Parse("[" + string.Join(",", tools.Select(t => t.Json)) + "]");
        Dictionary<string, object?> context = new()
        {
            ["messages"] = Values.FromJson(doc.RootElement),
            ["tools"] = Values.FromJson(toolDoc.RootElement),
            ["add_generation_prompt"] = false,
            ["current_date"] = "2026-10-15",
        };
        string rendered = new JinjaEngine(File.ReadAllText(TemplatePath)).Render(context);
        const string prompt = "<|start|>assistant";
        int user = rendered.LastIndexOf("<|start|>user<|message|>", StringComparison.Ordinal);
        int turn = rendered.IndexOf(prompt, user, StringComparison.Ordinal);
        return rendered[(turn + prompt.Length)..];
    }
}
