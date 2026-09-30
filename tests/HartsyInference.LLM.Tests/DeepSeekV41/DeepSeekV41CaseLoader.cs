using System.Text.Json;
using HartsyInference.LLM.ChatTemplates;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Converts an upstream OpenAI-format encoding case (the JSON shape of encoding/tests/test_input_N.json) into the C# conversation model, mirroring upstream <c>load_cases</c>.</summary>
internal static class DeepSeekV41CaseLoader
{
    public static (List<ChatMessage> Messages, EncodeOptions Options) Load(JsonElement testCase, IReadOnlyList<ImageGrid> grids)
    {
        JsonElement messagesElement = testCase.ValueKind == JsonValueKind.Array ? testCase : testCase.GetProperty("messages");
        int imageCounter = 0;
        List<ChatMessage> messages = [];
        foreach (JsonElement message in messagesElement.EnumerateArray())
            messages.Add(ReadMessage(message, ref imageCounter));

        if (testCase.ValueKind == JsonValueKind.Object && testCase.TryGetProperty("tools", out JsonElement tools))
            messages[0] = messages[0] with { Tools = ReadTools(tools) };

        bool thinking = ReadString(testCase, "thinking_mode") == "thinking";
        int? effort = null;
        bool? drop = null;
        if (testCase.ValueKind == JsonValueKind.Object)
        {
            if (testCase.TryGetProperty("reasoning_effort", out JsonElement e))
                effort = e.ValueKind == JsonValueKind.Number ? e.GetInt32() : EncodeOptions.ParseReasoningEffort(e.GetString()!);
            if (testCase.TryGetProperty("drop_thinking", out JsonElement d)) drop = d.GetBoolean();
        }
        EncodeOptions options = new() { Thinking = thinking, ReasoningEffort = effort, DropThinking = drop, Images = grids };
        return (messages, options);
    }

    public static List<ImageGrid> ReadGrids(JsonElement grids) =>
        grids.EnumerateArray().Select(g => new ImageGrid(g[0].GetInt32(), g[1].GetInt32())).ToList();

    private static string? ReadString(JsonElement el, string name) =>
        el.ValueKind == JsonValueKind.Object && el.TryGetProperty(name, out JsonElement v) && v.ValueKind == JsonValueKind.String
            ? v.GetString() : null;

    private static ChatMessage ReadMessage(JsonElement el, ref int imageCounter)
    {
        string role = el.GetProperty("role").GetString()!;
        string content = string.Empty;
        List<ContentBlock>? blocks = null;
        if (el.TryGetProperty("content", out JsonElement c))
        {
            if (c.ValueKind == JsonValueKind.String) content = c.GetString()!;
            else if (c.ValueKind == JsonValueKind.Array) blocks = ReadBlocks(c, ref imageCounter);
        }
        return new ChatMessage(role, content)
        {
            Blocks = blocks,
            ToolCalls = el.TryGetProperty("tool_calls", out JsonElement calls) ? ReadCalls(calls) : null,
            ToolCallId = ReadString(el, "tool_call_id"),
            ReasoningContent = ReadString(el, "reasoning_content"),
            Task = ReadString(el, "task"),
            Tools = el.TryGetProperty("tools", out JsonElement tools) ? ReadTools(tools) : null,
            ResponseFormatJson = el.TryGetProperty("response_format", out JsonElement rf) ? rf.GetRawText() : null,
            WithoutEos = el.TryGetProperty("wo_eos", out JsonElement wo) && wo.GetBoolean(),
        };
    }

    private static List<ContentBlock> ReadBlocks(JsonElement array, ref int imageCounter)
    {
        List<ContentBlock> blocks = [];
        foreach (JsonElement block in array.EnumerateArray())
        {
            string type = block.GetProperty("type").GetString()!;
            if (type == "text") blocks.Add(new TextBlock(block.GetProperty("text").GetString() ?? string.Empty));
            else if (type is "image" or "image_url") blocks.Add(new ImageBlock(imageCounter++));
            else throw new NotSupportedException($"Fixture block type {type}.");
        }
        return blocks;
    }

    private static List<ToolSpec> ReadTools(JsonElement tools) =>
        tools.EnumerateArray().Select(t => ToolSpec.FromJson(t.GetRawText())).ToList();

    private static List<ChatToolCall> ReadCalls(JsonElement calls)
    {
        List<ChatToolCall> result = [];
        foreach (JsonElement call in calls.EnumerateArray())
        {
            JsonElement function = call.GetProperty("function");
            JsonElement arguments = function.GetProperty("arguments");
            string argumentsJson = arguments.ValueKind == JsonValueKind.String ? arguments.GetString()! : arguments.GetRawText();
            string? ns = ReadString(call, "namespace") ?? ReadString(function, "namespace");
            result.Add(new ChatToolCall(ReadString(call, "id") ?? string.Empty, function.GetProperty("name").GetString()!, argumentsJson)
            {
                Namespace = ns,
            });
        }
        return result;
    }
}
