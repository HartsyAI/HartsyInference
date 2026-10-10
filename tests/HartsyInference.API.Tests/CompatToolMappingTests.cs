using System.Text.Json;
using HartsyInference.API.Endpoints;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>OpenAI-compat tool mapping both ways: an assistant turn's <c>tool_calls</c> and a tool turn's <c>tool_call_id</c>/<c>name</c> reach the native <see cref="TextMessage"/>, and a native <see cref="NativeToolCall"/> serializes back in OpenAI's wire shape.</summary>
public sealed class CompatToolMappingTests
{
    private const string RequestJson = """
        {
          "model": "m",
          "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": null, "tool_calls": [
              {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "{\"temp\":21}"}
          ],
          "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
        }
        """;

    [Fact]
    public void RequestToolCallsAndToolResultsMapToTheNativeMessages()
    {
        ChatCompletionRequest req = JsonSerializer.Deserialize<ChatCompletionRequest>(RequestJson)!;
        TextRequest native = CompatEndpoints.ToTextRequest(req);
        Assert.Equal(3, native.Messages.Count);

        TextMessage assistant = native.Messages[1];
        Assert.Equal(TextRole.Assistant, assistant.Role);
        Assert.Equal("", assistant.Content);
        NativeToolCall call = Assert.Single(assistant.ToolCalls!);
        Assert.Equal(("call_1", "get_weather", "{\"city\":\"Paris\"}"), (call.Id, call.Name, call.Arguments));

        TextMessage tool = native.Messages[2];
        Assert.Equal(TextRole.Tool, tool.Role);
        Assert.Equal("call_1", tool.ToolCallId);
        Assert.Equal("get_weather", tool.Name);
        Assert.Equal("{\"temp\":21}", tool.Content);
        Assert.Null(native.Messages[0].ToolCalls);

        ToolDefinition definition = Assert.Single(native.Tools!);
        Assert.Equal("get_weather", definition.Name);
        Assert.Equal("{\"type\": \"object\"}", definition.JsonSchema);
    }

    [Fact]
    public void NativeToolCallSerializesInOpenAiWireShape()
    {
        ChatToolCallDto dto = CompatEndpoints.ToToolCallDto(new NativeToolCall { Id = "call_9", Name = "hang_up", Arguments = "{\"reason\":\"done\"}" }, index: 0);
        JsonElement json = JsonSerializer.SerializeToElement(dto);
        Assert.Equal("call_9", json.GetProperty("id").GetString());
        Assert.Equal("function", json.GetProperty("type").GetString());
        Assert.Equal(0, json.GetProperty("index").GetInt32());
        Assert.Equal("hang_up", json.GetProperty("function").GetProperty("name").GetString());
        Assert.Equal("{\"reason\":\"done\"}", json.GetProperty("function").GetProperty("arguments").GetString());
    }

    [Fact]
    public void ToolCallIdAndNameAreOmittedFromAnAssistantResponseWhenUnset()
    {
        ChatMessageDto message = new() { Role = "assistant", Content = "hi" };
        JsonElement json = JsonSerializer.SerializeToElement(message);
        Assert.False(json.TryGetProperty("tool_call_id", out _));
        Assert.False(json.TryGetProperty("name", out _));

        ChatMessageDto tool = new() { Role = "tool", Content = "r", ToolCallId = "call_1", Name = "f" };
        JsonElement toolJson = JsonSerializer.SerializeToElement(tool);
        Assert.Equal("call_1", toolJson.GetProperty("tool_call_id").GetString());
        Assert.Equal("f", toolJson.GetProperty("name").GetString());
    }
}
