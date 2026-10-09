using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.API;
using HartsyInference.API.Endpoints;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The streamed chat reply, from native chunks through <see cref="ChatStreamTranslator"/> to the wire: the golden transcript for a turn with two streamed tool
/// calls. The expected frames are written out from the OpenAI wire shape, so a change to the translator shows up as a difference from them.</summary>
public sealed class ChatStreamTranslatorTests
{
    /// <summary>The app's JSON setup for the chat frames: web defaults, with string enums. Every frame property has an explicit name, so the naming policy does not reach them.</summary>
    private static readonly JsonSerializerOptions s_options = new(JsonSerializerDefaults.Web) { Converters = { new JsonStringEnumConverter() } };

    private static string Frame(ChatCompletionChunk chunk) => $"data: {JsonSerializer.Serialize(chunk, s_options)}\n\n";

    [Fact]
    public void Two_Streamed_Tool_Calls_Produce_The_Golden_Transcript()
    {
        ChatStreamTranslator translator = new("chatcmpl-test", 1700000000, "hartsy-test", includeUsage: true);
        List<TextChunk> native =
        [
            new() { Kind = TextChunkKind.Reasoning, Text = "Need a tool." },
            new() { Kind = TextChunkKind.Chunk, Text = "Checking. " },
            new() { Kind = TextChunkKind.ToolCallDelta, ToolCallIndex = 0, ToolCall = new NativeToolCall { Id = "call_7_0", Name = "search", Arguments = "" } },
            new() { Kind = TextChunkKind.ToolCallDelta, ToolCallIndex = 0, Text = "{\"q\":" },
            new() { Kind = TextChunkKind.ToolCallDelta, ToolCallIndex = 0, Text = "\"cats\"}" },
            new() { Kind = TextChunkKind.NativeToolCall, ToolCallIndex = 0, ToolCall = new NativeToolCall { Id = "call_7_0", Name = "search", Arguments = "{\"q\":\"cats\"}" } },
            new() { Kind = TextChunkKind.ToolCallDelta, ToolCallIndex = 1, ToolCall = new NativeToolCall { Id = "call_7_1", Name = "lookup", Arguments = "" } },
            new() { Kind = TextChunkKind.ToolCallDelta, ToolCallIndex = 1, Text = "{\"id\":3}" },
            new() { Kind = TextChunkKind.NativeToolCall, ToolCallIndex = 1, ToolCall = new NativeToolCall { Id = "call_7_1", Name = "lookup", Arguments = "{\"id\":3}" } },
            new() { Kind = TextChunkKind.StopReason, Stop = StopReason.ToolCall },
            new() { Kind = TextChunkKind.Usage, Usage = new TextUsage(42, 17) },
        ];

        StringBuilder transcript = new();
        transcript.Append(Frame(translator.Start()));
        foreach (TextChunk chunk in native)
            foreach (ChatCompletionChunk frame in translator.Handle(chunk))
                transcript.Append(Frame(frame));
        foreach (ChatCompletionChunk frame in translator.End())
            transcript.Append(Frame(frame));
        transcript.Append("data: [DONE]\n\n");

        string[] golden =
        [
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"hartsy-test\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"hartsy-test\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Need a tool.\"},\"finish_reason\":null}]}",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"hartsy-test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Checking. \"},\"finish_reason\":null}]}",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"hartsy-test\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_7_0\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"hartsy-test\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\u0022q\\u0022:\"}}]},\"finish_reason\":null}]}",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"hartsy-test\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\u0022cats\\u0022}\"}}]},\"finish_reason\":null}]}",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"hartsy-test\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_7_1\",\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"hartsy-test\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"{\\u0022id\\u0022:3}\"}}]},\"finish_reason\":null}]}",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"hartsy-test\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}",
            "data: {\"id\":\"chatcmpl-test\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"hartsy-test\",\"choices\":[],\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":17,\"total_tokens\":59}}",
            "data: [DONE]",
        ];
        Assert.Equal(string.Concat(golden.Select(line => line + "\n\n")), transcript.ToString());
    }

    [Fact]
    public void A_Whole_Call_That_Reuses_An_Aborted_Calls_Index_Is_Still_Sent_With_Its_Arguments()
    {
        ChatStreamTranslator translator = new("chatcmpl-test", 1, "m", includeUsage: false);
        Assert.Single(translator.Handle(new TextChunk
        {
            Kind = TextChunkKind.ToolCallDelta, ToolCallIndex = 0, ToolCall = new NativeToolCall { Id = "call_1_0", Name = "search", Arguments = "" },
        }));
        Assert.Single(translator.Handle(new TextChunk { Kind = TextChunkKind.ToolCallDelta, ToolCallIndex = 0, Text = "{\"q\":" }));
        Assert.Empty(translator.Handle(new TextChunk { Kind = TextChunkKind.ToolCallAbort, ToolCallIndex = 0 }));

        ChatCompletionChunk frame = Assert.Single(translator.Handle(new TextChunk
        {
            Kind = TextChunkKind.NativeToolCall, ToolCallIndex = 0, ToolCall = new NativeToolCall { Id = "call_1_0", Name = "lookup", Arguments = "{\"id\":3}" },
        }));

        ChatToolCallDeltaDto call = Assert.Single(frame.Choices[0].Delta.ToolCalls!);
        Assert.Equal(0, call.Index);
        Assert.Equal("call_1_0", call.Id);
        Assert.Equal("lookup", call.Function.Name);
        Assert.Equal("{\"id\":3}", call.Function.Arguments);
    }

    [Fact]
    public void A_Call_Without_Streamed_Fragments_Is_Sent_Once_Whole_And_Usage_Stays_Off_Unless_Asked()
    {
        ChatStreamTranslator translator = new("chatcmpl-test", 1, "m", includeUsage: false);

        List<ChatCompletionChunk> frames = [.. translator.Handle(new TextChunk
        {
            Kind = TextChunkKind.NativeToolCall,
            ToolCall = new NativeToolCall { Id = "call_1_0", Name = "search", Arguments = "{}" },
        })];

        ChatCompletionChunk frame = Assert.Single(frames);
        ChatToolCallDeltaDto call = Assert.Single(frame.Choices[0].Delta.ToolCalls!);
        Assert.Equal(0, call.Index);
        Assert.Equal("call_1_0", call.Id);
        Assert.Equal("function", call.Type);
        Assert.Equal("search", call.Function.Name);
        Assert.Equal("{}", call.Function.Arguments);

        // Without include_usage the stream ends at the finish frame, whatever usage the native stream reported. Handle is lazy, so the chunk is enumerated.
        Assert.Empty(translator.Handle(new TextChunk { Kind = TextChunkKind.Usage, Usage = new TextUsage(3, 4) }));
        ChatCompletionChunk finish = Assert.Single(translator.End());
        Assert.Equal("stop", finish.Choices[0].FinishReason);
        Assert.Null(finish.Usage);
    }
}
