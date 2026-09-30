using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Tools.Tests;

/// <summary><see cref="ToolLoop.RunAsync"/> against a scripted service: a tool call is dispatched, the next round sees the assistant turn with <see cref="TextMessage.ToolCalls"/> and the <see cref="TextRole.Tool"/> result, the tool result surfaces as a documented <see cref="TextChunkKind.Status"/> chunk, and the round limit, error stops and cancellation end the loop.</summary>
public sealed class ToolLoopTests
{
    private static readonly ModelSpec Spec = new() { Requested = "fake", Modality = Modality.Text };

    private static TextRequest Request(IReadOnlyList<ToolDefinition>? tools = null) => new()
    {
        Messages = [new TextMessage { Role = TextRole.User, Content = "please hang up now" }],
        Tools = tools,
    };

    private static async Task<List<TextChunk>> Collect(IAsyncEnumerable<TextChunk> stream)
    {
        List<TextChunk> chunks = [];
        await foreach (TextChunk chunk in stream) chunks.Add(chunk);
        return chunks;
    }

    [Fact]
    public async Task DispatchesTheCallAppendsTheTurnsAndStreamsTheFinalAnswer()
    {
        int invoked = 0;
        string? seenArguments = null;
        ToolRegistry registry = new ToolRegistry().Add("hang_up", "Ends the call.", "{\"type\":\"object\"}", (args, _) =>
        {
            invoked++;
            seenArguments = args;
            return Task.FromResult("ok");
        });
        ScriptedTextService service = new ScriptedTextService()
            .Round(ScriptedTextService.Text("Sure. "), ScriptedTextService.Call("call_0", "hang_up", "{\"reason\": \"done\"}"),
                ScriptedTextService.Result("Sure. "), ScriptedTextService.Stop(StopReason.ToolCall))
            .Round(ScriptedTextService.Text("Done."), ScriptedTextService.Result("Done."), ScriptedTextService.Stop(StopReason.Stop));

        List<TextChunk> chunks = await Collect(ToolLoop.RunAsync(service, Spec, Request(), registry, maxRounds: 3));

        Assert.Equal(1, invoked);
        Assert.Equal("{\"reason\": \"done\"}", seenArguments);
        Assert.Equal(
            [TextChunkKind.Chunk, TextChunkKind.NativeToolCall, TextChunkKind.Status, TextChunkKind.Chunk, TextChunkKind.Result, TextChunkKind.StopReason],
            chunks.Select(c => c.Kind));
        TextChunk result = chunks[2];
        Assert.Equal(ToolLoop.ToolResultPrefix + "ok", result.Text);
        Assert.Equal(ToolLoop.ToolResultPhase, result.Status!.Value.Phase);
        Assert.Equal("hang_up", result.ToolCall!.Name);
        Assert.Equal(0, result.ToolCallIndex);
        Assert.Equal("Sure. Done.", chunks[4].Text);
        Assert.Equal(StopReason.Stop, chunks[5].Stop);

        Assert.Equal(2, service.Requests.Count);
        Assert.Same(registry.Definitions, service.Requests[0].Tools);
        Assert.Same(registry.Definitions, service.Requests[1].Tools);
        IReadOnlyList<TextMessage> second = service.Requests[1].Messages;
        Assert.Equal(3, second.Count);
        Assert.Equal(TextRole.User, second[0].Role);
        Assert.Equal(TextRole.Assistant, second[1].Role);
        Assert.Equal("Sure. ", second[1].Content);
        Assert.Equal("hang_up", Assert.Single(second[1].ToolCalls!).Name);
        Assert.Equal(TextRole.Tool, second[2].Role);
        Assert.Equal("call_0", second[2].ToolCallId);
        Assert.Equal("hang_up", second[2].Name);
        Assert.Equal("ok", second[2].Content);
        Assert.Single(service.Requests[0].Messages);
    }

    [Fact]
    public async Task RoundLimitEndsWithATerminalStatusAndDoesNotDispatch()
    {
        int invoked = 0;
        ToolRegistry registry = new ToolRegistry().Add("hang_up", () => { invoked++; return "ok"; });
        ScriptedTextService service = new ScriptedTextService()
            .Round(ScriptedTextService.Text("Sure."), ScriptedTextService.Call("call_0", "hang_up"), ScriptedTextService.Result("Sure."), ScriptedTextService.Stop(StopReason.ToolCall));

        List<TextChunk> chunks = await Collect(ToolLoop.RunAsync(service, Spec, Request(), registry, maxRounds: 1));

        Assert.Equal(0, invoked);
        Assert.Single(service.Requests);
        Assert.Equal([TextChunkKind.Chunk, TextChunkKind.NativeToolCall, TextChunkKind.Status, TextChunkKind.Result, TextChunkKind.StopReason], chunks.Select(c => c.Kind));
        Assert.Equal(ToolLoop.RoundLimitPrefix + "max_rounds=1", chunks[2].Text);
        Assert.Equal(ToolLoop.RoundLimitPhase, chunks[2].Status!.Value.Phase);
        Assert.Equal("Sure.", chunks[3].Text);
        Assert.Equal(StopReason.ToolCall, chunks[4].Stop);
    }

    [Fact]
    public async Task UnknownToolResultIsFedBackAndTheLoopContinues()
    {
        ToolRegistry registry = new ToolRegistry().Add("other", () => "x");
        ScriptedTextService service = new ScriptedTextService()
            .Round(ScriptedTextService.Call("call_0", "nope"), ScriptedTextService.Result(""), ScriptedTextService.Stop(StopReason.ToolCall))
            .Round(ScriptedTextService.Text("I cannot."), ScriptedTextService.Result("I cannot."), ScriptedTextService.Stop(StopReason.Stop));

        List<TextChunk> chunks = await Collect(ToolLoop.RunAsync(service, Spec, Request(), registry));

        TextMessage toolTurn = service.Requests[1].Messages[^1];
        Assert.Equal(TextRole.Tool, toolTurn.Role);
        Assert.StartsWith("{\"error\":\"Unknown tool 'nope'", toolTurn.Content, StringComparison.Ordinal);
        Assert.Equal(StopReason.Stop, chunks[^1].Stop);
        Assert.Equal("I cannot.", chunks[^2].Text);
    }

    [Fact]
    public async Task PlainAnswerEndsAfterOneRoundWithTheRequestToolsOfferedAsIs()
    {
        ToolRegistry registry = new ToolRegistry().Add("hang_up", () => "ok");
        ToolDefinition[] offered = [new ToolDefinition { Name = "only_this" }];
        ScriptedTextService service = new ScriptedTextService()
            .Round(ScriptedTextService.Text("Hi"), ScriptedTextService.Text("!"), ScriptedTextService.Result("Hi!"), ScriptedTextService.Stop(StopReason.Length));

        List<TextChunk> chunks = await Collect(ToolLoop.RunAsync(service, Spec, Request(offered), registry));

        Assert.Same(offered, Assert.Single(service.Requests).Tools);
        Assert.Equal([TextChunkKind.Chunk, TextChunkKind.Chunk, TextChunkKind.Result, TextChunkKind.StopReason], chunks.Select(c => c.Kind));
        Assert.Equal("Hi!", chunks[2].Text);
        Assert.Equal(StopReason.Length, chunks[3].Stop);
    }

    [Fact]
    public async Task ErrorStopIsRelayedAndEndsTheLoop()
    {
        ToolRegistry registry = new ToolRegistry().Add("hang_up", () => "ok");
        ScriptedTextService service = new ScriptedTextService()
            .Round(ScriptedTextService.Text("partial"), ScriptedTextService.Stop(StopReason.Error, "load failed"));

        List<TextChunk> chunks = await Collect(ToolLoop.RunAsync(service, Spec, Request(), registry));

        Assert.Equal([TextChunkKind.Chunk, TextChunkKind.StopReason], chunks.Select(c => c.Kind));
        Assert.Equal(StopReason.Error, chunks[1].Stop);
        Assert.Equal("load failed", chunks[1].Text);
    }

    [Fact]
    public async Task CancellationStopsTheLoop()
    {
        ToolRegistry registry = new ToolRegistry().Add("hang_up", () => "ok");
        ScriptedTextService service = new ScriptedTextService()
            .Round(ScriptedTextService.Text("never"), ScriptedTextService.Result("never"), ScriptedTextService.Stop(StopReason.Stop));
        using CancellationTokenSource cts = new();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await Collect(ToolLoop.RunAsync(service, Spec, Request(), registry, cancel: cts.Token)));
    }

    [Fact]
    public async Task InvalidMaxRoundsIsRejected()
    {
        ScriptedTextService service = new();
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () => await Collect(ToolLoop.RunAsync(service, Spec, Request(), new ToolRegistry(), maxRounds: 0)));
    }
}
