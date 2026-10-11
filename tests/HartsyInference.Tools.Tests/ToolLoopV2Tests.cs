using HartsyInference.Engine.Requests;
using HartsyInference.Tools;
using Xunit;

namespace HartsyInference.Tools.Tests;

/// <summary>The v2 loop over an arbitrary model stream: the host supplies each round, gates calls, reads the conversation afterwards, and the loop repairs ids it cannot trust.</summary>
public sealed class ToolLoopV2Tests
{
    /// <summary>A stream that plays the given rounds in order and records each round's request.</summary>
    private sealed class Rounds(params TextChunk[][] scripts)
    {
        private int _next;
        public List<TextRequest> Requests { get; } = [];

        public async IAsyncEnumerable<TextChunk> Stream(TextRequest request, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancel = default)
        {
            Requests.Add(request);
            TextChunk[] script = scripts[Math.Min(_next++, scripts.Length - 1)];
            foreach (TextChunk chunk in script)
            {
                cancel.ThrowIfCancellationRequested();
                yield return chunk;
            }
            await Task.CompletedTask;
        }
    }

    private static TextChunk Call(string id, string name, string args = "{}") => new() { Kind = TextChunkKind.NativeToolCall, ToolCall = new NativeToolCall { Id = id, Name = name, Arguments = args } };

    private static TextChunk Text(string text) => new() { Kind = TextChunkKind.Chunk, Text = text };

    private static TextChunk Stop(StopReason reason) => new() { Kind = TextChunkKind.StopReason, Stop = reason };

    private static TextRequest Request(string? forced = null) => new()
    {
        Messages = [new TextMessage { Role = TextRole.User, Content = "time?" }],
        Tools = [new ToolDefinition { Name = "get_time", Description = "time", JsonSchema = "{}" }],
        ForceToolId = forced,
    };

    private static async Task<List<TextChunk>> Collect(ToolLoopRun run)
    {
        List<TextChunk> chunks = [];
        await foreach (TextChunk chunk in run.RunAsync()) chunks.Add(chunk);
        return chunks;
    }

    [Fact]
    public async Task ArbitraryStreamOverloadRunsTheLoop()
    {
        Rounds rounds = new([Text("Checking. "), Call("c1", "get_time"), Stop(StopReason.ToolCall)], [Text("It is 14:05."), Stop(StopReason.Stop)]);
        ToolRegistry tools = new ToolRegistry().Add("get_time", "time", "{}", (_, _) => Task.FromResult("{\"now\":\"14:05\"}"));
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), tools);
        List<TextChunk> chunks = await Collect(run);
        Assert.Equal(2, run.Rounds);
        Assert.Equal([TextChunkKind.Chunk, TextChunkKind.NativeToolCall, TextChunkKind.ToolResult, TextChunkKind.Chunk, TextChunkKind.Result, TextChunkKind.StopReason],
            chunks.Select(c => c.Kind));
        Assert.Equal("Checking. It is 14:05.", run.VisibleText);
        Assert.Equal(StopReason.Stop, run.Stop);
    }

    [Fact]
    public async Task CallsWithPlainStopAreStillDispatched()
    {
        int invoked = 0;
        Rounds rounds = new([Call("c1", "get_time"), Stop(StopReason.Stop)], [Text("done"), Stop(StopReason.Stop)]);
        ToolRegistry tools = new ToolRegistry().Add("get_time", "time", "{}", (_, _) => { invoked++; return Task.FromResult("ok"); });
        await Collect(ToolLoop.Create(rounds.Stream, Request(), tools));
        Assert.Equal(1, invoked);
    }

    [Fact]
    public async Task DeniedCallFeedsTheDenialBackAndIsNotDispatched()
    {
        int invoked = 0;
        Rounds rounds = new([Call("c1", "get_time"), Stop(StopReason.ToolCall)], [Text("ok, I will not."), Stop(StopReason.Stop)]);
        ToolRegistry tools = new ToolRegistry().Add("get_time", "time", "{}", (_, _) => { invoked++; return Task.FromResult("ok"); });
        ToolLoopOptions options = new() { OnBeforeToolCall = (_, _) => ValueTask.FromResult(ToolCallDecision.Deny("{\"error\":\"not allowed\"}")) };
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), tools, options);
        List<TextChunk> chunks = await Collect(run);
        Assert.Equal(0, invoked);
        Assert.Equal("{\"error\":\"not allowed\"}", run.ToolResults.Single().Result);
        Assert.Equal("{\"error\":\"not allowed\"}", chunks.Single(c => c.Kind == TextChunkKind.ToolResult).Text);
        Assert.Equal("{\"error\":\"not allowed\"}", rounds.Requests[1].Messages.Single(m => m.Role == TextRole.Tool).Content);
    }

    [Fact]
    public async Task ConversationIsExposedAfterCompletion()
    {
        Rounds rounds = new([Call("c1", "get_time")], [Text("It is 14:05."), Stop(StopReason.Stop)]);
        ToolRegistry tools = new ToolRegistry().Add("get_time", "time", "{}", (_, _) => Task.FromResult("{\"now\":\"14:05\"}"));
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), tools);
        await Collect(run);
        Assert.Equal([TextRole.User, TextRole.Assistant, TextRole.Tool], run.Conversation.Select(m => m.Role));
        Assert.Equal("c1", run.Conversation[2].ToolCallId);
        Assert.Equal("c1", run.ToolResults.Single().Call.Id);
    }

    [Fact]
    public async Task MissingOrDuplicateIdsAreRewritten()
    {
        Rounds rounds = new([Call("", "get_time"), Call("dup", "get_time"), Call("dup", "get_time"), Stop(StopReason.ToolCall)], [Text("done"), Stop(StopReason.Stop)]);
        ToolRegistry tools = new ToolRegistry().Add("get_time", "time", "{}", (_, _) => Task.FromResult("ok"));
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), tools, new ToolLoopOptions { IdPrefix = "fix_" });
        List<TextChunk> chunks = await Collect(run);
        List<string> ids = chunks.Where(c => c.Kind == TextChunkKind.ToolResult).Select(c => c.ToolCall!.Id).ToList();
        Assert.Equal(3, ids.Distinct().Count());
        Assert.All(ids, id => Assert.False(string.IsNullOrEmpty(id)));
        Assert.StartsWith("fix_", ids[0], StringComparison.Ordinal);
        Assert.Equal("dup", ids[1]);
    }

    [Fact]
    public async Task ToolResultChunkCarriesCallAndText()
    {
        Rounds rounds = new([Call("c1", "get_time", "{\"tz\":\"UTC\"}"), Stop(StopReason.ToolCall)], [Text("ok"), Stop(StopReason.Stop)]);
        ToolRegistry tools = new ToolRegistry().Add("get_time", "time", "{}", (_, _) => Task.FromResult("14:05"));
        TextChunk result = (await Collect(ToolLoop.Create(rounds.Stream, Request(), tools))).Single(c => c.Kind == TextChunkKind.ToolResult);
        Assert.Equal("14:05", result.Text);
        Assert.Equal("get_time", result.ToolCall!.Name);
        Assert.Equal("{\"tz\":\"UTC\"}", result.ToolCall.Arguments);
        Assert.Equal(0, result.ToolCallIndex);
    }

    [Fact]
    public async Task ForceToolIdAppliesToRoundOneOnly()
    {
        Rounds rounds = new([Call("c1", "get_time"), Stop(StopReason.ToolCall)], [Text("ok"), Stop(StopReason.Stop)]);
        ToolRegistry tools = new ToolRegistry().Add("get_time", "time", "{}", (_, _) => Task.FromResult("14:05"));
        await Collect(ToolLoop.Create(rounds.Stream, Request("get_time"), tools));
        Assert.Equal("get_time", rounds.Requests[0].ForceToolId);
        Assert.Null(rounds.Requests[1].ForceToolId);
    }

    [Fact]
    public void MaxRoundsBelowOneIsRefused()
    {
        Rounds rounds = new([Stop(StopReason.Stop)]);
        Assert.Throws<ArgumentOutOfRangeException>(() => ToolLoop.Create(rounds.Stream, Request(), new ToolRegistry(), new ToolLoopOptions { MaxRounds = 0 }));
    }

    [Fact]
    public async Task ARunOnlyRunsOnce()
    {
        Rounds rounds = new([Text("hi"), Stop(StopReason.Stop)]);
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), new ToolRegistry());
        await Collect(run);
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await Collect(run));
    }
}
