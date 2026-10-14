using HartsyInference.Engine.Requests;
using HartsyInference.Tools;
using Xunit;

namespace HartsyInference.Tools.Tests;

/// <summary>The v2 loop over an arbitrary model stream: the host supplies each round, gates calls, reads the conversation
/// afterwards, the loop repairs ids it cannot trust, and a call the stream already resolved is recorded, not dispatched.</summary>
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

    /// <summary>A result the stream relays for a call it ran itself, under the stream's own index.</summary>
    private static TextChunk Relayed(string id, string result, int index = 7) => new()
    {
        Kind = TextChunkKind.ToolResult,
        Text = result,
        ToolCallIndex = index,
        ToolCall = new NativeToolCall { Id = id, Name = "get_time", Arguments = "{}" },
    };

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
        Assert.Equal([TextRole.User, TextRole.Assistant, TextRole.Tool, TextRole.Assistant], run.Conversation.Select(m => m.Role));
        Assert.Equal("c1", run.Conversation[2].ToolCallId);
        Assert.Equal("c1", run.ToolResults.Single().Call.Id);
    }

    [Fact]
    public async Task ResultOnlyRoundKeepsItsTextInTheTurn()
    {
        Rounds rounds = new([new TextChunk { Kind = TextChunkKind.Result, Text = "14:05" }, Stop(StopReason.Stop)]);
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), new ToolRegistry());
        List<TextChunk> chunks = await Collect(run);
        Assert.Equal("14:05", run.VisibleText);
        Assert.Contains(chunks, c => c.Kind == TextChunkKind.Chunk && c.Text == "14:05");
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

    [Fact]
    public async Task AResultThatCorrectsItsDeltasIsAuthoritative()
    {
        Rounds rounds = new([Text("It is 14:0"), new TextChunk { Kind = TextChunkKind.Result, Text = "It is 14:05." }, Stop(StopReason.Stop)]);
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), new ToolRegistry());
        List<TextChunk> chunks = await Collect(run);
        Assert.Equal("It is 14:05.", run.VisibleText);
        Assert.Equal("It is 14:05.", Assert.Single(chunks, c => c.Kind == TextChunkKind.Result).Text);
        Assert.Equal("It is 14:05.", run.Conversation[^1].Content);
    }

    [Fact]
    public async Task TheFinalAnswerIsInTheConversation()
    {
        Rounds rounds = new([Call("c1", "get_time"), Stop(StopReason.ToolCall)], [Text("It is 14:05."), Stop(StopReason.Stop)]);
        ToolRegistry tools = new ToolRegistry().Add("get_time", "time", "{}", (_, _) => Task.FromResult("14:05"));
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), tools);
        await Collect(run);
        TextMessage last = run.Conversation[^1];
        Assert.Equal(TextRole.Assistant, last.Role);
        Assert.Equal("It is 14:05.", last.Content);
    }

    [Fact]
    public async Task TheRoundLimitKeepsTheLastTurnsTextWithoutItsCalls()
    {
        Rounds rounds = new([Text("Let me check."), Call("c1", "get_time"), Stop(StopReason.ToolCall)]);
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), new ToolRegistry().Add("get_time", "time", "{}", (_, _) => Task.FromResult("x")),
            new ToolLoopOptions { MaxRounds = 1 });
        await Collect(run);
        TextMessage last = run.Conversation[^1];
        Assert.Equal(TextRole.Assistant, last.Role);
        Assert.Equal("Let me check.", last.Content);
        Assert.True(last.ToolCalls is null || last.ToolCalls.Count == 0);
    }

    [Fact]
    public async Task AThrowingDispatcherBecomesTheCallsErrorResult()
    {
        Rounds rounds = new([Call("c1", "get_time"), Stop(StopReason.ToolCall)], [Text("sorry"), Stop(StopReason.Stop)]);
        ToolRegistry tools = new ToolRegistry().Add("get_time", "time", "{}", (_, _) => throw new InvalidOperationException("clock down"));
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), tools);
        await Collect(run);
        Assert.Contains("clock down", run.ToolResults.Single().Result);
        Assert.Contains(run.Conversation, m => m.Role == TextRole.Tool && m.ToolCallId == "c1");
        Assert.Equal(StopReason.Stop, run.Stop);
    }

    [Fact]
    public async Task AnIndexlessCallKeepsTheIdItsDeltaSettled()
    {
        TextChunk begin = new() { Kind = TextChunkKind.ToolCallDelta, ToolCall = new NativeToolCall { Id = "c1", Name = "get_time", Arguments = "" } };
        Rounds rounds = new([begin, Call("c1", "get_time"), Stop(StopReason.ToolCall)], [Text("ok"), Stop(StopReason.Stop)]);
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), new ToolRegistry().Add("get_time", "time", "{}", (_, _) => Task.FromResult("x")));
        await Collect(run);
        Assert.Equal("c1", run.ToolResults.Single().Call.Id);
    }

    [Theory]
    [InlineData("")]
    [InlineData("c1")]
    public async Task AnIndexlessCallKeepsItsDeltasRepairedId(string id)
    {
        // "" has no id at all; "c1" was already used in round one, so round two's delta is repaired.
        TextChunk begin = new() { Kind = TextChunkKind.ToolCallDelta, ToolCall = new NativeToolCall { Id = id, Name = "get_time", Arguments = "" } };
        Rounds rounds = new([Call("c1", "get_time"), Stop(StopReason.ToolCall)], [begin, Call(id, "get_time"), Stop(StopReason.ToolCall)], [Text("ok"), Stop(StopReason.Stop)]);
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), new ToolRegistry().Add("get_time", "time", "{}", (_, _) => Task.FromResult("x")));
        List<TextChunk> chunks = await Collect(run);
        string deltaId = chunks.Single(c => c.Kind == TextChunkKind.ToolCallDelta).ToolCall!.Id;
        Assert.NotEqual("c1", deltaId);
        Assert.False(string.IsNullOrEmpty(deltaId));
        Assert.Equal(deltaId, run.ToolResults[1].Call.Id);
    }

    [Fact]
    public async Task ARepairedIdCoversTheCallsDeltasToo()
    {
        // Both calls arrive without an id; the begin delta, the completed call and the tool result must share each repaired id.
        TextChunk Begin(int index) => new() { Kind = TextChunkKind.ToolCallDelta, ToolCallIndex = index, ToolCall = new NativeToolCall { Id = "", Name = "get_time", Arguments = "" } };
        TextChunk Done(int index) => new() { Kind = TextChunkKind.NativeToolCall, ToolCallIndex = index, ToolCall = new NativeToolCall { Id = "", Name = "get_time", Arguments = "{}" } };
        Rounds rounds = new([Begin(0), Done(0), Begin(1), Done(1), Stop(StopReason.ToolCall)], [Text("ok"), Stop(StopReason.Stop)]);
        ToolRegistry tools = new ToolRegistry().Add("get_time", "time", "{}", (_, _) => Task.FromResult("14:05"));
        List<TextChunk> chunks = await Collect(ToolLoop.Create(rounds.Stream, Request(), tools));
        for (int index = 0; index < 2; index++)
        {
            string begin = chunks.Single(c => c.Kind == TextChunkKind.ToolCallDelta && c.ToolCallIndex == index).ToolCall!.Id;
            string done = chunks.Single(c => c.Kind == TextChunkKind.NativeToolCall && c.ToolCallIndex == index).ToolCall!.Id;
            Assert.False(string.IsNullOrEmpty(begin));
            Assert.Equal(begin, done);
        }
        string[] resultIds = [.. chunks.Where(c => c.Kind == TextChunkKind.ToolResult).Select(c => c.ToolCall!.Id)];
        Assert.Equal(2, resultIds.Distinct().Count());
    }

    [Fact]
    public async Task ARelayedResultResolvesItsCallWithoutDispatchOrAnotherRound()
    {
        int invoked = 0;
        Rounds rounds = new([Text("Checking. "), Call("c1", "get_time"), Relayed("c1", "{\"now\":\"14:05\"}"), Text("It is 14:05."),
            new TextChunk { Kind = TextChunkKind.Result, Text = "Checking. It is 14:05." }, Stop(StopReason.Stop)]);
        ToolRegistry tools = new ToolRegistry().Add("get_time", "time", "{}", (_, _) => { invoked++; return Task.FromResult("local"); });
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), tools);
        List<TextChunk> chunks = await Collect(run);
        Assert.Equal(0, invoked);
        Assert.Equal(1, run.Rounds);
        Assert.Single(rounds.Requests);
        Assert.Equal([TextChunkKind.Chunk, TextChunkKind.NativeToolCall, TextChunkKind.ToolResult, TextChunkKind.Chunk,
            TextChunkKind.Result, TextChunkKind.StopReason], chunks.Select(c => c.Kind));
        TextChunk result = Assert.Single(chunks, c => c.Kind == TextChunkKind.ToolResult);
        Assert.Equal("{\"now\":\"14:05\"}", result.Text);
        Assert.Equal(0, result.ToolCallIndex);
        Assert.Equal("c1", result.ToolCall!.Id);
        Assert.Equal("{\"now\":\"14:05\"}", Assert.Single(run.ToolResults).Result);
        Assert.Equal([TextRole.User, TextRole.Assistant, TextRole.Tool, TextRole.Assistant], run.Conversation.Select(m => m.Role));
        Assert.Equal("Checking. ", run.Conversation[1].Content);
        Assert.Equal("c1", Assert.Single(run.Conversation[1].ToolCalls!).Id);
        Assert.Equal("c1", run.Conversation[2].ToolCallId);
        Assert.Equal("It is 14:05.", run.Conversation[3].Content);
        Assert.Equal("Checking. It is 14:05.", run.VisibleText);
        Assert.Equal(StopReason.Stop, run.Stop);
    }

    [Fact]
    public async Task OnlyTheCallsWithoutARelayedResultAreDispatched()
    {
        List<string> dispatched = [];
        Rounds rounds = new([Call("a", "get_time"), Call("b", "get_time"), Relayed("a", "remote"), Text("Half done."), Stop(StopReason.ToolCall)],
            [Text("Done."), Stop(StopReason.Stop)]);
        ToolRegistry tools = new ToolRegistry().Add("get_time", "time", "{}", (_, _) => Task.FromResult("local"));
        ToolLoopOptions options = new()
        {
            OnBeforeToolCall = (call, _) =>
            {
                dispatched.Add(call.Id);
                return ValueTask.FromResult(ToolCallDecision.Allow);
            },
        };
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), tools, options);
        List<TextChunk> chunks = await Collect(run);
        Assert.Equal(["b"], dispatched);
        Assert.Equal(2, run.Rounds);
        TextChunk[] results = [.. chunks.Where(c => c.Kind == TextChunkKind.ToolResult)];
        Assert.Equal(["a", "b"], results.Select(c => c.ToolCall!.Id));
        Assert.Equal(["remote", "local"], results.Select(c => c.Text));
        Assert.Equal([0, 1], results.Select(c => c.ToolCallIndex!.Value));
        IReadOnlyList<TextMessage> second = rounds.Requests[1].Messages;
        Assert.Equal([TextRole.User, TextRole.Assistant, TextRole.Tool, TextRole.Assistant, TextRole.Tool], second.Select(m => m.Role));
        Assert.Equal("a", Assert.Single(second[1].ToolCalls!).Id);
        Assert.Equal("remote", second[2].Content);
        Assert.Equal("Half done.", second[3].Content);
        Assert.Equal("b", Assert.Single(second[3].ToolCalls!).Id);
        Assert.Equal("local", second[4].Content);
    }

    [Fact]
    public async Task ARelayedResultFollowsItsCallsRepairedId()
    {
        int invoked = 0;
        Rounds rounds = new([Call("", "get_time"), Relayed("", "r0"), Call("dup", "get_time"), Call("dup", "get_time"), Relayed("dup", "r1"),
            Relayed("dup", "r2"), Text("ok"), Stop(StopReason.Stop)]);
        ToolRegistry tools = new ToolRegistry().Add("get_time", "time", "{}", (_, _) => { invoked++; return Task.FromResult("local"); });
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), tools, new ToolLoopOptions { IdPrefix = "fix_" });
        List<TextChunk> chunks = await Collect(run);
        Assert.Equal(0, invoked);
        string[] callIds = [.. chunks.Where(c => c.Kind == TextChunkKind.NativeToolCall).Select(c => c.ToolCall!.Id)];
        Assert.StartsWith("fix_", callIds[0], StringComparison.Ordinal);
        Assert.Equal("dup", callIds[1]);
        Assert.StartsWith("fix_", callIds[2], StringComparison.Ordinal);
        Assert.Equal(callIds, chunks.Where(c => c.Kind == TextChunkKind.ToolResult).Select(c => c.ToolCall!.Id));
        Assert.Equal(["r0", "r1", "r2"], chunks.Where(c => c.Kind == TextChunkKind.ToolResult).Select(c => c.Text));
        Assert.Equal(callIds, run.ToolResults.Select(r => r.Call.Id));
        string[] recordedCalls = [.. run.Conversation.Where(m => m.ToolCalls is not null).SelectMany(m => m.ToolCalls!).Select(c => c.Id)];
        Assert.Equal(callIds, recordedCalls);
        Assert.Equal(callIds, run.Conversation.Where(m => m.Role == TextRole.Tool).Select(m => m.ToolCallId));
    }

    [Fact]
    public async Task ARelayedResultWithNoOpenCallIsDropped()
    {
        Rounds rounds = new([Call("c1", "get_time"), Relayed("c1", "first"), Relayed("c1", "again"), Relayed("ghost", "?"), Text("ok"),
            Stop(StopReason.Stop)]);
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), new ToolRegistry());
        List<TextChunk> chunks = await Collect(run);
        Assert.Equal("first", Assert.Single(chunks, c => c.Kind == TextChunkKind.ToolResult).Text);
        Assert.Equal("first", Assert.Single(run.Conversation, m => m.Role == TextRole.Tool).Content);
    }

    [Fact]
    public async Task APreResolvedLastRoundIsNotARoundLimit()
    {
        Rounds rounds = new([Call("c1", "get_time"), Relayed("c1", "14:05"), Text("It is 14:05."), Stop(StopReason.Stop)]);
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), new ToolRegistry(), new ToolLoopOptions { MaxRounds = 1 });
        List<TextChunk> chunks = await Collect(run);
        Assert.DoesNotContain(chunks, c => c.Kind == TextChunkKind.Status);
        Assert.Equal(StopReason.Stop, run.Stop);
        Assert.Equal("It is 14:05.", run.Conversation[^1].Content);
    }

    [Fact]
    public async Task AFailedRoundKeepsItsRelayedResultsOutOfTheConversation()
    {
        Rounds rounds = new([Text("Checking. "), Call("c1", "get_time"), Relayed("c1", "14:05"), Text("It is"), Stop(StopReason.Error)]);
        ToolLoopRun run = ToolLoop.Create(rounds.Stream, Request(), new ToolRegistry());
        await Collect(run);
        Assert.Equal([TextRole.User], run.Conversation.Select(m => m.Role));
        Assert.Empty(run.ToolResults);
        Assert.Equal(StopReason.Error, run.Stop);
    }
}
