using System.Diagnostics;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Stream lifetime through the real <c>TextService</c> on a CPU engine with a model path that does not exist: the pre-cancelled and load-failure cases end before any weight is touched, so they need no checkpoint. Abandonment after a first chunk needs a producing model and is covered at the pump seam (<c>TextStreamPumpTests</c>).</summary>
public sealed class TextServiceStreamLifetimeTests
{
    private static readonly TimeSpan Budget = TimeSpan.FromMilliseconds(100);

    private static ModelSpec MissingSpec() => new()
    {
        Requested = "missing",
        Modality = Modality.Text,
        LocalPath = "/nonexistent/hartsy-missing-model.gguf",
    };

    private static TextRequest Request() => new()
    {
        Messages = [new TextMessage { Role = TextRole.User, Content = "hi" }],
        MaxTokens = 4,
        Device = "cpu",
    };

    [Fact]
    public async Task PreCancelledTokenCompletesWithCancelledStopWithin100Ms()
    {
        using InferenceEngine engine = new("cpu");
        using CancellationTokenSource cts = new();
        cts.Cancel();
        Stopwatch watch = Stopwatch.StartNew();
        List<TextChunk> chunks = [];
        await foreach (TextChunk chunk in engine.Text.StreamAsync(MissingSpec(), Request(), cts.Token))
            chunks.Add(chunk);
        Assert.True(watch.Elapsed < Budget, $"stream took {watch.ElapsedMilliseconds} ms");
        TextChunk only = Assert.Single(chunks);
        Assert.Equal(TextChunkKind.StopReason, only.Kind);
        Assert.Equal(StopReason.Cancelled, only.Stop);
    }

    [Fact]
    public async Task LoadFailureEndsWithAnErrorChunkCarryingTheMessageThenCompletes()
    {
        using InferenceEngine engine = new("cpu");
        List<TextChunk> chunks = [];
        await foreach (TextChunk chunk in engine.Text.StreamAsync(MissingSpec(), Request()))
            chunks.Add(chunk);
        TextChunk only = Assert.Single(chunks);
        Assert.Equal(TextChunkKind.StopReason, only.Kind);
        Assert.Equal(StopReason.Error, only.Stop);
        Assert.False(string.IsNullOrEmpty(only.Text), "error chunk must carry the failure message");
    }

    [Fact]
    public async Task NonStreamingPreCancelledTokenThrowsCancellation()
    {
        using InferenceEngine engine = new("cpu");
        using CancellationTokenSource cts = new();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => engine.Text.GenerateAsync(MissingSpec(), Request(), cts.Token));
    }
}
