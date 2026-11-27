using System.Collections.Concurrent;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Tests.PlacementChecks;
using Xunit;

namespace HartsyInference.Engine.Tests.Services;

/// <summary>The level an audio job logs its failure at: a warning when it ran out of VRAM and its caller falls back, an error otherwise. No GPU and no weights.</summary>
[Collection(GlobalLogSink.Name)]
public sealed class AudioRuntimeFailureLogTests : IDisposable
{
    private readonly ConcurrentQueue<(LogLevel Level, string Message)> _lines = new();
    private readonly string _key = "failure-log-" + Guid.NewGuid().ToString("N");

    public AudioRuntimeFailureLogTests() => Logs.SetLogger((level, message) => _lines.Enqueue((level, message)));

    public void Dispose() => Logs.SetLogger((_, message) => Console.Error.WriteLine(message));

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task AFullCardTheCallerFallsBackFromIsAWarning(bool wrapped)
    {
        OutOfVramException full = new(1 << 30, 1 << 20);
        Exception thrown = wrapped ? new InvalidOperationException("kernel launch failed", full) : full;

        int attempts = await FailingRun(thrown, callerFallsBack: true);

        Assert.Equal(2, attempts);
        Assert.DoesNotContain(Lines(), line => line.Level == LogLevel.Error);
        Assert.Contains(Lines(), line => line.Level == LogLevel.Warning
            && line.Message.Contains("its caller falls back", StringComparison.Ordinal)
            && line.Message.Contains(full.Message, StringComparison.Ordinal));
    }

    [Fact]
    public async Task AFullCardNobodyFallsBackFromIsAnError()
    {
        await FailingRun(new OutOfVramException(1 << 30, 1 << 20), callerFallsBack: false);

        Assert.Single(Lines(), line => line.Level == LogLevel.Error);
    }

    [Fact]
    public async Task AnotherFailureIsAnErrorEvenWhenTheCallerFallsBackFromAFullCard()
    {
        int attempts = await FailingRun(new InvalidDataException("bad weights"), callerFallsBack: true);

        Assert.Equal(1, attempts);
        Assert.Single(Lines(), line => line.Level == LogLevel.Error);
    }

    /// <summary>The lines about this test's own job.</summary>
    private IEnumerable<(LogLevel Level, string Message)> Lines() => _lines.Where(line => line.Message.Contains(_key, StringComparison.Ordinal));

    /// <summary>Runs a job whose work always throws <paramref name="failure"/>, and returns how many times the work ran.</summary>
    private async Task<int> FailingRun(Exception failure, bool callerFallsBack)
    {
        AudioRuntime runtime = new();
        using CpuBackend backend = new();
        int attempts = 0;

        Exception thrown = await Assert.ThrowsAnyAsync<Exception>(() => runtime.RunAsync<int>(backend, new AudioJob(runtime.Mdx, _key),
            _ => { attempts++; return Task.FromException<int>(failure); }, CancellationToken.None,
            callerFallsBackFromOutOfVram: callerFallsBack));

        Assert.Same(failure, thrown);
        return attempts;
    }
}
