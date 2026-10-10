using System.Runtime.CompilerServices;
using HartsyInference.API;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Direct (non-HTTP) tests for <see cref="WorldSessionRegistry"/> — the one genuinely new piece of
/// infrastructure Phase 5 adds. Uses a short real idle timeout + a real delay (same pattern as
/// <c>ServerTests.InferenceQueue_RejectsWhenFull</c>'s <c>Task.Delay(50)</c>) rather than an injectable clock,
/// which would be more machinery than a first-pass registry warrants.</summary>
public sealed class WorldSessionRegistryTests
{
    [Fact]
    public void Register_ThenGet_ReturnsTheSameSession()
    {
        using WorldSessionRegistry registry = new WorldSessionRegistry(TimeSpan.FromMinutes(10));
        FakeWorldSession session = new FakeWorldSession();

        string id = registry.Register(session);
        IWorldSession? found = registry.Get(id);

        Assert.Same(session, found);
    }

    [Fact]
    public void Close_RemovesAndDisposesTheSession()
    {
        using WorldSessionRegistry registry = new WorldSessionRegistry(TimeSpan.FromMinutes(10));
        FakeWorldSession session = new FakeWorldSession();
        string id = registry.Register(session);

        Assert.True(registry.Close(id));
        Assert.True(session.Disposed);
        Assert.Null(registry.Get(id));
    }

    [Fact]
    public async Task IdleSession_IsEvictedAndDisposedAfterTimeout()
    {
        using WorldSessionRegistry registry = new WorldSessionRegistry(TimeSpan.FromMilliseconds(100));
        FakeWorldSession session = new FakeWorldSession();
        string id = registry.Register(session);

        await Task.Delay(400);

        Assert.True(session.Disposed);
        Assert.Null(registry.Get(id));
    }

    [Fact]
    public async Task Get_TouchesLastActivity_SoAnActivelyPolledSessionOutlivesTheTimeout()
    {
        using WorldSessionRegistry registry = new WorldSessionRegistry(TimeSpan.FromMilliseconds(150));
        FakeWorldSession session = new FakeWorldSession();
        string id = registry.Register(session);

        // Poll it twice, each within the idle window, spanning longer than one timeout period total.
        await Task.Delay(80);
        Assert.NotNull(registry.Get(id));
        await Task.Delay(80);
        Assert.NotNull(registry.Get(id));

        Assert.False(session.Disposed);
    }

    private sealed class FakeWorldSession : IWorldSession
    {
        public bool Disposed { get; private set; }

        public void SendAction(string action)
        {
        }

        public async IAsyncEnumerable<VideoFrame> StreamAsync([EnumeratorCancellation] CancellationToken cancel)
        {
            await Task.CompletedTask;
            yield break;
        }

        public void Dispose() => Disposed = true;
    }
}
