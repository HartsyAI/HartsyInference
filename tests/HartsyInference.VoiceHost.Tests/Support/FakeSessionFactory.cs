using System.Collections.Concurrent;
using HartsyInference.Tools;
using HartsyInference.VoiceHost.Calls;

namespace HartsyInference.VoiceHost.Tests.Support;

/// <summary>Hands the host <see cref="FakeCallSession"/>s and keeps them, in creation order, for the test to drive.</summary>
internal sealed class FakeSessionFactory : IVoiceCallSessionFactory
{
    private readonly ConcurrentQueue<FakeCallSession> _sessions = new();

    public int OutboundSampleRate => 16_000;

    /// <summary>When set, <see cref="Create"/> throws it.</summary>
    public Exception? CreateFailure { get; set; }

    /// <summary>When set, the next session's start throws it.</summary>
    public Exception? StartFailure { get; set; }

    /// <summary>When set, the next session's start waits for it.</summary>
    public Task? StartGate { get; set; }

    public IReadOnlyList<FakeCallSession> Sessions => [.. _sessions];

    public IVoiceCallSession Create(uint callId, ToolRegistry tools)
    {
        if (CreateFailure is { } failure)
        {
            CreateFailure = null;
            throw failure;
        }
        FakeCallSession session = new(callId, tools) { StartFailure = StartFailure, StartGate = StartGate };
        StartFailure = null;
        StartGate = null;
        _sessions.Enqueue(session);
        return session;
    }

    /// <summary>The <paramref name="index"/>-th session, once the host has started it.</summary>
    public async Task<FakeCallSession> WaitForSessionAsync(int index = 0, int timeoutMs = FakeGateway.WaitMs)
    {
        long deadline = Environment.TickCount64 + timeoutMs;
        while (Sessions.Count <= index)
        {
            if (Environment.TickCount64 > deadline)
            {
                throw new TimeoutException($"The host created {Sessions.Count} session(s), not {index + 1}.");
            }
            await Task.Delay(2);
        }
        FakeCallSession session = Sessions[index];
        await session.Started.WaitAsync(TimeSpan.FromMilliseconds(timeoutMs));
        return session;
    }
}
