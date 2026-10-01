using HartsyInference.Core.Runtime;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>The session's single outbound consumer for a test, on a thread of its own as the host's sender is: reads
/// <c>chunk</c> samples, keeps the real ones with the time they were read, pauses, repeats. <see cref="Paused"/> stops
/// reading without ending the consumer; <see cref="ReadAllocatedBytes"/> counts what the reads themselves allocated.</summary>
internal sealed class OutboundReader : IAsyncDisposable
{
    private readonly VoiceAgentSession _session;
    private readonly int _chunk;
    private readonly TimeSpan _pause;
    private readonly List<float> _samples = [];
    private readonly List<(long Ns, int Count)> _reads = [];
    private readonly CancellationTokenSource _stop = new();
    private readonly Task _loop;
    private volatile bool _paused;
    private long _readAllocated;

    public OutboundReader(VoiceAgentSession session, int chunk, TimeSpan pause)
    {
        _session = session;
        _chunk = chunk;
        _pause = pause;
        _loop = Task.Factory.StartNew(Run, TaskCreationOptions.LongRunning);
    }

    public bool Paused
    {
        get => _paused;
        set => _paused = value;
    }

    /// <summary>Managed bytes allocated on the reader thread inside <see cref="VoiceAgentSession.ReadOutbound"/> calls.</summary>
    public long ReadAllocatedBytes => Volatile.Read(ref _readAllocated);

    /// <summary>Every real sample read so far.</summary>
    public float[] Samples
    {
        get
        {
            lock (_samples)
            {
                return [.. _samples];
            }
        }
    }

    /// <summary>Monotonic time and count of every read that returned real samples.</summary>
    public IReadOnlyList<(long Ns, int Count)> Reads
    {
        get
        {
            lock (_samples)
            {
                return [.. _reads];
            }
        }
    }

    /// <summary>Waits until at least <paramref name="count"/> real samples were read.</summary>
    public async Task WaitForSamplesAsync(int count, double seconds = 20)
    {
        DateTime deadline = DateTime.UtcNow.AddSeconds(seconds);
        while (Samples.Length < count)
        {
            if (DateTime.UtcNow > deadline)
            {
                throw new TimeoutException($"Read {Samples.Length} of {count} samples within {seconds} s.");
            }
            await Task.Delay(5);
        }
    }

    private void Run()
    {
        float[] buffer = new float[_chunk];
        while (!_stop.IsCancellationRequested)
        {
            if (!_paused)
            {
                long before = GC.GetAllocatedBytesForCurrentThread();
                int real = _session.ReadOutbound(buffer);
                Volatile.Write(ref _readAllocated, _readAllocated + GC.GetAllocatedBytesForCurrentThread() - before);
                if (real > 0)
                {
                    lock (_samples)
                    {
                        _samples.AddRange(buffer.AsSpan(0, real));
                        _reads.Add((MonotonicClock.NowNs(), real));
                    }
                }
            }
            _stop.Token.WaitHandle.WaitOne(_pause);
        }
    }

    public async ValueTask DisposeAsync()
    {
        _stop.Cancel();
        await _loop;
        _stop.Dispose();
    }
}
