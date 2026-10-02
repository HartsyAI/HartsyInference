using System.Collections.Concurrent;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tests.MemoryManagement;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests;

/// <summary>A large allocation's mandatory zero-fill goes through <see cref="CpuParallel"/>. Inside an
/// <see cref="CpuParallel.InlineScope"/> — the voice session's real-time audio thread — every chunk is cleared on the
/// calling thread; the <c>numerics.cpuThreads</c> cap bounds how many threads clear at once; the chunks tile the buffer
/// exactly whatever the cap; and a buffer comes back zeroed under every schedule. Serialized with the other classes that
/// change process-wide knobs.</summary>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed unsafe class NativeBufferZeroFillTests
{
    private const nuint Chunk = (nuint)(2UL << 20);
    private const nuint Large = 32 * Chunk;

    private readonly ITestOutputHelper _output;

    public NativeBufferZeroFillTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void InsideAnInlineScope_EveryChunkIsClearedOnTheCallingThread()
    {
        using Recorder recorder = new(Large);
        int caller = Environment.CurrentManagedThreadId;

        using (CpuParallel.EnterInline())
        {
            NativeBuffer.Clear(recorder.Pointer, Large, recorder.Clear);
        }

        Assert.Equal(32, recorder.Calls.Count);
        Assert.All(recorder.Calls, call => Assert.Equal(caller, call.Thread));
        recorder.AssertTiledAndZeroed();
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void TheCpuThreadsCap_BoundsHowManyThreadsClearAtOnce(int cap)
    {
        using Recorder recorder = new(Large);
        int caller = Environment.CurrentManagedThreadId;

        WithCpuThreads(cap, () => NativeBuffer.Clear(recorder.Pointer, Large, recorder.Clear));

        _output.WriteLine($"cap {cap}: at most {recorder.MaxConcurrent} chunks in flight on "
            + $"{recorder.Calls.Select(c => c.Thread).Distinct().Count()} thread(s)");
        Assert.InRange(recorder.MaxConcurrent, 1, cap);
        if (cap == 1)
        {
            Assert.All(recorder.Calls, call => Assert.Equal(caller, call.Thread));
        }
        recorder.AssertTiledAndZeroed();
    }

    [Fact]
    public void TheLastChunkIsShort_AndABufferUnderTheThresholdIsOneCallOnTheCallingThread()
    {
        nuint justOver = 4 * Chunk + 1;
        using (Recorder recorder = new(justOver))
        {
            NativeBuffer.Clear(recorder.Pointer, justOver, recorder.Clear);
            Assert.Equal(5, recorder.Calls.Count);
            Assert.Equal((nuint)1, recorder.Calls.Single(c => c.Offset == 4 * Chunk).Length);
            recorder.AssertTiledAndZeroed();
        }

        nuint small = Chunk / 2;
        using (Recorder recorder = new(small))
        {
            NativeBuffer.Clear(recorder.Pointer, small, recorder.Clear);
            (nuint offset, nuint length, int thread) = Assert.Single(recorder.Calls);
            Assert.Equal((nuint)0, offset);
            Assert.Equal(small, length);
            Assert.Equal(Environment.CurrentManagedThreadId, thread);
            recorder.AssertTiledAndZeroed();
        }
    }

    [Fact]
    public void ANewBufferIsZeroed_UnderTheDefaultTheCapAndAnInlineScope()
    {
        nuint bytes = 12 * Chunk + 3;
        AssertZeroed(bytes);
        WithCpuThreads(1, () => AssertZeroed(bytes));
        using (CpuParallel.EnterInline())
        {
            AssertZeroed(bytes);
        }
    }

    private static void AssertZeroed(nuint bytes)
    {
        using NativeBuffer buffer = new(bytes);
        Assert.Equal(-1, buffer.AsReadOnlySpan<byte>().IndexOfAnyExcept((byte)0));
    }

    private static void WithCpuThreads(int cap, Action run)
    {
        bool hadOverride = KnobStore.HasOverride(EngineKnobs.CpuThreads);
        int previous = EngineKnobs.CpuThreads.Value;
        try
        {
            KnobStore.Set(EngineKnobs.CpuThreads, cap);
            run();
        }
        finally
        {
            if (hadOverride) KnobStore.Set(EngineKnobs.CpuThreads, previous);
            else KnobStore.Clear(EngineKnobs.CpuThreads);
        }
    }

    /// <summary>A buffer filled with a non-zero pattern, and a clear that records each range it is handed, the thread
    /// that cleared it and how many clears were in flight at once.</summary>
    private sealed class Recorder : IDisposable
    {
        private readonly nuint _length;
        private int _active;
        private int _maxConcurrent;

        public Recorder(nuint length)
        {
            _length = length;
            Pointer = NativeMemory.AlignedAlloc(length, 64);
            NativeMemory.Fill(Pointer, length, 0xAB);
        }

        public void* Pointer { get; }

        public ConcurrentBag<(nuint Offset, nuint Length, int Thread)> Calls { get; } = [];

        public int MaxConcurrent => Volatile.Read(ref _maxConcurrent);

        public void Clear(nint start, nuint length)
        {
            int active = Interlocked.Increment(ref _active);
            int seen = Volatile.Read(ref _maxConcurrent);
            while (active > seen && Interlocked.CompareExchange(ref _maxConcurrent, active, seen) != seen)
            {
                seen = Volatile.Read(ref _maxConcurrent);
            }
            Calls.Add(((nuint)(start - (nint)Pointer), length, Environment.CurrentManagedThreadId));
            // Long enough that chunks handed to different workers overlap, so a cap that did not hold would show.
            Thread.SpinWait(20_000);
            NativeMemory.Clear((void*)start, length);
            Interlocked.Decrement(ref _active);
        }

        /// <summary>The recorded ranges cover the buffer end to end with no gap or overlap, and it is all zero.</summary>
        public void AssertTiledAndZeroed()
        {
            nuint next = 0;
            foreach ((nuint offset, nuint length, int _) in Calls.OrderBy(c => c.Offset))
            {
                Assert.Equal(next, offset);
                next = offset + length;
            }
            Assert.Equal(_length, next);
            Assert.Equal(-1, new ReadOnlySpan<byte>(Pointer, checked((int)_length)).IndexOfAnyExcept((byte)0));
        }

        public void Dispose() => NativeMemory.AlignedFree(Pointer);
    }
}
