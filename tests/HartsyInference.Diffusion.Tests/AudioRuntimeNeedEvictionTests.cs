using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The switch check sizes the incoming model instead of comparing free VRAM against a constant, and a run that
/// still runs out of VRAM evicts the other models and retries once. Driven through a backend that reports scripted VRAM,
/// with fake runners loaded into the real caches, so nothing touches a device. Every test relaxes the host-RAM floor
/// first: a box genuinely short of RAM would otherwise evict through the other path and blur what is being measured.</summary>
public sealed class AudioRuntimeNeedEvictionTests
{
    private const long GiB = 1L << 30;

    [Fact]
    public void SwitchThreshold_IsTheFloorWithoutANeed_AndCoversTheWeightsPlusRoomBesideThem()
    {
        long floor = EngineKnobs.AudioEvictFreeVramFloorMb.Value << 20;
        long headroom = EngineKnobs.AutopromoteHeadroomMb.Value << 20;
        Assert.Equal(floor, AudioRuntime.SwitchThresholdBytes(0));
        Assert.Equal(Math.Max(floor, GiB + headroom), AudioRuntime.SwitchThresholdBytes(GiB));
        // Below ~7.5 GB the promotion headroom is the larger margin, above it the fifth.
        Assert.Equal(6 * GiB + Math.Max(headroom, 6 * GiB / 5), AudioRuntime.SwitchThresholdBytes(6 * GiB));
        Assert.Equal(12 * GiB + 12 * GiB / 5, AudioRuntime.SwitchThresholdBytes(12 * GiB));
    }

    [Fact]
    public async Task Switch_EvictsOthers_WhenFreeVramIsBelowTheIncomingDiskEstimate()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        IBackend backend = FakeVramBackend.Create(free: 5 * GiB, total: 24 * GiB, out FakeVramBackend device);
        AudioRuntime runtime = new();
        FakeTts bark = await SeedAsync(runtime, "bark");

        // 5 GB free clears the old 3 GB constant, but not a 6 GB model plus the room beside it.
        await RunLoadingAsync(runtime, backend, "dia", estimate: 6 * GiB);

        Assert.True(bark.Disposed);
        Assert.False(runtime.Tts.IsResident("bark"));
        Assert.True(runtime.Tts.IsResident("dia"));
        Assert.Equal(1, device.FreeAllCalls);
    }

    [Theory]
    [InlineData(5L, false)]
    [InlineData(2L, true)]
    public async Task Switch_WithNothingToEstimateFrom_UsesTheFloor(long freeGb, bool evicts)
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        IBackend backend = FakeVramBackend.Create(free: freeGb * GiB, total: 24 * GiB, out _);
        AudioRuntime runtime = new();
        FakeTts bark = await SeedAsync(runtime, "bark");

        await RunLoadingAsync(runtime, backend, "kokoro", estimate: null);

        Assert.Equal(evicts, bark.Disposed);
    }

    [Fact]
    public async Task Switch_BackToAResidentModel_UsesTheFloor_NotItsEstimate()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        IBackend backend = FakeVramBackend.Create(free: 5 * GiB, total: 24 * GiB, out FakeVramBackend device);
        AudioRuntime runtime = new();
        FakeTts dia = await SeedAsync(runtime, "dia");
        FakeTts bark = await SeedAsync(runtime, "bark");
        await RunLoadingAsync(runtime, backend, "bark", estimate: null);

        // Dia's weights are already loaded, so switching back adds nothing they need.
        await RunLoadingAsync(runtime, backend, "dia", estimate: 6 * GiB);

        Assert.False(bark.Disposed);
        Assert.False(dia.Disposed);
        Assert.Equal(0, device.FreeAllCalls);
    }

    [Fact]
    public async Task NeedBasedEviction_KeepsPinnedRunners()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        IBackend backend = FakeVramBackend.Create(free: 5 * GiB, total: 24 * GiB, out _);
        AudioRuntime runtime = new();
        FakeTts kokoro = await SeedAsync(runtime, "kokoro");
        FakeTts bark = await SeedAsync(runtime, "bark");
        using IDisposable pin = runtime.Tts.Pin("kokoro");

        await RunLoadingAsync(runtime, backend, "dia", estimate: 6 * GiB);

        Assert.True(bark.Disposed);
        Assert.False(kokoro.Disposed);
        Assert.True(runtime.Tts.IsResident("kokoro"));
    }

    [Fact]
    public async Task LearnedFootprint_FromALoad_RaisesTheNeedOnTheNextLoad()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        IBackend backend = FakeVramBackend.Create(free: 20 * GiB, total: 24 * GiB, out FakeVramBackend device);
        AudioRuntime runtime = new();
        // The files say 1 GB; the load really leaves 8 GB promoted on the card.
        await RunLoadingAsync(runtime, backend, "orpheus", estimate: GiB, onLoad: () => device.FreeBytes -= 8 * GiB);
        FakeTts bark = await SeedAsync(runtime, "bark");
        await RunLoadingAsync(runtime, backend, "bark", estimate: null);
        runtime.Tts.UnloadAllExcept("bark");
        device.FreeBytes = 9 * GiB;

        await RunLoadingAsync(runtime, backend, "orpheus", estimate: GiB);

        Assert.True(bark.Disposed);
    }

    [Fact]
    public async Task OutOfVram_EvictsEveryOtherUnpinnedModel_AndRetriesOnce()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        IBackend backend = FakeVramBackend.Create(free: 20 * GiB, total: 24 * GiB, out FakeVramBackend device);
        AudioRuntime runtime = new();
        FakeTts kokoro = await SeedAsync(runtime, "kokoro");
        FakeTts bark = await SeedAsync(runtime, "bark");
        using IDisposable pin = runtime.Tts.Pin("kokoro");
        int attempts = 0;
        ConcurrentQueue<string> lines = new();
        Logs.SetLogger((level, message) => lines.Enqueue($"[{level}] {message}"));
        int result;
        try
        {
            result = await runtime.RunAsync(backend, new AudioJob(runtime.Tts, "dia"), async ct =>
            {
                await LoadAsync(runtime, "dia", ct);
                return ++attempts == 1 ? throw Oom() : 42;
            }, CancellationToken.None);
        }
        finally
        {
            Logs.SetLogger(null!);
        }

        Assert.Equal(42, result);
        Assert.Equal(2, attempts);
        Assert.True(bark.Disposed);
        Assert.False(kokoro.Disposed);
        Assert.True(runtime.Tts.IsResident("dia"));
        Assert.Equal(1, device.FreeAllCalls);
        string retry = Assert.Single(lines, line => line.Contains("retrying once", StringComparison.Ordinal));
        Assert.Contains("tts:bark", retry, StringComparison.Ordinal);
        Assert.DoesNotContain("tts:kokoro", retry, StringComparison.Ordinal);
    }

    [Fact]
    public async Task OutOfVram_Twice_Propagates_AndReleasesTheLock()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        IBackend backend = FakeVramBackend.Create(free: 20 * GiB, total: 24 * GiB, out _);
        AudioRuntime runtime = new();
        int attempts = 0;

        await Assert.ThrowsAsync<OutOfVramException>(() => runtime.RunAsync<int>(backend, new AudioJob(runtime.Tts, "dia"),
            _ => { attempts++; throw Oom(); }, CancellationToken.None));

        Assert.Equal(2, attempts);
        Assert.Equal(7, await runtime.RunAsync(backend, new AudioJob(runtime.Tts, "dia"), _ => Task.FromResult(7), CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(10)));
    }

    [Fact]
    public async Task OtherFailures_AreNotRetried()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        IBackend backend = FakeVramBackend.Create(free: 20 * GiB, total: 24 * GiB, out FakeVramBackend device);
        AudioRuntime runtime = new();
        FakeTts bark = await SeedAsync(runtime, "bark");
        int attempts = 0;

        await Assert.ThrowsAsync<InvalidOperationException>(() => runtime.RunAsync<int>(backend, new AudioJob(runtime.Tts, "dia"),
            _ => { attempts++; throw new InvalidOperationException("not a capacity failure"); }, CancellationToken.None));

        Assert.Equal(1, attempts);
        Assert.False(bark.Disposed);
        Assert.Equal(0, device.FreeAllCalls);
    }

    [Fact]
    public async Task OutOfVram_AfterCancellation_IsNotRetried()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        IBackend backend = FakeVramBackend.Create(free: 20 * GiB, total: 24 * GiB, out _);
        AudioRuntime runtime = new();
        using CancellationTokenSource cancel = new();
        int attempts = 0;

        await Assert.ThrowsAsync<OutOfVramException>(() => runtime.RunAsync<int>(backend, new AudioJob(runtime.Tts, "dia"),
            _ => { attempts++; cancel.Cancel(); throw Oom(); }, cancel.Token));

        Assert.Equal(1, attempts);
    }

    [Fact]
    public async Task Stream_OutOfVramBeforeTheFirstItem_IsRetried()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        IBackend backend = FakeVramBackend.Create(free: 20 * GiB, total: 24 * GiB, out FakeVramBackend device);
        AudioRuntime runtime = new();
        FakeTts bark = await SeedAsync(runtime, "bark");
        int attempts = 0;

        List<int> items = await CollectAsync(runtime.RunStreamAsync(backend, new AudioJob(runtime.Tts, "dia"),
            ct => Items(++attempts == 1 ? 0 : 3, failAfter: attempts == 1 ? 0 : -1, ct), CancellationToken.None));

        Assert.Equal(new[] { 0, 1, 2 }, items);
        Assert.Equal(2, attempts);
        Assert.True(bark.Disposed);
        Assert.Equal(1, device.FreeAllCalls);
    }

    [Fact]
    public async Task Stream_OutOfVramAfterAnItem_IsNotRetried_AndReleasesTheLock()
    {
        using IDisposable calm = AudioEvictionPressure.Relax();
        IBackend backend = FakeVramBackend.Create(free: 20 * GiB, total: 24 * GiB, out FakeVramBackend device);
        AudioRuntime runtime = new();
        int attempts = 0;
        List<int> received = [];

        await Assert.ThrowsAsync<OutOfVramException>(async () =>
        {
            await foreach (int item in runtime.RunStreamAsync(backend, new AudioJob(runtime.Tts, "dia"),
                ct => { attempts++; return Items(3, failAfter: 1, ct); }, CancellationToken.None))
            {
                received.Add(item);
            }
        });

        Assert.Equal(1, attempts);
        Assert.Equal(new[] { 0 }, received);
        Assert.Equal(0, device.FreeAllCalls);
        Assert.Equal(7, await runtime.RunAsync(backend, new AudioJob(runtime.Tts, "dia"), _ => Task.FromResult(7), CancellationToken.None)
            .WaitAsync(TimeSpan.FromSeconds(10)));
    }

    [Fact]
    public void FindOutOfVram_LooksThroughWrappers_AndNothingElse()
    {
        OutOfVramException oom = Oom();
        Assert.Same(oom, AudioRuntime.FindOutOfVram(oom));
        Assert.Same(oom, AudioRuntime.FindOutOfVram(new InvalidOperationException("outer", new AggregateException(oom))));
        Assert.Same(oom, AudioRuntime.FindOutOfVram(new AggregateException(new TimeoutException(), oom)));
        Assert.Null(AudioRuntime.FindOutOfVram(new InvalidOperationException("outer", new TimeoutException())));
        Assert.Null(AudioRuntime.FindOutOfVram(null));
    }

    /// <summary>The driver refusal the Dia sweep failure carried.</summary>
    private static OutOfVramException Oom() => new(128L << 20, 201L << 20, 24082L << 20);

    private static async Task<FakeTts> SeedAsync(AudioRuntime runtime, string key)
    {
        FakeTts runner = new();
        await runtime.Tts.GetOrLoadAsync(key, _ => Task.FromResult<ITtsRunner>(runner), CancellationToken.None);
        return runner;
    }

    private static Task<ITtsRunner> LoadAsync(AudioRuntime runtime, string key, CancellationToken cancel) =>
        runtime.Tts.GetOrLoadAsync(key, _ => Task.FromResult<ITtsRunner>(new FakeTts()), cancel);

    /// <summary>One job that loads <paramref name="key"/> the way a service call does, running <paramref name="onLoad"/>
    /// only when the load actually happens.</summary>
    private static Task<int> RunLoadingAsync(AudioRuntime runtime, IBackend backend, string key, long? estimate,
        Action? onLoad = null)
    {
        Func<long>? estimator = estimate is long bytes ? () => bytes : null;
        return runtime.RunAsync(backend, new AudioJob(runtime.Tts, key, estimator), async ct =>
        {
            await runtime.Tts.GetOrLoadAsync(key, _ =>
            {
                onLoad?.Invoke();
                return Task.FromResult<ITtsRunner>(new FakeTts());
            }, ct);
            return 0;
        }, CancellationToken.None);
    }

    /// <summary>Yields 0..count-1, throwing an out-of-VRAM failure once <paramref name="failAfter"/> items are out
    /// (negative never fails).</summary>
    private static async IAsyncEnumerable<int> Items(int count, int failAfter, [EnumeratorCancellation] CancellationToken cancel)
    {
        for (int i = 0; i < count || i == failAfter; i++)
        {
            await Task.Yield();
            cancel.ThrowIfCancellationRequested();
            if (i == failAfter)
            {
                throw Oom();
            }
            yield return i;
        }
    }

    private static async Task<List<int>> CollectAsync(IAsyncEnumerable<int> stream)
    {
        List<int> items = [];
        await foreach (int item in stream)
        {
            items.Add(item);
        }
        return items;
    }

    private sealed class FakeTts : ITtsRunner
    {
        public bool Disposed { get; private set; }

        public int SampleRate => 24_000;

        public float[] Synthesize(IBackend backend, TtsJob job) => [];

        public void Dispose() => Disposed = true;
    }

    /// <summary>A backend that reports scripted VRAM and counts the releases the runtime asks for. It says it is a CPU,
    /// so the device gate stays out of the way; nothing the runtime does not call is reachable.</summary>
    public class FakeVramBackend : DispatchProxy
    {
        public long FreeBytes { get; set; }

        public long TotalBytes { get; set; }

        public int FreeAllCalls { get; private set; }

        public static IBackend Create(long free, long total, out FakeVramBackend fake)
        {
            IBackend backend = Create<IBackend, FakeVramBackend>();
            fake = (FakeVramBackend)backend;
            fake.FreeBytes = free;
            fake.TotalBytes = total;
            return backend;
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            ArgumentNullException.ThrowIfNull(targetMethod);
            switch (targetMethod.Name)
            {
                case "get_Device":
                    return DeviceKind.Cpu;
                case nameof(IBackend.GetVramInfo):
                    return (FreeBytes, TotalBytes);
                case nameof(IBackend.FreeAllDeviceMemory):
                    FreeAllCalls++;
                    return null;
                case nameof(IBackend.FreeActivations):
                case nameof(IBackend.TrimMemoryPool):
                    return null;
                default:
                    throw new NotSupportedException($"{targetMethod.Name} is not reachable from AudioRuntime.");
            }
        }
    }
}
