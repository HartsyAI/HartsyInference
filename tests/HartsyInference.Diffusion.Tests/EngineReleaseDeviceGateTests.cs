using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>An engine's between-jobs free must not release device memory while a job runs on that device.
///
/// <para>The live failure: <c>FreeMemory</c> unloaded the audio runners, released the audio generation lock, then ran
/// <c>FreeAllDeviceMemory</c> without the device gate. An STT request that started in that gap was inside
/// <c>CopyToDevice</c> when the sweep emptied the residency cache, and the cache's dictionaries were corrupted for
/// every later request. Each test uses its own GPU ordinal, because the device gate is process-wide.</para></summary>
public sealed class EngineReleaseDeviceGateTests
{
    private static readonly TimeSpan Blocked = TimeSpan.FromMilliseconds(400);

    private static readonly TimeSpan Generous = TimeSpan.FromSeconds(30);

    /// <summary>Pins the gate on for one test. It is process-wide, so a test that left <c>vram.sameGpuConcurrent</c> set
    /// would run these with no gate at all.</summary>
    private static IDisposable GateOn() => KnobProfile.Create("device-gate").With(EngineKnobs.SameGpuConcurrent, false).Push();

    /// <summary>The incident's shape: an audio job holds the device when the engine is asked to free memory.</summary>
    [Fact]
    public async Task FreeMemory_WaitsForAnAudioJobOnTheDevice_BeforeReleasingDeviceMemory()
    {
        // A host short of RAM must not evict the fake from inside the job's own setup: that frees device memory before the
        // test measures anything.
        using IDisposable calm = AudioEvictionPressure.Relax();
        using IDisposable gates = GateOn();
        IBackend backend = GateRecordingBackend.Create(ordinal: 9101, out GateRecordingBackend device);
        using InferenceEngine engine = new(backend, "cuda:9101");
        AudioRuntime jobs = new();
        using ManualResetEventSlim running = new(false);
        using ManualResetEventSlim finish = new(false);

        Task<int> job = jobs.RunAsync(backend, new AudioJob(jobs.Stt, "whisper"), async _ =>
        {
            device.JobRunning = true;
            running.Set();
            await Task.Run(() => finish.Wait(Generous));
            device.JobRunning = false;
            return 0;
        }, CancellationToken.None);
        Assert.True(running.Wait(Generous));

        Task free = Task.Run(engine.FreeMemory);
        await Task.WhenAny(free, Task.Delay(Blocked));
        bool freedDuringJob = free.IsCompleted || device.FreeAllCalls > 0;
        finish.Set();
        await job;
        await free.WaitAsync(Generous);

        Assert.False(freedDuringJob, "FreeMemory released device memory while an audio job held the device.");
        Assert.Equal(1, device.FreeAllCalls);
        Assert.Equal(0, device.FreeAllDuringJob);
    }

    /// <summary>A job that outlasts the wait keeps its device memory: the free skips the release instead of running it
    /// under the job.</summary>
    [Fact]
    public void FreeMemory_SkipsTheDeviceRelease_WhenAJobOutlastsTheWait()
    {
        using IDisposable gates = GateOn();
        IBackend backend = GateRecordingBackend.Create(ordinal: 9102, out GateRecordingBackend device);
        using InferenceEngine engine = new(backend, "cuda:9102") { DeviceReleaseWait = TimeSpan.FromMilliseconds(100) };

        using (DeviceGate.AcquireForCudaOrdinal(9102))
        {
            engine.FreeMemory();
        }

        Assert.Equal(0, device.FreeAllCalls);
        engine.FreeMemory();
        Assert.Equal(1, device.FreeAllCalls);
    }

    /// <summary>A teardown that cannot wait for a job stops the job at its next op before it disposes the backend,
    /// instead of freeing device memory under it.</summary>
    [Fact]
    public void Teardown_StopsAJobThatOutlastsTheWait_BeforeDisposingTheBackend()
    {
        using IDisposable gates = GateOn();
        IBackend backend = GateRecordingBackend.Create(ordinal: 9103, out GateRecordingBackend device);
        InferenceEngine engine = new(backend, "cuda:9103") { DeviceReleaseWait = TimeSpan.FromMilliseconds(100) };

        using (DeviceGate.AcquireForCudaOrdinal(9103))
        {
            engine.Dispose();
        }

        Assert.Equal(["RefuseNewOps", "Dispose"], device.Lifecycle);
    }

    /// <summary>A GPU-looking backend that records when the engine asks it to release everything.</summary>
    public class GateRecordingBackend : DispatchProxy
    {
        private int _freeAllCalls;
        private int _freeAllDuringJob;

        public int Ordinal { get; private set; }

        public volatile bool JobRunning;

        public int FreeAllCalls => Volatile.Read(ref _freeAllCalls);

        public int FreeAllDuringJob => Volatile.Read(ref _freeAllDuringJob);

        /// <summary>The refusal and disposal calls, in order.</summary>
        public List<string> Lifecycle { get; } = [];

        public static IBackend Create(int ordinal, out GateRecordingBackend fake)
        {
            IBackend backend = Create<IBackend, GateRecordingBackend>();
            fake = (GateRecordingBackend)backend;
            fake.Ordinal = ordinal;
            return backend;
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            ArgumentNullException.ThrowIfNull(targetMethod);
            switch (targetMethod.Name)
            {
                case "get_Device":
                    return DeviceKind.Cuda(Ordinal);
                case nameof(IBackend.GetVramInfo):
                    return (0L, 0L);
                case nameof(IBackend.FreeAllDeviceMemory):
                    Interlocked.Increment(ref _freeAllCalls);
                    if (JobRunning)
                    {
                        Interlocked.Increment(ref _freeAllDuringJob);
                    }
                    return null;
                case nameof(IBackend.FreeActivations):
                case nameof(IBackend.TrimMemoryPool):
                    return null;
                case nameof(IBackend.RefuseNewOps):
                case nameof(IDisposable.Dispose):
                    lock (Lifecycle)
                    {
                        Lifecycle.Add(targetMethod.Name);
                    }
                    return null;
                default:
                    throw new NotSupportedException($"{targetMethod.Name} is not reachable from an engine release.");
            }
        }
    }
}
