using System.Reflection;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Gpu.Tests;

/// <summary>A whole-cache release on one thread against ops on another, on a fake device.
///
/// <para>The live failure: an engine-level free ran <see cref="GpuBackendBase.FreeAllDeviceMemory"/> while a Whisper
/// decode on another thread was inside <c>CopyToDevice</c>. The cache's plain dictionaries were mutated from both
/// threads, one of them was left with a cycle in a bucket chain, and every later lookup on that backend threw
/// "Operations that change non-concurrent collections must have exclusive access" until the process restarted.</para>
///
/// <para>The handshake tests park one side inside the cache with an event and check what the other side can do
/// meanwhile, so they fail the same way on every run rather than when the scheduler happens to interleave.</para></summary>
public sealed class ResidencyReleaseRaceTests
{
    /// <summary>Long enough that a release free to run finishes well inside it.</summary>
    private static readonly TimeSpan Blocked = TimeSpan.FromMilliseconds(300);

    private static readonly TimeSpan Generous = TimeSpan.FromSeconds(30);

    /// <summary>The deterministic form of the live race: a release issued while an op is inside an upload must wait
    /// for that op to finish, instead of clearing the cache under it.</summary>
    [Fact]
    public void A_Full_Release_Waits_For_The_Op_In_Flight()
    {
        using FakeBackend backend = new();
        using Tensor weight = NewTensor();
        using Tensor output = NewTensor();
        backend.Cache.PauseUploads();

        Worker op = Worker.Start(() => backend.RunOp(() =>
        {
            backend.Cache.CopyToDevice(weight);
            backend.Cache.CacheActivation(output, backend.Cache.AllocateForTest(Size(output)), Size(output));
        }));
        Assert.True(backend.Cache.UploadEntered.Wait(Generous));

        Worker release = Worker.Start(backend.FreeAllDeviceMemory);
        bool releasedDuringOp = release.Join(Blocked);

        backend.Cache.ResumeUploads();
        Assert.True(op.Join(Generous));
        Assert.True(release.Join(Generous));
        Assert.False(releasedDuringOp, "FreeAllDeviceMemory ran while another thread's op was inside the cache.");
        Assert.Equal(0, backend.Cache.ActivationCountForTest);
    }

    /// <summary>The other direction: an op that starts while a release is sweeping waits for the sweep to finish.</summary>
    [Fact]
    public void An_Op_Waits_For_A_Release_In_Progress()
    {
        using FakeBackend backend = new();
        using Tensor resident = NewTensor();
        using Tensor lookup = NewTensor();
        backend.RunOp(() => backend.Cache.PreloadWeights([resident]));
        backend.Cache.PauseFrees();

        Worker release = Worker.Start(backend.FreeAllDeviceMemory);
        Assert.True(backend.Cache.FreeEntered.Wait(Generous));

        Worker op = Worker.Start(() => backend.RunOp(() => backend.Cache.CopyToDevice(lookup)));
        bool ranDuringRelease = op.Join(Blocked);

        backend.Cache.ResumeFrees();
        Assert.True(release.Join(Generous));
        Assert.True(op.Join(Generous));
        Assert.False(ranDuringRelease, "An op touched the cache while FreeAllDeviceMemory was sweeping it.");
        Assert.Equal(0, backend.Cache.WeightCountForTest);
    }

    /// <summary>A tensor disposed on another thread reaches the cache through its binding callback, outside any op.
    /// It must not touch the cache while another thread's op is inside it, and it must not wait either: the release
    /// is queued and runs at the cache's next op.</summary>
    [Fact]
    public void A_Dispose_During_Another_Threads_Op_Is_Deferred_To_The_Next_Op()
    {
        using FakeBackend backend = new();
        using Tensor weight = NewTensor();
        Tensor activation = NewTensor();
        backend.RunOp(() =>
            backend.Cache.CacheActivation(activation, backend.Cache.AllocateForTest(Size(activation)), Size(activation)));
        backend.Cache.PauseUploads();

        Worker op = Worker.Start(() => backend.RunOp(() => backend.Cache.CopyToDevice(weight)));
        Assert.True(backend.Cache.UploadEntered.Wait(Generous));

        Worker dispose = Worker.Start(activation.Dispose);
        Assert.True(dispose.Join(Generous), "A tensor dispose waited for another thread's op.");
        int cachedAfterDispose = backend.Cache.ActivationCountForTest;

        backend.Cache.ResumeUploads();
        Assert.True(op.Join(Generous));
        Assert.Equal(1, cachedAfterDispose);
        backend.RunOp(() => { });
        Assert.Equal(0, backend.Cache.ActivationCountForTest);
    }

    /// <summary>A host read on another thread needs the bytes now, so it waits for the op in flight instead.</summary>
    [Fact]
    public void A_Host_Read_Waits_For_The_Op_In_Flight()
    {
        using FakeBackend backend = new();
        using Tensor weight = NewTensor();
        using Tensor activation = NewTensor();
        backend.RunOp(() =>
            backend.Cache.CacheActivation(activation, backend.Cache.AllocateForTest(Size(activation)), Size(activation)));
        backend.Cache.PauseUploads();

        Worker op = Worker.Start(() => backend.RunOp(() => backend.Cache.CopyToDevice(weight)));
        Assert.True(backend.Cache.UploadEntered.Wait(Generous));

        Worker read = Worker.Start(() => ReadHost(activation));
        bool readDuringOp = read.Join(Blocked);

        backend.Cache.ResumeUploads();
        Assert.True(op.Join(Generous));
        Assert.True(read.Join(Generous));
        Assert.False(readDuringOp, "A host read synced from the cache while another thread's op was inside it.");
        Assert.Equal(0, backend.Cache.ActivationCountForTest);
    }

    /// <summary>Disposing the backend is a whole-cache release too, and waits for the op in flight.</summary>
    [Fact]
    public void Disposing_The_Backend_Waits_For_The_Op_In_Flight()
    {
        FakeBackend backend = new();
        using Tensor weight = NewTensor();
        backend.Cache.PauseUploads();

        Worker op = Worker.Start(() => backend.RunOp(() => backend.Cache.CopyToDevice(weight)));
        Assert.True(backend.Cache.UploadEntered.Wait(Generous));

        Worker dispose = Worker.Start(((IDisposable)backend).Dispose);
        bool disposedDuringOp = dispose.Join(Blocked);

        backend.Cache.ResumeUploads();
        Assert.True(op.Join(Generous));
        Assert.True(dispose.Join(Generous));
        Assert.False(disposedDuringOp, "The backend was disposed while another thread's op was inside its cache.");
    }

    /// <summary>A free that throws partway through a full release still hands back every other buffer and leaves the
    /// cache empty, then reports the failure.</summary>
    [Fact]
    public void A_Failed_Free_Does_Not_Stop_The_Full_Release()
    {
        using FakeBackend backend = new();
        Tensor[] weights = [NewTensor(), NewTensor(), NewTensor()];
        backend.RunOp(() => backend.Cache.PreloadWeights(weights));
        backend.Cache.FailFreesOf = 1;

        Assert.Throws<InvalidOperationException>(backend.FreeAllDeviceMemory);

        Assert.Equal(3, backend.Cache.FreeAttempts);
        Assert.Equal(0, backend.Cache.WeightCountForTest);
        backend.Cache.FailFreesOf = 0;
        backend.RunOp(() => backend.Cache.PreloadWeights(weights));
        Assert.Equal(3, backend.Cache.WeightCountForTest);
        foreach (Tensor weight in weights)
        {
            weight.Dispose();
        }
    }

    private static unsafe void ReadHost(Tensor tensor) => _ = tensor.DataPointer;

    /// <summary>The unscripted form: ops and full releases hammered from two threads leave the cache usable.</summary>
    [Fact]
    public void Ops_And_Releases_On_Two_Threads_Never_Corrupt_The_Cache()
    {
        const int Iterations = 4000;
        using FakeBackend backend = new();
        Tensor[] weights = [.. Enumerable.Range(0, 64).Select(_ => NewTensor(16))];
        using Barrier start = new(2);
        Exception? failure = null;

        Thread ops = new(() =>
        {
            try
            {
                start.SignalAndWait();
                for (int i = 0; i < Iterations; i++)
                {
                    backend.RunOp(() =>
                    {
                        Tensor weight = weights[i % weights.Length];
                        backend.Cache.PreloadWeights([weight]);
                        backend.Cache.CopyToDevice(weight);
                        using Tensor output = NewTensor(16);
                        backend.Cache.CacheActivation(output, backend.Cache.AllocateForTest(Size(output)), Size(output));
                    });
                }
            }
            catch (Exception ex)
            {
                failure ??= ex;
            }
        });
        Thread releases = new(() =>
        {
            try
            {
                start.SignalAndWait();
                for (int i = 0; i < Iterations; i++)
                {
                    backend.FreeAllDeviceMemory();
                }
            }
            catch (Exception ex)
            {
                failure ??= ex;
            }
        });
        ops.Start();
        releases.Start();
        Assert.True(ops.Join(Generous) && releases.Join(Generous), "The two threads deadlocked.");

        Assert.Null(failure);
        backend.FreeAllDeviceMemory();
        foreach (Tensor weight in weights)
        {
            backend.RunOp(() => backend.Cache.CopyToDevice(weight));
            weight.Dispose();
        }
    }

    /// <summary>A cache whose bookkeeping is already corrupted must fail the request that finds it and then come back
    /// clean, rather than throwing the same error at every later request on the backend.</summary>
    [Fact]
    public void A_Corrupted_Cache_Fails_Once_And_Recovers_At_The_Next_Op()
    {
        using FakeBackend backend = new();
        using Tensor resident = NewTensor();
        using Tensor lookup = NewTensor();
        backend.RunOp(() => backend.Cache.PreloadWeights([resident]));
        backend.Cache.CorruptWeightChains();

        Assert.Throws<InvalidOperationException>(() => backend.RunOp(() => backend.Cache.CopyToDevice(lookup)));

        backend.RunOp(() => backend.Cache.CopyToDevice(lookup));
        backend.RunOp(() => backend.Cache.PreloadWeights([resident]));
        Assert.Equal(1, backend.Cache.WeightCountForTest);
    }

    /// <summary>The same recovery when the broken collection is the owned-buffer set, reached through an op's own
    /// temporary release rather than a lookup.</summary>
    [Fact]
    public void A_Corrupted_Owned_Buffer_Set_Recovers_At_The_Next_Op()
    {
        using FakeBackend backend = new();
        using Tensor resident = NewTensor();
        backend.RunOp(() => backend.Cache.PreloadWeights([resident]));
        backend.Cache.CorruptOwnedBufferChains();
        FakeCache.Buffer temporary = backend.Cache.AllocateForTest(64);

        Assert.Throws<InvalidOperationException>(() => backend.RunOp(() => backend.Cache.ReleaseIfNotCached(temporary, 64)));

        backend.RunOp(() => backend.Cache.ReleaseIfNotCached(temporary, 64));
        Assert.Equal(0, backend.Cache.WeightCountForTest);
    }

    private static Tensor NewTensor(int elements = 64) => new(new TensorShape(elements), DType.F32);

    private static long Size(Tensor tensor) => GpuResidencyCache<FakeCache.Buffer>.ByteSize(tensor);

    /// <summary>A dedicated thread that keeps its failure, so a test can join it with a timeout and rethrow.</summary>
    private sealed class Worker
    {
        private readonly Thread _thread;
        private Exception? _failure;

        private Worker(Action body)
        {
            _thread = new Thread(() =>
            {
                try
                {
                    body();
                }
                catch (Exception ex)
                {
                    _failure = ex;
                }
            })
            {
                IsBackground = true,
            };
        }

        public static Worker Start(Action body)
        {
            Worker worker = new(body);
            worker._thread.Start();
            return worker;
        }

        /// <summary>Whether the body finished within <paramref name="timeout"/>; rethrows its failure if it did.</summary>
        public bool Join(TimeSpan timeout)
        {
            if (!_thread.Join(timeout))
            {
                return false;
            }
            if (_failure is not null)
            {
                throw new InvalidOperationException("The worker thread failed.", _failure);
            }
            return true;
        }
    }

    /// <summary>A backend that is only its residency cache, with a way to run an op.</summary>
    private sealed class FakeBackend : GpuBackendBase, IDisposable
    {
        public FakeCache Cache { get; } = new();

        protected override IGpuResidency Residency => Cache;

        public void RunOp(Action body)
        {
            using OpScope _op = EnterOp();
            body();
        }

        public override (long FreeBytes, long TotalBytes) GetVramInfo() => (0, 0);

        protected override void TrimMemoryPoolCore() { }

        protected override void DisposeCore() => Cache.Dispose();
    }

    /// <summary>A device whose uploads and frees can be parked mid-call, so a test can hold one thread inside the
    /// cache while it checks what another thread may do.</summary>
    private sealed class FakeCache : GpuResidencyCache<FakeCache.Buffer>
    {
        internal sealed class Buffer(long bytes)
        {
            public long Bytes { get; } = bytes;
        }

        private readonly ManualResetEventSlim _uploadsOpen = new(true);
        private readonly ManualResetEventSlim _freesOpen = new(true);

        public ManualResetEventSlim UploadEntered { get; } = new(false);

        public ManualResetEventSlim FreeEntered { get; } = new(false);

        public int ActivationCountForTest => Activations.Count;

        public int WeightCountForTest => Weights.Count;

        public void PauseUploads() => _uploadsOpen.Reset();

        public void ResumeUploads() => _uploadsOpen.Set();

        public void PauseFrees() => _freesOpen.Reset();

        public void ResumeFrees() => _freesOpen.Set();

        public Buffer AllocateForTest(long bytes) => AllocateDevice(bytes);

        /// <summary>Leaves the weight dictionary in the state a concurrent insert can: every bucket leads to an entry
        /// whose chain points back at itself, so a lookup of any other key walks the loop until the dictionary
        /// reports the concurrent use it cannot otherwise tell from corruption.</summary>
        public void CorruptWeightChains() => CorruptChains(Weights);

        /// <summary>As <see cref="CorruptWeightChains"/>, for the set of buffers the cache owns.</summary>
        public void CorruptOwnedBufferChains() => CorruptChains(CachedBuffers);

        private static void CorruptChains(object collection)
        {
            const BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;
            Type type = collection.GetType();
            int[] buckets = (int[])type.GetField("_buckets", Private)!.GetValue(collection)!;
            Array entries = (Array)type.GetField("_entries", Private)!.GetValue(collection)!;
            Array.Fill(buckets, 1);
            object first = entries.GetValue(0)!;
            FieldInfo next = first.GetType().GetField("next", BindingFlags.Public | BindingFlags.NonPublic
                | BindingFlags.Instance | BindingFlags.IgnoreCase)!;
            next.SetValue(first, 0);
            entries.SetValue(first, 0);
        }

        protected override Buffer AllocateDevice(long bytes) => new(bytes);

        /// <summary>How many of the next frees throw, standing in for a driver error partway through a sweep.</summary>
        public int FailFreesOf { get; set; }

        public int FreeAttempts { get; private set; }

        protected override void FreeDevice(Buffer buffer, long bytes)
        {
            FreeEntered.Set();
            _freesOpen.Wait();
            FreeAttempts++;
            if (FailFreesOf > 0)
            {
                FailFreesOf--;
                throw new InvalidOperationException("The fake device refused a free.");
            }
        }

        protected override void Upload(Buffer destination, Tensor source, long bytes)
        {
            UploadEntered.Set();
            _uploadsOpen.Wait();
        }

        protected override void DownloadSynced(nint hostDestination, Buffer source, long bytes) { }

        protected override void MakeCurrent() { }
    }
}
