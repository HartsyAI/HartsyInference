using System.Collections.Concurrent;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine;

namespace HartsyInference.Voice.Gpu;

/// <summary>T2, the voice-GPU thread: the only thread that runs the audio device's models. Jobs (recognize one
/// utterance, synthesize one sentence, warm up, shut down) run one at a time in arrival order.</summary>
/// <remarks>Per job: a job whose token is already cancelled is skipped; otherwise the device's
/// <see cref="DeviceGate"/> slot is taken for that one job only (never across an await, never together with the
/// language model's device, so the ascending-ordinal rule of <see cref="DeviceGate.AcquireAll"/> can never be broken
/// from here), the job runs, <see cref="IBackend.FreeActivations()"/> releases its activations, and its completion
/// is set with continuations forced off this thread. A caller that stops waiting is released at once even while its
/// job is still running here; the job's result is then dropped. <see cref="RequestTrim"/> returns pool memory to the
/// driver once the queue is empty, because <see cref="IBackend.TrimMemoryPool"/> synchronizes the stream and must not
/// sit between two sentences of a reply.
/// <para>A job that throws <see cref="ObjectDisposedException"/> found its models released by their owner (an engine
/// release revokes runner leases). The worker then runs the reopen hook once, outside the gate because reopening
/// takes the gate itself, and retries the job once; if either fails the job fails, its caller reports it, and the
/// thread carries on with the next job.</para></remarks>
internal sealed class VoiceGpuWorker : IDisposable
{
    private readonly IBackend _device;
    private readonly Action? _reopen;
    private readonly BlockingCollection<Job> _queue = new(new ConcurrentQueue<Job>());
    private readonly Thread _thread;
    private Task? _stopped;
    private int _trimRequested;
    private long _completed;
    private int _reopens;
    private int _disposed;

    /// <summary>Starts the thread for <paramref name="device"/>, which it borrows. <paramref name="reopen"/> reloads the
    /// models after a job found them released; null fails such a job at once.</summary>
    public VoiceGpuWorker(IBackend device, Action? reopen = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        _device = device;
        _reopen = reopen;
        _thread = new Thread(Run) { Name = "voice-gpu", IsBackground = true };
        _thread.Start();
    }

    /// <summary>The device the jobs run on.</summary>
    public IBackend Device => _device;

    /// <summary>The worker thread's managed id, so callers can tell whether they are on it.</summary>
    public int ManagedThreadId => _thread.ManagedThreadId;

    /// <summary>Jobs that ran to completion or failure.</summary>
    public long CompletedJobs => Interlocked.Read(ref _completed);

    /// <summary>Times the models were reopened after a release.</summary>
    public int Reopens => Volatile.Read(ref _reopens);

    /// <summary>Queues <paramref name="work"/> and returns its result once it ran on the GPU thread.</summary>
    public async Task<T> RunAsync<T>(VoiceGpuJobKind kind, Func<T> work, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(work);
        if (kind == VoiceGpuJobKind.Shutdown)
        {
            throw new ArgumentException("Shutdown is queued by StopAsync.", nameof(kind));
        }
        cancel.ThrowIfCancellationRequested();
        Job<T> job = new(kind, work, cancel);
        Enqueue(job);
        using CancellationTokenRegistration registration = cancel.UnsafeRegister(static state => ((Job)state!).Cancel(), job);
        return await job.Completion.Task.ConfigureAwait(false);
    }

    /// <summary>Asks for the device's pool to be trimmed once the queue drains; call once per turn, not per sentence.</summary>
    public void RequestTrim() => Interlocked.Exchange(ref _trimRequested, 1);

    /// <summary>Queues the last job, which runs <paramref name="finalWork"/> on the GPU thread under the device gate
    /// (releasing the models there), cancels anything queued after it, and ends the thread.</summary>
    public Task StopAsync(Action? finalWork = null)
    {
        Job<bool> shutdown = new(VoiceGpuJobKind.Shutdown, () =>
        {
            finalWork?.Invoke();
            return true;
        }, CancellationToken.None);
        Task? prior = Interlocked.CompareExchange(ref _stopped, shutdown.Completion.Task, null);
        if (prior is not null)
        {
            return prior;
        }
        _queue.Add(shutdown);
        _queue.CompleteAdding();
        return shutdown.Completion.Task;
    }

    private void Enqueue(Job job)
    {
        try
        {
            _queue.Add(job);
        }
        catch (InvalidOperationException ex)
        {
            throw new ObjectDisposedException("The voice GPU thread has stopped.", ex);
        }
    }

    private void Run()
    {
        foreach (Job job in _queue.GetConsumingEnumerable())
        {
            if (job.IsCompleted || job.Cancellation.IsCancellationRequested)
            {
                job.Cancel();
                continue;
            }
            Execute(job);
            if (job.Kind == VoiceGpuJobKind.Shutdown)
            {
                break;
            }
            if (_queue.Count == 0 && Interlocked.Exchange(ref _trimRequested, 0) != 0)
            {
                Trim();
            }
        }
        while (_queue.TryTake(out Job? abandoned))
        {
            abandoned.Cancel();
        }
    }

    private void Execute(Job job)
    {
        try
        {
            try
            {
                RunGated(job);
            }
            catch (ObjectDisposedException released) when (_reopen is not null && job.Kind != VoiceGpuJobKind.Shutdown)
            {
                Logs.Warning($"[Voice] A {job.Kind} job found the speech models released ({released.Message}); reopening them and retrying once.");
                _reopen();
                Interlocked.Increment(ref _reopens);
                RunGated(job);
            }
        }
        catch (OperationCanceledException) when (job.Cancellation.IsCancellationRequested)
        {
            job.Cancel();
        }
        catch (Exception ex)
        {
            job.Fail(ex);
        }
        Interlocked.Increment(ref _completed);
    }

    private void RunGated(Job job)
    {
        using IDisposable gate = DeviceGate.Acquire(_device, job.Cancellation);
        try
        {
            job.Execute();
        }
        finally
        {
            _device.FreeActivations();
        }
    }

    private void Trim()
    {
        try
        {
            using IDisposable gate = DeviceGate.Acquire(_device);
            _device.TrimMemoryPool();
        }
        catch (Exception ex)
        {
            // Only returns cached memory to the driver; a failure leaves the pool as it was and the next job unaffected.
            Logs.Warning($"[Voice] Trimming the audio device's memory pool failed: {ex.Message}");
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        Task stopped = StopAsync();
        if (Thread.CurrentThread != _thread)
        {
            _thread.Join();
        }
        if (stopped.IsFaulted)
        {
            Logs.Error("[Voice] The GPU thread's shutdown work failed.", stopped.Exception!);
        }
        _queue.Dispose();
    }

    private abstract class Job(VoiceGpuJobKind kind, CancellationToken cancel)
    {
        public VoiceGpuJobKind Kind { get; } = kind;

        public CancellationToken Cancellation { get; } = cancel;

        public abstract bool IsCompleted { get; }

        public abstract void Execute();

        public abstract void Fail(Exception error);

        public abstract void Cancel();
    }

    private sealed class Job<T>(VoiceGpuJobKind kind, Func<T> work, CancellationToken cancel) : Job(kind, cancel)
    {
        public TaskCompletionSource<T> Completion { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public override bool IsCompleted => Completion.Task.IsCompleted;

        public override void Execute() => Completion.TrySetResult(work());

        public override void Fail(Exception error) => Completion.TrySetException(error);

        public override void Cancel() => Completion.TrySetCanceled(Cancellation);
    }
}
