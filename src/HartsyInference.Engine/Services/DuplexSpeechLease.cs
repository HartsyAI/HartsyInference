using HartsyInference.Core.Backends;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary><see cref="IDuplexSpeechSession"/> over a pinned <see cref="IDuplexSpeechRunner"/>: every call runs under the lease's call lock
/// and the device gate, so an engine release waits for the frame in flight and other services run between frames.</summary>
internal sealed class DuplexSpeechLease : AudioRunnerLease, IDuplexSpeechSession
{
    private readonly IDuplexSpeechSession _inner;
    // Never stale: a backend change revokes the lease before the engine drops this backend.
    private readonly IBackend _backend;

    internal DuplexSpeechLease(AudioRuntime runtime, string key, IDuplexSpeechSession inner, IBackend backend)
        : base(runtime, runtime.Tts, key)
    {
        _inner = inner;
        _backend = backend;
    }

    /// <inheritdoc/>
    public int SampleRate => _inner.SampleRate;

    /// <inheritdoc/>
    public int FrameSamples => _inner.FrameSamples;

    /// <inheritdoc/>
    public DuplexFrame Start()
    {
        ThrowIfClosed();
        lock (CallLock)
        {
            ThrowIfClosed();
            using IDisposable gate = DeviceGate.Acquire(_backend, Closing);
            return _inner.Start();
        }
    }

    /// <inheritdoc/>
    public DuplexFrame Step(ReadOnlySpan<float> userFrame)
    {
        ThrowIfClosed();
        lock (CallLock)
        {
            ThrowIfClosed();
            using IDisposable gate = DeviceGate.Acquire(_backend, Closing);
            return _inner.Step(userFrame);
        }
    }

    /// <summary>Closes the conversation, then releases the pin.</summary>
    public new void Dispose()
    {
        base.Dispose();
        _inner.Dispose();
    }
}
