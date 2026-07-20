using HartsyInference.Core.Backends;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>Wraps a transcribe delegate plus the disposables a loaded model owns, so each model is a descriptor.</summary>
internal sealed class SttRunner(Func<IBackend, float[], AudioRequest, string> transcribe, params IDisposable?[] disposables) : ISttRunner
{
    /// <summary>Optional timestamped-decode delegate; left null by models with no timestamp tokens.</summary>
    internal Func<IBackend, float[], AudioRequest, IReadOnlyList<SttSegment>>? Timed { get; init; }

    /// <inheritdoc/>
    public string Transcribe(IBackend backend, float[] audioMono, AudioRequest request) => transcribe(backend, audioMono, request);

    /// <inheritdoc/>
    public IReadOnlyList<SttSegment>? TranscribeTimed(IBackend backend, float[] audioMono, AudioRequest request)
        => Timed?.Invoke(backend, audioMono, request);

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (IDisposable? disposable in disposables)
        {
            disposable?.Dispose();
        }
    }
}
