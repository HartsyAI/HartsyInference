using HartsyInference.Audio.Pipelines;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Backends;

namespace HartsyInference.Cli.Dispatch.Handlers;

/// <summary>A loaded Whisper speech-to-text pipeline. Holds a borrowed backend reference (not owned) because
/// <see cref="WhisperPipeline.TranscribeWav"/> takes the backend per call.</summary>
public sealed class TranscribeRunner : IModalityRunner
{
    /// <summary>Creates a runner over <paramref name="pipeline"/>, borrowing <paramref name="backend"/> for per-call use.</summary>
    public TranscribeRunner(string modelId, WhisperPipeline pipeline, IBackend backend)
    {
        ModelId = modelId;
        Pipeline = pipeline;
        Backend = backend;
    }

    /// <inheritdoc/>
    public Modality Modality => Modality.Transcribe;

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <summary>The Whisper pipeline to drive.</summary>
    public WhisperPipeline Pipeline { get; }

    /// <summary>The borrowed compute backend (owned and disposed by the caller, not this runner).</summary>
    public IBackend Backend { get; }

    /// <inheritdoc/>
    public void Dispose() => Pipeline.Dispose();
}
