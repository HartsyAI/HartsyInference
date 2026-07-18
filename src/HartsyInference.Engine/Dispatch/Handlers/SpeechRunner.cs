using HartsyInference.Audio.Pipelines;
using HartsyInference.Engine;
using HartsyInference.Core.Backends;

namespace HartsyInference.Engine.Dispatch.Handlers;

/// <summary>A loaded Piper text-to-speech voice. Holds a borrowed backend because <see cref="PiperPipeline.SynthesizeText"/>
/// takes the backend per call.</summary>
public sealed class SpeechRunner : IModalityRunner
{
    /// <summary>Creates a runner over <paramref name="pipeline"/>, borrowing <paramref name="backend"/>.</summary>
    public SpeechRunner(string modelId, PiperPipeline pipeline, IBackend backend)
    {
        ModelId = modelId;
        Pipeline = pipeline;
        Backend = backend;
    }

    /// <inheritdoc/>
    public Modality Modality => Modality.Speech;

    /// <inheritdoc/>
    public string ModelId { get; }

    /// <summary>The Piper pipeline to drive.</summary>
    public PiperPipeline Pipeline { get; }

    /// <summary>The borrowed compute backend (owned by the caller).</summary>
    public IBackend Backend { get; }

    /// <inheritdoc/>
    public void Dispose() => Pipeline.Dispose();
}
