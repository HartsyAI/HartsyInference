using System.Globalization;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Text-to-music service. Phase 1 drives the existing music handler over the flat request fields; the full
/// lift (MusicGen/AudioGen/ACE-Step/YuE/HeartMuLa + continuation/repaint/cover) lands with E-AUD-2 / E-IMG-3.</summary>
public sealed class MusicService : IMusicService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal MusicService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<AudioResult> GenerateAsync(ModelSpec spec, MusicRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default)
    {
        if (request.Continuation is not null || request.Repaint is not null || request.Cover is not null)
        {
            throw new NotSupportedException(
                "Music continuation/repaint/cover editing modes are wired by E-AUD-2; not yet available.");
        }

        ParamState parameters = new ParamState(Modality.Music);
        parameters.Put("duration", request.Duration.ToString(CultureInfo.InvariantCulture));
        parameters.Put("seed", request.Seed.ToString(CultureInfo.InvariantCulture));

        SinkBridge sink = new SinkBridge(progress, null);
        return Task.Run(
            () =>
            {
                GeneratedArtifact artifact = _engine.RunHandler(spec, request.Prompt, parameters, sink, cancel);
                return AudioResults.From(artifact);
            },
            cancel);
    }
}
