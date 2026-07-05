using HartsyInference.Audio.Pipelines;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Backends;

namespace HartsyInference.Cli.Dispatch.Handlers;

/// <summary>Speech-to-text via Whisper. The model selector is an HF repo id (or a "whisper-*" shorthand); the checkpoint
/// is downloaded and cached on first use. The generation "prompt" is the path to a WAV file.</summary>
public sealed class TranscribeHandler : IModalityHandler
{
    /// <inheritdoc/>
    public Modality Modality => Modality.Transcribe;

    /// <inheritdoc/>
    public IModalityRunner Load(ModelSpec spec, IBackend backend, IProgressSink progress)
    {
        string repo = ResolveRepo(spec.Requested);
        progress.Stage($"Loading Whisper '{repo}' (downloading on first use) …");
        WhisperPipeline pipeline = WhisperPipeline.LoadAsync(repo).GetAwaiter().GetResult();
        return new TranscribeRunner(repo, pipeline, backend);
    }

    /// <inheritdoc/>
    public GeneratedArtifact Run(IModalityRunner runner, string prompt, ParamState parameters, IProgressSink progress, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        TranscribeRunner transcribe = (TranscribeRunner)runner;

        string wavPath = prompt.Trim();
        if (!File.Exists(wavPath))
            throw new FileNotFoundException($"Audio file not found: {wavPath}");

        WhisperOptions options = new WhisperOptions
        {
            Language = parameters.Get("language") is { Length: > 0 } lang ? lang : "en",
            Translate = string.Equals(parameters.Get("translate"), "true", StringComparison.OrdinalIgnoreCase),
            WithTimestamps = string.Equals(parameters.Get("timestamps"), "true", StringComparison.OrdinalIgnoreCase),
        };

        progress.Stage($"Transcribing {Path.GetFileName(wavPath)} …");
        string transcript = transcribe.Pipeline.TranscribeWav(transcribe.Backend, wavPath, options);

        GeneratedArtifact artifact = new GeneratedArtifact { Kind = ArtifactKind.Text, Text = transcript.Trim(), Extension = "txt" };
        artifact.Meta["model"] = transcribe.ModelId;
        artifact.Meta["audio"] = wavPath;
        return artifact;
    }

    private static string ResolveRepo(string model)
    {
        if (string.IsNullOrWhiteSpace(model) || string.Equals(model, "whisper", StringComparison.OrdinalIgnoreCase))
            return "openai/whisper-tiny";
        if (model.Contains('/'))
            return model;
        return model.ToLowerInvariant() switch
        {
            "whisper-tiny" => "openai/whisper-tiny",
            "whisper-base" => "openai/whisper-base",
            "whisper-small" => "openai/whisper-small",
            "whisper-medium" => "openai/whisper-medium",
            "whisper-large" or "whisper-large-v3" => "openai/whisper-large-v3",
            _ => model,
        };
    }
}
