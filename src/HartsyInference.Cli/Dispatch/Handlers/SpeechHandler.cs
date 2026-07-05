using System.Globalization;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cli.Infra;
using HartsyInference.Core.Backends;

namespace HartsyInference.Cli.Dispatch.Handlers;

/// <summary>Text-to-speech via Piper: the model selector is a Piper voice id (downloaded on first use) or a local
/// <c>.onnx</c> voice; the phoneme frontend (espeak) is built in, so it synthesizes raw text directly.</summary>
public sealed class SpeechHandler : IModalityHandler
{
    private const string DefaultVoice = "en_US-lessac-medium";

    /// <inheritdoc/>
    public Modality Modality => Modality.Speech;

    /// <inheritdoc/>
    public IModalityRunner Load(ModelSpec spec, IBackend backend, IProgressSink progress)
    {
        if (spec.LocalPath is { } path && File.Exists(path) && path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            string configJson = File.Exists(path + ".json") ? path + ".json" : Path.ChangeExtension(path, ".onnx.json");
            progress.Stage($"Loading Piper voice {Path.GetFileName(path)} …");
            PiperPipeline fromFile = PiperPipeline.LoadFromFiles(path, configJson);
            return new SpeechRunner(Path.GetFileNameWithoutExtension(path), fromFile, backend);
        }

        string voiceId = string.IsNullOrWhiteSpace(spec.Requested) ? DefaultVoice : spec.Requested;
        progress.Stage($"Loading Piper voice '{voiceId}' (downloading on first use) …");
        PiperPipeline pipeline = PiperPipeline.LoadAsync(voiceId).GetAwaiter().GetResult();
        return new SpeechRunner(voiceId, pipeline, backend);
    }

    /// <inheritdoc/>
    public GeneratedArtifact Run(IModalityRunner runner, string prompt, ParamState parameters, IProgressSink progress, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        SpeechRunner speech = (SpeechRunner)runner;

        float speed = parameters.GetFloat("speed", 1f);
        float? lengthScale = speed > 0f ? 1f / speed : null;
        int seed = parameters.GetInt("seed", 0);

        progress.Stage("Synthesizing …");
        float[] audio = speech.Pipeline.SynthesizeText(speech.Backend, prompt, lengthScale: lengthScale, noiseScale: null, seed: seed < 0 ? 0 : seed);
        int sampleRate = speech.Pipeline.SampleRate;

        byte[] wav = ToWav(audio, sampleRate);
        double seconds = audio.Length / (double)sampleRate;

        GeneratedArtifact artifact = new GeneratedArtifact
        {
            Kind = ArtifactKind.Audio,
            FileBytes = wav,
            Extension = "wav",
            Text = $"{seconds:F1}s of speech ({sampleRate} Hz)",
        };
        artifact.Meta["model"] = speech.ModelId;
        artifact.Meta["seconds"] = seconds.ToString("F1", CultureInfo.InvariantCulture);
        artifact.Meta["sample_rate"] = sampleRate.ToString(CultureInfo.InvariantCulture);
        return artifact;
    }

    private static byte[] ToWav(float[] samples, int sampleRate)
    {
        using MemoryStream ms = new MemoryStream();
        WavFile.WriteMono16(ms, samples, sampleRate);
        return ms.ToArray();
    }
}
