using System.Globalization;
using HartsyInference.Audio.Io;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Text-to-speech service: picks a descriptor from the model spec, materializes the optional voice reference,
/// and runs the synthesis on the shared audio device under the generation lock.</summary>
public sealed class SpeechService : ISpeechService
{
    /// <summary>Voice references are decoded at 24 kHz — the rate the cloning models expect.</summary>
    private const int ReferenceSampleRate = 24_000;

    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal SpeechService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<AudioResult> SynthesizeAsync(ModelSpec spec, SpeechRequest request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (string.IsNullOrWhiteSpace(request.Text))
        {
            throw new ArgumentException("No text supplied to synthesize.", nameof(request));
        }
        AudioModelSelector selector = AudioModelSelector.Parse(spec);
        TtsModelDescriptor descriptor = TtsCatalog.Resolve(selector.Id);
        string repo = descriptor.ResolveRepo(selector.Variant);
        IBackend backend = _engine.Backend;

        return AudioRuntime.RunAsync(backend, $"tts:{repo}", async ct =>
        {
            // Materialize the reference once: mono 24 kHz samples plus a temp WAV for pipelines that take a path.
            float[]? referenceMono = null;
            string? referenceWavPath = null;
            if (request.Reference is not null && request.Reference.Data.Length > 0)
            {
                referenceMono = AudioClipCodec.DecodeMono(request.Reference, ReferenceSampleRate);
                referenceWavPath = Path.Combine(Path.GetTempPath(), $"hartsy_voiceref_{Guid.NewGuid():N}.wav");
                using FileStream file = new FileStream(referenceWavPath, FileMode.Create, FileAccess.Write);
                WavFile.WriteMono16(file, referenceMono, ReferenceSampleRate);
            }
            try
            {
                ct.ThrowIfCancellationRequested();
                ITtsRunner runner = await TtsCatalog.Cache
                    .GetOrLoadAsync(repo, token => descriptor.LoadAsync(selector.Variant, token), ct).ConfigureAwait(false);
                TtsJob job = new TtsJob
                {
                    Text = request.Text,
                    RefText = request.RefText,
                    Reference = request.Reference,
                    ReferenceMono24k = referenceMono,
                    ReferenceWavPath = referenceWavPath,
                    // "default" is a placeholder some callers send, not a real voice name.
                    Voice = string.IsNullOrEmpty(request.Voice) || request.Voice.Equals("default", StringComparison.OrdinalIgnoreCase)
                        ? null
                        : request.Voice,
                    Speed = request.Speed,
                    Exaggeration = request.Exaggeration,
                    NfeStep = request.NfeStep,
                    CfgScale = request.CfgScale,
                    Seed = request.Seed,
                };
                long started = Environment.TickCount64;
                float[] samples = runner.Synthesize(backend, job);
                if (samples is null || samples.Length == 0)
                {
                    throw new InvalidOperationException("The text-to-speech model produced no audio.");
                }
                double seconds = AudioClipCodec.Seconds(samples.Length, runner.SampleRate);
                Logs.Verbose($"[Audio][TTS] Synthesized {seconds:0.0}s @ {runner.SampleRate} Hz in {Environment.TickCount64 - started}ms.");
                return new AudioResult
                {
                    Data = AudioClipCodec.EncodeWav(samples, null, runner.SampleRate),
                    Format = "wav",
                    DurationSeconds = seconds,
                    SampleRate = runner.SampleRate,
                    Meta = new Dictionary<string, string>(StringComparer.Ordinal)
                    {
                        ["model"] = repo,
                        ["seed"] = request.Seed.ToString(CultureInfo.InvariantCulture),
                    },
                };
            }
            finally
            {
                DeleteTempReference(referenceWavPath);
            }
        }, cancel);
    }

    private static void DeleteTempReference(string? path)
    {
        if (path is null)
        {
            return;
        }
        try
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Audio][TTS] Failed to delete the temp voice reference '{path}': {ex.Message}");
        }
    }
}
