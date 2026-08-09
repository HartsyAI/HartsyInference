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
        // A descriptor whose weights ARE the voice (Piper ships one .onnx per voice) needs the voice in the
        // variant, or every voice would share the first-loaded pipeline.
        string variant = descriptor.VoiceSelectsWeights && !string.IsNullOrWhiteSpace(request.Voice)
            && !request.Voice.Equals("default", StringComparison.OrdinalIgnoreCase)
            ? request.Voice
            : selector.Variant;
        string repo = descriptor.ResolveRepo(variant);
        IBackend backend = _engine.Backend;
        TtsLoadContext loadContext = BuildLoadContext(backend);
        string key = repo + (descriptor.VoiceSelectsWeights ? "|" + variant : "") + loadContext.CacheSuffix();

        return _engine.AudioRuntime.RunAsync(backend, $"tts:{key}", async ct =>
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
                ITtsRunner runner = await _engine.AudioRuntime.Tts
                    .GetOrLoadAsync(key, token => descriptor.LoadAsync(loadContext, variant, token), ct).ConfigureAwait(false);
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
                    SpeakerId = request.SpeakerId,
                    NormalizeLoudness = request.NormalizeLoudness,
                    Temperature = request.Temperature,
                    WaveformTemperature = request.WaveformTemperature,
                    TopP = request.TopP,
                    TopK = request.TopK,
                    MaxTokens = request.MaxTokens,
                    Emotion = request.Emotion,
                    SpeakingRate = request.SpeakingRate,
                    PitchStd = request.PitchStd,
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
                        ["model"] = key,
                        ["seed"] = request.Seed.ToString(CultureInfo.InvariantCulture),
                    },
                };
            }
            finally
            {
                DeleteTempReference(referenceWavPath);
            }
        }, cancel, stageBackends: loadContext.ShardStages is { Count: >= 2 } stages ? [.. stages.Select(s => s.Backend)] : null);
    }

    /// <summary>Builds the load-time context: single-device (byte-identical to pre-placement behavior) unless the
    /// engine placement has ≥2 <c>ShardDevices</c>, in which case CosyVoice's Qwen2 LM gets the resolved shard
    /// backends and layer-splits across them; every other TTS family ignores the shard stages. Mirrors
    /// <c>MusicService.BuildLoadContext</c>.</summary>
    private TtsLoadContext BuildLoadContext(IBackend primary)
    {
        IReadOnlyList<string> shardDevices = _engine.Placement.ShardDevices;
        bool sharded = shardDevices.Count >= 2;
        List<(string Selector, IBackend Backend)>? stages = null;
        if (sharded)
        {
            stages = new List<(string, IBackend)>(shardDevices.Count);
            foreach (string device in shardDevices)
            {
                stages.Add((device, _engine.EnsureBackend(device)));
            }
        }
        return new TtsLoadContext
        {
            Backend = primary,
            ShardStages = stages,
            ShardRatios = sharded ? _engine.Placement.ShardRatios : null,
        };
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
