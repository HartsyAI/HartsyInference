using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Voice-conversion service: decodes the source (and optional target) at the descriptor's input rate and runs the conversion on the shared audio device under the generation lock.</summary>
public sealed class VoiceConversionService : IVoiceConversionService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal VoiceConversionService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<AudioResult> ConvertAsync(ModelSpec spec, VoiceConversionRequest request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        AudioModelSelector selector = AudioModelSelector.Parse(spec);
        VcModelDescriptor descriptor = VcCatalog.Resolve(selector.Id);
        string key = descriptor.CacheKey(selector);
        IBackend backend = _engine.Backend;

        return _engine.AudioRuntime.RunAsync(backend, $"vc:{key}", async ct =>
        {
            float[] source = AudioClipCodec.DecodeMono(request.Source, descriptor.InputSampleRate);
            if (source.Length == 0)
            {
                throw new ArgumentException("The source audio decoded to no samples.", nameof(request));
            }
            // Optional target voice for tone-color transfer models; source-only models ignore it.
            float[]? target = request.Target is null ? null : AudioClipCodec.DecodeMono(request.Target, descriptor.InputSampleRate);
            ct.ThrowIfCancellationRequested();

            IVcRunner runner = await _engine.AudioRuntime.Vc
                .GetOrLoadAsync(key, token => descriptor.LoadAsync(selector, token), ct).ConfigureAwait(false);
            long started = Environment.TickCount64;
            float[] audio = runner.Convert(backend, source, target, request);
            if (audio is null || audio.Length == 0)
            {
                throw new InvalidOperationException("The voice-conversion model produced no audio.");
            }
            double seconds = AudioClipCodec.Seconds(audio.Length, runner.SampleRate);
            Logs.Verbose($"[Audio][VC] Converted {seconds:0.0}s @ {runner.SampleRate} Hz in {Environment.TickCount64 - started}ms.");
            return new AudioResult
            {
                Data = AudioClipCodec.EncodeWav(audio, null, runner.SampleRate),
                Format = "wav",
                DurationSeconds = seconds,
                SampleRate = runner.SampleRate,
                Meta = new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    ["model"] = key,
                    ["pitch_shift"] = request.PitchShift.ToString(CultureInfo.InvariantCulture),
                },
            };
        }, cancel);
    }
}
