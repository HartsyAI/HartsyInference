using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Speech-to-text service: decodes the clip to the descriptor's input rate and runs the transcription on the
/// shared audio device under the generation lock. Timestamps come from Whisper's native <c>&lt;|t|&gt;</c> tokens
/// (segment granularity — see <see cref="BuildWords"/>) and diarization from <see cref="SpeakerDiarizer"/>.</summary>
public sealed class TranscribeService : ITranscribeService
{
    private readonly InferenceEngine _engine;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal TranscribeService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<TranscriptResult> RunAsync(ModelSpec spec, AudioRequest request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        AudioModelSelector selector = AudioModelSelector.Parse(spec);
        SttModelDescriptor descriptor = SttCatalog.Resolve(selector.Id);
        string repo = descriptor.ResolveRepo(selector.Variant);
        IBackend backend = _engine.Backend;

        return _engine.AudioRuntime.RunAsync(backend, $"stt:{repo}", async ct =>
        {
            // Decode straight to the rate the pipeline wants (it would resample otherwise).
            float[] audio = AudioClipCodec.DecodeMono(request.Audio, descriptor.InputSampleRate);
            if (audio.Length == 0)
            {
                throw new ArgumentException("The audio to transcribe decoded to no samples.", nameof(request));
            }
            ct.ThrowIfCancellationRequested();

            ISttRunner runner = await _engine.AudioRuntime.Stt
                .GetOrLoadAsync(repo, token => descriptor.LoadAsync(repo, token), ct).ConfigureAwait(false);
            long started = Environment.TickCount64;

            IReadOnlyList<SttSegment>? segments = null;
            if (request.WordTimestamps || request.Diarization)
            {
                segments = runner.TranscribeTimed(backend, audio, request);
                if (segments is null && request.WordTimestamps)
                {
                    Logs.Warning($"[Audio][STT] '{repo}' emits no timestamp tokens — returning Words = null. "
                        + "Only the Whisper family (whisper / distilwhisper / whisperstreaming) can time a transcript.");
                }
                else if (segments is { Count: 0 } && request.WordTimestamps)
                {
                    Logs.Warning($"[Audio][STT] '{repo}' produced no well-formed timestamp pair — returning Words = null.");
                }
            }
            ct.ThrowIfCancellationRequested();

            // A timestamped decode already produced the text; re-running the plain decode would double the cost.
            string text = segments is { Count: > 0 }
                ? string.Join(" ", segments.Select(segment => segment.Text))
                : runner.Transcribe(backend, audio, request);
            Logs.Verbose($"[Audio][STT] Transcribed {AudioClipCodec.Seconds(audio.Length, descriptor.InputSampleRate):0.0}s "
                + $"via {repo} in {Environment.TickCount64 - started}ms.");

            IReadOnlyList<DiarizedSegment>? speakers = null;
            if (request.Diarization)
            {
                // CAM++ is a 16 kHz model, so the clip is re-decoded rather than resampled from the STT rate.
                float[] mono16k = descriptor.InputSampleRate == SpeakerDiarizer.SampleRate
                    ? audio
                    : AudioClipCodec.DecodeMono(request.Audio, SpeakerDiarizer.SampleRate);
                speakers = SpeakerDiarizer.Shared().Diarize(backend, mono16k, segments, ct);
            }

            return new TranscriptResult
            {
                Text = text?.Trim() ?? string.Empty,
                Language = request.Language,
                Words = request.WordTimestamps ? BuildWords(segments, speakers) : null,
                Speakers = speakers,
            };
        }, cancel);
    }

    /// <summary>Projects the model's timestamp spans onto <see cref="WordSegment"/>s. <b>These are segment-level, not
    /// word-level:</b> Whisper's <c>&lt;|t|&gt;</c> tokens delimit phrases, and true word alignment would need
    /// cross-attention DTW, which the decoder does not expose — so <see cref="WordSegment.Word"/> carries the whole
    /// span's text. Speaker indices are filled from the diarized span with the largest time overlap.</summary>
    private static IReadOnlyList<WordSegment>? BuildWords(IReadOnlyList<SttSegment>? segments,
        IReadOnlyList<DiarizedSegment>? speakers)
    {
        if (segments is not { Count: > 0 })
        {
            return null;
        }
        List<WordSegment> words = new List<WordSegment>(segments.Count);
        foreach (SttSegment segment in segments)
        {
            words.Add(new WordSegment
            {
                Word = segment.Text,
                Start = segment.Start,
                End = segment.End,
                Speaker = SpeakerFor(segment, speakers),
            });
        }
        return words;
    }

    /// <summary>The diarized speaker whose span overlaps <paramref name="segment"/> most, or null when none does.</summary>
    private static int? SpeakerFor(SttSegment segment, IReadOnlyList<DiarizedSegment>? speakers)
    {
        if (speakers is not { Count: > 0 })
        {
            return null;
        }
        int? best = null;
        double bestOverlap = 0d;
        foreach (DiarizedSegment span in speakers)
        {
            double overlap = Math.Min(segment.End, span.End) - Math.Max(segment.Start, span.Start);
            if (overlap > bestOverlap)
            {
                bestOverlap = overlap;
                best = span.Speaker;
            }
        }
        return best;
    }
}
