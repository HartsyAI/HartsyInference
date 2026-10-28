using HartsyInference.Audio.Dsp.Windowing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>Transcribes a clip longer than a model's window: the clip is planned into windows cut in silence, each
/// window is decoded whole, and the results are joined with segment times moved onto the clip's timeline.</summary>
internal static class LongFormTranscription
{
    /// <summary>Refuses a clip over <c>vram.transcribeMaxSeconds</c> when the model is windowed; a model with no window is not limited.</summary>
    internal static void RequireWithinLimit(double? maxInputSeconds, double clipSeconds, string paramName)
    {
        int limit = EngineKnobs.TranscribeMaxSeconds.Value;
        if (maxInputSeconds is not null && clipSeconds > limit)
        {
            throw new ArgumentException($"The audio is {clipSeconds:0} s; this model transcribes up to {limit} s per request "
                + "(vram.transcribeMaxSeconds).", paramName);
        }
    }

    /// <summary>Plans <paramref name="audio"/> against a model's window; a model with no window, or a clip that fits it, gets one window.</summary>
    internal static IReadOnlyList<SpeechWindow> Plan(float[] audio, int sampleRate, double? maxInputSeconds)
    {
        if (maxInputSeconds is not double seconds || audio.Length <= (int)(seconds * sampleRate))
        {
            return [new SpeechWindow(0, audio.Length, Silent: false, HardCut: false)];
        }
        return SpeechWindowPlanner.Plan(audio, sampleRate, (int)(seconds * sampleRate));
    }

    /// <summary>Decodes each non-silent window and joins the results. Timed segments are offset onto the clip timeline,
    /// and a window the timed decode leaves without segments is decoded again as plain text, recorded as one span when it has any.</summary>
    internal static LongFormTranscript Transcribe(ISttRunner runner, IBackend backend, float[] audio, int sampleRate,
        IReadOnlyList<SpeechWindow> windows, AudioRequest request, bool timed, CancellationToken cancel)
    {
        List<string> texts = [];
        List<SttSegment>? segments = timed ? [] : null;
        int silent = 0;
        int hardCuts = 0;
        int empty = 0;
        foreach (SpeechWindow window in windows)
        {
            if (window.HardCut)
            {
                hardCuts++;
            }
            if (window.Silent)
            {
                silent++;
                continue;
            }
            cancel.ThrowIfCancellationRequested();

            float[] piece = audio.AsSpan(window.Start, window.Length).ToArray();
            double offset = window.Start / (double)sampleRate;
            double pieceSeconds = window.Length / (double)sampleRate;
            IReadOnlyList<SttSegment>? decoded = segments is null ? null : runner.TranscribeTimed(backend, piece, request);
            if (segments is null || decoded is null)
            {
                segments = null;
                string plain = runner.Transcribe(backend, piece, request).Trim();
                if (plain.Length == 0)
                {
                    empty++;
                }
                texts.Add(plain);
                continue;
            }
            // A span that starts past the window end is text invented for Whisper's zero padding.
            List<SttSegment> spans = decoded.Where(span => span.Start < pieceSeconds).ToList();
            if (spans.Count > 0)
            {
                string spoken = string.Join(" ", spans.Select(span => span.Text.Trim())).Trim();
                if (spoken.Length == 0)
                {
                    empty++;
                }
                texts.Add(spoken);
                segments.AddRange(spans.Select(span => span with
                {
                    Start = offset + Math.Clamp(span.Start, 0.0, pieceSeconds),
                    End = offset + Math.Clamp(span.End, 0.0, pieceSeconds),
                }));
                continue;
            }
            string text = runner.Transcribe(backend, piece, request).Trim();
            if (text.Length > 0)
            {
                texts.Add(text);
                segments.Add(new SttSegment { Text = text, Start = offset, End = offset + pieceSeconds });
            }
            else
            {
                empty++;
            }
        }

        if (silent > 0 || hardCuts > 0)
        {
            Logs.Info($"[Audio][STT] Long-form: {windows.Count} windows, {silent} silent skipped, {hardCuts} hard cuts (a word may be split there).");
        }
        if (empty > 0)
        {
            Logs.Warning($"[Audio][STT] Long-form: {empty} non-silent window(s) decoded to no text, so any speech in them is missing from the transcript.");
        }
        string joined = string.Join(" ", texts.Where(text => text.Length > 0));
        return new LongFormTranscript(joined, segments, windows.Count, silent, hardCuts, empty);
    }
}
