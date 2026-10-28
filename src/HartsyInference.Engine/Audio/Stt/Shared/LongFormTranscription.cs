using HartsyInference.Audio.Dsp.Windowing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>Transcribes a clip longer than a model's window: the clip is planned into windows cut in silence, each
/// window is decoded whole, and the results are joined with segment times moved onto the clip's timeline.</summary>
internal static class LongFormTranscription
{
    /// <summary>Plans <paramref name="audio"/> against a model's window; a model with no window gets one window.</summary>
    internal static IReadOnlyList<SpeechWindow> Plan(float[] audio, int sampleRate, double? maxInputSeconds)
    {
        if (maxInputSeconds is not double seconds)
        {
            return [new SpeechWindow(0, audio.Length, Silent: false, HardCut: false)];
        }
        return SpeechWindowPlanner.Plan(audio, sampleRate, (int)(seconds * sampleRate));
    }

    /// <summary>Decodes each non-silent window and joins the results. Timed segments are offset onto the clip timeline,
    /// and a window the timed decode leaves without segments is decoded again as plain text, recorded as one span.</summary>
    internal static LongFormTranscript Transcribe(ISttRunner runner, IBackend backend, float[] audio, int sampleRate,
        IReadOnlyList<SpeechWindow> windows, AudioRequest request, bool timed, CancellationToken cancel)
    {
        List<string> texts = [];
        List<SttSegment>? segments = timed ? [] : null;
        int silent = 0;
        int hardCuts = 0;
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
            IReadOnlyList<SttSegment>? spans = timed ? runner.TranscribeTimed(backend, piece, request) : null;
            if (segments is null || spans is null)
            {
                segments = null;
                texts.Add(runner.Transcribe(backend, piece, request).Trim());
                continue;
            }
            if (spans.Count > 0)
            {
                texts.Add(string.Join(" ", spans.Select(span => span.Text.Trim())).Trim());
                segments.AddRange(spans.Select(span => span with
                {
                    Start = offset + Math.Clamp(span.Start, 0.0, pieceSeconds),
                    End = offset + Math.Clamp(span.End, 0.0, pieceSeconds),
                }));
                continue;
            }
            string text = runner.Transcribe(backend, piece, request).Trim();
            texts.Add(text);
            segments.Add(new SttSegment { Text = text, Start = offset, End = offset + pieceSeconds });
        }

        if (silent > 0 || hardCuts > 0)
        {
            Logs.Info($"[Audio][STT] Long-form: {windows.Count} windows, {silent} silent skipped, {hardCuts} hard cuts (a word may be split there).");
        }
        string joined = string.Join(" ", texts.Where(text => text.Length > 0));
        return new LongFormTranscript(joined, segments, windows.Count, silent, hardCuts);
    }
}
