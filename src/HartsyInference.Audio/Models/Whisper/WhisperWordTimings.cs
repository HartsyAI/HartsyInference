using HartsyInference.Audio.Dsp.Alignment;
using HartsyInference.Audio.Pipelines;

namespace HartsyInference.Audio.Models.Whisper;

/// <summary>Turns the mean cross-attention a probe recorded into a start and end time for each word, by the method of
/// OpenAI's Whisper word timestamps: warp the text onto the audio frames with dynamic time warping and read where
/// each word's tokens begin.</summary>
internal static class WhisperWordTimings
{
    /// <summary>Encoder positions per second of audio (a 20 ms frame).</summary>
    internal const double FramesPerSecond = 50d;

    /// <summary>Aligns <paramref name="words"/> using a probe filled by a pass over <c>[prompt, text tokens,
    /// end-of-text]</c>. <paramref name="promptLength"/> counts the prompt up to and including its last token, the
    /// one whose attention row introduces the first word.</summary>
    internal static IReadOnlyList<WhisperWord> Compute(WhisperAttentionProbe probe, int promptLength, IReadOnlyList<WhisperWordTokens> words)
    {
        int textTokens = 0;
        foreach (WhisperWordTokens word in words)
        {
            textTokens += word.TokenCount;
        }
        int rows = probe.Rows;
        int frames = probe.Frames;
        if (rows != promptLength + textTokens + 1)
        {
            throw new ArgumentException($"The probe recorded {rows} rows but a {promptLength}-token prompt, {textTokens} text tokens "
                + $"and an end-of-text token need {promptLength + textTokens + 1}.", nameof(probe));
        }

        float[] average = probe.MeanAttention();

        // The row of the prompt's last token and each text token's row, without the end-of-text row: row i then says
        // where the audio is while the decoder is about to read text token i.
        int used = textTokens + 1;
        float[] cost = new float[used * frames];
        for (int r = 0; r < used; r++)
        {
            ReadOnlySpan<float> source = average.AsSpan((promptLength - 1 + r) * frames, frames);
            Span<float> target = cost.AsSpan(r * frames, frames);
            for (int f = 0; f < frames; f++)
            {
                target[f] = -source[f];
            }
        }

        (int[] pathRows, int[] pathCols) = DynamicTimeWarping.Align(cost, used, frames);
        int[] rowStart = new int[used];
        int row = -1;
        for (int k = 0; k < pathRows.Length; k++)
        {
            if (pathRows[k] != row)
            {
                row = pathRows[k];
                rowStart[row] = pathCols[k];
            }
        }

        List<WhisperWord> timed = new List<WhisperWord>(words.Count);
        int token = 0;
        foreach (WhisperWordTokens word in words)
        {
            int start = rowStart[token + word.LeadTokens];
            int end = rowStart[token + word.TokenCount - word.TrailTokens];
            timed.Add(new WhisperWord(word.Text, start / FramesPerSecond, end / FramesPerSecond));
            token += word.TokenCount;
        }
        return timed;
    }
}
