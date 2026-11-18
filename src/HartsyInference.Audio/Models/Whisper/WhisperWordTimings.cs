using HartsyInference.Audio.Dsp.Alignment;
using HartsyInference.Audio.Pipelines;

namespace HartsyInference.Audio.Models.Whisper;

/// <summary>Turns the cross-attention a probe recorded into a start and end time for each word, by the method of OpenAI's Whisper word timestamps: normalize each frame's attention across the decoded tokens, smooth it along time, average the heads, then warp the text onto the audio frames with dynamic time warping and read where each word's tokens begin.</summary>
internal static class WhisperWordTimings
{
    /// <summary>Encoder positions per second of audio (a 20 ms frame).</summary>
    internal const double FramesPerSecond = 50d;

    /// <summary>The median window along time, in frames.</summary>
    private const int SmoothingWidth = 7;

    /// <summary>Aligns <paramref name="words"/> using a probe filled by a pass over <c>[prompt, text tokens, end-of-text]</c>. <paramref name="promptLength"/> counts the prompt up to and including its last token, the one whose attention row introduces the first word.</summary>
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
            throw new ArgumentException($"The probe recorded {rows} rows but a {promptLength}-token prompt, {textTokens} text tokens and an end-of-text token need {promptLength + textTokens + 1}.", nameof(probe));
        }

        float[] average = AverageHeads(probe, rows, frames);

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
            timed.Add(new WhisperWord(word.Text, rowStart[token] / FramesPerSecond, rowStart[token + word.TokenCount] / FramesPerSecond));
            token += word.TokenCount;
        }
        return timed;
    }

    /// <summary>The mean over the probe's heads of each head's attention, normalized across rows per frame and median-filtered along time; row-major <c>[rows, frames]</c>.</summary>
    private static float[] AverageHeads(WhisperAttentionProbe probe, int rows, int frames)
    {
        float[] sum = new float[rows * frames];
        float[] head = new float[rows * frames];
        for (int h = 0; h < probe.Heads.Count; h++)
        {
            probe.HeadWeights(h).CopyTo(head);
            NormalizeColumns(head, rows, frames);
            MedianFilter.Rows(head, rows, frames, SmoothingWidth);
            for (int i = 0; i < sum.Length; i++)
            {
                sum[i] += head[i];
            }
        }
        float inverse = 1f / probe.Heads.Count;
        for (int i = 0; i < sum.Length; i++)
        {
            sum[i] *= inverse;
        }
        return sum;
    }

    /// <summary>Gives each column of the row-major <c>[rows, cols]</c> matrix zero mean and unit variance across its rows, so a frame every token attends to alike carries no signal; a flat column becomes zero.</summary>
    private static void NormalizeColumns(Span<float> data, int rows, int cols)
    {
        for (int c = 0; c < cols; c++)
        {
            double mean = 0d;
            for (int r = 0; r < rows; r++)
            {
                mean += data[r * cols + c];
            }
            mean /= rows;
            double variance = 0d;
            for (int r = 0; r < rows; r++)
            {
                double delta = data[r * cols + c] - mean;
                variance += delta * delta;
            }
            double deviation = Math.Sqrt(variance / rows);
            for (int r = 0; r < rows; r++)
            {
                data[r * cols + c] = deviation > 1e-9 ? (float)((data[r * cols + c] - mean) / deviation) : 0f;
            }
        }
    }
}
