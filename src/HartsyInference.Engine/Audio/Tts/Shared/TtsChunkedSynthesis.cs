using HartsyInference.Audio.Text;

namespace HartsyInference.Engine.Audio;

/// <summary>Model-agnostic long-text synthesis: splits the text with <see cref="TtsTextSegmenter"/>, synthesizes each
/// chunk through a caller-supplied function, and joins the chunks with a short silence. Any model that can synthesize
/// one bounded chunk can use it; the model keeps whatever per-request state it needs (e.g. an RNG) inside its function.</summary>
internal static class TtsChunkedSynthesis
{
    /// <summary>Silence between chunks, in seconds (matches the Hurston reference's 0.25 s join).</summary>
    internal const double GapSeconds = 0.25;

    /// <summary>Synthesizes <paramref name="text"/> chunk by chunk. <paramref name="synthesizeChunk"/> gets the chunk's
    /// index and text and returns its mono PCM at <paramref name="sampleRate"/>.</summary>
    internal static float[] Synthesize(string text, int chunkWords, int sampleRate, Func<int, string, float[]> synthesizeChunk)
    {
        ArgumentNullException.ThrowIfNull(synthesizeChunk);
        IReadOnlyList<string> chunks = TtsTextSegmenter.Pack(text, chunkWords);
        if (chunks.Count == 0)
        {
            return [];
        }

        int gap = (int)Math.Round(sampleRate * GapSeconds);
        List<float[]> parts = new(chunks.Count);
        for (int i = 0; i < chunks.Count; i++)
        {
            parts.Add(synthesizeChunk(i, chunks[i]));
        }

        int total = 0;
        foreach (float[] part in parts)
        {
            total += part.Length;
        }
        total += gap * (parts.Count - 1);

        float[] joined = new float[total];
        int at = 0;
        for (int i = 0; i < parts.Count; i++)
        {
            parts[i].CopyTo(joined, at);
            at += parts[i].Length;
            if (i < parts.Count - 1)
            {
                at += gap; // silence stays zero
            }
        }
        return joined;
    }
}
