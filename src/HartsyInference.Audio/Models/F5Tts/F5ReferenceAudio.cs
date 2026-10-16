using HartsyInference.Audio.Dsp;

namespace HartsyInference.Audio.Models.F5Tts;

/// <summary>F5-TTS's <c>preprocess_ref_audio_text</c> clipping, which runs before the reference loudness scale: a
/// reference of more than 12 s is cut at its long pauses (then its short pauses, then at 12 s), and its leading and
/// trailing silence below -42 dBFS is removed with no keep margin before 50 ms of silence is appended.</summary>
internal static class F5ReferenceAudio
{
    private const int MaxMs = 12000;
    private const int MinSplitLongMs = 1000;
    private const int MinSplitShortMs = 100;
    private const int SplitKeepMs = 1000;
    private const double SplitThresholdLongDb = -50.0;
    private const double SplitThresholdShortDb = -40.0;
    private const double EdgeThresholdDb = -42.0;
    private const int TailSilenceMs = 50;
    private const int ClipAccumulateMs = 6000;

    /// <summary>The processed reference at <paramref name="sampleRate"/> (float samples in [-1, 1)). pydub's chain
    /// works on the 16-bit samples, so the input is quantized first.</summary>
    internal static float[] Prepare(float[] mono, int sampleRate)
    {
        short[] pcm = PydubSilence.Quantize(mono);
        short[] clip = ClipLong(pcm, sampleRate, MinSplitLongMs, SplitThresholdLongDb);
        if (PydubSilence.LengthMs(clip.Length, sampleRate) > MaxMs)
            clip = ClipLong(pcm, sampleRate, MinSplitShortMs, SplitThresholdShortDb);
        if (PydubSilence.LengthMs(clip.Length, sampleRate) > MaxMs)
            clip = PydubSilence.Take(clip, sampleRate, MaxMs);

        short[] edged = PydubSilence.RemoveEdges(clip, sampleRate, EdgeThresholdDb, keepMs: 0);
        float[] tail = new float[PydubSilence.ResampledSilenceFrames(TailSilenceMs, sampleRate)];
        float[] result = new float[edged.Length + tail.Length];
        PydubSilence.ToFloat(edged).CopyTo(result, 0);
        return result;
    }

    /// <summary>Root mean square over the whole clip, as upstream's <c>infer_batch_process</c> computes it.</summary>
    internal static double Rms(ReadOnlySpan<float> audio)
    {
        if (audio.Length == 0) return 0;
        double sum = 0;
        foreach (float v in audio) sum += (double)v * v;
        return Math.Sqrt(sum / audio.Length);
    }

    // The accumulation of pydub's stages: pieces are added in order until the clip passes 6 s and the next piece
    // would take it past 12 s.
    private static short[] ClipLong(short[] pcm, int sampleRate, int minSilenceMs, double thresholdDb)
    {
        List<short[]> pieces = PydubSilence.SplitOnSilence(pcm, sampleRate, minSilenceMs, thresholdDb, SplitKeepMs);
        List<short[]> kept = new();
        int keptSamples = 0;
        foreach (short[] piece in pieces)
        {
            // pydub measures the joined clip, so the length is taken from the sample count, not from each piece.
            int keptMs = PydubSilence.LengthMs(keptSamples, sampleRate);
            int joinedMs = PydubSilence.LengthMs(keptSamples + piece.Length, sampleRate);
            if (keptMs > ClipAccumulateMs && joinedMs > MaxMs) break;
            kept.Add(piece);
            keptSamples += piece.Length;
        }
        return PydubSilence.Concat(kept);
    }
}
