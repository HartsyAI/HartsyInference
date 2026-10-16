namespace HartsyInference.Audio.Dsp;

/// <summary>pydub's silence rules for the reference clips of ZipVoice and F5-TTS: 16-bit samples, millisecond positions
/// rounded as <c>AudioSegment</c> rounds them, integer RMS in 10 ms chunks and 1 s windows, and a trailing silence that
/// pydub resamples from 11.025 kHz.</summary>
internal static class PydubSilence
{
    private const int ChunkMs = 10;
    private const int KeepMs = 100;
    private const int MinSilenceMs = 1000;
    private const int KeepAroundSegmentsMs = 1000;
    private const double ThresholdDb = -50.0;
    private const int DefaultSilentRate = 11025;
    private const double FullScale = 32768.0;

    /// <summary>Drops silences over one second (unless <paramref name="onlyEdge"/>) and edge silences over 100 ms, then
    /// appends <paramref name="trailSilenceMs"/> of silence. Input and output are float samples in [-1, 1).</summary>
    internal static float[] RemoveSilence(float[] audio, int sampleRate, bool onlyEdge, int trailSilenceMs)
    {
        short[] pcm = Quantize(audio);
        if (!onlyEdge) pcm = Concat(SplitOnSilence(pcm, sampleRate, MinSilenceMs, ThresholdDb, KeepAroundSegmentsMs));
        pcm = RemoveEdges(pcm, sampleRate, ThresholdDb, KeepMs);
        int trail = ResampledSilenceFrames(trailSilenceMs, sampleRate);
        float[] result = new float[pcm.Length + trail];
        for (int i = 0; i < pcm.Length; i++) result[i] = (float)(pcm[i] / FullScale);
        return result;
    }

    /// <summary>Samples in <paramref name="ms"/> of silence after pydub builds it at 11.025 kHz and converts it to
    /// <paramref name="sampleRate"/>; the conversion drops the last two frames.</summary>
    internal static int ResampledSilenceFrames(int ms, int sampleRate)
    {
        int frames = (int)(DefaultSilentRate * (ms / 1000.0));
        if (frames == 0 || sampleRate == DefaultSilentRate) return frames;
        long gcd = Gcd(DefaultSilentRate, sampleRate);
        long inRate = DefaultSilentRate / gcd;
        long outRate = sampleRate / gcd;
        return (int)(outRate * (frames - 1) / inRate) + 1;
    }

    /// <summary>pydub's <c>split_on_silence</c> with a 10 ms seek step: the non-silent pieces, each widened by
    /// <paramref name="keepMs"/> on both sides, where neighbours that overlap are split at their midpoint.</summary>
    internal static List<short[]> SplitOnSilence(
        short[] pcm, int sampleRate, int minSilenceMs, double thresholdDb, int keepMs)
    {
        List<(int Start, int End)> ranges = DetectNonsilent(pcm, sampleRate, minSilenceMs, thresholdDb);
        for (int i = 0; i < ranges.Count; i++)
            ranges[i] = (ranges[i].Start - keepMs, ranges[i].End + keepMs);
        for (int i = 0; i + 1 < ranges.Count; i++)
        {
            int lastEnd = ranges[i].End;
            int nextStart = ranges[i + 1].Start;
            if (nextStart >= lastEnd) continue;
            int middle = (int)Math.Floor((lastEnd + nextStart) / 2.0);
            ranges[i] = (ranges[i].Start, middle);
            ranges[i + 1] = (middle, ranges[i + 1].End);
        }
        int lengthMs = LengthMs(pcm.Length, sampleRate);
        List<short[]> pieces = new(ranges.Count);
        foreach ((int start, int end) in ranges)
            pieces.Add(Slice(pcm, sampleRate, Math.Max(start, 0), Math.Min(end, lengthMs)));
        return pieces;
    }

    /// <summary>pydub's <c>remove_silence_edges</c>-style trim: leading and trailing silence in 10 ms chunks below
    /// <paramref name="thresholdDb"/>, keeping <paramref name="keepMs"/> of each edge.</summary>
    internal static short[] RemoveEdges(short[] pcm, int sampleRate, double thresholdDb, int keepMs)
    {
        short[] trimmed = DropLeading(pcm, sampleRate, thresholdDb, keepMs);
        Array.Reverse(trimmed);
        trimmed = DropLeading(trimmed, sampleRate, thresholdDb, keepMs);
        Array.Reverse(trimmed);
        return trimmed;
    }

    /// <summary>Joins pieces end to end, as pydub's <c>+</c> does at a shared frame rate.</summary>
    internal static short[] Concat(IEnumerable<short[]> pieces)
    {
        List<short> joined = new();
        foreach (short[] piece in pieces) joined.AddRange(piece);
        return [.. joined];
    }

    /// <summary>The first <paramref name="endMs"/> milliseconds (pydub's <c>segment[:endMs]</c>).</summary>
    internal static short[] Take(short[] pcm, int sampleRate, int endMs) => Slice(pcm, sampleRate, 0, endMs);

    /// <summary>The 16-bit samples of a float signal in [-1, 1), as pydub's export would write them.</summary>
    internal static short[] Quantize(float[] audio)
    {
        short[] pcm = new short[audio.Length];
        for (int i = 0; i < audio.Length; i++) pcm[i] = (short)Math.Clamp(audio[i] * 32768f, -32768f, 32767f);
        return pcm;
    }

    /// <summary>Back to float samples in [-1, 1).</summary>
    internal static float[] ToFloat(short[] pcm)
    {
        float[] audio = new float[pcm.Length];
        for (int i = 0; i < pcm.Length; i++) audio[i] = (float)(pcm[i] / FullScale);
        return audio;
    }

    // pydub's len(): whole milliseconds, rounded half to even.
    internal static int LengthMs(int samples, int sampleRate) =>
        (int)Math.Round(1000.0 * (samples / (double)sampleRate), MidpointRounding.ToEven);

    private static short[] DropLeading(short[] pcm, int sampleRate, double thresholdDb, int keepMs)
    {
        int start = Math.Max(0, LeadingSilenceMs(pcm, sampleRate, thresholdDb) - keepMs);
        return Slice(pcm, sampleRate, start, LengthMs(pcm.Length, sampleRate));
    }

    private static int LeadingSilenceMs(short[] pcm, int sampleRate, double thresholdDb)
    {
        int lengthMs = LengthMs(pcm.Length, sampleRate);
        int trim = 0;
        while (trim < lengthMs && IsQuiet(IntegerRms(pcm, sampleRate, trim, trim + ChunkMs), thresholdDb))
            trim += ChunkMs;
        return Math.Min(trim, lengthMs);
    }

    private static bool IsQuiet(int rms, double thresholdDb) =>
        rms == 0 || 20 * Math.Log(rms / FullScale, 10) < thresholdDb;

    private static List<(int Start, int End)> DetectNonsilent(
        short[] pcm, int sampleRate, int minSilenceMs, double thresholdDb)
    {
        int lengthMs = LengthMs(pcm.Length, sampleRate);
        List<(int Start, int End)> silent = DetectSilence(pcm, sampleRate, lengthMs, minSilenceMs, thresholdDb);
        if (silent.Count == 0) return [(0, lengthMs)];
        if (silent[0].Start == 0 && silent[0].End == lengthMs) return [];
        List<(int Start, int End)> nonsilent = new();
        int previousEnd = 0;
        foreach ((int start, int end) in silent)
        {
            nonsilent.Add((previousEnd, start));
            previousEnd = end;
        }
        if (silent[^1].End != lengthMs) nonsilent.Add((previousEnd, lengthMs));
        if (nonsilent[0] == (0, 0)) nonsilent.RemoveAt(0);
        return nonsilent;
    }

    private static List<(int Start, int End)> DetectSilence(
        short[] pcm, int sampleRate, int lengthMs, int minSilenceMs, double thresholdDb)
    {
        List<(int Start, int End)> ranges = new();
        if (lengthMs < minSilenceMs) return ranges;
        double threshold = Math.Pow(10, thresholdDb / 20) * FullScale;
        int lastStart = lengthMs - minSilenceMs;
        List<int> starts = new();
        for (int i = 0; i <= lastStart; i += ChunkMs)
        {
            if (IntegerRms(pcm, sampleRate, i, i + minSilenceMs) <= threshold) starts.Add(i);
        }
        if (lastStart % ChunkMs != 0 && IntegerRms(pcm, sampleRate, lastStart, lastStart + minSilenceMs) <= threshold)
            starts.Add(lastStart);
        if (starts.Count == 0) return ranges;
        int previous = starts[0];
        int rangeStart = previous;
        for (int k = 1; k < starts.Count; k++)
        {
            int current = starts[k];
            bool continuous = current == previous + ChunkMs;
            bool gap = current > previous + minSilenceMs;
            if (!continuous && gap)
            {
                ranges.Add((rangeStart, previous + minSilenceMs));
                rangeStart = current;
            }
            previous = current;
        }
        ranges.Add((rangeStart, previous + minSilenceMs));
        return ranges;
    }

    private static int SampleIndex(int ms, int sampleRate) => (int)(ms * (sampleRate / 1000.0));

    private static short[] Slice(short[] pcm, int sampleRate, int startMs, int endMs)
    {
        int lengthMs = LengthMs(pcm.Length, sampleRate);
        int start = SampleIndex(Math.Min(startMs, lengthMs), sampleRate);
        int end = SampleIndex(Math.Min(endMs, lengthMs), sampleRate);
        short[] slice = new short[Math.Max(0, end - start)];
        int available = Math.Min(end, pcm.Length) - start;
        if (available > 0) Array.Copy(pcm, start, slice, 0, available);
        return slice;
    }

    // audioop.rms of the slice: the root mean square truncated to an integer, counting padded samples as zeros.
    private static int IntegerRms(short[] pcm, int sampleRate, int startMs, int endMs)
    {
        int lengthMs = LengthMs(pcm.Length, sampleRate);
        int start = SampleIndex(Math.Min(startMs, lengthMs), sampleRate);
        int end = SampleIndex(Math.Min(endMs, lengthMs), sampleRate);
        int count = Math.Max(0, end - start);
        if (count == 0) return 0;
        long sumSquares = 0;
        for (int i = start; i < Math.Min(end, pcm.Length); i++) sumSquares += (long)pcm[i] * pcm[i];
        return (int)Math.Sqrt((double)sumSquares / count);
    }

    private static long Gcd(long a, long b) => b == 0 ? a : Gcd(b, a % b);
}
