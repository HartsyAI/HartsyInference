namespace HartsyInference.Audio.Models.VoiceIsolator;

/// <summary>The band split of the isolator: bin ranges that cover every STFT bin once, narrow at low frequency.</summary>
public static class VoiceIsolatorBands
{
    /// <summary>Bin ranges [start, end) of the bands of <paramref name="config"/>, as the reference's <c>band_edges</c> cuts them.</summary>
    public static (int Start, int End)[] Edges(VoiceIsolatorConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        int bins = config.NFft / 2 + 1;
        double hzPerBin = (double)config.SampleRate / config.NFft;
        List<(int, int)> edges = [];
        int start = 0;
        foreach ((int widthHz, int upToHz) in config.BandPlan)
        {
            int width = Math.Max(1, (int)Math.Round(widthHz / hzPerBin, MidpointRounding.ToEven));
            int upTo = Math.Min(bins, (int)Math.Round(upToHz / hzPerBin, MidpointRounding.ToEven));
            while (start < upTo)
            {
                int end = Math.Min(start + width, upTo);
                edges.Add((start, end));
                start = end;
            }
        }
        if (start < bins)
        {
            edges.Add((start, bins));
        }
        return [.. edges];
    }
}
