namespace HartsyInference.ThreeD.Paint;

/// <summary>One camera of the Paint view set: elevation and azimuth in degrees and the weight its pixels carry in the bake.</summary>
public readonly record struct PaintViewCandidate(double Elevation, double Azimuth, double Weight)
{
    /// <summary>The six views every Paint run uses first: front, right, back, left, top, bottom.</summary>
    public static IReadOnlyList<PaintViewCandidate> Fixed { get; } =
    [
        new(0, 0, 1.0), new(0, 90, 0.1), new(0, 180, 0.5), new(0, 270, 0.1), new(90, 0, 0.05), new(-90, 180, 0.05),
    ];

    /// <summary>The fixed views followed by twelve azimuths at +20 and at -20 degrees elevation (weight 0.01), upstream's candidate order.</summary>
    public static IReadOnlyList<PaintViewCandidate> All { get; } = BuildAll();

    private static List<PaintViewCandidate> BuildAll()
    {
        List<PaintViewCandidate> all = [.. Fixed];
        for (int azimuth = 0; azimuth < 360; azimuth += 30)
        {
            all.Add(new PaintViewCandidate(20, azimuth, 0.01));
            all.Add(new PaintViewCandidate(-20, azimuth, 0.01));
        }
        return all;
    }
}
