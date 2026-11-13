using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>Says when the mesh given to Paint unwrapped into so many tiny charts that the texture will come out speckled, and what to do about it.</summary>
/// <remarks>A clean mesh gives a few hundred charts of tens to thousands of faces. A mesh torn into thousands of small pieces (a decimated mesh whose seams were not kept, a triangle soup welded only by position)
/// gives about one chart per few faces: the atlas is mostly padding, the views reach few of its texels and the rest is inpainted. One 38,613-face bust with 6,335 connected pieces gave 10,453 charts
/// (3.7 faces each) and a baked texture covered in dark and coloured texels.</remarks>
public static class PaintInputAudit
{
    /// <summary>Fewer charts than this are never a problem, whatever their size (a tiny mesh).</summary>
    public const int MinCharts = 100;

    /// <summary>A mesh whose charts average fewer faces than this is torn.</summary>
    public const double MinFacesPerChart = 12;

    /// <summary>The warning for an unwrap of <paramref name="faceCount"/> faces into <paramref name="chartCount"/> charts, or null when the charts are plausible.</summary>
    public static string? TornMeshWarning(int faceCount, int chartCount)
    {
        if (chartCount < MinCharts || faceCount >= chartCount * MinFacesPerChart)
            return null;
        double average = (double)faceCount / chartCount;
        return $"The mesh unwrapped into {chartCount:N0} charts of {average:F1} faces on average (a clean mesh gives a few hundred), so the texture will be speckled with dark and coloured texels. "
            + "The mesh is probably torn into many small pieces, or decimated without keeping its seams. Remove the small pieces and fill the holes first (the remove-small-components and fill-holes mesh operations), "
            + "or decimate a clean copy again (the decimate operation), then texture the result.";
    }

    /// <summary>The warning for <paramref name="unwrap"/>, the unwrap of <paramref name="faceCount"/> faces.</summary>
    public static string? TornMeshWarning(UvUnwrapResult unwrap, int faceCount)
    {
        ArgumentNullException.ThrowIfNull(unwrap);
        return TornMeshWarning(faceCount, unwrap.ChartCount);
    }
}
