using SharpInference.Core.Exceptions;

namespace SharpInference.Diffusion.Prompting;

/// <summary>Normalized 0–1000 bounding box. Field order matches Ideogram 4's JSON convention: <c>[y_min, x_min, y_max, x_max]</c> (row/y first).</summary>
public readonly record struct BoundingBox(int YMin, int XMin, int YMax, int XMax)
{
    /// <summary>Validates the box is inside <c>[0, 1000]</c> on both axes with <c>min &lt; max</c>. Throws <see cref="SharpInferenceException"/> otherwise.</summary>
    public void Validate()
    {
        if (YMin < 0 || XMin < 0 || YMax > 1000 || XMax > 1000)
            throw new SharpInferenceException($"BoundingBox out of [0,1000]: [{YMin},{XMin},{YMax},{XMax}].");
        if (YMin >= YMax || XMin >= XMax)
            throw new SharpInferenceException($"BoundingBox must have min < max: [{YMin},{XMin},{YMax},{XMax}].");
    }

    /// <summary>Converts the normalized box to a pixel-space rectangle <c>(x, y, width, height)</c> at the target resolution.</summary>
    public (int X, int Y, int Width, int Height) ToPixels(int width, int height)
    {
        int x0 = XMin * width / 1000;
        int y0 = YMin * height / 1000;
        int x1 = XMax * width / 1000;
        int y1 = YMax * height / 1000;
        return (x0, y0, Math.Max(1, x1 - x0), Math.Max(1, y1 - y0));
    }
}
