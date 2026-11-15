namespace HartsyInference.ThreeD.Paint;

/// <summary>The reference picture laid over the front view of a mesh: an image at the bake's render resolution and a mask of the pixels it covers.
/// The mapping from view to picture is one uniform scale plus an offset: <c>x = Scale * u + OffsetX</c>, <c>y = Scale * v + OffsetY</c>, where u and v run
/// from 0 to 1 across the view and x and y are picture pixels.</summary>
/// <param name="Image">RGB floats in [0, 1], <c>Resolution^2 * 3</c>, laid out like the generated views (row 0 first).</param>
/// <param name="Covered">Pixels that take the picture's colour: covered by the mesh and inside the picture's subject.</param>
/// <param name="Resolution">Side of <paramref name="Image"/>.</param>
/// <param name="Fit">Intersection over union of the mesh's front silhouette and the placed subject, 0 to 1.</param>
/// <param name="MirroredFit">The same measure with the picture flipped left to right (a diagnostic: it should be lower).</param>
/// <param name="Scale">Picture pixels per unit of view.</param>
/// <param name="OffsetX">Picture x at the view's left edge.</param>
/// <param name="OffsetY">Picture y at the view's first row.</param>
public sealed record PaintReferenceProjection(float[] Image, bool[] Covered, int Resolution, double Fit, double MirroredFit, double Scale, double OffsetX, double OffsetY);

/// <summary>Places the reference picture on the mesh's front view so its pixels can be baked at their own resolution. Paint repaints every view at 512 px, so
/// on a full figure a face is a few dozen pixels wide and its eyes, nose and brows are lost; the front view is the one the picture was drawn from, so the
/// picture itself can stand in for that view wherever the mesh and the picture's subject overlap.
/// The subject's outline (from the picture's alpha) is fitted to the mesh's front silhouette: the boxes are matched by height and centre, then a small
/// search on the overlap refines the scale and offset, which keeps a staff or a hat brim the mesh lost from pulling the fit off.</summary>
public static class PaintReferenceProjector
{
    /// <summary>Below this overlap of silhouette and subject the picture is not used: the mesh does not follow it closely enough for its details to land in place.</summary>
    public const double MinimumFit = 0.8;

    /// <summary>Alpha at or above this counts as subject.</summary>
    public const byte SubjectAlpha = 128;

    // Side of the grid the overlap is measured on.
    private const int FitResolution = 256;

    // The refinement keeps the scale within this share of the box match and the offsets within this share of the subject's height.
    private const double ScaleRange = 0.1, OffsetRange = 0.05;

    /// <summary>Fits the picture's subject to the front view (elevation 0, azimuth 0) of <paramref name="renderer"/>'s mesh and lays the picture out at
    /// <paramref name="resolution"/>. Returns null when the picture has no subject or the mesh has no front silhouette.</summary>
    /// <param name="rgb">The picture, RGB bytes, row 0 first.</param>
    /// <param name="alpha">One byte per pixel; at or above <see cref="SubjectAlpha"/> is subject.</param>
    public static PaintReferenceProjection? Project(PaintViewRenderer renderer, ReadOnlySpan<byte> rgb, ReadOnlySpan<byte> alpha, int width, int height, int resolution,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(renderer);
        if (width <= 0 || height <= 0 || rgb.Length != width * height * 3 || alpha.Length != width * height)
            throw new ArgumentException("The picture's colour and alpha planes do not match its size.");
        if (resolution <= 0)
            throw new ArgumentOutOfRangeException(nameof(resolution));

        // The picture's edge pixels mix in the old background, so the subject is pulled in by about a pixel per 512 of picture before it is sampled.
        bool[] subject = new bool[width * height];
        for (int i = 0; i < subject.Length; i++)
            subject[i] = alpha[i] >= SubjectAlpha;
        int erosion = Math.Max(1, Math.Max(width, height) / 512);
        bool[] inner = Erode(subject, width, height, erosion);

        bool[] silhouette = Silhouette(renderer.Rasterize(0, 0, FitResolution, cancellationToken));
        if (!Bounds(silhouette, FitResolution, FitResolution, out int sx0, out int sy0, out int sx1, out int sy1)
            || !Bounds(subject, width, height, out int px0, out int py0, out int px1, out int py1))
            return null;

        double u0 = sx0 / (double)FitResolution, u1 = (sx1 + 1) / (double)FitResolution, v0 = sy0 / (double)FitResolution, v1 = (sy1 + 1) / (double)FitResolution;
        double scale = (py1 + 1 - py0) / (v1 - v0);
        double offsetX = (px0 + px1 + 1) * 0.5 - scale * (u0 + u1) * 0.5;
        double offsetY = py0 - scale * v0;
        (scale, offsetX, offsetY, double fit) = Refine(silhouette, subject, width, height, scale, offsetX, offsetY, py1 + 1 - py0, cancellationToken);
        double mirroredFit = Overlap(silhouette, subject, width, height, -scale, (px0 + px1 + 1) - offsetX, offsetY, cancellationToken);

        ViewRaster raster = renderer.Rasterize(0, 0, resolution, cancellationToken);
        float[] image = new float[resolution * resolution * 3];
        bool[] covered = new bool[resolution * resolution];
        byte[] colors = rgb.ToArray();
        Parallel.For(0, resolution, new ParallelOptions { CancellationToken = cancellationToken }, row =>
        {
            double y = scale * ((row + 0.5) / resolution) + offsetY - 0.5;
            for (int column = 0; column < resolution; column++)
            {
                int pixel = row * resolution + column;
                if (raster.FaceIds[pixel] == 0)
                    continue;
                double x = scale * ((column + 0.5) / resolution) + offsetX - 0.5;
                int nx = (int)Math.Round(x), ny = (int)Math.Round(y);
                if (nx < 0 || ny < 0 || nx >= width || ny >= height || !inner[ny * width + nx])
                    continue;
                Sample(colors, width, height, x, y, image, pixel * 3);
                covered[pixel] = true;
            }
        });
        return new PaintReferenceProjection(image, covered, resolution, fit, mirroredFit, scale, offsetX, offsetY);
    }

    /// <summary>Puts the picture over a view image: covered pixels take the picture's colour, the rest keep <paramref name="view"/>'s. Both are RGB floats at the projection's resolution.</summary>
    public static void Overlay(PaintReferenceProjection projection, float[] view)
    {
        ArgumentNullException.ThrowIfNull(projection);
        ArgumentNullException.ThrowIfNull(view);
        if (view.Length != projection.Image.Length)
            throw new ArgumentException("The view image does not match the projection's resolution.", nameof(view));
        for (int p = 0; p < projection.Covered.Length; p++)
            if (projection.Covered[p])
                for (int c = 0; c < 3; c++)
                    view[p * 3 + c] = projection.Image[p * 3 + c];
    }

    // Pattern search on the overlap: each step tries the scale and both offsets one step up and down and keeps any gain; when none gains, the steps halve.
    private static (double Scale, double OffsetX, double OffsetY, double Fit) Refine(bool[] silhouette, bool[] subject, int width, int height, double scale, double offsetX,
        double offsetY, int subjectHeight, CancellationToken cancellationToken)
    {
        double baseScale = scale, baseX = offsetX, baseY = offsetY, range = subjectHeight * OffsetRange;
        double best = Overlap(silhouette, subject, width, height, scale, offsetX, offsetY, cancellationToken);
        double scaleStep = baseScale * 0.02, offsetStep = subjectHeight * 0.01;
        for (int round = 0; round < 64 && offsetStep > subjectHeight * 0.0005; round++)
        {
            bool improved = false;
            for (int parameter = 0; parameter < 3; parameter++)
                foreach (int sign in (ReadOnlySpan<int>)[1, -1])
                {
                    double s = scale, x = offsetX, y = offsetY;
                    if (parameter == 0)
                    {
                        // A scale change keeps the subject centred where it is.
                        s = scale + sign * scaleStep;
                        x = offsetX + (scale - s) * 0.5;
                        y = offsetY + (scale - s) * 0.5;
                    }
                    else if (parameter == 1)
                        x = offsetX + sign * offsetStep;
                    else
                        y = offsetY + sign * offsetStep;
                    if (Math.Abs(s - baseScale) > baseScale * ScaleRange || Math.Abs(x - baseX) > range || Math.Abs(y - baseY) > range)
                        continue;
                    double fit = Overlap(silhouette, subject, width, height, s, x, y, cancellationToken);
                    if (fit > best)
                    {
                        (best, scale, offsetX, offsetY, improved) = (fit, s, x, y, true);
                    }
                }
            if (!improved)
            {
                scaleStep *= 0.5;
                offsetStep *= 0.5;
            }
        }
        return (scale, offsetX, offsetY, best);
    }

    // Intersection over union of the silhouette and the placed subject, on the fit grid. A negative scale flips the picture left to right.
    internal static double Overlap(bool[] silhouette, bool[] subject, int width, int height, double scale, double offsetX, double offsetY, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        int both = 0, either = 0;
        double yScale = Math.Abs(scale);
        for (int row = 0; row < FitResolution; row++)
        {
            int y = (int)Math.Floor(yScale * ((row + 0.5) / FitResolution) + offsetY);
            for (int column = 0; column < FitResolution; column++)
            {
                int x = (int)Math.Floor(scale * ((column + 0.5) / FitResolution) + offsetX);
                bool inSubject = x >= 0 && y >= 0 && x < width && y < height && subject[y * width + x];
                bool inSilhouette = silhouette[row * FitResolution + column];
                if (inSubject && inSilhouette)
                    both++;
                if (inSubject || inSilhouette)
                    either++;
            }
        }
        return either == 0 ? 0 : both / (double)either;
    }

    private static bool[] Silhouette(ViewRaster raster)
    {
        bool[] mask = new bool[raster.FaceIds.Length];
        for (int p = 0; p < mask.Length; p++)
            mask[p] = raster.FaceIds[p] > 0;
        return mask;
    }

    internal static bool Bounds(bool[] mask, int width, int height, out int x0, out int y0, out int x1, out int y1)
    {
        x0 = width;
        y0 = height;
        x1 = y1 = -1;
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                if (mask[y * width + x])
                {
                    x0 = Math.Min(x0, x);
                    x1 = Math.Max(x1, x);
                    y0 = Math.Min(y0, y);
                    y1 = Math.Max(y1, y);
                }
        return x1 >= 0;
    }

    // A pixel stays only when every pixel within the square of the given radius is set.
    internal static bool[] Erode(bool[] mask, int width, int height, int radius)
    {
        bool[] inverse = new bool[mask.Length];
        for (int i = 0; i < mask.Length; i++)
            inverse[i] = !mask[i];
        bool[] grown = PaintImageFilters.Dilate(inverse, width, height, radius);
        for (int i = 0; i < grown.Length; i++)
            grown[i] = !grown[i];
        return grown;
    }

    private static void Sample(byte[] rgb, int width, int height, double x, double y, float[] target, int offset)
    {
        int x0 = Math.Clamp((int)Math.Floor(x), 0, width - 1), y0 = Math.Clamp((int)Math.Floor(y), 0, height - 1);
        int x1 = Math.Min(x0 + 1, width - 1), y1 = Math.Min(y0 + 1, height - 1);
        double wx = Math.Clamp(x - x0, 0, 1), wy = Math.Clamp(y - y0, 0, 1);
        for (int c = 0; c < 3; c++)
        {
            double top = rgb[(y0 * width + x0) * 3 + c] * (1 - wx) + rgb[(y0 * width + x1) * 3 + c] * wx;
            double bottom = rgb[(y1 * width + x0) * 3 + c] * (1 - wx) + rgb[(y1 * width + x1) * 3 + c] * wx;
            target[offset + c] = (float)((top * (1 - wy) + bottom * wy) / 255.0);
        }
    }
}
