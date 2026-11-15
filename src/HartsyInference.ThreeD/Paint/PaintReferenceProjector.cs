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
/// <param name="Facing">Per pixel, how squarely the surface there faces the front camera: the cosine between its face normal and the view direction (0 to 1). Null treats every pixel as facing it.</param>
public sealed record PaintReferenceProjection(float[] Image, bool[] Covered, int Resolution, double Fit, double MirroredFit, double Scale, double OffsetX, double OffsetY, float[]? Facing = null);

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

        PaintViewProjection front = renderer.Project(0, 0);
        ViewRaster raster = ViewRasterizer.Rasterize(front.Clip, renderer.Frame.Indices, resolution, resolution, cancellationToken);
        float[] faceFacing = FaceFacing(front.Camera, renderer.Frame.Indices);
        float[] facing = new float[resolution * resolution];
        for (int p = 0; p < facing.Length; p++)
            if (raster.FaceIds[p] > 0)
                facing[p] = faceFacing[raster.FaceIds[p] - 1];
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
        return new PaintReferenceProjection(image, covered, resolution, fit, mirroredFit, scale, offsetX, offsetY, facing);
    }

    /// <summary>The picture is used fully on surfaces facing the front camera at least this squarely (cosine; about 50 degrees off the view direction).</summary>
    public const float FacingFull = 0.65f;

    /// <summary>Surfaces turned further from the front camera than this (cosine; about 70 degrees) do not take the picture: the front picture has almost no pixels for them, so it
    /// would be stretched into streaks (a brim's underside, the sides of a face recessed under it). The painted views cover them. In between, the two are blended.</summary>
    public const float FacingNone = 0.35f;

    // How squarely each triangle faces the camera: |cos| of the angle between its normal and the view axis (camera space looks down -z). A triangle that is drawn faces the camera,
    // whatever its winding, so the sign is dropped.
    private static float[] FaceFacing(float[] camera, int[] indices)
    {
        float[] facing = new float[indices.Length / 3];
        for (int t = 0; t < facing.Length; t++)
        {
            int a = indices[t * 3] * 3, b = indices[t * 3 + 1] * 3, c = indices[t * 3 + 2] * 3;
            float e1x = camera[b] - camera[a], e1y = camera[b + 1] - camera[a + 1], e1z = camera[b + 2] - camera[a + 2];
            float e2x = camera[c] - camera[a], e2y = camera[c + 1] - camera[a + 1], e2z = camera[c + 2] - camera[a + 2];
            float nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
            float length = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
            facing[t] = length > 1e-12f ? MathF.Abs(nz) / length : 0f;
        }
        return facing;
    }

    /// <summary>Blurred colours farther apart than this (RGB distance, channels in [0, 1]) after alignment are different surfaces (skin where the mesh has a hat brim): the generated view is kept there.</summary>
    public const float DisagreeDistance = 0.5f;

    /// <summary>Largest local shift, in pixels at 512, that <see cref="Overlay"/> applies to line the picture's features up with the generated view's.</summary>
    public const int MaxShift = 10;

    /// <summary>Puts the picture over the generated front view (RGB floats at the projection's resolution). The silhouette fit places the picture as a whole; inside, the mesh follows it only
    /// roughly (a hat brim the mesh brings lower than drawn covers the eyes), while the generated view was painted on the mesh's own geometry. So the picture is first moved locally onto the
    /// generated view: a displacement of up to <see cref="MaxShift"/> px at 512 per 8 px block, found by matching blurred patches and smoothed, pulls its features to where the mesh has them.
    /// Its small details stay its own (the generated view's are the 512 px repaint the picture is here to replace). Where the aligned picture still shows a different surface than the
    /// generated view (blurred colours further apart than <see cref="DisagreeDistance"/>, widened by 2 px at 512 and softened by 1 px), the generated view is kept.
    /// Returns how many pixels took the picture fully.</summary>
    public static int Overlay(PaintReferenceProjection projection, float[] view)
    {
        ArgumentNullException.ThrowIfNull(projection);
        ArgumentNullException.ThrowIfNull(view);
        if (view.Length != projection.Image.Length)
            throw new ArgumentException("The view image does not match the projection's resolution.", nameof(view));
        int size = projection.Resolution;
        (float[] image, bool[] covered) = Align(projection, view);
        int radius = Math.Max(1, size / 128);
        float[] picture = BoxBlur(image, size, radius);
        float[] generated = BoxBlur(view, size, radius);
        bool[] rejected = new bool[size * size];
        for (int p = 0; p < covered.Length; p++)
        {
            if (!covered[p])
                continue;
            float sum = 0;
            for (int c = 0; c < 3; c++)
            {
                float d = picture[p * 3 + c] - generated[p * 3 + c];
                sum += d * d;
            }
            rejected[p] = MathF.Sqrt(sum) > DisagreeDistance;
        }
        rejected = PaintImageFilters.Dilate(rejected, size, size, Math.Max(1, size / 256));
        float[] keep = new float[size * size * 3];
        for (int p = 0; p < rejected.Length; p++)
            keep[p * 3] = keep[p * 3 + 1] = keep[p * 3 + 2] = rejected[p] ? 0f : 1f;
        keep = BoxBlur(keep, size, Math.Max(1, size / 512));
        int full = 0;
        for (int p = 0; p < covered.Length; p++)
        {
            if (!covered[p])
                continue;
            float weight = keep[p * 3];
            if (projection.Facing is { } facing)
                weight *= Math.Clamp((facing[p] - FacingNone) / (FacingFull - FacingNone), 0f, 1f);
            if (weight >= 1f)
                full++;
            for (int c = 0; c < 3; c++)
                view[p * 3 + c] += weight * (image[p * 3 + c] - view[p * 3 + c]);
        }
        return full;
    }

    // Side of the grid the alignment matches on, the block step, the patch radius (all in pixels of that grid), and the cost of a shift: matching is on colours in [0, 1], so a block
    // moved MaxShift px must explain about 0.06 of mean colour difference more than staying put.
    private const int AlignResolution = 512, AlignStep = 8, AlignPatch = 7;
    private const double ShiftCost = 0.06 * 0.06 / (MaxShift * MaxShift);

    // The picture moved locally onto the generated view, with its coverage moved alike. See Overlay.
    internal static (float[] Image, bool[] Covered) Align(PaintReferenceProjection projection, float[] view)
    {
        int size = projection.Resolution, factor = Math.Max(1, size / AlignResolution), small = size / factor;
        // Outside the picture's coverage the generated view stands in, so a patch at the silhouette compares like with like.
        float[] filled = (float[])view.Clone();
        for (int p = 0; p < projection.Covered.Length; p++)
            if (projection.Covered[p])
                for (int c = 0; c < 3; c++)
                    filled[p * 3 + c] = projection.Image[p * 3 + c];
        float[] picture = BoxBlur(Downsample(filled, size, factor), small, 1);
        float[] generated = BoxBlur(Downsample(view, size, factor), small, 1);
        bool[] coveredSmall = new bool[small * small];
        for (int y = 0; y < small; y++)
            for (int x = 0; x < small; x++)
                coveredSmall[y * small + x] = projection.Covered[(y * factor + factor / 2) * size + x * factor + factor / 2];

        int blocks = (small + AlignStep - 1) / AlignStep;
        float[] shiftX = new float[blocks * blocks], shiftY = new float[blocks * blocks];
        Parallel.For(0, blocks * blocks, block =>
        {
            int cx = block % blocks * AlignStep + AlignStep / 2, cy = block / blocks * AlignStep + AlignStep / 2;
            if (cx >= small || cy >= small || !coveredSmall[cy * small + cx])
                return;
            double best = double.MaxValue;
            int bestX = 0, bestY = 0;
            for (int dy = -MaxShift; dy <= MaxShift; dy++)
                for (int dx = -MaxShift; dx <= MaxShift; dx++)
                {
                    double cost = PatchCost(generated, picture, small, cx, cy, dx, dy);
                    if (double.IsNaN(cost))
                        continue;
                    cost += ShiftCost * (dx * dx + dy * dy);
                    if (cost < best)
                        (best, bestX, bestY) = (cost, dx, dy);
                }
            shiftX[block] = bestX;
            shiftY[block] = bestY;
        });
        shiftX = Median3(shiftX, blocks);
        shiftY = Median3(shiftY, blocks);

        float[] image = new float[projection.Image.Length];
        bool[] covered = new bool[projection.Covered.Length];
        Parallel.For(0, size, row =>
        {
            for (int column = 0; column < size; column++)
            {
                int p = row * size + column;
                if (!projection.Covered[p])
                    continue;
                // The block grid's value at this pixel, bilinear between block centres, scaled from the alignment grid to this one.
                double gx = Math.Clamp(((column + 0.5) / factor - AlignStep / 2.0) / AlignStep, 0, blocks - 1);
                double gy = Math.Clamp(((row + 0.5) / factor - AlignStep / 2.0) / AlignStep, 0, blocks - 1);
                double sx = Bilinear(shiftX, blocks, gx, gy) * factor, sy = Bilinear(shiftY, blocks, gx, gy) * factor;
                double x = column + sx, y = row + sy;
                int nx = (int)Math.Round(x), ny = (int)Math.Round(y);
                if (nx < 0 || ny < 0 || nx >= size || ny >= size || !projection.Covered[ny * size + nx])
                    continue;
                for (int c = 0; c < 3; c++)
                    image[p * 3 + c] = (float)BilinearChannel(projection.Image, size, x, y, c);
                covered[p] = true;
            }
        });
        return (image, covered);
    }

    // Mean squared colour difference between the generated patch at (cx, cy) and the picture patch shifted by (dx, dy); NaN when the shifted patch leaves the grid.
    private static double PatchCost(float[] generated, float[] picture, int size, int cx, int cy, int dx, int dy)
    {
        if (cx - AlignPatch + dx < 0 || cy - AlignPatch + dy < 0 || cx + AlignPatch + dx >= size || cy + AlignPatch + dy >= size
            || cx - AlignPatch < 0 || cy - AlignPatch < 0 || cx + AlignPatch >= size || cy + AlignPatch >= size)
            return double.NaN;
        double sum = 0;
        for (int y = -AlignPatch; y <= AlignPatch; y++)
        {
            int g = ((cy + y) * size + cx - AlignPatch) * 3, q = ((cy + y + dy) * size + cx - AlignPatch + dx) * 3;
            for (int k = 0; k < (2 * AlignPatch + 1) * 3; k++)
            {
                double d = generated[g + k] - picture[q + k];
                sum += d * d;
            }
        }
        return sum / ((2 * AlignPatch + 1) * (2 * AlignPatch + 1) * 3);
    }

    private static float[] Downsample(float[] image, int size, int factor)
    {
        int small = size / factor;
        float[] result = new float[small * small * 3];
        for (int y = 0; y < small; y++)
            for (int x = 0; x < small; x++)
                for (int c = 0; c < 3; c++)
                {
                    double sum = 0;
                    for (int j = 0; j < factor; j++)
                        for (int i = 0; i < factor; i++)
                            sum += image[((y * factor + j) * size + x * factor + i) * 3 + c];
                    result[(y * small + x) * 3 + c] = (float)(sum / (factor * factor));
                }
        return result;
    }

    private static float[] Median3(float[] grid, int side)
    {
        float[] result = new float[grid.Length];
        float[] window = new float[9];
        for (int y = 0; y < side; y++)
            for (int x = 0; x < side; x++)
            {
                int n = 0;
                for (int j = -1; j <= 1; j++)
                    for (int i = -1; i <= 1; i++)
                        window[n++] = grid[Math.Clamp(y + j, 0, side - 1) * side + Math.Clamp(x + i, 0, side - 1)];
                Array.Sort(window);
                result[y * side + x] = window[4];
            }
        return result;
    }

    private static double Bilinear(float[] grid, int side, double x, double y)
    {
        int x0 = (int)Math.Floor(x), y0 = (int)Math.Floor(y), x1 = Math.Min(x0 + 1, side - 1), y1 = Math.Min(y0 + 1, side - 1);
        double wx = x - x0, wy = y - y0;
        return (grid[y0 * side + x0] * (1 - wx) + grid[y0 * side + x1] * wx) * (1 - wy) + (grid[y1 * side + x0] * (1 - wx) + grid[y1 * side + x1] * wx) * wy;
    }

    private static double BilinearChannel(float[] image, int size, double x, double y, int channel)
    {
        int x0 = Math.Clamp((int)Math.Floor(x), 0, size - 1), y0 = Math.Clamp((int)Math.Floor(y), 0, size - 1);
        int x1 = Math.Min(x0 + 1, size - 1), y1 = Math.Min(y0 + 1, size - 1);
        double wx = Math.Clamp(x - x0, 0, 1), wy = Math.Clamp(y - y0, 0, 1);
        return (image[(y0 * size + x0) * 3 + channel] * (1 - wx) + image[(y0 * size + x1) * 3 + channel] * wx) * (1 - wy)
            + (image[(y1 * size + x0) * 3 + channel] * (1 - wx) + image[(y1 * size + x1) * 3 + channel] * wx) * wy;
    }

    // Box blur of a square RGB float image, separable, edges clamped.
    internal static float[] BoxBlur(float[] image, int size, int radius)
    {
        float[] across = new float[image.Length], result = new float[image.Length];
        Parallel.For(0, size, row =>
        {
            for (int c = 0; c < 3; c++)
            {
                double sum = 0;
                for (int k = -radius; k <= radius; k++)
                    sum += image[(row * size + Math.Clamp(k, 0, size - 1)) * 3 + c];
                for (int x = 0; x < size; x++)
                {
                    across[(row * size + x) * 3 + c] = (float)(sum / (2 * radius + 1));
                    sum += image[(row * size + Math.Min(x + radius + 1, size - 1)) * 3 + c] - image[(row * size + Math.Max(x - radius, 0)) * 3 + c];
                }
            }
        });
        Parallel.For(0, size, column =>
        {
            for (int c = 0; c < 3; c++)
            {
                double sum = 0;
                for (int k = -radius; k <= radius; k++)
                    sum += across[(Math.Clamp(k, 0, size - 1) * size + column) * 3 + c];
                for (int y = 0; y < size; y++)
                {
                    result[(y * size + column) * 3 + c] = (float)(sum / (2 * radius + 1));
                    sum += across[(Math.Min(y + radius + 1, size - 1) * size + column) * 3 + c] - across[(Math.Max(y - radius, 0) * size + column) * 3 + c];
                }
            }
        });
        return result;
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
