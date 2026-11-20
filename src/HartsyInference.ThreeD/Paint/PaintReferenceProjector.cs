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
/// <param name="Elevation">Elevation in degrees of the camera the picture was taken from; 0 for the front picture.</param>
/// <param name="Azimuth">Azimuth in degrees of that camera (<see cref="PaintCamera"/>'s convention); 0 for the front picture.</param>
public sealed record PaintReferenceProjection(float[] Image, bool[] Covered, int Resolution, double Fit, double MirroredFit, double Scale, double OffsetX, double OffsetY,
    float[]? Facing = null, double Elevation = 0, double Azimuth = 0);

/// <summary>A picture to bake with <see cref="PaintReferenceProjector.ApplyToTexels(IReadOnlyList{PaintReferencePicture}, PaintViewRenderer, PaintTexelSurface, float[], bool[], double, CancellationToken)"/>:
/// its placement on its own camera and the picture itself, RGB bytes and subject mask at its own size.</summary>
public sealed record PaintReferencePicture(PaintReferenceProjection Placement, byte[] Rgb, byte[] Alpha, int Width, int Height);

/// <summary>Bakes the reference picture onto the surfaces the front camera sees, at the picture's own resolution. Paint repaints every view at 512 px, so
/// on a full figure a face is a few dozen pixels wide and its eyes, nose and brows are lost; the front view is the one the picture was drawn from, so the
/// picture itself stands in for it wherever the mesh and the picture's subject overlap (<see cref="Project"/> places it, <see cref="ApplyToTexels"/> bakes it).
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
        CancellationToken cancellationToken = default) =>
        Project(renderer, rgb, alpha, width, height, resolution, 0, 0, cancellationToken);

    /// <summary>Fits the picture's subject to the view of <paramref name="renderer"/>'s mesh from the camera at <paramref name="elevation"/> and
    /// <paramref name="azimuth"/> (degrees) and lays the picture out at <paramref name="resolution"/>. Returns null when the picture has no subject or the
    /// mesh has no silhouette from that camera.</summary>
    public static PaintReferenceProjection? Project(PaintViewRenderer renderer, ReadOnlySpan<byte> rgb, ReadOnlySpan<byte> alpha, int width, int height, int resolution,
        double elevation, double azimuth, CancellationToken cancellationToken = default)
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

        bool[] silhouette = Silhouette(renderer.Rasterize(elevation, azimuth, FitResolution, cancellationToken));
        if (!Bounds(silhouette, FitResolution, FitResolution, out int sx0, out int sy0, out int sx1, out int sy1)
            || !Bounds(subject, width, height, out int px0, out int py0, out int px1, out int py1))
            return null;

        double u0 = sx0 / (double)FitResolution, u1 = (sx1 + 1) / (double)FitResolution, v0 = sy0 / (double)FitResolution, v1 = (sy1 + 1) / (double)FitResolution;
        double scale = (py1 + 1 - py0) / (v1 - v0);
        double offsetX = (px0 + px1 + 1) * 0.5 - scale * (u0 + u1) * 0.5;
        double offsetY = py0 - scale * v0;
        (scale, offsetX, offsetY, double fit) = Refine(silhouette, subject, width, height, scale, offsetX, offsetY, py1 + 1 - py0, cancellationToken);
        double mirroredFit = Overlap(silhouette, subject, width, height, -scale, (px0 + px1 + 1) - offsetX, offsetY, cancellationToken);

        PaintViewProjection front = renderer.Project(elevation, azimuth);
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
                SampleBilinear(colors, width, height, x, y, image.AsSpan(pixel * 3, 3));
                covered[pixel] = true;
            }
        });
        return new PaintReferenceProjection(image, covered, resolution, fit, mirroredFit, scale, offsetX, offsetY, facing, elevation, azimuth);
    }

    /// <summary>Writes the picture into the albedo texels the front camera sees, at the picture's own resolution: each texel is placed in the picture through the front
    /// camera and <paramref name="placement"/>'s scale and offset, depth-tested against the front view so only the first surface takes it, and sampled once (bicubic).
    /// The picture replaces the merged views on surfaces facing the camera at least <see cref="TexelFacingFull"/> and fades out by facing alone down to <see cref="TexelFacingNone"/>,
    /// so no texel mixes the picture's features with the generated views' differently placed ones; along the subject's outline it needs a squarer facing, so the picture's drawn
    /// outline is not stretched over surfaces turning away. A texel no view painted takes the picture outright.
    /// Returns how many texels took the picture.</summary>
    /// <param name="texture">The merged albedo, three floats per texel in [0, 1], <c>surface.Width * surface.Height</c> texels; changed in place.</param>
    /// <param name="known">Which texels the views painted; texels the picture paints are set.</param>
    /// <param name="rgb">The picture, RGB bytes, row 0 first, at its own size (<paramref name="width"/> by <paramref name="height"/>).</param>
    /// <param name="alpha">The picture's subject mask, one byte per pixel.</param>
    public static int ApplyToTexels(PaintReferenceProjection placement, PaintViewRenderer renderer, PaintTexelSurface surface, float[] texture, bool[] known,
        byte[] rgb, byte[] alpha, int width, int height, double depthThreshold, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(placement);
        return ApplyToTexels([new PaintReferencePicture(placement, rgb, alpha, width, height)], renderer, surface, texture, known, depthThreshold, cancellationToken);
    }

    /// <summary>Bakes several pictures of the subject, each through its own camera, as <see cref="ApplyToTexels(PaintReferenceProjection, PaintViewRenderer,
    /// PaintTexelSurface, float[], bool[], byte[], byte[], int, int, double, CancellationToken)"/> bakes one. Where several pictures see a texel they are
    /// laid over each other in rank, each taking its facing weight of what the pictures above it left uncovered, and the stack replaces the views by the
    /// share it covers. The first picture (the caller's own, before any drawn side) leads wherever it faces its camera at least <see cref="FirstPictureLeads"/>;
    /// on surfaces it sees more obliquely the pictures go by facing, squarest first, with a short blend between the two orders over
    /// <see cref="FirstPictureLeadBand"/>. Blending two pictures where both are at full weight laid their differently drawn features over each other (a cheek
    /// at 45 degrees took half the front picture's face and half a side drawing's); letting the front lead everywhere it reached put the front picture's fur
    /// collar on a head's side, cut off where the collar ends, where a side picture sees the head squarely. Returns how many texels took a picture.</summary>
    public static int ApplyToTexels(IReadOnlyList<PaintReferencePicture> pictures, PaintViewRenderer renderer, PaintTexelSurface surface, float[] texture, bool[] known,
        double depthThreshold, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(pictures);
        ArgumentNullException.ThrowIfNull(renderer);
        ArgumentNullException.ThrowIfNull(surface);
        if (texture.Length != surface.Width * surface.Height * 3 || known.Length != surface.Width * surface.Height)
            throw new ArgumentException("The texture does not match the surface.");
        foreach (PaintReferencePicture picture in pictures)
        {
            ArgumentNullException.ThrowIfNull(picture);
            ArgumentNullException.ThrowIfNull(picture.Placement);
            if (picture.Rgb.Length != picture.Width * picture.Height * 3 || picture.Alpha.Length != picture.Width * picture.Height)
                throw new ArgumentException("The picture's colour and alpha planes do not match its size.");
        }

        // The depth buffer at least as fine as the texture, so a texel's visibility is decided at its own scale.
        int side = Math.Max(2048, Math.Max(surface.Width, surface.Height));
        PictureCamera[] cameras = new PictureCamera[pictures.Count];
        for (int k = 0; k < cameras.Length; k++)
            cameras[k] = PictureCamera.Create(pictures[k], renderer, side, cancellationToken);

        int written = 0;
        Parallel.For(0, surface.Count, new ParallelOptions { CancellationToken = cancellationToken }, i =>
        {
            int n = cameras.Length;
            Span<float> colours = stackalloc float[n * 3], weights = stackalloc float[n], facings = stackalloc float[n], colour = stackalloc float[3],
                other = stackalloc float[3];
            Span<int> order = stackalloc int[n];
            int contributors = 0;
            float firstFacing = -1f;
            for (int k = 0; k < n; k++)
            {
                if (!cameras[k].Sample(surface.Positions, i, depthThreshold, colours.Slice(contributors * 3, 3), out float w, out float facing))
                    continue;
                weights[contributors] = w;
                facings[contributors] = facing;
                if (k == 0)
                    firstFacing = facing;
                contributors++;
            }
            if (contributors == 0)
                return;
            float weight;
            if (contributors == 1)
            {
                // One picture keeps its own colour exactly, at its own weight.
                colours[..3].CopyTo(colour);
                weight = weights[0];
            }
            else
            {
                // How far the first picture leads: fully where it faces its camera squarely enough, not at all below the band.
                float lead = firstFacing < 0f ? 0f : Math.Clamp((firstFacing - (FirstPictureLeads - FirstPictureLeadBand)) / FirstPictureLeadBand, 0f, 1f);
                weight = Stack(colours, weights, facings, order, contributors, lead > 0f, colour);
                if (lead < 1f)
                {
                    float byFacing = Stack(colours, weights, facings, order, contributors, false, other);
                    for (int c = 0; c < 3; c++)
                        colour[c] = lead * colour[c] + (1f - lead) * other[c];
                    weight = lead * weight + (1f - lead) * byFacing;
                }
            }
            int texel = surface.Texels[i];
            if (!known[texel])
                weight = 1f;
            for (int c = 0; c < 3; c++)
                texture[texel * 3 + c] += weight * (colour[c] - texture[texel * 3 + c]);
            known[texel] = true;
            Interlocked.Increment(ref written);
        });
        return written;
    }

    /// <summary>In the multi-picture <see cref="ApplyToTexels(IReadOnlyList{PaintReferencePicture}, PaintViewRenderer, PaintTexelSurface, float[], bool[], double, CancellationToken)"/>,
    /// the first picture leads on surfaces facing its camera at least this squarely (smoothed cosine; about 57 degrees off its axis), which keeps a face's cheeks
    /// on the front picture.</summary>
    public const float FirstPictureLeads = 0.55f;

    /// <summary>The facing band below <see cref="FirstPictureLeads"/> over which the first picture's lead fades into the order by facing.</summary>
    public const float FirstPictureLeadBand = 0.1f;

    // Lays the sampled pictures over each other, each taking its weight of what the ones above left uncovered: the first sampled picture first when
    // firstLeads (it is the caller's first picture whenever that one was sampled), the rest by facing, squarest first. Writes the colour and returns the
    // share covered.
    private static float Stack(Span<float> colours, Span<float> weights, Span<float> facings, Span<int> order, int count, bool firstLeads, Span<float> colour)
    {
        for (int r = 0; r < count; r++)
            order[r] = r;
        int start = firstLeads ? 1 : 0;
        for (int a = start + 1; a < count; a++)
            for (int b = a; b > start && facings[order[b]] > facings[order[b - 1]]; b--)
                (order[b], order[b - 1]) = (order[b - 1], order[b]);
        float uncovered = 1f;
        colour.Clear();
        for (int r = 0; r < count; r++)
        {
            int s = order[r];
            float share = weights[s] * uncovered;
            for (int c = 0; c < 3; c++)
                colour[c] += share * colours[s * 3 + c];
            uncovered *= 1f - weights[s];
        }
        float covered = 1f - uncovered;
        if (covered > 0f)
            for (int c = 0; c < 3; c++)
                colour[c] /= covered;
        else
            colours.Slice(order[0] * 3, 3).CopyTo(colour);
        return covered;
    }

    // One picture's camera for the texel bake: the depth buffer and facing from that camera, the picture's subject masks and its placement.
    private sealed class PictureCamera
    {
        private readonly PaintReferencePicture _picture;
        private readonly ViewRaster _raster;
        private readonly float[] _depth, _faceFacing;
        private readonly bool[] _inner, _deep;
        private readonly PaintOccluderAlignment? _alignment;
        private readonly double[] _worldToCamera;
        private readonly int _side;

        private PictureCamera(PaintReferencePicture picture, ViewRaster raster, float[] depth, float[] faceFacing, bool[] inner, bool[] deep,
            PaintOccluderAlignment? alignment, double[] worldToCamera, int side)
        {
            _picture = picture;
            _raster = raster;
            _depth = depth;
            _faceFacing = faceFacing;
            _inner = inner;
            _deep = deep;
            _alignment = alignment;
            _worldToCamera = worldToCamera;
            _side = side;
        }

        public static PictureCamera Create(PaintReferencePicture picture, PaintViewRenderer renderer, int side, CancellationToken cancellationToken)
        {
            PaintReferenceProjection placement = picture.Placement;
            PaintViewProjection view = renderer.Project(placement.Elevation, placement.Azimuth);
            int[] indices = renderer.Frame.Indices;
            ViewRaster raster = ViewRasterizer.Rasterize(view.Clip, indices, side, side, cancellationToken);
            float[] z = new float[view.Camera.Length / 3];
            for (int i = 0; i < z.Length; i++)
                z[i] = view.Camera[i * 3 + 2];
            float[] depth = raster.Interpolate(z, indices, 1);
            float[] faceFacing = SmoothFacing(view.Camera, indices, FacingSmoothingPasses);

            int width = picture.Width, height = picture.Height;
            bool[] subject = new bool[width * height];
            for (int i = 0; i < subject.Length; i++)
                subject[i] = picture.Alpha[i] >= SubjectAlpha;
            bool[] inner = Erode(subject, width, height, Math.Max(1, Math.Max(width, height) / 512));
            // A drawn picture outlines its subject with a dark line a few pixels wide. Where the surface turns away near the silhouette (the sides of a head), that line would be
            // stretched across it, so near the outline only surfaces facing the camera squarely take the picture.
            bool[] deep = Erode(subject, width, height, Math.Max(2, Math.Max(width, height) / OutlineShare));
            PaintOccluderAlignment? alignment = PaintOccluderAlignment.Create(depth, raster.FaceIds, side, picture.Rgb, subject, width, height, placement,
                cancellationToken);
            return new PictureCamera(picture, raster, depth, faceFacing, inner, deep, alignment, view.WorldToCamera, side);
        }

        // The picture's colour at surface point i, its facing weight (0 to 1, by facing alone) and the smoothed facing; false where the picture does not reach.
        public bool Sample(float[] positions, int i, double depthThreshold, Span<float> colour, out float weight, out float facing)
        {
            weight = 0f;
            facing = 0f;
            int side = _side;
            double toView = 1.0 / (PaintCamera.OrthoScale * 0.5);
            (double cx, double cy, double cz, _) = PaintCamera.Transform(_worldToCamera, positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
            double u = cx * toView * 0.5 + 0.5, v = cy * toView * 0.5 + 0.5;
            if (u < 0 || u >= 1 || v < 0 || v >= 1)
                return false;
            int pixel = (int)(v * side) * side + (int)(u * side);
            int face = _raster.FaceIds[pixel];
            if (face == 0 || !VisibleNear(_raster.FaceIds, _depth, side, pixel, cz, depthThreshold))
                return false;
            facing = _faceFacing[face - 1];
            weight = Math.Clamp((facing - TexelFacingNone) / (TexelFacingFull - TexelFacingNone), 0f, 1f);
            if (weight <= 0f)
                return false;
            // The same mapping Project fitted: picture pixel = Scale * view coordinate + Offset, at pixel centres.
            // A nearer part the mesh put off the drawing samples it shifted onto its own edge.
            PaintReferenceProjection placement = _picture.Placement;
            int width = _picture.Width, height = _picture.Height;
            (double du, double dv) = _alignment is null ? (0, 0) : _alignment.Shift(u, v, cz);
            double x = placement.Scale * (u + du) + placement.OffsetX - 0.5, y = placement.Scale * (v + dv) + placement.OffsetY - 0.5;
            int nx = (int)Math.Round(x), ny = (int)Math.Round(y);
            if (nx < 0 || ny < 0 || nx >= width || ny >= height || !_inner[ny * width + nx]
                || (!_deep[ny * width + nx] && facing < OutlineFacing))
                return false;
            SampleBicubic(_picture.Rgb, width, height, x, y, colour);
            return true;
        }
    }

    // Passes of neighbour averaging on the vertex facing in ApplyToTexels.
    private const int FacingSmoothingPasses = 3;

    /// <summary>In <see cref="ApplyToTexels"/>, the picture fully replaces the views on surfaces facing the front camera at least this squarely (smoothed cosine; about 63 degrees off
    /// the view axis, where the picture is stretched about twice).</summary>
    public const float TexelFacingFull = 0.45f;

    /// <summary>In <see cref="ApplyToTexels"/>, surfaces turned further than this (about 78 degrees) keep the views: the picture would be a streak there.</summary>
    public const float TexelFacingNone = 0.2f;

    // The band along the subject's outline, as a share of the picture's longer side (about 11 px on a 1400 px picture), where only surfaces facing the camera at least
    // OutlineFacing (about 45 degrees) take the picture.
    private const int OutlineShare = 128;
    private const float OutlineFacing = 0.7f;

    // A texel is visible when its depth matches the front depth buffer at its pixel or at one of the eight around it: on a fold the nearest pixel can belong to the other side of the
    // edge while the texel itself is in plain view.
    private static bool VisibleNear(int[] faceIds, float[] depth, int side, int pixel, double z, double threshold)
    {
        int px = pixel % side, py = pixel / side;
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
            {
                int x = px + dx, y = py + dy;
                if (x < 0 || y < 0 || x >= side || y >= side)
                    continue;
                int p = y * side + x;
                if (faceIds[p] > 0 && Math.Abs(z - depth[p]) < threshold)
                    return true;
            }
        return false;
    }

    // How squarely each triangle faces the camera, from vertex normals (area-weighted face normals) averaged with their neighbours over a few passes: the shape models' meshes are
    // bumpy at the scale of a few triangles, and a per-triangle facing turns small creases (an eyelid, a brow line) into slivers that drop out of the projection.
    private static float[] SmoothFacing(float[] camera, int[] indices, int passes)
    {
        int vertices = camera.Length / 3, faces = indices.Length / 3;
        float[] nx = new float[vertices], ny = new float[vertices], nz = new float[vertices];
        for (int t = 0; t < faces; t++)
        {
            int a = indices[t * 3], b = indices[t * 3 + 1], c = indices[t * 3 + 2];
            float e1x = camera[b * 3] - camera[a * 3], e1y = camera[b * 3 + 1] - camera[a * 3 + 1], e1z = camera[b * 3 + 2] - camera[a * 3 + 2];
            float e2x = camera[c * 3] - camera[a * 3], e2y = camera[c * 3 + 1] - camera[a * 3 + 1], e2z = camera[c * 3 + 2] - camera[a * 3 + 2];
            float fx = e1y * e2z - e1z * e2y, fy = e1z * e2x - e1x * e2z, fz = e1x * e2y - e1y * e2x;
            // A drawn triangle faces the camera whatever its winding, so every face normal is turned toward it before summing.
            if (fz < 0)
                (fx, fy, fz) = (-fx, -fy, -fz);
            foreach (int v in (ReadOnlySpan<int>)[a, b, c])
            {
                nx[v] += fx;
                ny[v] += fy;
                nz[v] += fz;
            }
        }
        float[] facing = new float[vertices];
        for (int v = 0; v < vertices; v++)
        {
            float length = MathF.Sqrt(nx[v] * nx[v] + ny[v] * ny[v] + nz[v] * nz[v]);
            facing[v] = length > 1e-12f ? MathF.Abs(nz[v]) / length : 0f;
        }
        float[] sum = new float[vertices];
        int[] count = new int[vertices];
        for (int pass = 0; pass < passes; pass++)
        {
            Array.Clear(sum);
            Array.Clear(count);
            for (int t = 0; t < faces; t++)
            {
                int a = indices[t * 3], b = indices[t * 3 + 1], c = indices[t * 3 + 2];
                float mean = (facing[a] + facing[b] + facing[c]) / 3f;
                foreach (int v in (ReadOnlySpan<int>)[a, b, c])
                {
                    sum[v] += mean;
                    count[v]++;
                }
            }
            for (int v = 0; v < vertices; v++)
                if (count[v] > 0)
                    facing[v] = sum[v] / count[v];
        }
        float[] faceFacing = new float[faces];
        for (int t = 0; t < faces; t++)
            faceFacing[t] = (facing[indices[t * 3]] + facing[indices[t * 3 + 1]] + facing[indices[t * 3 + 2]]) / 3f;
        return faceFacing;
    }

    // Catmull-Rom bicubic sample of an RGB byte image at a continuous pixel position (integers are pixel centres), edges clamped; colours in [0, 1].
    private static void SampleBicubic(byte[] rgb, int width, int height, double x, double y, Span<float> target)
    {
        int x0 = (int)Math.Floor(x), y0 = (int)Math.Floor(y);
        double fx = x - x0, fy = y - y0;
        Span<double> wx = stackalloc double[4], wy = stackalloc double[4];
        CatmullRom(fx, wx);
        CatmullRom(fy, wy);
        for (int c = 0; c < 3; c++)
        {
            double sum = 0;
            for (int j = 0; j < 4; j++)
            {
                int row = Math.Clamp(y0 - 1 + j, 0, height - 1) * width;
                double line = 0;
                for (int k = 0; k < 4; k++)
                    line += wx[k] * rgb[(row + Math.Clamp(x0 - 1 + k, 0, width - 1)) * 3 + c];
                sum += wy[j] * line;
            }
            target[c] = (float)Math.Clamp(sum / 255.0, 0, 1);
        }
    }

    private static void CatmullRom(double t, Span<double> w)
    {
        double t2 = t * t, t3 = t2 * t;
        w[0] = 0.5 * (-t3 + 2 * t2 - t);
        w[1] = 0.5 * (3 * t3 - 5 * t2 + 2);
        w[2] = 0.5 * (-3 * t3 + 4 * t2 + t);
        w[3] = 0.5 * (t3 - t2);
    }

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

    /// <summary>Bilinear sample of an RGB byte image at a pixel position (integers are pixel centres), edges clamped, in [0, 1].</summary>
    internal static void SampleBilinear(byte[] rgb, int width, int height, double x, double y, Span<float> target)
    {
        int x0 = Math.Clamp((int)Math.Floor(x), 0, width - 1), y0 = Math.Clamp((int)Math.Floor(y), 0, height - 1);
        int x1 = Math.Min(x0 + 1, width - 1), y1 = Math.Min(y0 + 1, height - 1);
        double wx = Math.Clamp(x - x0, 0, 1), wy = Math.Clamp(y - y0, 0, 1);
        for (int c = 0; c < 3; c++)
        {
            double top = rgb[(y0 * width + x0) * 3 + c] * (1 - wx) + rgb[(y0 * width + x1) * 3 + c] * wx;
            double bottom = rgb[(y1 * width + x0) * 3 + c] * (1 - wx) + rgb[(y1 * width + x1) * 3 + c] * wx;
            target[c] = (float)((top * (1 - wy) + bottom * wy) / 255.0);
        }
    }
}
