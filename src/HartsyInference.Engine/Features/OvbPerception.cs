using HartsyInference.Core.Imaging;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Vision.OpenVisionBanana;

namespace HartsyInference.Engine.Features;

/// <summary>A semantic segmentation by Open Vision Banana: one label per source pixel (an index into <see cref="Classes"/>, or
/// <see cref="OvbDecoder.Ignore"/> for background), with the share of legend draws that agreed on it.</summary>
public sealed record OvbSegmentation(IReadOnlyList<string> Classes, byte[] Labels, float[] Agreement, int Width, int Height,
    ImageData Painted);

/// <summary>A relative inverse-depth map in <c>[0, 1]</c>, closer is larger, one value per source pixel, row-major.</summary>
public sealed record OvbDepth(float[] InverseDepth, int Width, int Height, ImageData Painted);

/// <summary>Camera-space unit normals <c>[height, width, 3]</c> (x right, y up, z toward the camera) with a validity mask.</summary>
public sealed record OvbNormals(float[] Normals, bool[] Valid, int Width, int Height, ImageData Painted);

/// <summary>Open Vision Banana as a perception model: builds the task prompt, runs the engine's image path with the photo as the
/// reference, and decodes the painted answer at the photo's own size. It adds no model code: the FLUX.2 reference-edit recipe
/// does the generation, and <see cref="OvbDecoder"/> reads the picture. The generate delegate is the engine's image service,
/// so a caller can substitute any source of pictures (the tests do).</summary>
public static class OvbPerception
{
    /// <summary>The release's sampling seed, so a run is repeatable.</summary>
    public const int Seed = 0;

    /// <summary>Whether <paramref name="spec"/> is one of the Open Vision Banana catalog entries.</summary>
    public static bool Handles(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        return spec.Catalog?.Id is "open-vision-banana-4b" or "open-vision-banana-9b";
    }

    /// <summary>Semantic segmentation of <paramref name="image"/> into <paramref name="classes"/>. <paramref name="legends"/> above
    /// one runs that many generations under different colour legends and votes per pixel (legend ensembling, which the paper
    /// reports as the largest test-time gain); each costs a full generation.</summary>
    public static async Task<OvbSegmentation> SegmentAsync(InferenceEngine engine, ModelSpec spec, ImageData image,
        IReadOnlyList<string> classes, int legends, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(engine);
        RequireHandled(spec);
        return await SegmentAsync(Generator(engine, spec), image, classes, legends, Seed, OvbSemsegStyle.Quickstart, cancel).ConfigureAwait(false);
    }

    /// <summary>Segmentation through any picture source (<paramref name="generate"/>).</summary>
    public static async Task<OvbSegmentation> SegmentAsync(Func<ImageRequest, CancellationToken, Task<ImageResult>> generate,
        ImageData image, IReadOnlyList<string> classes, int legends, int seed, OvbSemsegStyle style, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(generate);
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(classes);
        if (classes.Count is < 1 or >= OvbDecoder.Ignore)
            throw new ArgumentOutOfRangeException(nameof(classes), classes.Count, "Name between 1 and 254 classes.");
        if (legends is < 1 or > 32)
            throw new ArgumentOutOfRangeException(nameof(legends), legends, "Legend draws are 1 to 32.");
        List<byte[]> maps = new(legends);
        ImageData? first = null;
        for (int draw = 0; draw < legends; draw++)
        {
            cancel.ThrowIfCancellationRequested();
            IReadOnlyList<OvbLegendEntry> legend = OvbLegend.Build(classes, unchecked(seed * 7919 + draw));
            ImageResult result = await generate(Request(OvbPrompts.SemanticSegmentation(legend, style), image, seed + draw), cancel).ConfigureAwait(false);
            // The README resizes a mask back with NEAREST, so painted colours stay exact for the snap.
            ImageData painted = NearestTo(result, image.Width, image.Height);
            first ??= painted;
            (byte[] labels, _) = OvbDecoder.SnapToLegend(painted.Rgb, painted.Width, painted.Height,
                legend.Select(e => (e.R, e.G, e.B)).ToArray());
            maps.Add(labels);
        }
        (byte[] voted, float[] agreement) = OvbDecoder.MajorityVote(maps, classes.Count);
        return new OvbSegmentation(classes, voted, agreement, image.Width, image.Height, first!);
    }

    /// <summary>The region of <paramref name="image"/> that matches <paramref name="description"/>, as a 0/255 mask the size of
    /// the photo (referring segmentation: the model paints the match red on black).</summary>
    public static async Task<(byte[] Mask, ImageData Painted)> ReferAsync(Func<ImageRequest, CancellationToken, Task<ImageResult>> generate,
        ImageData image, string description, int seed, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(generate);
        ArgumentNullException.ThrowIfNull(image);
        ImageResult result = await generate(Request(OvbPrompts.ReferringSegmentation(description), image, seed), cancel).ConfigureAwait(false);
        ImageData painted = NearestTo(result, image.Width, image.Height);
        (byte R, byte G, byte B) red = OvbPrompts.ReferringColor;
        (byte[] labels, _) = OvbDecoder.SnapToLegend(painted.Rgb, painted.Width, painted.Height, [red]);
        byte[] mask = new byte[labels.Length];
        for (int i = 0; i < mask.Length; i++) mask[i] = labels[i] == 0 ? (byte)255 : (byte)0;
        return (mask, painted);
    }

    /// <summary>Relative inverse depth of <paramref name="image"/>.</summary>
    public static async Task<OvbDepth> DepthAsync(InferenceEngine engine, ModelSpec spec, ImageData image, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(engine);
        RequireHandled(spec);
        return await DepthAsync(Generator(engine, spec), image, Seed, cancel).ConfigureAwait(false);
    }

    /// <summary>Depth through any picture source.</summary>
    public static async Task<OvbDepth> DepthAsync(Func<ImageRequest, CancellationToken, Task<ImageResult>> generate, ImageData image,
        int seed, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(generate);
        ArgumentNullException.ThrowIfNull(image);
        ImageResult result = await generate(Request(OvbPrompts.Depth, image, seed), cancel).ConfigureAwait(false);
        // The README resizes depth and normals back with BILINEAR.
        ImageData painted = BilinearTo(result, image.Width, image.Height);
        return new OvbDepth(OvbDecoder.DepthFromRgb(painted.Rgb, painted.Width, painted.Height), painted.Width, painted.Height, painted);
    }

    /// <summary>Camera-space surface normals of <paramref name="image"/>.</summary>
    public static async Task<OvbNormals> NormalsAsync(InferenceEngine engine, ModelSpec spec, ImageData image, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(engine);
        RequireHandled(spec);
        return await NormalsAsync(Generator(engine, spec), image, Seed, cancel).ConfigureAwait(false);
    }

    /// <summary>Normals through any picture source.</summary>
    public static async Task<OvbNormals> NormalsAsync(Func<ImageRequest, CancellationToken, Task<ImageResult>> generate, ImageData image,
        int seed, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(generate);
        ArgumentNullException.ThrowIfNull(image);
        ImageResult result = await generate(Request(OvbPrompts.Normals, image, seed), cancel).ConfigureAwait(false);
        ImageData painted = BilinearTo(result, image.Width, image.Height);
        (float[] normals, bool[] valid) = OvbDecoder.NormalsFromRgb(painted.Rgb, painted.Width, painted.Height);
        return new OvbNormals(normals, valid, painted.Width, painted.Height, painted);
    }

    /// <summary>The image request for one task: the task prompt verbatim, the photo as the reference, the model's own sampling.</summary>
    public static ImageRequest Request(string prompt, ImageData image, int seed) => new()
    {
        Prompt = prompt,
        Seed = seed,
        Img2Img = new Img2Img { InitImage = image, Mode = Img2ImgMode.Reference },
    };

    private static Func<ImageRequest, CancellationToken, Task<ImageResult>> Generator(InferenceEngine engine, ModelSpec spec) =>
        (request, cancel) => engine.Images.GenerateAsync(spec with { Modality = Modality.Image }, request, null, cancel);

    private static void RequireHandled(ModelSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        if (!Handles(spec))
            throw new ArgumentException($"'{spec.Requested}' is not an Open Vision Banana model.", nameof(spec));
    }

    /// <summary>The generated picture at <paramref name="width"/>×<paramref name="height"/> by nearest neighbour.</summary>
    internal static ImageData NearestTo(ImageResult picture, int width, int height)
    {
        if (picture.Width == width && picture.Height == height)
            return new ImageData { Rgb = picture.Rgb, Width = width, Height = height };
        byte[] resized = new byte[(long)width * height * 3];
        for (int y = 0; y < height; y++)
        {
            int sy = Math.Min(picture.Height - 1, (int)((y + 0.5) * picture.Height / height));
            for (int x = 0; x < width; x++)
            {
                int sx = Math.Min(picture.Width - 1, (int)((x + 0.5) * picture.Width / width));
                Buffer.BlockCopy(picture.Rgb, (sy * picture.Width + sx) * 3, resized, (y * width + x) * 3, 3);
            }
        }
        return new ImageData { Rgb = resized, Width = width, Height = height };
    }

    /// <summary>The generated picture at <paramref name="width"/>×<paramref name="height"/> by bilinear interpolation.</summary>
    internal static ImageData BilinearTo(ImageResult picture, int width, int height)
    {
        if (picture.Width == width && picture.Height == height)
            return new ImageData { Rgb = picture.Rgb, Width = width, Height = height };
        // PIL BILINEAR semantics (antialiased when shrinking, each pass quantised to 8 bits), as the README's resize.
        float[] wide = new float[(long)width * height * 3];
        Resample.BilinearHwc8(picture.Rgb, picture.Width, picture.Height, 3, wide, width, height, antialias: true, quantizeU8: true);
        byte[] resized = new byte[wide.Length];
        for (int i = 0; i < resized.Length; i++)
            resized[i] = (byte)Math.Clamp((int)MathF.Round(wide[i]), 0, 255);
        return new ImageData { Rgb = resized, Width = width, Height = height };
    }
}
