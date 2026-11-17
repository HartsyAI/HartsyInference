using HartsyInference.Core.Backends;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.ThreeD.Preprocessing;

/// <summary>How much room Idol's square crop leaves around the object: the crop side is the object's box times this margin.</summary>
public enum IdolFraming
{
    /// <summary>Renders and generated pictures of one object: the object fills 1 / 1.2 of the frame (idol-sdk's <c>synthetic</c> mode).</summary>
    Object = 0,

    /// <summary>Photographs: the object fills 1 / 1.7 of the frame (idol-sdk's <c>real</c> mode).</summary>
    Photo = 1,
}

/// <summary>Idol's input preparation (<c>idol.preprocess.preprocess_image</c>): resize so the short side is 1024 with a Lanczos filter, take
/// the matte from the picture's alpha or from a matting model when it has none, erode the alpha by one pixel so no background-coloured fringe
/// survives, crop a square of side box × margin around the object (padding with transparent black), resize to 1024, and composite on black.
/// The result is what idol-sdk hands its two image encoders.</summary>
public static class IdolImageProcessor
{
    /// <summary>The edge of the prepared square, and of the encoders' input, in pixels.</summary>
    public const int Size = 1024;

    /// <summary>The crop margin of <paramref name="framing"/>.</summary>
    public static double Margin(IdolFraming framing) => framing == IdolFraming.Photo ? 1.7 : 1.2;

    /// <summary>Prepares one image. <paramref name="alpha"/> is its coverage when it has a real one (some pixel below 255); otherwise
    /// (null or fully opaque) <paramref name="matter"/> estimates it on the resized picture. Whether it has one is decided on the image as given, before the resize.</summary>
    /// <exception cref="ArgumentException">The image is malformed, has no matte, or the matte is empty.</exception>
    public static PreparedImage Process(IBackend backend, byte[] rgb, byte[]? alpha, int width, int height, IdolFraming framing, IForegroundMatter? matter)
    {
        byte[] coverage = ForegroundPixels.Validate(rgb, alpha, width, height);
        double scale = (double)Size / Math.Min(width, height);
        int scaledWidth = Math.Max(1, (int)Math.Round(width * scale, MidpointRounding.ToEven));
        int scaledHeight = Math.Max(1, (int)Math.Round(height * scale, MidpointRounding.ToEven));
        byte[] scaled = PilRgba.ResizeLanczos(ForegroundPixels.ToRgba(rgb, coverage), width, height, scaledWidth, scaledHeight);

        // The decision is the original image's, the one the engine's validator and memory planner made; the resize cannot flip it.
        if (!HasRealAlpha(coverage))
        {
            if (matter is null)
                throw new ArgumentException("The image has no alpha and no matting model is available to cut the object out.", nameof(alpha));
            byte[] colours = new byte[scaledWidth * scaledHeight * 3];
            for (int i = 0; i < scaledWidth * scaledHeight; i++)
            {
                colours[i * 3] = scaled[i * 4];
                colours[i * 3 + 1] = scaled[i * 4 + 1];
                colours[i * 3 + 2] = scaled[i * 4 + 2];
            }

            float[] matte = matter.Alpha(backend, colours, scaledWidth, scaledHeight);
            for (int i = 0; i < matte.Length; i++)
                scaled[i * 4 + 3] = (byte)Math.Clamp((int)Math.Round(matte[i] * 255f, MidpointRounding.ToEven), 0, 255);
        }

        byte[] plane = new byte[scaledWidth * scaledHeight];
        for (int i = 0; i < plane.Length; i++)
            plane[i] = scaled[i * 4 + 3];
        byte[] eroded = PilRgba.MinFilter3(plane, scaledWidth, scaledHeight);
        for (int i = 0; i < eroded.Length; i++)
            scaled[i * 4 + 3] = eroded[i];

        (int MinX, int MinY, int MaxX, int MaxY)? box = ForegroundPixels.Bounds(eroded, scaledWidth, scaledHeight, 0);
        if (box is null)
            throw new ArgumentException("No object found in the image: its alpha channel is empty.", nameof(alpha));
        double centerX = (box.Value.MinX + box.Value.MaxX) / 2.0, centerY = (box.Value.MinY + box.Value.MaxY) / 2.0;
        double half = Math.Max(box.Value.MaxX - box.Value.MinX, box.Value.MaxY - box.Value.MinY) / 2.0 * Margin(framing);
        // Python's int() truncates toward zero, as a C# cast does.
        int x0 = (int)(centerX - half), y0 = (int)(centerY - half), x1 = (int)(centerX + half), y1 = (int)(centerY + half);
        if (x1 <= x0 || y1 <= y0)
            throw new ArgumentException("The object is too small to crop.", nameof(alpha));

        byte[] crop = ForegroundPixels.CropRgba(scaled, scaledWidth, scaledHeight, x0, y0, x1, y1);
        byte[] square = PilRgba.ResizeLanczos(crop, x1 - x0, y1 - y0, Size, Size);
        return new PreparedImage(PilRgba.CompositeOnBlack(square, Size, Size), Size, Size);
    }

    /// <summary>Whether the coverage plane carries a matte: some pixel is not fully opaque (idol-sdk's <c>has_real_alpha</c>).</summary>
    internal static bool HasRealAlpha(ReadOnlySpan<byte> coverage)
    {
        foreach (byte a in coverage)
        {
            if (a < 255)
                return true;
        }

        return false;
    }
}
