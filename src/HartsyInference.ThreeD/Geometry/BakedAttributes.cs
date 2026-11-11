namespace HartsyInference.ThreeD.Geometry;

/// <summary>Float attribute image produced by the texture baker: texel-major, <see cref="Channels"/> interleaved values per texel,
/// row 0 first (row index grows with UV v), plus a mask of texels that received a value.</summary>
public sealed class BakedAttributes
{
    /// <summary>Creates an image with every texel uncovered and zero.</summary>
    public BakedAttributes(int width, int height, int channels)
    {
        if (width <= 0 || height <= 0 || channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "Image dimensions and channel count must be positive.");
        Width = width;
        Height = height;
        Channels = channels;
        Data = new float[checked(width * height * channels)];
        Covered = new bool[width * height];
    }

    /// <summary>Width in texels.</summary>
    public int Width { get; }

    /// <summary>Height in texels.</summary>
    public int Height { get; }

    /// <summary>Values per texel.</summary>
    public int Channels { get; }

    /// <summary>Texel values, <c>(y * Width + x) * Channels + c</c>.</summary>
    public float[] Data { get; }

    /// <summary>True for texels the rasterizer covered and the volume answered.</summary>
    public bool[] Covered { get; }
}
