namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Per-texel result of rasterizing a mesh in UV space: which triangle covers the texel center and the barycentric weights there.</summary>
public sealed class RasterizedUvMap
{
    /// <summary>Creates an empty map (no texel covered).</summary>
    public RasterizedUvMap(int width, int height)
    {
        Width = width;
        Height = height;
        TriangleIds = new int[width * height];
        Array.Fill(TriangleIds, -1);
        Barycentric = new float[width * height * 2];
    }

    /// <summary>Width in texels.</summary>
    public int Width { get; }

    /// <summary>Height in texels.</summary>
    public int Height { get; }

    /// <summary>Covering triangle per texel (row-major, row 0 at v = 0), or -1.</summary>
    public int[] TriangleIds { get; }

    /// <summary>Per texel (b1, b2): weights of the triangle's second and third vertex; the first is <c>1 - b1 - b2</c>.</summary>
    public float[] Barycentric { get; }
}
