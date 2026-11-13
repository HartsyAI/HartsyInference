using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Paint;

/// <summary>The 3D position, in the render frame, of every UV-covered texel of a mesh (<c>MeshRender.extract_textiles</c>): the points the
/// bake looks up in each view.</summary>
public sealed class PaintTexelSurface
{
    private PaintTexelSurface(int width, int height, int[] texels, float[] positions)
    {
        Width = width;
        Height = height;
        Texels = texels;
        Positions = positions;
    }

    /// <summary>Texture width.</summary>
    public int Width { get; }

    /// <summary>Texture height.</summary>
    public int Height { get; }

    /// <summary>Linear index (<c>y * Width + x</c>) of each covered texel.</summary>
    public int[] Texels { get; }

    /// <summary>Position per covered texel, three floats each, same order as <see cref="Texels"/>.</summary>
    public float[] Positions { get; }

    /// <summary>Number of covered texels.</summary>
    public int Count => Texels.Length;

    /// <summary>A surface from texels and positions that are already known (a reference dump); <paramref name="texels"/> are linear indices <c>y * width + x</c> and <paramref name="positions"/> three floats each.</summary>
    internal static PaintTexelSurface FromTexels(int width, int height, int[] texels, float[] positions)
    {
        ArgumentNullException.ThrowIfNull(texels);
        ArgumentNullException.ThrowIfNull(positions);
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), "The texture needs a positive size.");
        if (positions.Length != texels.Length * 3 || texels.Any(t => t < 0 || t >= (long)width * height) || texels.Distinct().Count() != texels.Length)
            throw new ArgumentException("The texels must be distinct, lie in the texture and have one position each.");
        return new PaintTexelSurface(width, height, texels, positions);
    }

    /// <summary>Rasterizes the frame's triangles in UV space (one UV pair per vertex of <paramref name="uvs"/>) and interpolates positions.</summary>
    public static PaintTexelSurface Create(PaintMeshFrame frame, ReadOnlySpan<float> uvs, int width, int height, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(frame);
        if (uvs.Length != frame.Vertices.Length / 3 * 2)
            throw new ArgumentException("The mesh needs one UV per vertex.", nameof(uvs));
        RasterizedUvMap map = Rasterizer.Rasterize(uvs, frame.Indices, width, height, cancellationToken);
        int count = 0;
        foreach (int id in map.TriangleIds)
            if (id >= 0)
                count++;
        int[] texels = new int[count];
        float[] positions = new float[count * 3];
        int n = 0;
        for (int texel = 0; texel < map.TriangleIds.Length; texel++)
        {
            int tri = map.TriangleIds[texel];
            if (tri < 0)
                continue;
            float b1 = map.Barycentric[texel * 2], b2 = map.Barycentric[texel * 2 + 1], b0 = 1f - b1 - b2;
            int a = frame.Indices[tri * 3] * 3, b = frame.Indices[tri * 3 + 1] * 3, c = frame.Indices[tri * 3 + 2] * 3;
            texels[n] = texel;
            for (int k = 0; k < 3; k++)
                positions[n * 3 + k] = b0 * frame.Vertices[a + k] + b1 * frame.Vertices[b + k] + b2 * frame.Vertices[c + k];
            n++;
        }
        return new PaintTexelSurface(width, height, texels, positions);
    }
}
