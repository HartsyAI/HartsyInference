namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>One parametrized chart: a connected set of source faces with local vertices and planar coordinates.</summary>
internal sealed class UvChart
{
    public UvChart(int[] faces, int[] sourceVertices, double[] uv, int[] localTriangles, double area3d, double area2d)
    {
        Faces = faces;
        SourceVertices = sourceVertices;
        Uv = uv;
        LocalTriangles = localTriangles;
        Area3d = area3d;
        Area2d = area2d;
    }

    /// <summary>Source face indices, in the order they were collected.</summary>
    public int[] Faces { get; }

    /// <summary>Source vertex of each local vertex.</summary>
    public int[] SourceVertices { get; }

    /// <summary>Planar coordinates, two per local vertex; rewritten by the packer.</summary>
    public double[] Uv { get; }

    /// <summary>Local vertex indices, three per entry of <see cref="Faces"/>.</summary>
    public int[] LocalTriangles { get; }

    /// <summary>Surface area of the chart's faces.</summary>
    public double Area3d { get; }

    /// <summary>Total absolute area of the projected triangles.</summary>
    public double Area2d { get; }

    /// <summary>Packed width after <see cref="UvAtlasPacker"/> normalized the chart.</summary>
    public double Width { get; set; }

    /// <summary>Packed height after <see cref="UvAtlasPacker"/> normalized the chart.</summary>
    public double Height { get; set; }
}
