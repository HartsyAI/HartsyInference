namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Output of <see cref="UvUnwrapper"/>: the mesh with split vertices and UVs, plus the maps back to the source mesh.</summary>
public sealed class UvUnwrapResult
{
    /// <summary>Creates a result; the arrays are referenced, not copied.</summary>
    public UvUnwrapResult(Mesh mesh, int[] vertexMap, int[] faceChart, int chartCount)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        ArgumentNullException.ThrowIfNull(vertexMap);
        ArgumentNullException.ThrowIfNull(faceChart);
        Mesh = mesh;
        VertexMap = vertexMap;
        FaceChart = faceChart;
        ChartCount = chartCount;
    }

    /// <summary>Unwrapped mesh: same faces in the same order as the source, vertices duplicated along chart seams, one UV each in [0,1].</summary>
    public Mesh Mesh { get; }

    /// <summary>For each new vertex, the index of the source vertex it was split from.</summary>
    public int[] VertexMap { get; }

    /// <summary>Chart index of each face.</summary>
    public int[] FaceChart { get; }

    /// <summary>Number of charts in the atlas.</summary>
    public int ChartCount { get; }
}
