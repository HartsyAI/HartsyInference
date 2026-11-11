namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Points drawn from a prepared mesh, each remembering where on the mesh it came from so per-vertex data can be interpolated to it.</summary>
public sealed record UniRigPointSample
{
    /// <summary>Positions, length <c>3 * Count</c>.</summary>
    public required float[] Points { get; init; }

    /// <summary>Normals, length <c>3 * Count</c>: the vertex normal for vertex samples, the face normal for surface samples.</summary>
    public required float[] Normals { get; init; }

    /// <summary>Source vertex of a vertex sample, <c>-1</c> for a surface sample.</summary>
    public required int[] Vertex { get; init; }

    /// <summary>Face of a surface sample, <c>-1</c> for a vertex sample.</summary>
    public required int[] Face { get; init; }

    /// <summary>Barycentric weights (towards the face's second and third corner), length <c>2 * Count</c>; zero for vertex samples.</summary>
    public required float[] Barycentric { get; init; }

    public int Count => Points.Length / 3;
}
