namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>A chain of boundary edges traced by <see cref="MeshBoundary.Trace"/>: the vertex indices in walking order, closed when the walk returned to its first edge.</summary>
public sealed class BoundaryLoop
{
    /// <summary>Creates a loop over <paramref name="vertices"/>.</summary>
    public BoundaryLoop(int[] vertices, bool closed, double length)
    {
        ArgumentNullException.ThrowIfNull(vertices);
        Vertices = vertices;
        Closed = closed;
        Length = length;
    }

    /// <summary>Start vertex of each edge in walking order; a closed loop's last edge returns to the first vertex, an open chain's last vertex is its end.</summary>
    public int[] Vertices { get; }

    /// <summary>The walk came back to its first edge; false when it ended at a non-manifold edge.</summary>
    public bool Closed { get; }

    /// <summary>Sum of the edge lengths in mesh units.</summary>
    public double Length { get; }

    /// <summary>Number of edges.</summary>
    public int EdgeCount => Closed ? Vertices.Length : Vertices.Length - 1;

    /// <summary>A closed loop cut into cycles that repeat no vertex: a loop through a pinch vertex twice (a figure-eight) gives its two lobes. Cycles under three vertices are dropped; an open chain gives none.</summary>
    public IReadOnlyList<int[]> SimpleCycles()
    {
        if (!Closed) return [];
        List<int[]> cycles = [];
        List<int> stack = [];
        Dictionary<int, int> position = [];
        foreach (int v in Vertices)
        {
            if (!position.TryGetValue(v, out int at))
            {
                position[v] = stack.Count;
                stack.Add(v);
                continue;
            }
            if (stack.Count - at >= 3) cycles.Add([.. stack.GetRange(at, stack.Count - at)]);
            for (int i = at + 1; i < stack.Count; i++) position.Remove(stack[i]);
            stack.RemoveRange(at + 1, stack.Count - at - 1);
        }
        if (stack.Count >= 3) cycles.Add([.. stack]);
        return cycles;
    }
}
