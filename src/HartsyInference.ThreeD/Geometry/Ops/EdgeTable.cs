namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Undirected edges of an indexed triangle list, grouped into runs of half-edges that share the same two vertices.</summary>
internal sealed class EdgeTable
{
    private readonly int[] _indices;

    private EdgeTable(int[] indices, long[] keys, int[] halfEdges)
    {
        _indices = indices;
        Keys = keys;
        HalfEdges = halfEdges;
    }

    /// <summary>Half-edge slots ordered by edge key then slot; a slot is <c>3 * face + corner</c>, running from corner to corner + 1.</summary>
    public int[] HalfEdges { get; }

    /// <summary>Edge key of each entry of <see cref="HalfEdges"/>, ascending.</summary>
    public long[] Keys { get; }

    /// <summary>Number of half-edges (3 per face).</summary>
    public int Count => HalfEdges.Length;

    /// <summary>Builds the table, skipping faces whose entry in <paramref name="faceAlive"/> is false.</summary>
    public static EdgeTable Build(int[] indices, int vertexCount, bool[]? faceAlive = null)
    {
        int faces = indices.Length / 3;
        int live = 0;
        for (int f = 0; f < faces; f++)
            if (faceAlive is null || faceAlive[f]) live++;
        long[] keys = new long[live * 3];
        int[] slots = new int[live * 3];
        int n = 0;
        for (int f = 0; f < faces; f++)
        {
            if (faceAlive is not null && !faceAlive[f]) continue;
            for (int c = 0; c < 3; c++)
            {
                int a = indices[f * 3 + c], b = indices[f * 3 + (c + 1) % 3];
                keys[n] = Key(a, b, vertexCount);
                slots[n++] = f * 3 + c;
            }
        }
        Array.Sort(keys, slots);
        return new EdgeTable(indices, keys, slots);
    }

    /// <summary>Key of the undirected edge (a, b).</summary>
    public static long Key(int a, int b, int vertexCount) =>
        a < b ? (long)a * vertexCount + b : (long)b * vertexCount + a;

    /// <summary>End of the run that starts at <paramref name="start"/>.</summary>
    public int RunEnd(int start)
    {
        int end = start + 1;
        while (end < Keys.Length && Keys[end] == Keys[start]) end++;
        return end;
    }

    /// <summary>Vertex the half-edge in <paramref name="slot"/> starts from.</summary>
    public int From(int slot) => _indices[slot];

    /// <summary>Vertex the half-edge in <paramref name="slot"/> ends at.</summary>
    public int To(int slot) => _indices[slot / 3 * 3 + (slot % 3 + 1) % 3];
}
