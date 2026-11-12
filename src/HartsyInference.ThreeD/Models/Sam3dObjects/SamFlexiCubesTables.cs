namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// The case tables of the FlexiCubes extraction (NVIDIA's FlexiCubes, Apache-2.0): for each of the 256 corner-sign patterns of a cube, how many dual vertices it emits
/// (<see cref="NumVd"/>), which cube edges each dual vertex averages (<see cref="Dmc"/>: up to 7 edge indices per vertex, -1 padded), and the ambiguity resolution
/// (<see cref="Check"/>: a flag, the offset of the neighbour cube to consult and the case to use when that neighbour is ambiguous too). The data is supplied by the caller; this type only
/// validates its shape.
/// </summary>
public sealed class SamFlexiCubesTables
{
    /// <summary>Edge indices per dual vertex.</summary>
    public const int EdgesPerVertex = 7;

    /// <summary>Number of corner-sign patterns of a cube.</summary>
    public const int Cases = 256;

    /// <param name="dmc"><c>[256, maxVertices, 7]</c> flat.</param>
    /// <param name="maxVertices">Dual vertices a case can emit at most (the second extent of <paramref name="dmc"/>).</param>
    /// <param name="numVd"><c>[256]</c>.</param>
    /// <param name="check"><c>[256, 5]</c> flat: flag, dx, dy, dz, replacement case.</param>
    /// <exception cref="ArgumentException">A table has the wrong length, or a case emits more vertices than the table has room for.</exception>
    public SamFlexiCubesTables(int[] dmc, int maxVertices, int[] numVd, int[] check)
    {
        ArgumentNullException.ThrowIfNull(dmc);
        ArgumentNullException.ThrowIfNull(numVd);
        ArgumentNullException.ThrowIfNull(check);
        if (maxVertices <= 0 || dmc.Length != Cases * maxVertices * EdgesPerVertex)
            throw new ArgumentException($"The dual-vertex table holds {Cases} x {maxVertices} x {EdgesPerVertex} entries; got {dmc.Length}.", nameof(dmc));
        if (numVd.Length != Cases)
            throw new ArgumentException($"The vertex-count table holds {Cases} entries; got {numVd.Length}.", nameof(numVd));
        if (check.Length != Cases * 5)
            throw new ArgumentException($"The check table holds {Cases} x 5 entries; got {check.Length}.", nameof(check));
        if (numVd.Any(n => n < 0 || n > maxVertices))
            throw new ArgumentException($"A case emits more than {maxVertices} dual vertices.", nameof(numVd));
        if (dmc.Any(e => e < -1 || e > 11))
            throw new ArgumentException("A dual-vertex edge index is outside -1 (padding) to 11.", nameof(dmc));
        for (int c = 0; c < Cases; c++)
        {
            if (check[c * 5] is not (0 or 1))
                throw new ArgumentException($"The check flag of case {c} is {check[c * 5]}; it is 0 or 1.", nameof(check));
            if (check[c * 5 + 4] < 0 || check[c * 5 + 4] >= Cases)
                throw new ArgumentException($"The replacement case of case {c} is {check[c * 5 + 4]}; it is 0 to {Cases - 1}.", nameof(check));
            for (int g = 0; g < numVd[c]; g++)
            {
                bool any = false;
                for (int t = 0; t < EdgesPerVertex; t++)
                    any |= dmc[(c * maxVertices + g) * EdgesPerVertex + t] >= 0;
                if (!any)
                    throw new ArgumentException($"Case {c} emits dual vertex {g} but lists no edge for it.", nameof(dmc));
            }
        }

        Dmc = dmc;
        MaxVertices = maxVertices;
        NumVd = numVd;
        Check = check;
    }

    /// <summary>Dual-vertex edge lists, <c>[256, MaxVertices, 7]</c> flat.</summary>
    public int[] Dmc { get; }

    /// <summary>Second extent of <see cref="Dmc"/>.</summary>
    public int MaxVertices { get; }

    /// <summary>Dual vertices per case.</summary>
    public int[] NumVd { get; }

    /// <summary>Ambiguity table, <c>[256, 5]</c> flat.</summary>
    public int[] Check { get; }
}
