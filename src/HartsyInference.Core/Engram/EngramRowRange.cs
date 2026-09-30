namespace HartsyInference.Core.Engram;

/// <summary>A half-open range of table rows, <c>[Start, Start + Count)</c>.</summary>
public readonly record struct EngramRowRange(long Start, long Count)
{
    /// <summary>First row past the range.</summary>
    public long End => Start + Count;

    /// <summary>The range covering <paramref name="rows"/> rows from row 0.</summary>
    public static EngramRowRange All(long rows) => new(0, rows);

    /// <summary>True when <paramref name="row"/> lies inside the range.</summary>
    public bool Contains(long row) => row >= Start && row < Start + Count;
}
