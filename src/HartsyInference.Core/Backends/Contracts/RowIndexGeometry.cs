namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The extents of a row-index op: the rows the index names (<c>Rows</c> entries), their width, and how many rows the
/// indexed tensor holds, every index lying in <c>[0, IndexedRows)</c>.</summary>
public readonly record struct RowIndexGeometry(long Rows, int Width, long IndexedRows);
