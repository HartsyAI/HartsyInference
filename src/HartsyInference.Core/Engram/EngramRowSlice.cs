namespace HartsyInference.Core.Engram;

/// <summary>One contiguous, row-major byte plane of an Engram table: row <c>r</c> occupies <c>BaseOffset + r * BytesPerRow</c> in its source.</summary>
/// <param name="BaseOffset">Absolute file offset of row 0 of this plane.</param>
/// <param name="BytesPerRow">Bytes one row takes in this plane.</param>
/// <param name="SourceIndex">Which byte source holds the plane (0 unless the table spans several shard files).</param>
public readonly record struct EngramRowSlice(long BaseOffset, int BytesPerRow, int SourceIndex = 0);
