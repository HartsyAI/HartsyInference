namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The extents of a neighborhood attention call: the <c>[Batch, Depth, Height, Width, Heads, HeadDim]</c> grid and the
/// window after each extent is clamped to its axis.</summary>
public readonly record struct NeighborhoodGeometry(int Batch, int Depth, int Height, int Width, int Heads, int HeadDim, int WindowDepth, int WindowHeight, int WindowWidth);
