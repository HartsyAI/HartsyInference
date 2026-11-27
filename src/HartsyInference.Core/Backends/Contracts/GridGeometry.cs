namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The extents of a grid scatter or gather: the active voxels, their channels, and the cells of one channel plane.</summary>
public readonly record struct GridGeometry(int Voxels, int Channels, long Cells);
