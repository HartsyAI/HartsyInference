namespace HartsyInference.Core.Backends.Layout;

/// <summary>A dense <c>[channels, Depth, Height, Width]</c> grid addressed by voxel coordinates for <see cref="IBackend.GridScatter"/>
/// and <see cref="IBackend.GridGather"/>: a coordinate row holds <see cref="CoordStride"/> integers and its <c>(d, h, w)</c> are
/// the three starting at <see cref="FirstCoordColumn"/>; voxel <c>(d, h, w)</c> is cell <c>d · Height · Width + h · Width + w</c>
/// of every channel plane.</summary>
public readonly record struct GridSpec(int Depth, int Height, int Width, int CoordStride, int FirstCoordColumn)
{
    /// <summary>A cubic grid of <paramref name="resolution"/>³ cells (TRELLIS: coordinate rows <c>[batch, d, h, w]</c>, stride 4, first column 1).</summary>
    public static GridSpec Cube(int resolution, int coordStride, int firstCoordColumn) => new(resolution, resolution, resolution, coordStride, firstCoordColumn);

    /// <summary>Cells in one channel plane.</summary>
    public long Cells => (long)Depth * Height * Width;
}
