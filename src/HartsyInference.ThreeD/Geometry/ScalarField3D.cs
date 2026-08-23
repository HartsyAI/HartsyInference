namespace HartsyInference.ThreeD.Geometry;

/// <summary>A dense occupancy/SDF volume on a regular 3D grid, laid out <c>[z, y, x]</c> row-major (x fastest), spanning the axis-aligned box <c>[Min, Max]</c> in object space; consumed by <see cref="Ops.MarchingCubes"/>.</summary>
public sealed class ScalarField3D
{
    /// <summary>Field values, length <c>ResX*ResY*ResZ</c>, indexed <c>x + ResX*(y + ResY*z)</c>.</summary>
    public required float[] Values { get; init; }

    /// <summary>Grid resolution along X (number of samples, not cells).</summary>
    public required int ResX { get; init; }

    /// <summary>Grid resolution along Y.</summary>
    public required int ResY { get; init; }

    /// <summary>Grid resolution along Z.</summary>
    public required int ResZ { get; init; }

    /// <summary>Object-space minimum corner (x,y,z) of the sampled box.</summary>
    public required (float X, float Y, float Z) Min { get; init; }

    /// <summary>Object-space maximum corner (x,y,z) of the sampled box.</summary>
    public required (float X, float Y, float Z) Max { get; init; }

    /// <summary>Flattened index of grid sample (x,y,z).</summary>
    public int Index(int x, int y, int z) => x + ResX * (y + ResY * z);

    /// <summary>Value at grid sample (x,y,z).</summary>
    public float At(int x, int y, int z) => Values[Index(x, y, z)];

    /// <summary>Object-space world coordinate of grid sample (x,y,z).</summary>
    public (float X, float Y, float Z) WorldOf(int x, int y, int z) =>
    (
        Min.X + (ResX > 1 ? x / (float)(ResX - 1) : 0f) * (Max.X - Min.X),
        Min.Y + (ResY > 1 ? y / (float)(ResY - 1) : 0f) * (Max.Y - Min.Y),
        Min.Z + (ResZ > 1 ? z / (float)(ResZ - 1) : 0f) * (Max.Z - Min.Z)
    );
}
