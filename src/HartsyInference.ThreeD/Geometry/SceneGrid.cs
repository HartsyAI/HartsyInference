using System.Numerics;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Geometry;

/// <summary>
/// Cubic dual-grid frame that fits a scene mesh: voxel index <c>i</c> spans world <c>Origin + i·VoxelSize</c> to the next
/// corner. The scene's largest extent covers all but one voxel of margin on each side of the grid.
/// </summary>
public sealed record SceneGrid
{
    /// <summary>Smallest grid the frame accepts; below this the margin leaves no interior.</summary>
    public const int MinResolution = 16;

    /// <summary>Voxels per axis.</summary>
    public required int Resolution { get; init; }

    /// <summary>World-space corner of voxel index (0, 0, 0).</summary>
    public required Vector3 Origin { get; init; }

    /// <summary>World-space edge length of one voxel.</summary>
    public required float VoxelSize { get; init; }

    /// <summary>Fits a grid of <paramref name="resolution"/> voxels per axis around <paramref name="scene"/>.</summary>
    public static SceneGrid Fit(Mesh scene, int resolution)
    {
        ArgumentNullException.ThrowIfNull(scene);
        if (resolution < MinResolution)
            throw new ArgumentOutOfRangeException(nameof(resolution), resolution, $"Grid resolution must be at least {MinResolution}.");
        if (scene.VertexCount == 0 || scene.TriangleCount == 0)
            throw new HartsyInferenceException("The scene mesh has no triangles to voxelize.");

        Vector3 min = new(float.MaxValue), max = new(float.MinValue);
        float[] v = scene.Vertices;
        for (int i = 0; i < v.Length; i += 3)
        {
            Vector3 p = new(v[i], v[i + 1], v[i + 2]);
            min = Vector3.Min(min, p);
            max = Vector3.Max(max, p);
        }

        Vector3 extent = max - min;
        float span = MathF.Max(extent.X, MathF.Max(extent.Y, extent.Z));
        if (!float.IsFinite(span) || span <= 0f)
            throw new HartsyInferenceException("The scene mesh has no positive extent.");

        float voxelSize = span / (resolution - 2);
        Vector3 center = (min + max) * 0.5f;
        Vector3 origin = center - new Vector3(span * 0.5f + voxelSize);
        return new SceneGrid { Resolution = resolution, Origin = origin, VoxelSize = voxelSize };
    }

    /// <summary>World position in voxel units, where index <c>i</c> spans <c>[i, i+1)</c>.</summary>
    public Vector3 ToVoxelUnits(Vector3 world) => (world - Origin) / VoxelSize;
}
