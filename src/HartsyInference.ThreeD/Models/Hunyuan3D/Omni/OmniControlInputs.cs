using HartsyInference.ThreeD.Models.Hunyuan3D.Part;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Omni;

/// <summary>Builds the rows of each Hunyuan3D-Omni control signal from user geometry the way upstream's <c>inference.py</c> prepares them: a point control is the vertices of the loaded file normalised into the cube, a voxel control is
/// the mesh turned upright, normalised and sampled before it is quantised to 16 voxels per axis, a bounding box is three extents and a pose is the bones as given.</summary>
public static class OmniControlInputs
{
    /// <summary>Upstream's normalisation of a point control: the longest side of the bounding box becomes <c>2 * 0.98</c> (<c>normalize_mesh(mesh, scale=0.98)</c>).</summary>
    public const float PointScale = 0.98f;

    /// <summary>Upstream's normalisation of a voxel control (<c>normalize_mesh(mesh)</c>, default scale).</summary>
    public const float VoxelScale = 0.9999f;

    /// <summary>Surface samples a voxel control is built from (<c>mesh.sample(81920)</c>).</summary>
    public const int VoxelSamples = 81920;

    /// <summary>Most points a point control keeps: each one is a token in every cross-attention of the DiT (upstream's demos use 2,048), so a denser cloud is subsampled.</summary>
    public const int MaxPoints = 8192;

    /// <summary>The point control: <paramref name="vertices"/> (<c>3 * n</c> values, a mesh's or a point cloud's) centred on their bounding box and scaled so the longest side is <c>1.96</c>, with identical points merged (trimesh
    /// merges the duplicate vertices of a loaded mesh) and, above <see cref="MaxPoints"/>, a subset of that many points picked with <paramref name="seed"/>, in their original order; one row, hence one token, per point.</summary>
    public static float[] Point(ReadOnlySpan<float> vertices, int seed = 0)
    {
        Require(vertices);
        float[] normalised = Normalise(vertices, PointScale);
        HashSet<(float, float, float)> seen = [];
        List<int> kept = [];
        for (int i = 0; i < normalised.Length; i += 3)
        {
            if (seen.Add((normalised[i], normalised[i + 1], normalised[i + 2])))
                kept.Add(i);
        }

        if (kept.Count > MaxPoints)
        {
            Random random = new(seed);
            int[] order = [.. Enumerable.Range(0, kept.Count)];
            for (int i = 0; i < MaxPoints; i++)
            {
                int j = i + random.Next(order.Length - i);
                (order[i], order[j]) = (order[j], order[i]);
            }

            kept = [.. order.Take(MaxPoints).Order().Select(k => kept[k])];
        }

        float[] points = new float[kept.Count * 3];
        for (int k = 0; k < kept.Count; k++)
            Array.Copy(normalised, kept[k], points, k * 3, 3);
        return OmniControlEncoder.PointRows(points);
    }

    /// <summary>The voxel control: the mesh is turned by -90 degrees about x (<c>(x, y, z)</c> becomes <c>(x, z, -y)</c>, upstream's rotation for its voxel files), centred and scaled to <c>2 * 0.9999</c> on the longest side,
    /// sampled at <see cref="VoxelSamples"/> area-weighted surface points with <paramref name="seed"/>, and quantised to the 16 x 16 x 16 grid; one row per occupied voxel.</summary>
    public static float[] Voxel(float[] vertices, int[] indices, int seed)
    {
        ArgumentNullException.ThrowIfNull(vertices);
        ArgumentNullException.ThrowIfNull(indices);
        Require(vertices);
        if (indices.Length == 0 || indices.Length % 3 != 0)
            throw new ArgumentException("A voxel control needs a triangle mesh with faces.", nameof(indices));
        foreach (int index in indices)
        {
            if ((uint)index >= (uint)(vertices.Length / 3))
                throw new ArgumentException("A face refers to a vertex that does not exist.", nameof(indices));
        }

        float[] upright = new float[vertices.Length];
        for (int i = 0; i < vertices.Length; i += 3)
        {
            upright[i] = vertices[i];
            upright[i + 1] = vertices[i + 2];
            upright[i + 2] = -vertices[i + 1];
        }

        float[] normalised = Normalise(upright, VoxelScale);
        (double[] points, _, _) = P3SamAutoMask.Sample(normalised, indices, VoxelSamples, new Random(seed));
        float[] samples = new float[points.Length];
        for (int i = 0; i < samples.Length; i++)
            samples[i] = (float)points[i];
        return OmniControlEncoder.VoxelRows(samples);
    }

    /// <summary>The bounding-box control from the three extents (length, height, width), each in <c>(0, 1]</c>.</summary>
    public static float[] BoundingBox(float length, float height, float width)
    {
        foreach ((string name, float value) in new[] { ("length", length), ("height", height), ("width", width) })
        {
            if (!float.IsFinite(value) || value <= 0f || value > 1f)
                throw new ArgumentOutOfRangeException(name, value, "A bounding-box extent must be in (0, 1].");
        }

        return OmniControlEncoder.BoundingBoxRow(length, height, width);
    }

    /// <summary>The pose control from <paramref name="bones"/>, 6 values per bone (<c>x1 y1 z1 x2 y2 z2</c>, the lines of upstream's bone files).</summary>
    public static float[] Pose(ReadOnlySpan<float> bones)
    {
        Require(bones, 6, "bone");
        return OmniControlEncoder.PoseRows(bones);
    }

    private static void Require(ReadOnlySpan<float> values, int width = 3, string unit = "point")
    {
        if (values.Length == 0 || values.Length % width != 0)
            throw new ArgumentException($"The control needs at least one {unit} of {width} values; got {values.Length} values.", nameof(values));
        foreach (float value in values)
        {
            if (!float.IsFinite(value))
                throw new ArgumentException("The control has a value that is not finite.", nameof(values));
        }
    }

    private static float[] Normalise(ReadOnlySpan<float> points, float scale)
    {
        float[] lo = [float.MaxValue, float.MaxValue, float.MaxValue], hi = [float.MinValue, float.MinValue, float.MinValue];
        for (int i = 0; i < points.Length; i += 3)
        {
            for (int c = 0; c < 3; c++)
            {
                lo[c] = MathF.Min(lo[c], points[i + c]);
                hi[c] = MathF.Max(hi[c], points[i + c]);
            }
        }

        float longest = MathF.Max(hi[0] - lo[0], MathF.Max(hi[1] - lo[1], hi[2] - lo[2]));
        if (!(longest > 0f))
            throw new ArgumentException("The control geometry has no extent.", nameof(points));
        float factor = 2f * scale / longest;
        float[] result = new float[points.Length];
        for (int i = 0; i < points.Length; i += 3)
        {
            for (int c = 0; c < 3; c++)
                result[i + c] = (points[i + c] - (lo[c] + hi[c]) / 2f) * factor;
        }

        return result;
    }
}
