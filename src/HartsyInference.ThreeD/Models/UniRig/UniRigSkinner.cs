using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Runs stage 2 end to end for a prepared mesh and a generated skeleton: frame, voxel prior, point sampling, network, and the move back onto the mesh's vertices.</summary>
public static class UniRigSkinner
{
    /// <summary>Points sampled for the network; upstream's skin transform uses 8,192 vertices among 32,768.</summary>
    public const int SamplePoints = 32768;

    public const int SampleVertices = 8192;

    /// <param name="mesh">The mesh as prepared for stage 1, in the same normalized frame as <paramref name="skeleton"/>.</param>
    public static UniRigSkinResult Skin(UniRigSkinModel model, UniRigPreparedMesh mesh, UniRigSkeleton skeleton, UniRigVoxelSkinOptions? voxelOptions = null,
        int seed = 12345, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(mesh);
        ArgumentNullException.ThrowIfNull(skeleton);
        int j = skeleton.JointCount;
        if (j < 1)
        {
            throw new HartsyInferenceException("UniRig skinning needs a skeleton with at least one joint.");
        }

        // Upstream renormalizes vertices and joints together before the network sees them.
        (UniRigPreparedMesh framed, float[] joints) = Reframe(mesh, skeleton.Joints);
        float[] prior = VoxelPrior(framed, joints, voxelOptions ?? new UniRigVoxelSkinOptions(), cancel);
        UniRigPointSample sample = UniRigSurfaceSampler.Sample(framed, SamplePoints, SampleVertices, seed);
        float[] samplePrior = InterpolatePrior(framed, sample, prior, j);
        float[] sampleSkin = model.Predict(sample.Points, sample.Normals, joints, skeleton.Parents, samplePrior, seed, cancel);
        float[] weights = UniRigReskin.Apply(sample.Points, sampleSkin, framed.Vertices, framed.Faces, skeleton.Parents);
        return new UniRigSkinResult { Weights = weights, VertexCount = framed.VertexCount, JointCount = j };
    }

    /// <summary>Normalizes the union of the mesh vertices and the joints into [-1, 1]³ by its bounding box's longest side.</summary>
    internal static (UniRigPreparedMesh Mesh, float[] Joints) Reframe(UniRigPreparedMesh mesh, float[] joints)
    {
        float[] lo = [float.MaxValue, float.MaxValue, float.MaxValue];
        float[] hi = [float.MinValue, float.MinValue, float.MinValue];
        foreach (float[] set in new[] { mesh.Vertices, joints })
        {
            for (int i = 0; i < set.Length; i++)
            {
                lo[i % 3] = Math.Min(lo[i % 3], set[i]);
                hi[i % 3] = Math.Max(hi[i % 3], set[i]);
            }
        }

        float[] center = [(lo[0] + hi[0]) / 2f, (lo[1] + hi[1]) / 2f, (lo[2] + hi[2]) / 2f];
        float scale = Math.Max(hi[0] - lo[0], Math.Max(hi[1] - lo[1], hi[2] - lo[2])) / 2f;
        if (!(scale > 0f))
        {
            throw new HartsyInferenceException("UniRig skinning found a degenerate mesh and skeleton extent.");
        }

        return (mesh with { Vertices = Apply(mesh.Vertices, center, scale) }, Apply(joints, center, scale));
    }

    private static float[] VoxelPrior(UniRigPreparedMesh mesh, float[] joints, UniRigVoxelSkinOptions options, CancellationToken cancel)
    {
        float[] lo = [float.MaxValue, float.MaxValue, float.MaxValue];
        float[] hi = [float.MinValue, float.MinValue, float.MinValue];
        for (int i = 0; i < mesh.Vertices.Length; i++)
        {
            lo[i % 3] = Math.Min(lo[i % 3], mesh.Vertices[i]);
            hi[i % 3] = Math.Max(hi[i % 3], mesh.Vertices[i]);
        }

        float[] center = [(lo[0] + hi[0]) / 2f, (lo[1] + hi[1]) / 2f, (lo[2] + hi[2]) / 2f];
        float scale = Math.Max(hi[0] - lo[0], Math.Max(hi[1] - lo[1], hi[2] - lo[2])) / 2f;
        return UniRigVoxelSkin.Compute(Apply(mesh.Vertices, center, scale), mesh.Faces, Apply(joints, center, scale), options, cancel);
    }

    private static float[] InterpolatePrior(UniRigPreparedMesh mesh, UniRigPointSample sample, float[] prior, int j)
    {
        float[] result = new float[sample.Count * j];
        for (int i = 0; i < sample.Count; i++)
        {
            if (sample.Vertex[i] >= 0)
            {
                Array.Copy(prior, sample.Vertex[i] * j, result, i * j, j);
                continue;
            }

            int f = sample.Face[i];
            int a = mesh.Faces[3 * f] * j;
            int b = mesh.Faces[3 * f + 1] * j;
            int c = mesh.Faces[3 * f + 2] * j;
            float u = sample.Barycentric[2 * i];
            float v = sample.Barycentric[2 * i + 1];
            for (int k = 0; k < j; k++)
            {
                result[i * j + k] = prior[a + k] + u * (prior[b + k] - prior[a + k]) + v * (prior[c + k] - prior[a + k]);
            }
        }

        return result;
    }

    private static float[] Apply(float[] points, float[] center, float scale)
    {
        float[] result = new float[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            result[i] = (points[i] - center[i % 3]) / scale;
        }

        return result;
    }
}
