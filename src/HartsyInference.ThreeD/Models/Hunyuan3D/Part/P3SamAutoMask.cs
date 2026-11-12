using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.ThreeD.Geometry.Ops;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Part;

/// <summary>P3-SAM's automatic part segmentation of a mesh (<c>mesh_sam</c> in <c>P3-SAM/demo/auto_mask.py</c>): sample 100,000 surface points and normalise them to
/// the unit cube, compute the per-point Sonata features, pick 400 farthest-point prompts, predict three masks and their IoUs per prompt, keep the mask with the
/// highest predicted IoU (probability above 0.5), and hand the masks to <see cref="P3SamPartLabeler"/>.
/// <para>The mesh must already be cleaned the way upstream does before segmenting (<c>merge_vertices</c> and <c>process(validate=True)</c>: vertices merged,
/// degenerate and duplicate faces and unreferenced vertices removed); the part boxes and face labels refer to that mesh.</para></summary>
public sealed class P3SamAutoMask(IBackend backend, P3SamFeatures features, P3SamHeads heads)
{
    /// <summary>Settings of one run. The defaults are upstream defaults.</summary>
    public sealed record Options
    {
        /// <summary>Surface points sampled (<c>point_num</c>).</summary>
        public int PointCount { get; init; } = 100_000;

        /// <summary>Prompt points (<c>prompt_num</c>).</summary>
        public int PromptCount { get; init; } = 400;

        /// <summary>Seed of the surface sampling and of the voxel picks (the Python seed does not reproduce this generator's points).</summary>
        public int Seed { get; init; } = 42;

        /// <summary>Whether the small-region merge runs at the end (<c>post_process</c>, true in the released <c>AutoMask</c>).</summary>
        public bool PostProcess { get; init; } = true;

        /// <summary>Cumulative-area threshold of that merge (<c>threshold</c>).</summary>
        public double Threshold { get; init; } = 0.95;
    }

    /// <summary>Receives one line per stage with its wall time (sampling, features, prompts, masks, labelling).</summary>
    public Action<string>? StageLog { get; set; }

    /// <summary>Segments the mesh: <paramref name="vertices"/> (<c>[V, 3]</c>) and <paramref name="faces"/> (<c>[F, 3]</c>), already cleaned.</summary>
    public P3SamPartLabeler.Result Run(float[] vertices, int[] faces, Options? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new Options();
        Stopwatch clock = Stopwatch.StartNew();
        void Stage(string name)
        {
            StageLog?.Invoke($"{name}: {clock.Elapsed.TotalSeconds:F2} s");
            clock.Restart();
        }

        Random random = new(options.Seed);
        (double[] points, float[] normals, int[] pointFace) = Sample(vertices, faces, options.PointCount, random);
        double[] unit = Normalize(points, options.PointCount);
        Stage("surface points");
        cancellationToken.ThrowIfCancellationRequested();

        float[] feature = features.Extract(unit, normals, options.PointCount, P3SamVoxelPick.Random, random);
        Stage("sonata features");
        cancellationToken.ThrowIfCancellationRequested();

        float[] unitFloat = [.. unit.Select(v => (float)v)];
        int[] prompts = SurfaceSampler.FarthestPointIndices(unitFloat, options.PromptCount);
        using P3SamHeads.Prepared prepared = heads.Prepare(backend, feature, unitFloat, options.PointCount);
        Stage("prompts and head setup");

        bool[][] masks = new bool[prompts.Length][];
        float[] iou = new float[prompts.Length];
        for (int k = 0; k < prompts.Length; k++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            P3SamHeads.Result r = heads.Segment(backend, prepared, unitFloat.AsSpan(prompts[k] * 3, 3));
            int best = 0;
            for (int m = 1; m < 3; m++)
                if (r.Iou[m] > r.Iou[best])
                    best = m;
            masks[k] = [.. r.Masks[best].Select(p => p > 0.5f)];
            iou[k] = r.Iou[best];
            if ((k + 1) % 50 == 0)
                StageLog?.Invoke($"  prompt {k + 1}/{prompts.Length}: {clock.Elapsed.TotalSeconds:F2} s into the mask stage");
        }

        Stage($"masks for {prompts.Length} prompts");
        P3SamPartLabeler.Result result = P3SamPartLabeler.Label(vertices, faces, unitFloat, pointFace, masks, iou, options.PostProcess, options.Threshold);
        Stage("labels and boxes");
        return result;
    }

    /// <summary>Area-weighted uniform sampling of the surface (<c>trimesh.sample.sample_surface</c>): a face is drawn with probability proportional to its area and a
    /// point is placed uniformly inside it. Returns the points, the face normals and the face of each point.</summary>
    internal static (double[] Points, float[] Normals, int[] Face) Sample(float[] vertices, int[] faces, int count, Random random)
    {
        int faceCount = faces.Length / 3;
        double[] cumulative = new double[faceCount];
        float[] faceNormals = new float[faceCount * 3];
        double total = 0;
        for (int f = 0; f < faceCount; f++)
        {
            int a = faces[f * 3] * 3, b = faces[f * 3 + 1] * 3, c = faces[f * 3 + 2] * 3;
            double ux = vertices[b] - vertices[a], uy = vertices[b + 1] - vertices[a + 1], uz = vertices[b + 2] - vertices[a + 2];
            double vx = vertices[c] - vertices[a], vy = vertices[c + 1] - vertices[a + 1], vz = vertices[c + 2] - vertices[a + 2];
            double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx, length = Math.Sqrt(nx * nx + ny * ny + nz * nz);
            total += 0.5 * length;
            cumulative[f] = total;
            if (length > 0)
            {
                faceNormals[f * 3] = (float)(nx / length);
                faceNormals[f * 3 + 1] = (float)(ny / length);
                faceNormals[f * 3 + 2] = (float)(nz / length);
            }
        }

        double[] points = new double[count * 3];
        float[] normals = new float[count * 3];
        int[] face = new int[count];
        for (int i = 0; i < count; i++)
        {
            int f = Array.BinarySearch(cumulative, random.NextDouble() * total);
            f = f < 0 ? ~f : f;
            f = Math.Min(f, faceCount - 1);
            double u = random.NextDouble(), v = random.NextDouble();
            if (u + v > 1)
            {
                u = 1 - u;
                v = 1 - v;
            }

            int a = faces[f * 3] * 3, b = faces[f * 3 + 1] * 3, c = faces[f * 3 + 2] * 3;
            for (int k = 0; k < 3; k++)
            {
                points[i * 3 + k] = vertices[a + k] + u * (vertices[b + k] - vertices[a + k]) + v * (vertices[c + k] - vertices[a + k]);
                normals[i * 3 + k] = faceNormals[f * 3 + k];
            }

            face[i] = f;
        }

        return (points, normals, face);
    }

    /// <summary>normalize_pc: centre on the bounding box and divide by the largest half extent, so the points fill the unit cube.</summary>
    internal static double[] Normalize(double[] points, int count)
    {
        double[] min = [double.MaxValue, double.MaxValue, double.MaxValue], max = [double.MinValue, double.MinValue, double.MinValue];
        for (int i = 0; i < count; i++)
            for (int a = 0; a < 3; a++)
            {
                min[a] = Math.Min(min[a], points[i * 3 + a]);
                max[a] = Math.Max(max[a], points[i * 3 + a]);
            }

        double scale = Math.Max(Math.Max((max[0] - min[0]) / 2, (max[1] - min[1]) / 2), (max[2] - min[2]) / 2);
        double[] result = new double[count * 3];
        for (int i = 0; i < count; i++)
            for (int a = 0; a < 3; a++)
                result[i * 3 + a] = (points[i * 3 + a] - (min[a] + max[a]) / 2) / (scale + 1e-10);
        return result;
    }
}
