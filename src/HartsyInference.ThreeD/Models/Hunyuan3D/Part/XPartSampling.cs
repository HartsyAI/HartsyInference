using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Part;

/// <summary>The geometry steps of the X-Part pipeline around the networks (<c>partformer_pipeline.py</c>, <c>mesh_utils.py</c>): the mesh normalisation, the surface samples of the object and of
/// each part's box, and the part ids.</summary>
internal static class XPartSampling
{
    /// <summary>Upstream's <c>normalize_mesh</c>: the box centre moves to the origin and the longest side becomes 1.6 (<c>scale = max extent / 2 / 0.8</c>). Returns the normalised vertices, the
    /// centre and the scale (<c>vertex = normalised * scale + center</c> restores them).</summary>
    public static (float[] Vertices, double[] Center, double Scale) Normalize(float[] vertices)
    {
        if (vertices.Length == 0 || vertices.Length % 3 != 0)
            throw new ArgumentException("The mesh needs whole vertices.", nameof(vertices));
        double[] min = [double.MaxValue, double.MaxValue, double.MaxValue], max = [double.MinValue, double.MinValue, double.MinValue];
        for (int i = 0; i < vertices.Length; i += 3)
        {
            for (int c = 0; c < 3; c++)
            {
                min[c] = Math.Min(min[c], vertices[i + c]);
                max[c] = Math.Max(max[c], vertices[i + c]);
            }
        }

        double[] center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
        double scale = Math.Max(max[0] - min[0], Math.Max(max[1] - min[1], max[2] - min[2])) / 2 / 0.8;
        if (!(scale > 0))
            throw new ArgumentException("The mesh has no extent.", nameof(vertices));
        float[] result = new float[vertices.Length];
        for (int i = 0; i < vertices.Length; i += 3)
        {
            for (int c = 0; c < 3; c++)
                result[i + c] = (float)((vertices[i + c] - center[c]) / scale);
        }

        return (result, center, scale);
    }

    /// <summary>The inverse of <see cref="Normalize"/>.</summary>
    public static float[] Denormalize(float[] vertices, double[] center, double scale)
    {
        float[] result = new float[vertices.Length];
        for (int i = 0; i < vertices.Length; i += 3)
        {
            for (int c = 0; c < 3; c++)
                result[i + c] = (float)(vertices[i + c] * scale + center[c]);
        }

        return result;
    }

    /// <summary>The faces with at least one vertex inside <paramref name="box"/> (<c>[minX, minY, minZ, maxX, maxY, maxZ]</c>, inclusive), as upstream selects the faces of a part's box.</summary>
    public static int[] FacesInBox(float[] vertices, int[] indices, ReadOnlySpan<float> box)
    {
        List<int> faces = [];
        for (int f = 0; f < indices.Length / 3; f++)
        {
            for (int k = 0; k < 3; k++)
            {
                int v = indices[f * 3 + k] * 3;
                if (vertices[v] >= box[0] && vertices[v + 1] >= box[1] && vertices[v + 2] >= box[2] && vertices[v] <= box[3] && vertices[v + 1] <= box[4] && vertices[v + 2] <= box[5])
                {
                    faces.Add(f);
                    break;
                }
            }
        }

        return [.. faces];
    }

    /// <summary>The corners of the faces <paramref name="faces"/> as a triangle index list over the same vertices.</summary>
    public static int[] SubIndices(int[] indices, int[] faces)
    {
        int[] result = new int[faces.Length * 3];
        for (int i = 0; i < faces.Length; i++)
            Array.Copy(indices, faces[i] * 3, result, i * 3, 3);
        return result;
    }

    /// <summary>Surface samples as the conditioner takes them: <paramref name="count"/> area-weighted points of the triangles <paramref name="indices"/> (<c>[count, 3]</c>) and their
    /// features <c>[count, 4]</c>: the face normal and a sharp-edge label that is 0 (the release samples no sharp edges: <c>pc_sharpedge_size</c> is 0).</summary>
    public static (float[] Points, float[] Features) SurfaceSamples(float[] vertices, int[] indices, int count, int seed)
    {
        (double[] points, float[] normals, _) = P3SamAutoMask.Sample(vertices, indices, count, new Random(seed));
        float[] xyz = new float[count * 3], features = new float[count * 4];
        for (int i = 0; i < count; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                xyz[i * 3 + c] = (float)points[i * 3 + c];
                features[i * 4 + c] = normals[i * 3 + c];
            }
        }

        return (xyz, features);
    }

    /// <summary>The parts to generate when P3-SAM finds more than the release has embeddings for: the indices of the boxes with at least one face, at most <paramref name="max"/> of them, the ones
    /// holding the most faces (ties by index), returned in ascending order.</summary>
    public static int[] KeptParts(int[] faceCounts, int max) =>
        [.. Enumerable.Range(0, faceCounts.Length).Where(k => faceCounts[k] > 0).OrderByDescending(k => faceCounts[k]).ThenBy(k => k).Take(max).Order()];

    /// <summary>The first <paramref name="count"/> entries of a seeded permutation of <c>0 .. validParts - 1</c> (upstream: <c>torch.randperm(valid_num)[:num_parts]</c>), the ids of the parts' learned embeddings.</summary>
    public static int[] PartIds(int count, int validParts, int seed)
    {
        if (count > validParts)
            throw new ArgumentOutOfRangeException(nameof(count), $"{count} parts do not fit the {validParts} part embeddings of the release.");
        int[] ids = [.. Enumerable.Range(0, validParts)];
        Random random = new(seed);
        for (int i = ids.Length - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (ids[i], ids[j]) = (ids[j], ids[i]);
        }

        return ids[..count];
    }

    /// <summary>A standard-normal sample of <paramref name="count"/> values (the initial latents), Box-Muller from a seeded generator.</summary>
    public static float[] Gaussian(long count, int seed)
    {
        Random random = new(seed);
        float[] values = new float[count];
        for (long i = 0; i < count; i++)
        {
            double u = 1.0 - random.NextDouble(), v = random.NextDouble();
            values[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u)) * Math.Cos(2.0 * Math.PI * v));
        }

        return values;
    }
}
