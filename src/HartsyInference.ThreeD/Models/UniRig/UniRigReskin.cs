namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>
/// Moves skin weights predicted at sampled points onto the mesh's own vertices, as upstream's <c>reskin</c> does with the
/// <c>median</c> method: the per-channel median of the 7 nearest samples, one diffusion step along mesh edges in subtree-summed
/// space, a small-weight cut-off and renormalization.
/// </summary>
public static class UniRigReskin
{
    private const int NearestSamples = 7;
    private const float Alpha = 2f;
    private const float Threshold = 0.03f;

    /// <returns><c>[N, J]</c> weights, each row summing to 1.</returns>
    /// <param name="samplePoints">Positions of the predicted samples, length <c>3 * S</c>.</param>
    /// <param name="sampleSkin">Predicted weights <c>[S, J]</c>.</param>
    /// <param name="vertices">Mesh vertices, length <c>3 * N</c>.</param>
    /// <param name="faces">Triangles, length <c>3 * F</c>.</param>
    /// <param name="parents">Parent per joint, <c>-1</c> for roots; parents precede children.</param>
    public static float[] Apply(float[] samplePoints, float[] sampleSkin, float[] vertices, int[] faces, int[] parents)
    {
        int s = samplePoints.Length / 3;
        int n = vertices.Length / 3;
        int j = parents.Length;
        if (s < NearestSamples || sampleSkin.Length != s * j)
        {
            throw new ArgumentException($"Reskin needs at least {NearestSamples} samples and {j} weights per sample.");
        }

        float[] skin = MedianOfNearest(samplePoints, sampleSkin, vertices, n, j);
        float[] sum = new float[n * j];
        Array.Copy(skin, sum, skin.Length);
        for (int i = j - 1; i >= 0; i--)
        {
            if (parents[i] >= 0)
            {
                for (int v = 0; v < n; v++)
                {
                    sum[v * j + parents[i]] += sum[v * j + i];
                }
            }
        }

        float[] neighbour = new float[n * j];
        float[] coefficient = new float[n * j];
        for (int f = 0; f < faces.Length / 3; f++)
        {
            for (int k = 0; k < 3; k++)
            {
                int a = faces[3 * f + k];
                int b = faces[3 * f + (k + 1) % 3];
                Diffuse(vertices, sum, neighbour, coefficient, a, b, j);
                Diffuse(vertices, sum, neighbour, coefficient, b, a, j);
            }
        }

        for (int v = 0; v < n; v++)
        {
            for (int c = 0; c < j; c++)
            {
                sum[v * j + c] = (sum[v * j + c] + neighbour[v * j + c]) / (1f + coefficient[v * j + c]);
            }
        }

        for (int i = 0; i < j; i++)
        {
            if (parents[i] >= 0)
            {
                for (int v = 0; v < n; v++)
                {
                    sum[v * j + parents[i]] -= sum[v * j + i];
                }
            }
        }

        Normalize(sum, n, j);
        for (int v = 0; v < n; v++)
        {
            bool any = false;
            for (int c = 0; c < j; c++)
            {
                any |= sum[v * j + c] >= Threshold;
            }

            for (int c = 0; c < j && any; c++)
            {
                if (sum[v * j + c] < Threshold)
                {
                    sum[v * j + c] = 0f;
                }
            }
        }

        Normalize(sum, n, j);
        return sum;
    }

    private static float[] MedianOfNearest(float[] samplePoints, float[] sampleSkin, float[] vertices, int n, int j)
    {
        UniRigKdTree tree = new(samplePoints);
        float[] skin = new float[n * j];
        Parallel.For(0, n, () => (new int[NearestSamples], new float[NearestSamples], new float[NearestSamples]), (v, _, local) =>
        {
            (int[] idx, float[] dis, float[] values) = local;
            tree.Nearest(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2], NearestSamples, idx, dis);
            for (int c = 0; c < j; c++)
            {
                for (int q = 0; q < NearestSamples; q++)
                {
                    values[q] = sampleSkin[idx[q] * j + c];
                }

                Array.Sort(values);
                skin[v * j + c] = values[NearestSamples / 2];
            }

            return local;
        }, _ => { });
        return skin;
    }

    /// <summary>Hot-to-cool transfer along one directed edge: <paramref name="from"/> gives to <paramref name="to"/> only in channels where it holds more.</summary>
    private static void Diffuse(float[] vertices, float[] sum, float[] neighbour, float[] coefficient, int from, int to, int j)
    {
        float dx = vertices[3 * to] - vertices[3 * from];
        float dy = vertices[3 * to + 1] - vertices[3 * from + 1];
        float dz = vertices[3 * to + 2] - vertices[3 * from + 2];
        float co = MathF.Exp(-MathF.Sqrt(dx * dx + dy * dy + dz * dz) * Alpha);
        for (int c = 0; c < j; c++)
        {
            if (sum[to * j + c] < sum[from * j + c])
            {
                neighbour[to * j + c] += sum[from * j + c] * co;
                coefficient[to * j + c] += co;
            }
        }
    }

    private static void Normalize(float[] rows, int n, int j)
    {
        for (int v = 0; v < n; v++)
        {
            float total = 0f;
            for (int c = 0; c < j; c++)
            {
                total += rows[v * j + c];
            }

            for (int c = 0; c < j; c++)
            {
                rows[v * j + c] = total > 0f ? rows[v * j + c] / total : 1f / j;
            }
        }
    }
}
