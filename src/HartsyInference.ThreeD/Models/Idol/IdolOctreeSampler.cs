namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>Places Idol's anchor points by descending an octree (upstream <c>OctreeProbabilityFixedlenDecoder.sample</c>): at each level the parents'
/// counts are split among their eight children by systematic resampling of the decoder's softmax, children that received points are kept, and
/// the final points land uniformly inside their leaf cells. The draws come from the supplied <see cref="Random"/>, not torch's generator, so a
/// run is reproducible by seed but not equal to the reference's.</summary>
public static class IdolOctreeSampler
{
    /// <summary>The logits <c>[parents, 8]</c> of parent cells given their centres (<c>parents × 3</c>) and the resolution (2 to the level).</summary>
    public delegate float[] LogitsFunction(ReadOnlySpan<float> centres, int resolution);

    /// <summary>Samples <paramref name="numPoints"/> anchors down to <paramref name="level"/>; returns <c>numPoints × 3</c> coordinates in [0, 1].</summary>
    public static float[] Sample(LogitsFunction logits, int numPoints, int level, Random random)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(random);
        if (numPoints <= 0)
            throw new ArgumentOutOfRangeException(nameof(numPoints), numPoints, "The number of anchors must be positive.");

        // Child order of upstream's offsets: [[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)].
        int[] childX = new int[8], childY = new int[8], childZ = new int[8];
        for (int c = 0; c < 8; c++)
        {
            childX[c] = c & 1;
            childY[c] = (c >> 1) & 1;
            childZ[c] = (c >> 2) & 1;
        }

        Span<float> probs = stackalloc float[8];
        Span<int> split = stackalloc int[8];
        int parents = 1;
        int[] coords = new int[3];
        int[] counts = [numPoints];
        for (int l = 1; l <= level; l++)
        {
            float[] centres = new float[parents * 3];
            float parentCells = 1 << (l - 1);
            for (int p = 0; p < parents * 3; p++)
                centres[p] = (coords[p] + 0.5f) / parentCells;
            float[] raw = logits(centres, 1 << l);
            if (raw.Length != parents * 8)
                throw new InvalidOperationException($"The octree decoder returned {raw.Length} logits for {parents} parents.");

            List<int> nextCoords = new(parents * 6), nextCounts = new(parents * 2);
            for (int p = 0; p < parents; p++)
            {
                Softmax(raw.AsSpan(p * 8, 8), probs);
                SystematicSample(probs, counts[p], random, split);
                for (int c = 0; c < 8; c++)
                {
                    if (split[c] <= 0)
                        continue;
                    nextCoords.Add(coords[p * 3] * 2 + childX[c]);
                    nextCoords.Add(coords[p * 3 + 1] * 2 + childY[c]);
                    nextCoords.Add(coords[p * 3 + 2] * 2 + childZ[c]);
                    nextCounts.Add(split[c]);
                }
            }

            coords = [.. nextCoords];
            counts = [.. nextCounts];
            parents = counts.Length;
        }

        float[] points = new float[numPoints * 3];
        float cells = 1 << level;
        int at = 0;
        for (int p = 0; p < parents; p++)
        {
            for (int n = 0; n < counts[p]; n++)
            {
                for (int a = 0; a < 3; a++)
                    points[at++] = (coords[p * 3 + a] + (float)random.NextDouble()) / cells;
            }
        }

        return points;
    }

    private static void Softmax(ReadOnlySpan<float> logits, Span<float> probs)
    {
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
            max = Math.Max(max, logits[i]);
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = MathF.Exp(logits[i] - max);
            sum += probs[i];
        }

        for (int i = 0; i < logits.Length; i++)
            probs[i] /= sum;
    }

    /// <summary>Splits <paramref name="count"/> items among the bins of <paramref name="probs"/> by systematic resampling: a random offset in
    /// [0, 1/count) and <c>count</c> evenly spaced points through the cumulative distribution (capped just below 1).</summary>
    internal static void SystematicSample(ReadOnlySpan<float> probs, int count, Random random, Span<int> result)
    {
        result.Clear();
        if (count <= 0)
            return;
        Span<double> cdf = stackalloc double[probs.Length];
        double total = 0;
        foreach (float p in probs)
            total += Math.Max(p, 0f);
        double running = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            running += total > 0 ? Math.Max(probs[i], 0f) / total : 1.0 / probs.Length;
            cdf[i] = Math.Min(running, 1.0 - 1e-12);
        }

        double offset = random.NextDouble() / count;
        int bin = 0;
        for (int i = 0; i < count; i++)
        {
            double u = Math.Min(offset + (double)i / count, 1.0 - 1e-12);
            while (bin < probs.Length - 1 && cdf[bin] < u)
                bin++;
            result[bin]++;
        }
    }
}
