namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Seeded random subset of point indices, the stand-in for upstream's <c>default_rng(0).choice</c> before farthest-point sampling.</summary>
public static class UniRigPointSelection
{
    /// <summary>
    /// <paramref name="count"/> distinct indices out of <paramref name="total"/> (a partial Fisher–Yates shuffle); when
    /// <paramref name="count"/> exceeds <paramref name="total"/> the draw repeats indices, as upstream's <c>replace=True</c> does.
    /// Not numpy's stream: the subset is statistically equivalent, never identical, to upstream's.
    /// </summary>
    public static int[] Choose(int total, int count, int seed)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(total);
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        Random random = new(seed);
        int[] result = new int[count];
        if (count > total)
        {
            for (int i = 0; i < count; i++)
            {
                result[i] = random.Next(total);
            }

            return result;
        }

        int[] pool = new int[total];
        for (int i = 0; i < total; i++)
        {
            pool[i] = i;
        }

        for (int i = 0; i < count; i++)
        {
            int j = i + random.Next(total - i);
            (pool[i], pool[j]) = (pool[j], pool[i]);
            result[i] = pool[i];
        }

        return result;
    }
}
