namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Turns detokenized joint heads and parent positions into parents and tails, as upstream's <c>make_skeleton</c> does for this tokenizer.</summary>
internal static class UniRigSkeletonBuilder
{
    private const float ExtrudeScale = 0.5f;
    private const float MinLength = 1e-9f;

    /// <summary>Resolves each joint's parent as the nearest earlier joint to its parent position, then fills tails.</summary>
    /// <param name="joints">Heads, length <c>3 * J</c>.</param>
    /// <param name="parentJoints">Parent positions, length <c>3 * J</c>.</param>
    /// <param name="tailsFromSequence">Tail set by a following consecutive joint, keyed by joint index.</param>
    /// <param name="parts">Part tokens in sequence order.</param>
    public static UniRigSkeleton Build(float[] joints, float[] parentJoints, Dictionary<int, float[]> tailsFromSequence, string?[] parts)
    {
        int count = joints.Length / 3;
        int[] parents = new int[count];
        parents[0] = -1;
        for (int i = 1; i < count; i++)
        {
            float best = 999999f;
            int parent = -1;
            for (int j = i - 1; j >= 0; j--)
            {
                float distance = SquaredDistance(joints, 3 * j, parentJoints, 3 * i);
                if (distance < best)
                {
                    best = distance;
                    parent = j;
                }
            }

            parents[i] = parent;
        }

        int[] childCount = new int[count];
        int[] onlyChild = new int[count];
        for (int i = 1; i < count; i++)
        {
            childCount[parents[i]]++;
            onlyChild[parents[i]] = i;
        }

        float[][] tails = new float[count][];
        foreach (KeyValuePair<int, float[]> pair in tailsFromSequence)
        {
            tails[pair.Key] = pair.Value;
        }

        for (int i = 0; i < count; i++)
        {
            if (childCount[i] != 0)
            {
                continue;
            }

            float[] direction = parents[i] < 0 ? [0f, 0f, 1f] : Difference(joints, 3 * i, 3 * parents[i]);
            if (Length(direction) <= MinLength)
            {
                direction = [0f, 0f, 1f];
            }

            tails[i] = Offset(joints, 3 * i, direction, ExtrudeScale);
        }

        for (int i = 0; i < count; i++)
        {
            if (childCount[i] <= 1)
            {
                continue;
            }

            float[] direction;
            if (parents[i] < 0)
            {
                float mean = 0f;
                for (int c = 1; c < count; c++)
                {
                    if (parents[c] == i)
                    {
                        mean += Length(Difference(joints, 3 * i, 3 * c));
                    }
                }

                mean /= childCount[i];
                direction = [0f, 0f, mean];
            }
            else
            {
                direction = Difference(joints, 3 * i, 3 * parents[i]);
                if (Length(direction) <= MinLength)
                {
                    direction = [0f, 0f, 1f];
                }
            }

            tails[i] = Offset(joints, 3 * i, direction, ExtrudeScale);
        }

        float[] flatTails = new float[3 * count];
        for (int i = 0; i < count; i++)
        {
            float[] tail = childCount[i] == 1 ? [joints[3 * onlyChild[i]], joints[3 * onlyChild[i] + 1], joints[3 * onlyChild[i] + 2]] : tails[i];
            Array.Copy(tail, 0, flatTails, 3 * i, 3);
        }

        string[] names = new string[count];
        for (int i = 0; i < count; i++)
        {
            names[i] = $"bone_{i}";
        }

        return new UniRigSkeleton { Joints = joints, Tails = flatTails, Parents = parents, Names = names, Parts = parts };
    }

    private static float SquaredDistance(float[] a, int ai, float[] b, int bi)
    {
        float dx = a[ai] - b[bi];
        float dy = a[ai + 1] - b[bi + 1];
        float dz = a[ai + 2] - b[bi + 2];
        return dx * dx + dy * dy + dz * dz;
    }

    private static float[] Difference(float[] points, int to, int from) =>
        [points[to] - points[from], points[to + 1] - points[from + 1], points[to + 2] - points[from + 2]];

    private static float Length(float[] v) => MathF.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

    private static float[] Offset(float[] points, int at, float[] direction, float scale) =>
        [points[at] + direction[0] * scale, points[at + 1] + direction[1] * scale, points[at + 2] + direction[2] * scale];
}
