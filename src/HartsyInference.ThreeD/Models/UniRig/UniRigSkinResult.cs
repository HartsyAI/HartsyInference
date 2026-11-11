namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>Skin weights of every mesh vertex for every joint of a skeleton.</summary>
public sealed record UniRigSkinResult
{
    /// <summary>Weights <c>[VertexCount, JointCount]</c>, row-major; each row is non-negative and sums to 1.</summary>
    public required float[] Weights { get; init; }

    public required int VertexCount { get; init; }

    public required int JointCount { get; init; }

    /// <summary>
    /// Keeps each vertex's <paramref name="maxInfluences"/> largest weights and renormalizes them, as upstream's export does
    /// (<c>group_per_vertex 4</c>). Returns joint indices and weights, <c>maxInfluences</c> per vertex, largest first; unused slots are weight 0 on joint 0.
    /// </summary>
    public (int[] Joints, float[] Weights) Limit(int maxInfluences = 4)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxInfluences, 1);
        int[] joints = new int[VertexCount * maxInfluences];
        float[] weights = new float[VertexCount * maxInfluences];
        int[] order = new int[JointCount];
        for (int v = 0; v < VertexCount; v++)
        {
            for (int j = 0; j < JointCount; j++)
            {
                order[j] = j;
            }

            ReadOnlySpan<float> row = Weights.AsSpan(v * JointCount, JointCount);
            int keep = Math.Min(maxInfluences, JointCount);
            for (int k = 0; k < keep; k++)
            {
                int best = k;
                for (int j = k + 1; j < JointCount; j++)
                {
                    if (row[order[j]] > row[order[best]])
                    {
                        best = j;
                    }
                }

                (order[k], order[best]) = (order[best], order[k]);
            }

            float total = 0f;
            for (int k = 0; k < keep; k++)
            {
                total += row[order[k]];
            }

            for (int k = 0; k < keep; k++)
            {
                joints[v * maxInfluences + k] = order[k];
                weights[v * maxInfluences + k] = total > 0f ? row[order[k]] / total : (k == 0 ? 1f : 0f);
            }
        }

        return (joints, weights);
    }
}
