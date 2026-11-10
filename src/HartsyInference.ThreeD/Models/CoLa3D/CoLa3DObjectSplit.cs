namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>Assigns scene voxels to decomposed object parts from the mask logits.</summary>
internal static class CoLa3DObjectSplit
{
    /// <summary>Fewest voxels a part needs to count as an object; smaller parts are dropped as noise.</summary>
    internal const int MinObjectVoxels = 8;

    /// <summary>Owner of each voxel: the part with the highest logit when that logit is positive (mask probability above one half), else -1.</summary>
    /// <param name="logits">Row-major <c>[voxels, parts]</c> mask logits.</param>
    internal static int[] AssignOwners(ReadOnlySpan<float> logits, int voxels, int parts)
    {
        if (logits.Length != voxels * parts)
            throw new ArgumentException($"Mask logits hold {logits.Length} values; expected {voxels} x {parts}.", nameof(logits));
        int[] owners = new int[voxels];
        for (int i = 0; i < voxels; i++)
        {
            int best = 0;
            float bestLogit = logits[i * parts];
            for (int k = 1; k < parts; k++)
            {
                float value = logits[i * parts + k];
                if (value > bestLogit)
                {
                    best = k;
                    bestLogit = value;
                }
            }

            owners[i] = bestLogit > 0f ? best : -1;
        }

        return owners;
    }

    /// <summary>Voxel rows of every part with at least <see cref="MinObjectVoxels"/> rows, in part order.</summary>
    internal static List<int[]> GroupRows(int[] owners, int parts)
    {
        int[] counts = new int[parts];
        foreach (int owner in owners)
        {
            if (owner >= 0)
                counts[owner]++;
        }

        List<int[]> groups = [];
        for (int k = 0; k < parts; k++)
        {
            if (counts[k] < MinObjectVoxels)
                continue;
            int[] rows = new int[counts[k]];
            int next = 0;
            for (int i = 0; i < owners.Length; i++)
            {
                if (owners[i] == k)
                    rows[next++] = i;
            }

            groups.Add(rows);
        }

        return groups;
    }
}
