using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>
/// Input transform of the CoLa3D feature path. Voxel indices map to metres at 0.01 per cell, then each sample is centred
/// in x and y and set to start at z = 0. Features are <c>[centred xyz, color, normal]</c>; color and normal default to zero.
/// </summary>
internal static class UtoniaDirectGrid
{
    /// <summary>Metres per voxel index.</summary>
    public const float CellSize = 0.01f;

    public const int FeatureChannels = 9;

    /// <summary>Builds the raw input point. <paramref name="coords"/> is <c>[N, 4]</c> rows of (batch, x, y, z), grouped by batch.</summary>
    public static unsafe UtoniaPoint Build(ReadOnlySpan<int> coords, int batchSize, float[]? colors, float[]? normals)
    {
        if (coords.Length % 4 != 0)
        {
            throw new HartsyInferenceException("Utonia coords must be rows of (batch, x, y, z).");
        }

        int n = coords.Length / 4;
        if (colors is not null && colors.Length != n * 3)
        {
            throw new HartsyInferenceException($"Utonia colors need {n * 3} values, got {colors.Length}.");
        }

        if (normals is not null && normals.Length != n * 3)
        {
            throw new HartsyInferenceException($"Utonia normals need {n * 3} values, got {normals.Length}.");
        }

        float[] coord = new float[n * 3];
        int[] gridCoord = new int[n * 3];
        int[] batch = new int[n];
        Tensor feat = new(new TensorShape(n, FeatureChannels), DType.F32);
        float* f = (float*)feat.DataPointer;
        int start = 0, previousBatch = -1;
        while (start < n)
        {
            int b = coords[start * 4];
            if (b < 0 || b >= batchSize || b <= previousBatch)
            {
                feat.Dispose();
                throw new HartsyInferenceException("Utonia coords must be grouped by increasing batch index within [0, batchSize).");
            }

            previousBatch = b;
            int end = start;
            while (end < n && coords[end * 4] == b)
            {
                end++;
            }

            float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue;
            for (int i = start; i < end; i++)
            {
                for (int a = 0; a < 3; a++)
                {
                    int idx = coords[i * 4 + 1 + a];
                    if (idx < 0)
                    {
                        feat.Dispose();
                        throw new HartsyInferenceException("Utonia voxel indices must be non-negative.");
                    }

                    coord[i * 3 + a] = idx * CellSize;
                }

                minX = Math.Min(minX, coord[i * 3]);
                maxX = Math.Max(maxX, coord[i * 3]);
                minY = Math.Min(minY, coord[i * 3 + 1]);
                maxY = Math.Max(maxY, coord[i * 3 + 1]);
                minZ = Math.Min(minZ, coord[i * 3 + 2]);
            }

            float shiftX = (minX + maxX) / 2f, shiftY = (minY + maxY) / 2f, shiftZ = minZ;
            for (int i = start; i < end; i++)
            {
                coord[i * 3] -= shiftX;
                coord[i * 3 + 1] -= shiftY;
                coord[i * 3 + 2] -= shiftZ;
                batch[i] = b;
                for (int a = 0; a < 3; a++)
                {
                    gridCoord[i * 3 + a] = coords[i * 4 + 1 + a];
                    f[(long)i * FeatureChannels + a] = coord[i * 3 + a];
                    f[(long)i * FeatureChannels + 3 + a] = colors is null ? 0f : colors[i * 3 + a];
                    f[(long)i * FeatureChannels + 6 + a] = normals is null ? 0f : normals[i * 3 + a];
                }
            }

            start = end;
        }

        return new UtoniaPoint(feat, coord, gridCoord, batch);
    }
}
