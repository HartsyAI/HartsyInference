using HartsyInference.Core.Exceptions;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>Coordinate bookkeeping between the TRELLIS.2 stages: structure occupancy to voxels, and the coarse shape latent's
/// upsampled voxels to the high-resolution flow's tokens.</summary>
public static class Trellis2Cascade
{
    /// <summary>Upstream's cap on high-resolution tokens (<c>max_num_tokens</c>).</summary>
    public const int DefaultMaxTokens = 49152;

    /// <summary>The resolution below which the cascade never retreats (upstream's literal 1024).</summary>
    public const int FloorResolution = 1024;

    /// <summary>The step by which the cascade retreats when a resolution has too many tokens.</summary>
    public const int ResolutionStep = 128;

    /// <summary>Latent grid cells per voxel edge of the dual grid: the shape VAE's 16x downsample.</summary>
    public const int LatentStride = 16;

    /// <summary>
    /// Voxels <c>(0, x, y, z)</c>, x-major, of the occupied cells of a dense logit grid <c>[R, R, R]</c> (occupied where the logit is
    /// above 0). When <paramref name="targetResolution"/> is coarser, a target cell is occupied if any source cell in its block is,
    /// which is upstream's max-pool followed by <c>&gt; 0.5</c>.
    /// </summary>
    public static int[] OccupiedVoxels(ReadOnlySpan<float> logits, int sourceResolution, int targetResolution)
    {
        if (sourceResolution <= 0 || targetResolution <= 0 || targetResolution > sourceResolution || sourceResolution % targetResolution != 0)
            throw new ArgumentException($"A {sourceResolution} grid cannot be pooled to {targetResolution}.", nameof(targetResolution));
        if (logits.Length != (long)sourceResolution * sourceResolution * sourceResolution)
            throw new ArgumentException($"Expected {sourceResolution}^3 logits; got {logits.Length}.", nameof(logits));
        int ratio = sourceResolution / targetResolution;
        bool[] occupied = new bool[targetResolution * targetResolution * targetResolution];
        int count = 0;
        for (int x = 0; x < sourceResolution; x++)
        {
            for (int y = 0; y < sourceResolution; y++)
            {
                for (int z = 0; z < sourceResolution; z++)
                {
                    if (!(logits[(x * sourceResolution + y) * sourceResolution + z] > 0f))
                        continue;
                    int cell = ((x / ratio) * targetResolution + y / ratio) * targetResolution + z / ratio;
                    if (!occupied[cell])
                    {
                        occupied[cell] = true;
                        count++;
                    }
                }
            }
        }

        int[] coords = new int[count * 4];
        int o = 0;
        for (int x = 0; x < targetResolution; x++)
        {
            for (int y = 0; y < targetResolution; y++)
            {
                for (int z = 0; z < targetResolution; z++)
                {
                    if (!occupied[(x * targetResolution + y) * targetResolution + z])
                        continue;
                    coords[o * 4 + 1] = x;
                    coords[o * 4 + 2] = y;
                    coords[o * 4 + 3] = z;
                    o++;
                }
            }
        }

        return coords;
    }

    /// <summary>
    /// The unique high-resolution flow voxels for a coarse latent upsampled to <paramref name="coarseResolution"/>: each voxel maps to
    /// <c>floor((c + 0.5) / coarseResolution · (hrResolution / 16))</c> per axis and duplicates collapse (the TRELLIS.2 mapping; see
    /// <see cref="Trellis2CascadeQuantizer"/> for the other). Rows are <c>(batch, x, y, z)</c> in lexicographic order, as <c>unique(dim=0)</c> gives.
    /// </summary>
    public static int[] QuantizeUnique(ReadOnlySpan<int> upsampled, int coarseResolution, int hrResolution)
        => QuantizeUnique(upsampled, coarseResolution, hrResolution, Trellis2CascadeQuantizer.Floor);

    /// <summary>As <see cref="QuantizeUnique(ReadOnlySpan{int}, int, int)"/> with the mapping chosen by <paramref name="quantizer"/>.</summary>
    public static int[] QuantizeUnique(ReadOnlySpan<int> upsampled, int coarseResolution, int hrResolution, Trellis2CascadeQuantizer quantizer)
    {
        if (coarseResolution <= 0 || hrResolution < LatentStride || hrResolution % LatentStride != 0)
            throw new ArgumentException($"Invalid cascade resolutions {coarseResolution} -> {hrResolution}.", nameof(hrResolution));
        if (upsampled.Length % 4 != 0)
            throw new ArgumentException("Coordinates must be (batch, x, y, z) rows.", nameof(upsampled));
        int n = upsampled.Length / 4;
        long grid = hrResolution / LatentStride;
        long[] keys = new long[n];
        for (int i = 0; i < n; i++)
        {
            long key = upsampled[i * 4];
            for (int a = 1; a <= 3; a++)
            {
                key = key * 4096 + Quantize(upsampled[i * 4 + a], coarseResolution, grid, quantizer);
            }

            keys[i] = key;
        }

        Array.Sort(keys);
        int unique = 0;
        for (int i = 0; i < n; i++)
        {
            if (i == 0 || keys[i] != keys[i - 1])
                keys[unique++] = keys[i];
        }

        int[] coords = new int[unique * 4];
        for (int i = 0; i < unique; i++)
        {
            long key = keys[i];
            for (int a = 3; a >= 1; a--)
            {
                coords[i * 4 + a] = (int)(key % 4096);
                key /= 4096;
            }

            coords[i * 4] = (int)key;
        }

        return coords;
    }

    // Both forms are exact integer arithmetic where upstream's float32 product is exact: (2c + 1) · cells / (2 · coarse), floored or rounded.
    private static long Quantize(int c, int coarse, long grid, Trellis2CascadeQuantizer quantizer)
    {
        long denominator = 2L * coarse;
        if (quantizer == Trellis2CascadeQuantizer.Floor)
            return (2L * c + 1) * grid / denominator;
        long numerator = (2L * c + 1) * (grid - 1), q = numerator / denominator, twiceRemainder = 2 * (numerator % denominator);
        // torch.round is half to even; for the supported grids (multiples of 8, so grid - 1 is odd) no tie can occur.
        return twiceRemainder > denominator || (twiceRemainder == denominator && (q & 1) == 1) ? q + 1 : q;
    }

    /// <summary>
    /// Upstream's token-cap loop: starting at <paramref name="targetResolution"/>, retreat by <see cref="ResolutionStep"/> until the unique
    /// voxel count is under <paramref name="maxTokens"/> or <see cref="FloorResolution"/> is reached. Returns the voxels and the resolution
    /// the high-resolution flow and the decoder run at.
    /// </summary>
    public static (int[] Coords, int Resolution) FitTokenCap(ReadOnlySpan<int> upsampled, int coarseResolution, int targetResolution, int maxTokens)
        => FitTokenCap(upsampled, coarseResolution, targetResolution, maxTokens, Trellis2CascadeQuantizer.Floor);

    /// <summary>As <see cref="FitTokenCap(ReadOnlySpan{int}, int, int, int)"/> with the mapping chosen by <paramref name="quantizer"/>.</summary>
    public static (int[] Coords, int Resolution) FitTokenCap(
        ReadOnlySpan<int> upsampled, int coarseResolution, int targetResolution, int maxTokens, Trellis2CascadeQuantizer quantizer)
    {
        if (targetResolution < FloorResolution)
            throw new ArgumentOutOfRangeException(
                nameof(targetResolution), targetResolution, $"The cascade target must be at least {FloorResolution}.");
        if ((targetResolution - FloorResolution) % ResolutionStep != 0)
            throw new ArgumentException(
                $"The cascade target {targetResolution} must be the {FloorResolution} floor plus a multiple of {ResolutionStep}.", nameof(targetResolution));
        if (maxTokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxTokens), maxTokens, "The token cap must be positive.");
        int resolution = targetResolution;
        while (true)
        {
            int[] coords = QuantizeUnique(upsampled, coarseResolution, resolution, quantizer);
            if (coords.Length == 0)
                throw new HartsyInferenceException("The upsampled structure has no voxels.");
            if (coords.Length / 4 < maxTokens || resolution == FloorResolution)
                return (coords, resolution);
            resolution -= ResolutionStep;
        }
    }
}
