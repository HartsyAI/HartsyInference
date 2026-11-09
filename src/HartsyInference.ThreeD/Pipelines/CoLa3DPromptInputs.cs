using System.Numerics;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.CoLa3D;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>
/// Turns a scene prompt into the decomposer's query inputs for one scene latent. Masks become an F32 tensor. Points map to
/// the latent voxel that holds them, then snap to the nearest occupied latent voxel, as the reference does for a
/// ground-truth prompt. Owns the mask tensor it builds.
/// </summary>
internal sealed class CoLa3DPromptInputs : IDisposable
{
    private readonly Tensor? _masks;
    private bool _disposed;

    private CoLa3DPromptInputs(CoLa3DQueryPrompts prompts, Tensor? masks)
    {
        Prompts = prompts;
        _masks = masks;
    }

    /// <summary>Decomposer inputs for the prompt, borrowing the mask tensor this object owns.</summary>
    internal CoLa3DQueryPrompts Prompts { get; }

    /// <summary>Validates <paramref name="prompt"/> and builds its decomposer inputs on <paramref name="latent"/>, which holds one scene.</summary>
    internal static CoLa3DPromptInputs Create(CoLa3DPrompt prompt, SceneGrid grid, SparseTensor latent)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(grid);
        ArgumentNullException.ThrowIfNull(latent);
        prompt.Validate();
        return prompt.MaskCoverage is not null
            ? CreateMasks(prompt, prompt.MaskCoverage)
            : CreatePoints(prompt, grid, latent);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _masks?.Dispose();
    }

    private static CoLa3DPromptInputs CreateMasks(CoLa3DPrompt prompt, byte[] coverage)
    {
        Tensor masks = new(new TensorShape(1, prompt.Queries, prompt.MaskHeight, prompt.MaskWidth), DType.F32);
        Span<float> values = masks.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = coverage[i] / 255f;
        CoLa3DQueryPrompts queries = new() { Batch = 1, Queries = prompt.Queries, Masks = masks };
        return new CoLa3DPromptInputs(queries, masks);
    }

    private static CoLa3DPromptInputs CreatePoints(CoLa3DPrompt prompt, SceneGrid grid, SparseTensor latent)
    {
        float[] world = prompt.Points!;
        int[] points = new int[prompt.Queries * 3];
        for (int q = 0; q < prompt.Queries; q++)
        {
            Vector3 voxel = grid.ToVoxelUnits(new Vector3(world[q * 3], world[q * 3 + 1], world[q * 3 + 2]));
            int x = LatentIndex(voxel.X, grid.Resolution, q);
            int y = LatentIndex(voxel.Y, grid.Resolution, q);
            int z = LatentIndex(voxel.Z, grid.Resolution, q);
            Nearest(latent, x, y, z, out int nx, out int ny, out int nz);
            points[q * 3] = nx;
            points[q * 3 + 1] = ny;
            points[q * 3 + 2] = nz;
        }

        CoLa3DQueryPrompts queries = new() { Batch = 1, Queries = prompt.Queries, Points = points };
        return new CoLa3DPromptInputs(queries, null);
    }

    // The dual-grid voxel containing a point, then its latent voxel: the shape VAE downsamples by LatentStride.
    private static int LatentIndex(float dualCoordinate, int dualResolution, int query)
    {
        float floor = MathF.Floor(dualCoordinate);
        if (!(floor >= 0f && floor < dualResolution))
            throw new HartsyInferenceException($"Prompt point {query} lies outside the scene grid.");
        return (int)floor / CoLa3DPipeline.LatentStride;
    }

    // The occupied latent voxel with the least L1 distance; an exact hit is distance zero and stops the scan.
    private static void Nearest(SparseTensor latent, int x, int y, int z, out int nx, out int ny, out int nz)
    {
        int[] coords = latent.Coords;
        int best = int.MaxValue;
        nx = x;
        ny = y;
        nz = z;
        for (int r = 0; r < latent.Count && best != 0; r++)
        {
            int cx = coords[r * 4 + 1], cy = coords[r * 4 + 2], cz = coords[r * 4 + 3];
            int distance = Math.Abs(cx - x) + Math.Abs(cy - y) + Math.Abs(cz - z);
            if (distance < best)
            {
                best = distance;
                nx = cx;
                ny = cy;
                nz = cz;
            }
        }
    }
}
