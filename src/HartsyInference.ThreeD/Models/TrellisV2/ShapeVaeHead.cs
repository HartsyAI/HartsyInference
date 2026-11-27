using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// Turns the raw shape-VAE decoder head into a dual-grid field, following TRELLIS.2's FlexiDualGrid decode. Channels 0–2
/// give absolute dual-vertex positions <c>coord + (1 + 2m)·sigmoid(raw) − m</c> with <c>m = 0.5</c>, in voxel units. Channels 3–5 are the edge logits, kept in <see cref="DualGridField.IntersectionLogits"/>; their sign is the flag.
/// Channel 6 is the quad split weight, <c>softplus(raw)</c>.
/// </summary>
public static class ShapeVaeHead
{
    private const float VoxelMargin = 0.5f;

    /// <summary>Splits a decoder head <c>[N, 7]</c> on its coordinates <c>(batch, x, y, z)</c> into a single-batch dual-grid field.</summary>
    public static DualGridField Split(SparseTensor head)
    {
        ArgumentNullException.ThrowIfNull(head);
        int n = head.Count;
        if (head.Feats.DType != DType.F32 || head.Channels != ShapeVaeKeyMap.DecoderOutputChannels
            || head.Feats.Shape.ElementCount != (long)n * ShapeVaeKeyMap.DecoderOutputChannels)
            throw new HartsyInferenceException(
                $"Shape VAE head has shape {head.Feats.Shape}; expected F32 with {n} rows of {ShapeVaeKeyMap.DecoderOutputChannels} channels.");

        if (head.Coords.Length != n * 4)
            throw new HartsyInferenceException($"Shape VAE head coords hold {head.Coords.Length} values; expected {n} rows of 4.");

        int[] coords = new int[n * 3];
        float[] vertices = new float[n * 3];
        float[] logits = new float[n * 3];
        bool[] intersected = new bool[n * 3];
        float[] split = new float[n];
        ReadOnlySpan<float> h = head.Feats.AsReadOnlySpan<float>();
        for (int i = 0; i < n; i++)
        {
            for (int a = 0; a < 3; a++)
            {
                coords[i * 3 + a] = head.Coords[i * 4 + 1 + a];
                vertices[i * 3 + a] = coords[i * 3 + a] + (1f + 2f * VoxelMargin) * Sigmoid(h[i * 7 + a]) - VoxelMargin;
                logits[i * 3 + a] = h[i * 7 + 3 + a];
                intersected[i * 3 + a] = logits[i * 3 + a] > 0f;
            }

            split[i] = Softplus(h[i * 7 + 6]);
        }

        return new DualGridField(new DualGridVoxels(coords, vertices, intersected), split, logits);
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float Softplus(float x) => x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
}
