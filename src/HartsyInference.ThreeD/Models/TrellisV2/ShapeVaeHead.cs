using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>
/// Turns the raw shape-VAE decoder head into a dual-grid field. Channels 0–2 are the dual-vertex offsets (sigmoid into
/// the voxel), 3–5 the edge intersection logits (positive means intersected), and 6 the quad split weight (softplus).
/// </summary>
/// <remarks>The split follows TRELLIS.2's FlexiDualGrid decode as recalled from upstream; it is not checked against a Python
/// reference dump in this repository.</remarks>
public static class ShapeVaeHead
{
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
        float[] dual = new float[n * 3];
        bool[] intersected = new bool[n * 3];
        float[] split = new float[n];
        ReadOnlySpan<float> h = head.Feats.AsReadOnlySpan<float>();
        for (int i = 0; i < n; i++)
        {
            for (int a = 0; a < 3; a++)
            {
                int x = head.Coords[i * 4 + 1 + a];
                coords[i * 3 + a] = x;
                dual[i * 3 + a] = x + Sigmoid(h[i * 7 + a]);
                intersected[i * 3 + a] = h[i * 7 + 3 + a] > 0f;
            }

            split[i] = Softplus(h[i * 7 + 6]);
        }

        return new DualGridField(new DualGridVoxels(coords, dual, intersected), split);
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float Softplus(float x) => x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
}
