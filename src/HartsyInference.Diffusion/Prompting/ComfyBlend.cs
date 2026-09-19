using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>SwarmUI's <see cref="PromptWeightingMode.ComfyBlend"/> mechanism: the weights survive tokenization, so
/// ComfyUI interpolates each token's encoder OUTPUT away from the empty-prompt baseline —
/// <c>z = (z − z_empty)·w + z_empty</c> (<c>comfy/sd1_clip.py:54-63</c>). <see cref="EmphasisMath.ApplyComfy"/> is the
/// host implementation the CLIP encoders already use; this is the tensor/backend form for the LLM and T5 encoders,
/// whose conditioning is produced as a single tensor rather than fixed 77-token chunks.</summary>
public static class ComfyBlend
{
    /// <summary>Blends <paramref name="hidden"/> toward <paramref name="empty"/> by a per-token weight. Returns null —
    /// and touches nothing — when every weight is 1, which is what keeps an unweighted prompt byte-identical.</summary>
    /// <param name="empty">The SAME encoder run on this family's empty-prompt token batch, identical in shape to
    /// <paramref name="hidden"/> and produced with the same layer selection.</param>
    /// <returns>A new tensor the caller owns, or null when there was nothing to apply.</returns>
    public static Tensor? Apply(IBackend backend, Tensor hidden, Tensor empty, ReadOnlySpan<float> weights)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(hidden);
        ArgumentNullException.ThrowIfNull(empty);
        if (hidden.DType != DType.F32 || empty.DType != DType.F32)
        {
            throw new ArgumentException($"ComfyBlend needs F32 conditioning, got {hidden.DType}/{empty.DType}.");
        }
        if (!hidden.Shape.Equals(empty.Shape))
        {
            throw new ArgumentException(
                $"The empty-prompt baseline must match the conditioning shape: {hidden.Shape} vs {empty.Shape}.", nameof(empty));
        }
        if (hidden.Shape.Rank < 2)
        {
            throw new ArgumentException($"Conditioning must be at least rank 2, got rank {hidden.Shape.Rank}.", nameof(hidden));
        }
        int rows = (int)hidden.Shape[hidden.Shape.Rank - 2];
        int batch = (int)(hidden.ElementCount / (rows * hidden.Shape[hidden.Shape.Rank - 1]));
        if (weights.Length != rows)
        {
            throw new ArgumentException($"Expected {rows} token weights for the sequence, got {weights.Length}.", nameof(weights));
        }
        // Held as (w - 1) so the algebraically identical z + (z - zEmpty)·(w - 1) form can be used below.
        float[] rowScales = new float[checked(batch * rows)];
        bool any = false;
        for (int i = 0; i < rows; i++)
        {
            if (weights[i] == 1f)
            {
                continue;
            }
            for (int b = 0; b < batch; b++)
            {
                rowScales[(b * rows) + i] = weights[i] - 1f;
            }
            any = true;
        }
        if (!any)
        {
            return null;
        }
        using Tensor mask = new Tensor(new TensorShape(rowScales.Length), DType.F32);
        rowScales.CopyTo(mask.AsSpan<float>());
        using Tensor negatedEmpty = new Tensor(hidden.Shape, DType.F32);
        using Tensor delta = new Tensor(hidden.Shape, DType.F32);
        using Tensor weightedDelta = new Tensor(hidden.Shape, DType.F32);
        Tensor result = new Tensor(hidden.Shape, DType.F32);
        try
        {
            // (z - zEmpty)·(w - 1) + z rather than the reference's (z - zEmpty)·w + zEmpty: identical for w != 1, and
            // for w == 1 the delta is multiplied by exactly 0, so those rows come back bit-for-bit unchanged.
            backend.Scale(negatedEmpty, empty, -1f);
            backend.Add(delta, hidden, negatedEmpty);
            backend.MaskRows(weightedDelta, delta, mask);
            backend.Add(result, hidden, weightedDelta);
            // Same reason as CondTokenWeights.ScaleRightAligned: land it on the host before the encoder stage's
            // activation sweep can reclaim a device buffer the denoise loop still needs.
            result.AsReadOnlySpan<float>();
        }
        catch
        {
            result.Dispose();
            throw;
        }
        return result;
    }
}
