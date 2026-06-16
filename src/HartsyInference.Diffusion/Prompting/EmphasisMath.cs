namespace HartsyInference.Diffusion.Prompting;

/// <summary>The ComfyUI "comfy" emphasis math applied to encoded CLIP hidden states. Each token is interpolated away from the empty-prompt baseline by its weight: <c>z = (z - zEmpty) * weight + zEmpty</c>. Isolated here so it can be reference-validated against ComfyUI's <c>encode_token_weights</c> independent of the encoder.</summary>
public static class EmphasisMath
{
    /// <summary>In-place weighting of <paramref name="hidden"/> (<c>[seq*dim]</c>) against <paramref name="empty"/> (<c>[seq*dim]</c>) using a per-token <paramref name="weights"/> (<c>[seq]</c>).</summary>
    public static void ApplyComfy(Span<float> hidden, ReadOnlySpan<float> empty, ReadOnlySpan<float> weights, int seq, int dim)
    {
        if (hidden.Length != seq * dim || empty.Length != seq * dim)
        {
            throw new ArgumentException($"hidden/empty length must be seq*dim = {seq * dim}.");
        }
        if (weights.Length != seq)
        {
            throw new ArgumentException($"weights length must be seq = {seq}.", nameof(weights));
        }
        for (int i = 0; i < seq; i++)
        {
            float w = weights[i];
            if (w == 1f)
            {
                continue;
            }
            int off = i * dim;
            for (int d = 0; d < dim; d++)
            {
                float e = empty[off + d];
                hidden[off + d] = (hidden[off + d] - e) * w + e;
            }
        }
    }
}
