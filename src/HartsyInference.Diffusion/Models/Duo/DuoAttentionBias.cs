namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Expands the sparse Duo visibility rectangles into the additive F32 score bias that the attention layer adds before softmax.
/// CPU reference and test use only. The dense matrix is 4 bytes per query-key pair: about 4 GiB per call at 32,768 x 32,768 tokens,
/// so the production path must not call <see cref="Build"/> at that size; it needs a tiled or per-block bias.</summary>
public static class DuoAttentionBias
{
    /// <summary>Row-major <c>[queryLength, keyLength]</c> bias: 0 where a query may read a key, negative infinity elsewhere.
    /// Costs O(queryLength * keyLength) memory and time for the matrix itself; the "row sees a key" check is O(queryLength).</summary>
    /// <exception cref="ArgumentOutOfRangeException">A rectangle leaves the matrix.</exception>
    /// <exception cref="InvalidOperationException">A query row sees no key; its softmax would be undefined.</exception>
    public static float[] Build(IReadOnlyList<DuoMaskRect> rects, int queryLength, int keyLength)
    {
        ArgumentNullException.ThrowIfNull(rects);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(queryLength);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(keyLength);
        float[] bias = new float[checked(queryLength * keyLength)];
        Array.Fill(bias, float.NegativeInfinity);
        bool[] seen = new bool[queryLength];
        foreach (DuoMaskRect rect in rects)
        {
            if (rect.QStart < 0 || rect.QEnd > queryLength || rect.QStart >= rect.QEnd
                || rect.KStart < 0 || rect.KEnd > keyLength || rect.KStart >= rect.KEnd)
            {
                throw new ArgumentOutOfRangeException(nameof(rects),
                    $"Rectangle [{rect.QStart},{rect.QEnd})x[{rect.KStart},{rect.KEnd}) leaves a {queryLength}x{keyLength} mask.");
            }
            for (int q = rect.QStart; q < rect.QEnd; q++)
            {
                Array.Fill(bias, 0f, q * keyLength + rect.KStart, rect.KEnd - rect.KStart);
                seen[q] = true;
            }
        }
        for (int q = 0; q < queryLength; q++)
        {
            if (!seen[q])
            {
                throw new InvalidOperationException($"Query row {q} sees no key; the mask is malformed.");
            }
        }
        return bias;
    }
}
