namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Expands the sparse Duo visibility rectangles into the additive F32 score bias that the attention layer adds before softmax.</summary>
public static class DuoAttentionBias
{
    /// <summary>Row-major <c>[queryLength, keyLength]</c> bias: 0 where a query may read a key, negative infinity elsewhere.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A rectangle leaves the matrix.</exception>
    /// <exception cref="InvalidOperationException">A query row sees no key; its softmax would be undefined.</exception>
    public static float[] Build(IReadOnlyList<DuoMaskRect> rects, int queryLength, int keyLength)
    {
        ArgumentNullException.ThrowIfNull(rects);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(queryLength);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(keyLength);
        float[] bias = new float[checked(queryLength * keyLength)];
        Array.Fill(bias, float.NegativeInfinity);
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
            }
        }
        for (int q = 0; q < queryLength; q++)
        {
            bool sees = false;
            for (int k = 0; k < keyLength && !sees; k++)
            {
                sees = bias[q * keyLength + k] == 0f;
            }
            if (!sees)
            {
                throw new InvalidOperationException($"Query row {q} sees no key; the mask is malformed.");
            }
        }
        return bias;
    }
}
