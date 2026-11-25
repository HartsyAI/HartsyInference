namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Expands the sparse Duo visibility rectangles into the additive F32 score bias that the attention layer adds before softmax.
/// <see cref="Build"/> is the whole matrix and costs 4 bytes per query-key pair, so it is for small layouts and tests. The production
/// path uses <see cref="BuildRows"/> one query tile at a time, which keeps the bias at tile height rather than full height.</summary>
public static class DuoAttentionBias
{
    /// <summary>Row-major <c>[queryLength, keyLength]</c> bias: 0 where a query may read a key, negative infinity elsewhere.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A rectangle leaves the matrix.</exception>
    /// <exception cref="InvalidOperationException">A query row sees no key; its softmax would be undefined.</exception>
    public static float[] Build(IReadOnlyList<DuoMaskRect> rects, int queryLength, int keyLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(queryLength);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(keyLength);
        ValidateRows(rects, queryLength, keyLength);
        return BuildRows(rects, queryLength, 0, queryLength, keyLength);
    }

    /// <summary>Rows <c>[rowStart, rowStart + rowCount)</c> of the <c>[queryLength, keyLength]</c> bias, row-major
    /// <c>[rowCount, keyLength]</c>. Rectangles are checked against the full matrix, so a tile sees the same mask as <see cref="Build"/>.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A rectangle leaves the matrix, or the tile leaves the query range.</exception>
    public static float[] BuildRows(IReadOnlyList<DuoMaskRect> rects, int queryLength, int rowStart, int rowCount, int keyLength)
    {
        ArgumentNullException.ThrowIfNull(rects);
        if (rowStart < 0 || rowCount <= 0 || rowStart + rowCount > queryLength)
        {
            throw new ArgumentOutOfRangeException(nameof(rowStart), rowStart, $"Rows [{rowStart}, {rowStart + rowCount}) leave a {queryLength}-row mask.");
        }
        float[] bias = new float[checked(rowCount * keyLength)];
        Array.Fill(bias, float.NegativeInfinity);
        foreach (DuoMaskRect rect in rects)
        {
            CheckRect(rect, queryLength, keyLength, rects);
            int from = Math.Max(rect.QStart, rowStart), to = Math.Min(rect.QEnd, rowStart + rowCount);
            for (int q = from; q < to; q++)
            {
                Array.Fill(bias, 0f, (q - rowStart) * keyLength + rect.KStart, rect.KEnd - rect.KStart);
            }
        }
        return bias;
    }

    /// <summary>Checks every rectangle and that every query row sees a key. O(rows + rectangles), no matrix.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A rectangle leaves the matrix.</exception>
    /// <exception cref="InvalidOperationException">A query row sees no key.</exception>
    public static void ValidateRows(IReadOnlyList<DuoMaskRect> rects, int queryLength, int keyLength)
    {
        ArgumentNullException.ThrowIfNull(rects);
        bool[] seen = new bool[queryLength];
        foreach (DuoMaskRect rect in rects)
        {
            CheckRect(rect, queryLength, keyLength, rects);
            for (int q = rect.QStart; q < rect.QEnd; q++)
            {
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
    }

    private static void CheckRect(DuoMaskRect rect, int queryLength, int keyLength, IReadOnlyList<DuoMaskRect> rects)
    {
        if (rect.QStart < 0 || rect.QEnd > queryLength || rect.QStart >= rect.QEnd
            || rect.KStart < 0 || rect.KEnd > keyLength || rect.KStart >= rect.KEnd)
        {
            throw new ArgumentOutOfRangeException(nameof(rects),
                $"Rectangle [{rect.QStart},{rect.QEnd})x[{rect.KStart},{rect.KEnd}) leaves a {queryLength}x{keyLength} mask.");
        }
    }
}
