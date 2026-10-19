namespace HartsyInference.Core.Rope;

/// <summary>Expands compact per-pair cos/sin into the full <c>[tokens, headDim]</c> rows each apply op reads.</summary>
/// <remarks>Input holds <c>effectiveRotaryDim / 2</c> pairs per token. Destinations are written in full, so they may be
/// uninitialized. Partial rotary leaves the entries a layout does not read to the padding policy.</remarks>
public static class RopeTableExpander
{
    /// <summary>Rotated width for a request. <paramref name="rotaryDim"/> 0 means the full head.</summary>
    public static int EffectiveRotaryDim(int headDim, int rotaryDim, RopeTableLayout layout)
    {
        if (headDim <= 0 || (headDim & 1) != 0)
            throw new ArgumentOutOfRangeException(nameof(headDim), headDim, "RoPE head dimension must be positive and even.");
        if (rotaryDim == 0)
            return headDim;
        if (rotaryDim < 2 || rotaryDim > headDim || (rotaryDim & 1) != 0)
            throw new ArgumentOutOfRangeException(nameof(rotaryDim), rotaryDim, "Partial rotary must be an even width in [2, headDim].");
        if (layout == RopeTableLayout.InterleavedPair && rotaryDim != headDim)
            throw new ArgumentException("Duplicated-pair tables only support full-width rotary.", nameof(layout));
        return rotaryDim;
    }

    /// <summary>Writes <paramref name="tokens"/> rows of <paramref name="headDim"/> entries into both destinations.</summary>
    public static void Expand(
        ReadOnlySpan<float> cos,
        ReadOnlySpan<float> sin,
        int tokens,
        int headDim,
        RopeTableLayout layout,
        int rotaryDim,
        RopeTablePadding padding,
        Span<float> cosRows,
        Span<float> sinRows)
    {
        if (tokens < 0)
            throw new ArgumentOutOfRangeException(nameof(tokens), tokens, "Token count cannot be negative.");
        int effective = EffectiveRotaryDim(headDim, rotaryDim, layout);
        int active = effective / 2;
        long inputLength = (long)tokens * active;
        if (cos.Length != inputLength || sin.Length != inputLength)
            throw new ArgumentException($"Compact input must hold {tokens} tokens of {active} pairs.", nameof(cos));
        long rowLength = (long)tokens * headDim;
        if (cosRows.Length != rowLength || sinRows.Length != rowLength)
            throw new ArgumentException($"Destination rows must hold {tokens} tokens of {headDim} channels.", nameof(cosRows));
        if (cos.Overlaps(cosRows) || cos.Overlaps(sinRows) || sin.Overlaps(cosRows) || sin.Overlaps(sinRows)
            || cosRows.Overlaps(sinRows))
            throw new ArgumentException("Input and destination buffers must not overlap.");
        if (tokens == 0)
            return;

        float paddingCos = padding == RopeTablePadding.Identity ? 1f : 0f;
        int written = layout == RopeTableLayout.InterleavedPrefix ? active : effective;
        for (int t = 0; t < tokens; t++)
        {
            int inputBase = t * active;
            int rowBase = t * headDim;
            switch (layout)
            {
                case RopeTableLayout.RotateHalf:
                    for (int i = 0; i < active; i++)
                    {
                        cosRows[rowBase + i] = cos[inputBase + i];
                        sinRows[rowBase + i] = sin[inputBase + i];
                        cosRows[rowBase + active + i] = cos[inputBase + i];
                        sinRows[rowBase + active + i] = sin[inputBase + i];
                    }
                    break;
                case RopeTableLayout.InterleavedPrefix:
                    for (int i = 0; i < active; i++)
                    {
                        cosRows[rowBase + i] = cos[inputBase + i];
                        sinRows[rowBase + i] = sin[inputBase + i];
                    }
                    break;
                case RopeTableLayout.InterleavedPair:
                    for (int i = 0; i < active; i++)
                    {
                        cosRows[rowBase + 2 * i] = cos[inputBase + i];
                        cosRows[rowBase + 2 * i + 1] = cos[inputBase + i];
                        sinRows[rowBase + 2 * i] = sin[inputBase + i];
                        sinRows[rowBase + 2 * i + 1] = sin[inputBase + i];
                    }
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(layout), layout, "Unknown RoPE table layout.");
            }

            for (int j = written; j < headDim; j++)
            {
                cosRows[rowBase + j] = paddingCos;
                sinRows[rowBase + j] = 0f;
            }
        }
    }
}
