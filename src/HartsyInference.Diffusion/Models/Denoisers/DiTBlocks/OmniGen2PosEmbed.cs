namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Position-id builders for OmniGen2's 3-axis <c>(time, height, width)</c> RoPE (<c>OmniGen2RotaryPosEmbed</c>).
/// Text token <c>i</c> sits at <c>(i, i, i)</c>, matching diffusers' <c>repeat "l -&gt; l 3"</c>. An image token at packed
/// grid <c>(row, col)</c> sits at <c>(timeOffset, row, col)</c>; the text-to-image path passes the caption length as the
/// offset (<c>pe_shift = cap_seq_len</c>). In the joint sequence the image rows take time <c>txtSeqLen</c>. Rows are
/// token-major, three floats each, the layout <see cref="Core.Rope.MultiAxisRopeTables"/> reads.</summary>
public static class OmniGen2PosEmbed
{
    /// <summary>Axes per token: time, height and width.</summary>
    public const int NumAxes = 3;

    /// <summary>Rows for a text-only sequence: token <c>s</c> at <c>(s, s, s)</c>.</summary>
    public static float[] TextRows(int txtSeqLen)
    {
        float[] rows = new float[checked(txtSeqLen * NumAxes)];
        for (int s = 0; s < txtSeqLen; s++)
        {
            float pos = s;
            rows[s * NumAxes] = pos;
            rows[s * NumAxes + 1] = pos;
            rows[s * NumAxes + 2] = pos;
        }
        return rows;
    }

    /// <summary>Rows for an image-only grid, row-major, at time <paramref name="timeOffset"/>.</summary>
    public static float[] ImageRows(int hPacked, int wPacked, int timeOffset)
    {
        float[] rows = new float[checked(hPacked * wPacked * NumAxes)];
        WriteImage(rows, 0, hPacked * wPacked, wPacked, timeOffset);
        return rows;
    }

    /// <summary>Rows for the joint <c>[text ; image]</c> sequence: text rows <c>(s, s, s)</c>, then image rows at time
    /// <paramref name="txtSeqLen"/>.</summary>
    public static float[] JointRows(int txtSeqLen, int hPacked, int wPacked)
    {
        int imgSeqLen = hPacked * wPacked;
        float[] rows = new float[checked((txtSeqLen + imgSeqLen) * NumAxes)];
        float[] text = TextRows(txtSeqLen);
        text.CopyTo(rows, 0);
        WriteImage(rows, txtSeqLen * NumAxes, imgSeqLen, wPacked, txtSeqLen);
        return rows;
    }

    /// <summary>Rows from explicit per-token ids. The three spans must have equal length.</summary>
    public static float[] FromIds(ReadOnlySpan<int> timeIds, ReadOnlySpan<int> heightIds, ReadOnlySpan<int> widthIds)
    {
        int seqLen = timeIds.Length;
        float[] rows = new float[checked(seqLen * NumAxes)];
        for (int s = 0; s < seqLen; s++)
        {
            rows[s * NumAxes] = timeIds[s];
            rows[s * NumAxes + 1] = heightIds[s];
            rows[s * NumAxes + 2] = widthIds[s];
        }
        return rows;
    }

    private static void WriteImage(float[] rows, int offset, int tokenCount, int wPacked, int time)
    {
        for (int s = 0; s < tokenCount; s++)
        {
            int row = s / wPacked;
            int col = s - row * wPacked;
            int at = offset + s * NumAxes;
            rows[at] = time;
            rows[at + 1] = row;
            rows[at + 2] = col;
        }
    }
}
