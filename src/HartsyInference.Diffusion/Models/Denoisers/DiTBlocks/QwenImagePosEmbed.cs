namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Position coordinates for Qwen-Image's 3-axis <c>(frame, height, width)</c> RoPE. Image tokens take frame 0 and
/// centred height and width (<c>scale_rope=True</c>, which diffusers hardcodes). Text tokens count up from a start
/// position on all three axes. Reference grids take frame <c>k + 1</c> and centre on their own grid. Rows are token-major,
/// three floats each, the layout <see cref="Core.Rope.MultiAxisRopeTables"/> reads.</summary>
public static class QwenImagePosEmbed
{
    /// <summary>Axes per token: frame, height and width.</summary>
    public const int NumAxes = 3;

    /// <summary>Text start position, <c>max(H//2, W//2)</c>: the largest index of the centred image grid (diffusers' <c>max_vid_index</c>).</summary>
    public static int TextPositionStart(int hPacked, int wPacked) => Math.Max(hPacked / 2, wPacked / 2);

    /// <summary>Rows for <paramref name="txtSeqLen"/> text tokens; token <c>s</c> sits at <c>positionStart + s</c> on every axis.</summary>
    public static float[] TextRows(int txtSeqLen, int positionStart)
    {
        float[] rows = new float[checked(txtSeqLen * NumAxes)];
        for (int s = 0; s < txtSeqLen; s++)
        {
            float pos = positionStart + s;
            rows[s * NumAxes] = pos;
            rows[s * NumAxes + 1] = pos;
            rows[s * NumAxes + 2] = pos;
        }
        return rows;
    }

    /// <summary>Rows for every token of a <paramref name="hPacked"/> × <paramref name="wPacked"/> image grid, row-major, at frame 0.</summary>
    public static float[] ImageRows(int hPacked, int wPacked)
    {
        float[] rows = new float[checked(hPacked * wPacked * NumAxes)];
        FillGrid(rows, hPacked, wPacked, tokenStart: 0, tokenCount: hPacked * wPacked, frame: 0f);
        return rows;
    }

    /// <summary>Rows for the joint <c>[text ; image ; reference grids]</c> sequence. Text rows use
    /// <paramref name="txtPositionStart"/> + s when <paramref name="ropeText"/> is set, and position 0 otherwise (identity
    /// rotation, Mage-Flow's <c>apply_text_rotary_emb=false</c>). Image rows are the global tokens
    /// <c>[imgTokenStart, imgTokenStart + imgTokenCount)</c>, so a context-parallel rank keeps global positions; a negative
    /// count means the whole grid. Each reference grid follows, at frame <c>index + 1</c>, centred on its own grid.</summary>
    public static float[] JointRows(int hPacked, int wPacked, int txtSeqLen, int txtPositionStart, bool ropeText,
        ReadOnlySpan<(int H, int W)> refGrids, int imgTokenStart, int imgTokenCount)
    {
        int imgCount = imgTokenCount < 0 ? hPacked * wPacked : imgTokenCount;
        int refSeqLen = 0;
        foreach ((int rh, int rw) in refGrids)
            refSeqLen += rh * rw;
        float[] rows = new float[checked((txtSeqLen + imgCount + refSeqLen) * NumAxes)];

        for (int s = 0; s < txtSeqLen; s++)
        {
            float pos = ropeText ? txtPositionStart + s : 0f;
            rows[s * NumAxes] = pos;
            rows[s * NumAxes + 1] = pos;
            rows[s * NumAxes + 2] = pos;
        }
        int offset = txtSeqLen;
        FillGrid(rows.AsSpan(offset * NumAxes, imgCount * NumAxes), hPacked, wPacked, imgTokenStart, imgCount, frame: 0f);
        offset += imgCount;
        for (int r = 0; r < refGrids.Length; r++)
        {
            (int refH, int refW) = refGrids[r];
            int refCount = refH * refW;
            FillGrid(rows.AsSpan(offset * NumAxes, refCount * NumAxes), refH, refW, tokenStart: 0, tokenCount: refCount, frame: r + 1);
            offset += refCount;
        }
        return rows;
    }

    /// <summary>Writes <paramref name="tokenCount"/> grid tokens starting at <paramref name="tokenStart"/> into
    /// <paramref name="rows"/>. Height and width are <c>row − (H − H/2)</c> and <c>col − (W − W/2)</c>.</summary>
    private static void FillGrid(Span<float> rows, int hPacked, int wPacked, int tokenStart, int tokenCount, float frame)
    {
        int hCenter = hPacked - hPacked / 2;
        int wCenter = wPacked - wPacked / 2;
        for (int i = 0; i < tokenCount; i++)
        {
            int si = tokenStart + i;
            int row = si / wPacked;
            int col = si - row * wPacked;
            rows[i * NumAxes] = frame;
            rows[i * NumAxes + 1] = row - hCenter;
            rows[i * NumAxes + 2] = col - wCenter;
        }
    }
}
