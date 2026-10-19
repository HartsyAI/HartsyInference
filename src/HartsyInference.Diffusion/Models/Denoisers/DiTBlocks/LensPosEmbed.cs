namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Position-id builders for Microsoft Lens's 3-axis <c>(frame, height, width)</c> RoPE, with <c>scale_rope=True</c>.
/// Image height and width are centred around 0 in two halves: the first <c>H − H/2</c> rows take negative positions and the
/// rest take non-negative ones, so the rotation centres on the middle pixel. Frame is 0 for image-only inference. Text tokens
/// count up from <see cref="TextPositionStart"/> on all three axes. The joint sequence is <c>[img, txt]</c>, which is Lens's
/// concat order. Rows are token-major, three floats each.</summary>
public static class LensPosEmbed
{
    /// <summary>Axes per token: frame, height and width.</summary>
    public const int NumAxes = 3;

    /// <summary>Text start, <c>max(H//2, W//2)</c>, matching upstream's <c>max_vid_index</c>.</summary>
    public static int TextPositionStart(int hPacked, int wPacked) => Math.Max(hPacked / 2, wPacked / 2);

    /// <summary>Rows for every token of an image grid, row-major, at frame 0 with centred height and width.</summary>
    public static float[] ImageRows(int hPacked, int wPacked)
    {
        float[] rows = new float[checked(hPacked * wPacked * NumAxes)];
        WriteImage(rows, 0, hPacked, wPacked, hPacked * wPacked);
        return rows;
    }

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

    /// <summary>Rows for the joint <c>[img ; txt]</c> sequence: image rows first, then text rows.</summary>
    public static float[] JointRows(int hPacked, int wPacked, int txtSeqLen, int txtPositionStart)
    {
        int imgSeqLen = hPacked * wPacked;
        float[] rows = new float[checked((imgSeqLen + txtSeqLen) * NumAxes)];
        WriteImage(rows, 0, hPacked, wPacked, imgSeqLen);
        float[] text = TextRows(txtSeqLen, txtPositionStart);
        text.CopyTo(rows, imgSeqLen * NumAxes);
        return rows;
    }

    /// <summary>Writes the image tokens of a grid into <paramref name="rows"/>, starting at <paramref name="offset"/> tokens.</summary>
    private static void WriteImage(float[] rows, int offset, int hPacked, int wPacked, int tokenCount)
    {
        int[] rowPositions = CenteredPositions(hPacked);
        int[] colPositions = CenteredPositions(wPacked);
        for (int s = 0; s < tokenCount; s++)
        {
            int row = s / wPacked;
            int col = s - row * wPacked;
            int at = (offset + s) * NumAxes;
            rows[at] = 0f;
            rows[at + 1] = rowPositions[row];
            rows[at + 2] = colPositions[col];
        }
    }

    /// <summary>Upstream <c>LensEmbedRope._compute_video_freqs</c> with <c>scale_rope=True</c>: rows <c>0 .. H − H/2 − 1</c> get
    /// <c>[−(H − H/2), ..., −1]</c> and the rest get <c>[0, ..., H/2 − 1]</c>.</summary>
    private static int[] CenteredPositions(int len)
    {
        int[] positions = new int[len];
        int halfLen = len / 2;
        int negRows = len - halfLen;
        for (int i = 0; i < negRows; i++)
            positions[i] = -(negRows - i);
        for (int i = 0; i < halfLen; i++)
            positions[negRows + i] = i;
        return positions;
    }
}
