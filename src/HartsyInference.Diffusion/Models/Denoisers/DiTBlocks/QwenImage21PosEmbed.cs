namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Position coordinates for Qwen-Image 2.1's 3-axis <c>(sequence, height, width)</c> RoPE (ComfyUI
/// <c>qwen_image21/model.py</c>). One running counter covers the whole sequence: a text run of <c>n</c> tokens takes
/// positions <c>pos … pos+n−1</c> on all three axes. The image then takes the counter's value on the sequence axis and
/// centred row and column indices on the other two. This differs from <see cref="QwenImagePosEmbed"/>, so the two cannot
/// share rows.</summary>
public static class QwenImage21PosEmbed
{
    /// <summary>Axes per token: sequence, height and width.</summary>
    public const int NumAxes = 3;

    /// <summary>Rows for the text prefix: token <c>i</c> sits at <c>i</c> on all three axes.</summary>
    public static float[] TextRows(int textLen)
    {
        float[] rows = new float[checked(textLen * NumAxes)];
        for (int i = 0; i < textLen; i++)
        {
            float pos = i;
            rows[i * NumAxes] = pos;
            rows[i * NumAxes + 1] = pos;
            rows[i * NumAxes + 2] = pos;
        }
        return rows;
    }

    /// <summary>Rows for the target image, row-major over <paramref name="imgH"/> × <paramref name="imgW"/>. The sequence
    /// axis is the constant <paramref name="textLen"/>, the counter's value after the text run. Height and width are
    /// centred: <c>r − (h − h/2)</c> and <c>c − (w − w/2)</c>.</summary>
    public static float[] ImageRows(int textLen, int imgH, int imgW)
    {
        float[] rows = new float[checked(imgH * imgW * NumAxes)];
        int hCenter = imgH - imgH / 2;
        int wCenter = imgW - imgW / 2;
        for (int r = 0; r < imgH; r++)
        {
            for (int c = 0; c < imgW; c++)
            {
                int token = r * imgW + c;
                rows[token * NumAxes] = textLen;
                rows[token * NumAxes + 1] = r - hCenter;
                rows[token * NumAxes + 2] = c - wCenter;
            }
        }
        return rows;
    }
}
