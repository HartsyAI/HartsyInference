using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Position-ID builders for Flux.2's 4-axis RoPE. The rotation math is identical to Flux.1 (delegated to <see cref="FluxRope"/>); this class only constructs the 4-axis (T, H, W, L) position IDs that distinguish text from image tokens. Text token i: (0,0,0,i); image patch (row,col): (0,row,col,0); reference image k at (row,col): (10*(k+1),row,col,0) — refs time-shifted from the primary image.</summary>
public static unsafe class Flux2PosEmbed
{
    /// <summary>The four RoPE axes.</summary>
    public const int NumAxes = 4;

    /// <summary>Builds combined position IDs for the concatenated [text, image] sequence used by Flux.2's double- and single-stream blocks. Output shape is <c>[txtSeqLen + imgSeqLen, 4]</c>.</summary>
    /// <param name="txtSeqLen">Number of text tokens (typically 512 for Klein).</param>
    /// <param name="latentHPacked">Patchified latent height (= image_height / 16).</param>
    /// <param name="latentWPacked">Patchified latent width (= image_width / 16).</param>
    public static Tensor BuildPositionIds(int txtSeqLen, int latentHPacked, int latentWPacked)
    {
        int imgSeqLen = latentHPacked * latentWPacked;
        int totalSeqLen = txtSeqLen + imgSeqLen;
        TensorShape shape = new TensorShape(totalSeqLen, NumAxes);
        Tensor posIds = new Tensor(shape, DType.F32);
        float* ptr = (float*)posIds.DataPointer;

        for (int i = 0; i < txtSeqLen; i++)
        {
            ptr[i * NumAxes + 0] = 0f;
            ptr[i * NumAxes + 1] = 0f;
            ptr[i * NumAxes + 2] = 0f;
            ptr[i * NumAxes + 3] = i;
        }

        for (int row = 0; row < latentHPacked; row++)
        {
            for (int col = 0; col < latentWPacked; col++)
            {
                int idx = txtSeqLen + row * latentWPacked + col;
                ptr[idx * NumAxes + 0] = 0f;
                ptr[idx * NumAxes + 1] = row;
                ptr[idx * NumAxes + 2] = col;
                ptr[idx * NumAxes + 3] = 0f;
            }
        }

        return posIds;
    }
}
