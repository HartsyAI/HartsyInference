using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Stable Audio's partial split-half (NeoX) RoPE — verified against <c>stable_audio_tools</c>'s
/// <c>RotaryEmbedding</c>/<c>apply_rotary_pos_emb</c>: only the leading <c>rotDim</c> channels of each
/// <c>headDim</c>-sized head rotate (GPT-J-style partial rotary, <c>rotDim &lt; headDim</c>); the remaining
/// channels pass through unchanged. Within the rotated span, pairing is split-half (index <c>d</c> with
/// <c>d + rotDim/2</c>), NOT interleaved adjacent pairs. Self-attention only — Stable Audio's cross-attention
/// never receives <c>rotary_pos_emb</c>.</summary>
public static class StableAudioRope
{
    /// <summary>Builds <c>cos/sin[seqLen, rotDim/2]</c> tables from the checkpoint's <c>inv_freq</c> (loaded, not
    /// recomputed from theta — Stable Audio ships it verbatim as <c>transformer.rotary_pos_emb.inv_freq</c>).
    /// Position 0 is the prepended global token (rotation is identity there since angle = 0·inv_freq = 0);
    /// positions <c>[1, seqLen)</c> are the latent tokens.</summary>
    public static unsafe (float[] Cos, float[] Sin) BuildTables(ReadOnlySpan<float> invFreq, int seqLen)
    {
        int half = invFreq.Length;
        float[] cos = new float[seqLen * half];
        float[] sin = new float[seqLen * half];
        for (int pos = 0; pos < seqLen; pos++)
            for (int i = 0; i < half; i++)
            {
                float angle = pos * invFreq[i];
                cos[pos * half + i] = MathF.Cos(angle);
                sin[pos * half + i] = MathF.Sin(angle);
            }
        return (cos, sin);
    }

    /// <summary>Applies the rotation in place to a <c>[1, heads, rows, headDim]</c> tensor. Only channels
    /// <c>[0, rotDim)</c> rotate (paired <c>d</c> ↔ <c>d + rotDim/2</c>); channels <c>[rotDim, headDim)</c> are
    /// left untouched (partial rotary).</summary>
    public static unsafe void Apply(Tensor t, int heads, int rows, int headDim, int rotDim, float[] cos, float[] sin)
    {
        int half = rotDim / 2;
        float* p = (float*)t.DataPointer;
        for (int h = 0; h < heads; h++)
            for (int s = 0; s < rows; s++)
            {
                long baseOff = ((long)h * rows + s) * headDim;
                int row = s * half;
                for (int d = 0; d < half; d++)
                {
                    float c = cos[row + d], sn = sin[row + d];
                    float x1 = p[baseOff + d], x2 = p[baseOff + d + half];
                    p[baseOff + d] = x1 * c - x2 * sn;
                    p[baseOff + d + half] = x2 * c + x1 * sn;
                }
            }
    }
}
