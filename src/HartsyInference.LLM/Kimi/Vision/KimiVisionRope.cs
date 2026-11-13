using HartsyInference.LLM.Multimodal;

namespace HartsyInference.LLM.Kimi.Vision;

/// <summary>MoonViT-V2 2D rotary tables. Pair <c>j</c> rotates elements <c>(2j, 2j+1)</c>: even pairs take the column angle and odd pairs the row
/// angle, both at frequency <c>θ^(-4·⌊j/2⌋/D)</c>. This is the interleaved layout of NVIDIA NeMo's <c>vision.py</c> (<c>apply_rope</c> on
/// <c>view_as_complex</c> of adjacent elements), not the rotate-half layout of the Qwen-VL towers.</summary>
internal static class KimiVisionRope
{
    /// <summary>Cos and sin per pair, <c>[gh·gw, headDim/2]</c>, in the merge-block token order of <see cref="VisionPatchOps.Patchify"/>.</summary>
    public static (float[] Cos, float[] Sin) Build(int gh, int gw, int headDim, int merge, float theta)
    {
        int np = gh * gw, pairs = headDim / 2, freqs = headDim / 4;
        int[] rows = new int[np], cols = new int[np];
        VisionPatchOps.BlockOrderPositions(gh, gw, merge, rows, cols);
        float[] inv = new float[freqs];
        for (int i = 0; i < freqs; i++) inv[i] = 1f / MathF.Pow(theta, 4f * i / headDim);

        float[] cos = new float[np * pairs], sin = new float[np * pairs];
        for (int k = 0; k < np; k++)
            for (int j = 0; j < pairs; j++)
            {
                float position = (j & 1) == 0 ? cols[k] : rows[k];
                float angle = position * inv[j >> 1];
                cos[k * pairs + j] = MathF.Cos(angle);
                sin[k * pairs + j] = MathF.Sin(angle);
            }
        return (cos, sin);
    }
}
