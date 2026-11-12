namespace HartsyInference.LLM.Multimodal;

/// <summary>Host-side 2D-RoPE tables shared by the Qwen-VL encoders (<see cref="Qwen25VlEncoder"/>, <see cref="Qwen3VlEncoder"/>), which differ in their block wiring but agree on the rotate-half layout and merge-block order.</summary>
internal static class QwenVlOps
{
    /// <summary>2D rotary cos/sin tables <c>[gh·gw, headDim]</c> in the merge-block order of <see cref="VisionPatchOps.Patchify"/>: h-frequencies on the first quarter, w-frequencies on the second, both mirrored on the upper half (the rotate_half layout Qwen-VL's vision attention expects).</summary>
    public static (float[] cos, float[] sin) BuildRope(int gh, int gw, int headDim, int merge)
    {
        int np = gh * gw, ropeDim = headDim / 2, freqN = ropeDim / 2;
        float[] inv = new float[freqN];
        for (int i = 0; i < freqN; i++) inv[i] = 1f / MathF.Pow(10000f, (2f * i) / ropeDim);
        int[] hpos = new int[np], wpos = new int[np];
        VisionPatchOps.BlockOrderPositions(gh, gw, merge, hpos, wpos);
        float[] cos = new float[np * headDim], sin = new float[np * headDim];
        for (int pos = 0; pos < np; pos++)
            for (int j = 0; j < freqN; j++)
            {
                float fh = hpos[pos] * inv[j], fw = wpos[pos] * inv[j];
                SetCosSin(cos, sin, pos, headDim, j, fh);
                SetCosSin(cos, sin, pos, headDim, freqN + j, fw);
                SetCosSin(cos, sin, pos, headDim, ropeDim + j, fh);
                SetCosSin(cos, sin, pos, headDim, ropeDim + freqN + j, fw);
            }
        return (cos, sin);
    }

    private static void SetCosSin(float[] cos, float[] sin, int pos, int hd, int e, float ang)
    {
        cos[pos * hd + e] = MathF.Cos(ang);
        sin[pos * hd + e] = MathF.Sin(ang);
    }
}
