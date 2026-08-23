using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Multimodal;

/// <summary>Host-side vision-tower preprocessing shared by the Qwen-VL encoders (<see cref="Qwen25VlEncoder"/>, <see cref="Qwen3VlEncoder"/>), which differ in their block wiring but agree exactly on patch ordering and 2D-RoPE layout.</summary>
internal static unsafe class QwenVlOps
{
    /// <summary>Patchifies <paramref name="pixelValues"/> <c>[1,3,H,W]</c> into <c>[gh·gw, 3·patch²]</c> in merge-block order (bh, bw, mh, mw), so groups of <c>merge²</c> consecutive patches form one merger block and merged tokens come out row-major over the <c>(gh/merge, gw/merge)</c> grid.</summary>
    public static Tensor Patchify(IBackend backend, Tensor pixelValues, int gh, int gw, int merge, int patch)
    {
        int h = (int)pixelValues.Shape[2], w = (int)pixelValues.Shape[3];
        int np = gh * gw, pin = 3 * patch * patch;
        Tensor patches = new(new TensorShape(np, pin), DType.F32);
        backend.Sync();
        float* px = (float*)pixelValues.DataPointer;
        float* pp = (float*)patches.DataPointer;
        int idx = 0;
        for (int bh = 0; bh < gh / merge; bh++)
            for (int bw = 0; bw < gw / merge; bw++)
                for (int mh = 0; mh < merge; mh++)
                    for (int mw = 0; mw < merge; mw++)
                    {
                        int ph = (bh * merge + mh) * patch, pw = (bw * merge + mw) * patch;
                        float* dst = pp + (long)idx * pin;
                        int o = 0;
                        for (int c = 0; c < 3; c++)
                            for (int yy = 0; yy < patch; yy++)
                                for (int xx = 0; xx < patch; xx++)
                                    dst[o++] = px[((long)c * h + (ph + yy)) * w + (pw + xx)];
                        idx++;
                    }
        return patches;
    }

    /// <summary>2D rotary cos/sin tables <c>[gh·gw, headDim]</c> in the same merge-block order as <see cref="Patchify"/>: h-frequencies on the first quarter, w-frequencies on the second, both mirrored on the upper half (the rotate_half layout Qwen-VL's vision attention expects).</summary>
    public static (float[] cos, float[] sin) BuildRope(int gh, int gw, int headDim, int merge)
    {
        int np = gh * gw, ropeDim = headDim / 2, freqN = ropeDim / 2;
        float[] inv = new float[freqN];
        for (int i = 0; i < freqN; i++) inv[i] = 1f / MathF.Pow(10000f, (2f * i) / ropeDim);
        int[] hpos = new int[np], wpos = new int[np];
        int idx = 0;
        for (int bh = 0; bh < gh / merge; bh++)
            for (int bw = 0; bw < gw / merge; bw++)
                for (int mh = 0; mh < merge; mh++)
                    for (int mw = 0; mw < merge; mw++)
                    { hpos[idx] = bh * merge + mh; wpos[idx] = bw * merge + mw; idx++; }
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
