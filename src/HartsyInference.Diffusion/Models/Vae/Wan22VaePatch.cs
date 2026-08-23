using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Wan2.2 VAE pixel-shuffle patchify / unpatchify (upstream <c>vae2_2.py</c> <c>patchify</c>/<c>unpatchify</c>), reusable by any pixel-shuffle space↔channel fold on 5-D <c>[B, C, T, H, W]</c> tensors. The encoder input and decoder output use this to trade spatial resolution for channels (RGB 3 ↔ 12 at patch 2).
///
/// <para>Layout (einops <c>b c f (h q) (w r) -> b (c r q) f h w</c>): packed channel index = <c>c·p² + r·p + q</c>, where <c>q</c> indexes the height sub-pixel and <c>r</c> the width sub-pixel. <c>Unpatchify</c> is the exact inverse.</para></summary>
public static unsafe class Wan22VaePatch
{
    /// <summary>Folds spatial <c>p×p</c> blocks into channels: <c>[B, C, T, H, W] → [B, C·p², T, H/p, W/p]</c>.</summary>
    public static Tensor Patchify(Tensor x, int patchSize)
    {
        if (patchSize == 1) return VaeOps.Clone(x);
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        if (h % patchSize != 0 || w % patchSize != 0)
            throw new ArgumentException($"H/W ({h}x{w}) must be divisible by patch {patchSize}.");
        int p = patchSize, outC = c * p * p, outH = h / p, outW = w / p;

        Tensor outT = new Tensor(new TensorShape([(long)b, outC, t, outH, outW]), DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int ti = 0; ti < t; ti++)
                    for (int hh = 0; hh < outH; hh++)
                        for (int ww = 0; ww < outW; ww++)
                            for (int q = 0; q < p; q++)
                                for (int r = 0; r < p; r++)
                                {
                                    int oc = ci * p * p + r * p + q;
                                    long srcOff = ((((long)bi * c + ci) * t + ti) * h + (hh * p + q)) * w + (ww * p + r);
                                    long dstOff = ((((long)bi * outC + oc) * t + ti) * outH + hh) * outW + ww;
                                    dst[dstOff] = src[srcOff];
                                }
        return outT;
    }

    /// <summary>GPU-resident overload: keeps unpatchify on-device (the host <see cref="Unpatchify(Tensor,int)"/> forces a D2H sync + H2D re-upload, once per decoded frame).</summary>
    public static Tensor Unpatchify(HartsyInference.Core.Backends.IBackend backend, Tensor x, int patchSize)
    {
        if (patchSize == 1) return VaeOps.Clone(x);
        int b = (int)x.Shape[0], packedC = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int p = patchSize;
        if (packedC % (p * p) != 0)
            throw new ArgumentException($"Channels {packedC} not divisible by p² ({p * p}).");
        int c = packedC / (p * p);
        Tensor outT = new Tensor(new TensorShape([(long)b, c, t, h * p, w * p]), DType.F32);
        backend.UnpatchifyVae(outT, x, patchSize);
        return outT;
    }

    /// <summary>Inverse of <see cref="Patchify"/>: <c>[B, C·p², T, H, W] → [B, C, T, H·p, W·p]</c>.</summary>
    public static Tensor Unpatchify(Tensor x, int patchSize)
    {
        if (patchSize == 1) return VaeOps.Clone(x);
        int b = (int)x.Shape[0], packedC = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int p = patchSize;
        if (packedC % (p * p) != 0)
            throw new ArgumentException($"Channels {packedC} not divisible by p² ({p * p}).");
        int c = packedC / (p * p), outH = h * p, outW = w * p;

        Tensor outT = new Tensor(new TensorShape([(long)b, c, t, outH, outW]), DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int ti = 0; ti < t; ti++)
                    for (int hh = 0; hh < h; hh++)
                        for (int ww = 0; ww < w; ww++)
                            for (int q = 0; q < p; q++)
                                for (int r = 0; r < p; r++)
                                {
                                    int oc = ci * p * p + r * p + q;
                                    long srcOff = ((((long)bi * packedC + oc) * t + ti) * h + hh) * w + ww;
                                    long dstOff = ((((long)bi * c + ci) * t + ti) * outH + (hh * p + q)) * outW + (ww * p + r);
                                    dst[dstOff] = src[srcOff];
                                }
        return outT;
    }
}
