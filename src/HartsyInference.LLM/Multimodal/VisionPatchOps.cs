using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Multimodal;

/// <summary>Host-side patch flattening and merge-block token order shared by the ViT vision towers that take a <c>[1,3,H,W]</c> image.</summary>
internal static unsafe class VisionPatchOps
{
    /// <summary>Patchifies <paramref name="pixelValues"/> <c>[1,3,H,W]</c> into <c>[gh·gw, 3·patch²]</c> rows in merge-block order (bh, bw, mh, mw), so each run of <c>merge²</c> rows is one merge block and merged tokens come out row-major over the <c>(gh/merge, gw/merge)</c> grid.</summary>
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

    /// <summary>Fills the grid row and column of every token, in the merge-block order <see cref="Patchify"/> emits.</summary>
    public static void BlockOrderPositions(int gh, int gw, int merge, int[] rows, int[] cols)
    {
        int idx = 0;
        for (int bh = 0; bh < gh / merge; bh++)
            for (int bw = 0; bw < gw / merge; bw++)
                for (int mh = 0; mh < merge; mh++)
                    for (int mw = 0; mw < merge; mw++)
                    {
                        rows[idx] = bh * merge + mh;
                        cols[idx] = bw * merge + mw;
                        idx++;
                    }
    }
}
