using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Shared layout helpers for 3D video VAEs: convert between the channel-major <c>[B, C, T, H, W]</c> tensor form and the per-frame <c>[B·T, C, H, W]</c> form that 2D conv/attention ops consume. Used by every block that processes frames independently (attention, spatial resample). Reusable across Wan/LTX video VAEs.</summary>
public static unsafe class Vae3dLayout
{
    /// <summary><c>[B, C, T, H, W] → [B·T, C, H, W]</c> (frames batched).</summary>
    public static Tensor ToFrames(Tensor x, int b, int c, int t, int h, int w)
    {
        Tensor outT = new Tensor(new TensorShape(b * t, c, h, w), DType.F32);
        float* s = (float*)x.DataPointer;
        float* d = (float*)outT.DataPointer;
        long frame = (long)h * w;
        for (int bi = 0; bi < b; bi++)
            for (int ti = 0; ti < t; ti++)
                for (int ci = 0; ci < c; ci++)
                {
                    long src = (((long)bi * c + ci) * t + ti) * frame;
                    long dst = (((long)bi * t + ti) * c + ci) * frame;
                    Buffer.MemoryCopy(s + src, d + dst, frame * 4, frame * 4);
                }
        return outT;
    }

    /// <summary>Slices <paramref name="count"/> temporal frames starting at <paramref name="tStart"/> from a <c>[B,C,T,H,W]</c> tensor → <c>[B,C,count,H,W]</c>.</summary>
    public static Tensor SliceFrames(Tensor x, int tStart, int count)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor outT = new Tensor(new TensorShape([(long)b, c, count, h, w]), DType.F32);
        float* s = (float*)x.DataPointer;
        float* d = (float*)outT.DataPointer;
        long frame = (long)h * w;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int ti = 0; ti < count; ti++)
                {
                    long src = (((long)bi * c + ci) * t + (tStart + ti)) * frame;
                    long dst = (((long)bi * c + ci) * count + ti) * frame;
                    Buffer.MemoryCopy(s + src, d + dst, frame * 4, frame * 4);
                }
        return outT;
    }

    /// <summary>Concatenates a list of <c>[B,C,Ti,H,W]</c> tensors along the temporal axis → <c>[B,C,ΣTi,H,W]</c>.</summary>
    public static Tensor ConcatFrames(IReadOnlyList<Tensor> parts)
    {
        int b = (int)parts[0].Shape[0], c = (int)parts[0].Shape[1], h = (int)parts[0].Shape[3], w = (int)parts[0].Shape[4];
        int totalT = 0;
        foreach (Tensor p in parts) totalT += (int)p.Shape[2];
        Tensor outT = new Tensor(new TensorShape([(long)b, c, totalT, h, w]), DType.F32);
        float* d = (float*)outT.DataPointer;
        long frame = (long)h * w;
        int tOff = 0;
        foreach (Tensor p in parts)
        {
            int pt = (int)p.Shape[2];
            float* s = (float*)p.DataPointer;
            for (int bi = 0; bi < b; bi++)
                for (int ci = 0; ci < c; ci++)
                    for (int ti = 0; ti < pt; ti++)
                    {
                        long src = (((long)bi * c + ci) * pt + ti) * frame;
                        long dst = (((long)bi * c + ci) * totalT + (tOff + ti)) * frame;
                        Buffer.MemoryCopy(s + src, d + dst, frame * 4, frame * 4);
                    }
            tOff += pt;
        }
        return outT;
    }

    /// <summary>Prepends a single frame (the last frame of <paramref name="prefix"/>) before <paramref name="x"/> along T → <c>[B,C,1+Tx,H,W]</c>.</summary>
    public static Tensor PrependLastFrameOf(Tensor prefix, Tensor x)
    {
        int pt = (int)prefix.Shape[2];
        using Tensor last = SliceFrames(prefix, pt - 1, 1);
        return ConcatFrames([last, x]);
    }

    /// <summary>Prepends a zero frame before <paramref name="x"/> along T → <c>[B,C,1+Tx,H,W]</c>.</summary>
    public static Tensor PrependZeroFrame(Tensor x)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor zero = new Tensor(new TensorShape([(long)b, c, 1, h, w]), DType.F32);
        new Span<float>((float*)zero.DataPointer, checked((int)zero.Shape.ElementCount)).Clear();
        Tensor outT = ConcatFrames([zero, x]);
        zero.Dispose();
        return outT;
    }

    /// <summary><c>[B·T, C, H, W] → [B, C, T, H, W]</c> (inverse of <see cref="ToFrames"/>; H/W may differ from the input to ToFrames after a spatial op).</summary>
    public static Tensor FromFrames(Tensor x, int b, int c, int t, int h, int w)
    {
        Tensor outT = new Tensor(new TensorShape([(long)b, c, t, h, w]), DType.F32);
        float* s = (float*)x.DataPointer;
        float* d = (float*)outT.DataPointer;
        long frame = (long)h * w;
        for (int bi = 0; bi < b; bi++)
            for (int ti = 0; ti < t; ti++)
                for (int ci = 0; ci < c; ci++)
                {
                    long src = (((long)bi * t + ti) * c + ci) * frame;
                    long dst = (((long)bi * c + ci) * t + ti) * frame;
                    Buffer.MemoryCopy(s + src, d + dst, frame * 4, frame * 4);
                }
        return outT;
    }
}
