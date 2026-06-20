using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Dia;

/// <summary>Multi-head layout helpers for Dia attention (flat ⇄ head-major, cache slice, GQA repeat). These
/// mirror the private reshape helpers in <c>Qwen2Attention</c>; kept local to Dia for now (a future pass can
/// hoist one shared <c>MultiHeadOps</c> across Qwen2 / Moshi-depth / Dia once all three are validated).</summary>
public static unsafe class DiaHeads
{
    /// <summary>[1, T, H*D] → [1, H, T, D].</summary>
    public static void FlatToHeads(Tensor outT, Tensor input, int t, int heads, int hd)
    {
        float* ip = (float*)input.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int s = 0; s < t; s++)
            for (int h = 0; h < heads; h++)
            {
                long inOff = ((long)s * heads + h) * hd;
                long outOff = ((long)h * t + s) * hd;
                Buffer.MemoryCopy(ip + inOff, op + outOff, hd * 4, hd * 4);
            }
    }

    /// <summary>[1, H, T, D] → [1, T, H*D].</summary>
    public static void HeadsToFlat(Tensor outT, Tensor input, int t, int heads, int hd)
    {
        float* ip = (float*)input.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int s = 0; s < t; s++)
            for (int h = 0; h < heads; h++)
            {
                long inOff = ((long)h * t + s) * hd;
                long outOff = ((long)s * heads + h) * hd;
                Buffer.MemoryCopy(ip + inOff, op + outOff, hd * 4, hd * 4);
            }
    }

    /// <summary>Copies the first <paramref name="kvLen"/> positions of a <c>[1, H, maxSeq, D]</c> cache into a
    /// packed <c>[1, H, kvLen, D]</c>.</summary>
    public static Tensor SliceLeading(Tensor cache, int heads, int kvLen, int hd)
    {
        int maxSeq = (int)cache.Shape[2];
        Tensor sliced = new(new TensorShape(1, heads, kvLen, hd), DType.F32);
        float* src = (float*)cache.DataPointer;
        float* dst = (float*)sliced.DataPointer;
        long srcStride = (long)maxSeq * hd, dstStride = (long)kvLen * hd;
        for (int h = 0; h < heads; h++)
            Buffer.MemoryCopy(src + h * srcStride, dst + h * dstStride, dstStride * 4, dstStride * 4);
        return sliced;
    }

    /// <summary>[1, kvH, S, D] → [1, kvH*group, S, D] (GQA head replication).</summary>
    public static void RepeatKv(Tensor outT, Tensor input, int kvHeads, int group, int s, int hd)
    {
        float* ip = (float*)input.DataPointer;
        float* op = (float*)outT.DataPointer;
        long perHead = (long)s * hd;
        for (int h = 0; h < kvHeads; h++)
            for (int g = 0; g < group; g++)
            {
                long srcOff = h * perHead;
                long dstOff = (long)(h * group + g) * perHead;
                Buffer.MemoryCopy(ip + srcOff, op + dstOff, perHead * 4, perHead * 4);
            }
    }
}
