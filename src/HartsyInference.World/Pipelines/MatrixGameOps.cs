using HartsyInference.Core.Tensors;

namespace HartsyInference.World.Pipelines;

/// <summary>Latent-frame copy primitives shared by the MatrixGame pipelines.</summary>
internal static unsafe class MatrixGameOps
{
    /// <summary>Copies frame <paramref name="frameIndex"/> of clip <c>[1, C, T, H, W]</c> into a new <c>[1, C, 1, H, W]</c> tensor.</summary>
    internal static Tensor SliceFrame(Tensor clip, int frameIndex, int h, int w)
    {
        int c = (int)clip.Shape[1], t = (int)clip.Shape[2];
        Tensor o = new Tensor(new TensorShape([1L, c, 1, h, w]), DType.F32);
        float* src = (float*)clip.DataPointer;
        float* dst = (float*)o.DataPointer;
        long frame = (long)h * w;
        for (int ci = 0; ci < c; ci++)
            Buffer.MemoryCopy(src + ((long)ci * t + frameIndex) * frame, dst + (long)ci * frame, frame * 4, frame * 4);
        return o;
    }

    /// <summary>Concatenates latent clips <c>[1, C, T_i, H, W]</c> along the temporal axis.</summary>
    internal static Tensor ConcatFrames(IReadOnlyList<Tensor> clips, int h, int w)
    {
        int c = (int)clips[0].Shape[1];
        int tTotal = 0;
        foreach (Tensor clip in clips) tTotal += (int)clip.Shape[2];
        Tensor o = new Tensor(new TensorShape([1L, c, tTotal, h, w]), DType.F32);
        float* dst = (float*)o.DataPointer;
        long frame = (long)h * w;
        for (int ci = 0; ci < c; ci++)
        {
            int fOut = 0;
            foreach (Tensor clip in clips)
            {
                int tc = (int)clip.Shape[2];
                float* src = (float*)clip.DataPointer;
                Buffer.MemoryCopy(src + (long)ci * tc * frame, dst + ((long)ci * tTotal + fOut) * frame, (long)tc * frame * 4, (long)tc * frame * 4);
                fOut += tc;
            }
        }
        return o;
    }

    /// <summary>RGB-rate action rows <c>[count, dim]</c> starting at <paramref name="start"/>, each index clamped to <c>[0, maxRowExclusive-1]</c>; rows shorter than <paramref name="dim"/> leave zeros.</summary>
    internal static Tensor ActionRows(float[][] rows, int start, int count, int dim, int maxRowExclusive)
    {
        Tensor o = new Tensor(new TensorShape(count, dim), DType.F32);
        float* p = (float*)o.DataPointer;
        for (int i = 0; i < count; i++)
        {
            float[] row = rows[Math.Clamp(start + i, 0, maxRowExclusive - 1)];
            for (int d = 0; d < dim && d < row.Length; d++) p[(long)i * dim + d] = row[d];
        }
        return o;
    }
}
