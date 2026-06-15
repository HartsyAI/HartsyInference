using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Multi-head attention helper shared by the Hunyuan3D DiT (self + cross) and ShapeVAE decoder
/// (cross). Wraps the head reshape + <see cref="IBackend.ScaledDotProductAttention"/> + merge so query and
/// key/value can have different sequence lengths (required for cross-attention to image/latent tokens).</summary>
internal static unsafe class Hunyuan3DAttention
{
    /// <summary>Attends queries <paramref name="q"/> <c>[B,Sq,D]</c> over keys/values <paramref name="k"/>,
    /// <paramref name="v"/> <c>[B,Sk,D]</c> with <paramref name="numHeads"/> heads. Returns <c>[B,Sq,D]</c>.</summary>
    public static Tensor Attend(IBackend backend, Tensor q, Tensor k, Tensor v, int numHeads)
    {
        int b = (int)q.Shape[0], sq = (int)q.Shape[1], d = (int)q.Shape[2];
        int sk = (int)k.Shape[1];
        int headDim = d / numHeads;

        Tensor qh = ToHeads(q, b, sq, numHeads, headDim);
        Tensor kh = ToHeads(k, b, sk, numHeads, headDim);
        Tensor vh = ToHeads(v, b, sk, numHeads, headDim);

        Tensor o = new(new TensorShape(b, numHeads, sq, headDim), DType.F32);
        backend.ScaledDotProductAttention(o, qh, kh, vh, null, 1f / MathF.Sqrt(headDim));
        qh.Dispose(); kh.Dispose(); vh.Dispose();

        Tensor merged = FromHeads(o, b, sq, numHeads, headDim);
        o.Dispose();
        return merged;
    }

    private static Tensor ToHeads(Tensor input, int b, int s, int numHeads, int headDim)
    {
        int d = numHeads * headDim;
        Tensor o = new(new TensorShape(b, numHeads, s, headDim), DType.F32);
        float* sp = (float*)input.DataPointer; float* dp = (float*)o.DataPointer;
        for (int bb = 0; bb < b; bb++) for (int ss = 0; ss < s; ss++) for (int h = 0; h < numHeads; h++)
            new ReadOnlySpan<float>(sp + (bb * s + ss) * d + h * headDim, headDim)
                .CopyTo(new Span<float>(dp + ((bb * numHeads + h) * s + ss) * headDim, headDim));
        return o;
    }

    private static Tensor FromHeads(Tensor input, int b, int s, int numHeads, int headDim)
    {
        int d = numHeads * headDim;
        Tensor o = new(new TensorShape(b, s, d), DType.F32);
        float* sp = (float*)input.DataPointer; float* dp = (float*)o.DataPointer;
        for (int bb = 0; bb < b; bb++) for (int ss = 0; ss < s; ss++) for (int h = 0; h < numHeads; h++)
            new ReadOnlySpan<float>(sp + ((bb * numHeads + h) * s + ss) * headDim, headDim)
                .CopyTo(new Span<float>(dp + (bb * s + ss) * d + h * headDim, headDim));
        return o;
    }
}
