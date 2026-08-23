using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Multi-head attention helper shared by the Hunyuan3D DiT and ShapeVAE decoder, wrapping head reshape + <see cref="IBackend.ScaledDotProductAttention"/> + merge with independent query/key-value sequence lengths.</summary>
internal static unsafe class Hunyuan3DAttention
{
    /// <summary>Attends queries <paramref name="q"/> <c>[1,Sq,D]</c> over keys/values <paramref name="k"/>, <paramref name="v"/> <c>[1,Sk,D]</c> with <paramref name="numHeads"/> heads, returning <c>[1,Sq,D]</c>.</summary>
    public static Tensor Attend(IBackend backend, Tensor q, Tensor k, Tensor v, int numHeads)
    {
        int sq = (int)q.Shape[1], d = (int)q.Shape[2], sk = (int)k.Shape[1];
        int headDim = d / numHeads;
        TensorShape qMh = new(1, numHeads, sq, headDim), kvMh = new(1, numHeads, sk, headDim);

        Tensor qh = new(qMh, DType.F32); backend.Permute0213(qh, q, sq, numHeads, headDim);
        Tensor kh = new(kvMh, DType.F32); backend.Permute0213(kh, k, sk, numHeads, headDim);
        Tensor vh = new(kvMh, DType.F32); backend.Permute0213(vh, v, sk, numHeads, headDim);

        Tensor o = new(qMh, DType.F32);
        backend.ScaledDotProductAttention(o, qh, kh, vh, null, 1f / MathF.Sqrt(headDim), allowF16: true);
        qh.Dispose(); kh.Dispose(); vh.Dispose();

        Tensor merged = new(new TensorShape(1, sq, d), DType.F32);
        backend.Permute0213(merged, o, numHeads, sq, headDim);
        o.Dispose();
        return merged;
    }
}
