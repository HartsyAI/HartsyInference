using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Shared multi-head scaled-dot-product self/cross-attention core used by both the AIFI
/// encoder block and the decoder's query self-attention. Takes already-projected q/k/v (each
/// <c>[1, S, heads·headDim]</c>), reshapes to <c>[1, heads, S, headDim]</c>, runs
/// <see cref="IBackend.ScaledDotProductAttention"/>, and merges heads back — so the two call sites
/// don't each re-derive the head-split glue.</summary>
public static class RtDetrAttention
{
    /// <summary>Runs attention. <paramref name="q"/> is <c>[1, Sq, W]</c>, <paramref name="k"/>/<paramref name="v"/>
    /// are <c>[1, Skv, W]</c> with <c>W = heads·headDim</c>. Returns <c>[1, Sq, W]</c> (pre output-projection).
    /// Owns the returned tensor; the caller keeps ownership of q/k/v, which stay alive until this returns.</summary>
    /// <param name="additiveMask">Optional <c>[1, heads, Sq, Skv]</c> added to the pre-softmax scores; null for the unmasked RT-DETR paths.</param>
    public static Tensor MultiHead(IBackend backend, Tensor q, Tensor k, Tensor v, int heads, int headDim, Tensor? additiveMask = null)
    {
        int sq = (int)q.Shape[1];
        int skv = (int)k.Shape[1];
        int width = heads * headDim;
        float scale = 1f / MathF.Sqrt(headDim);

        Tensor qh = new Tensor(new TensorShape(1, heads, sq, headDim), DType.F32);
        Tensor kh = new Tensor(new TensorShape(1, heads, skv, headDim), DType.F32);
        Tensor vh = new Tensor(new TensorShape(1, heads, skv, headDim), DType.F32);
        backend.Permute0213(qh, q, sq, heads, headDim);
        backend.Permute0213(kh, k, skv, heads, headDim);
        backend.Permute0213(vh, v, skv, heads, headDim);

        Tensor attn = new Tensor(new TensorShape(1, heads, sq, headDim), DType.F32);
        backend.ScaledDotProductAttention(attn, qh, kh, vh, mask: additiveMask, scale: scale);
        qh.Dispose(); kh.Dispose(); vh.Dispose();

        // Merge heads: [1, heads, Sq, headDim] -> [1, Sq, heads*headDim].
        Tensor merged = new Tensor(new TensorShape(1, sq, width), DType.F32);
        backend.Permute0213(merged, attn, heads, sq, headDim);
        attn.Dispose();
        return merged;
    }
}
