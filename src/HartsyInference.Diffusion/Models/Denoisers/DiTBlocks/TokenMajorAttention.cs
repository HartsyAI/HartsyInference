using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Full (unmasked) self-attention over token-major Q <c>[1, qSeq, heads, headDim]</c> and K/V
/// <c>[1, kvSeq, heads, headDim]</c>: the token-major SDPA where the backend serves it (no permutes either side), else
/// head-major through <see cref="IBackend.Permute0213"/>. A longer K/V carries extra keys, such as cached reference
/// tokens. Callers pass <c>allowF16</c> only when Q/K are normalized, which bounds the scores.</summary>
public static class TokenMajorAttention
{
    /// <summary>Attends and returns token-major <c>[1, qSeq, heads, headDim]</c> in the query dtype, which a following
    /// Linear reads as <c>[qSeq, heads·headDim]</c> rows. Disposes <paramref name="q"/>, <paramref name="k"/> and
    /// <paramref name="v"/> unless <paramref name="disposeInputs"/> is false.</summary>
    public static Tensor Run(IBackend backend, Tensor q, Tensor k, Tensor v, int qSeq, int kvSeq, int heads, int headDim,
        bool allowF16 = true, bool disposeInputs = true)
    {
        DType dtype = q.DType;
        float scale = 1f / MathF.Sqrt(headDim);
        TensorShape tokenMajor = new(1, qSeq, heads, headDim);
        try
        {
            if (backend.SupportsTokenMajorAttention)
            {
                Tensor output = new(tokenMajor, dtype);
                backend.ScaledDotProductAttentionTokenMajor(output, q, k, v, null, heads, headDim, scale, allowF16);
                return output;
            }
            TensorShape qShape = new(1, heads, qSeq, headDim);
            TensorShape kvShape = new(1, heads, kvSeq, headDim);
            Tensor qMh = new(qShape, dtype);
            Tensor kMh = new(kvShape, dtype);
            Tensor vMh = new(kvShape, dtype);
            backend.Permute0213(qMh, q, qSeq, heads, headDim);
            backend.Permute0213(kMh, k, kvSeq, heads, headDim);
            backend.Permute0213(vMh, v, kvSeq, heads, headDim);
            Tensor attnMh = new(qShape, dtype);
            backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMh, null, scale, allowF16);
            qMh.Dispose();
            kMh.Dispose();
            vMh.Dispose();
            Tensor attn = new(tokenMajor, dtype);
            backend.Permute0213(attn, attnMh, heads, qSeq, headDim);
            attnMh.Dispose();
            return attn;
        }
        finally
        {
            if (disposeInputs)
            {
                q.Dispose();
                k.Dispose();
                v.Dispose();
            }
        }
    }
}
