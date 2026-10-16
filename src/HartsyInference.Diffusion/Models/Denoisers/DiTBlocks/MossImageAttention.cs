using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Full (unmasked) self-attention over token-major Q/K/V <c>[1, S, heads, headDim]</c> for the MOSS-Image
/// blocks: the token-major SDPA where the backend serves it (no permutes either side), else head-major through
/// <see cref="IBackend.Permute0213"/>. Q/K are RMS-normalized per head, which bounds the scores, so F16 attention is
/// allowed.</summary>
public static class MossImageAttention
{
    /// <summary>Attends and returns token-major <c>[1, S, heads, headDim]</c> in the input dtype, which a following
    /// Linear reads as <c>[S, heads·headDim]</c> rows. Disposes <paramref name="q"/>, <paramref name="k"/> and
    /// <paramref name="v"/>.</summary>
    public static Tensor Run(IBackend backend, Tensor q, Tensor k, Tensor v, int seq, int heads, int headDim)
    {
        DType dtype = q.DType;
        float scale = 1f / MathF.Sqrt(headDim);
        TensorShape tokenMajor = new(1, seq, heads, headDim);
        try
        {
            if (backend.SupportsTokenMajorAttention)
            {
                Tensor output = new(tokenMajor, dtype);
                backend.ScaledDotProductAttentionTokenMajor(output, q, k, v, null, heads, headDim, scale, allowF16: true);
                return output;
            }
            TensorShape mh = new(1, heads, seq, headDim);
            Tensor qMh = new(mh, dtype);
            Tensor kMh = new(mh, dtype);
            Tensor vMh = new(mh, dtype);
            backend.Permute0213(qMh, q, seq, heads, headDim);
            backend.Permute0213(kMh, k, seq, heads, headDim);
            backend.Permute0213(vMh, v, seq, heads, headDim);
            Tensor attnMh = new(mh, dtype);
            backend.ScaledDotProductAttention(attnMh, qMh, kMh, vMh, null, scale, allowF16: true);
            qMh.Dispose();
            kMh.Dispose();
            vMh.Dispose();
            Tensor attn = new(tokenMajor, dtype);
            backend.Permute0213(attn, attnMh, heads, seq, headDim);
            attnMh.Dispose();
            return attn;
        }
        finally
        {
            q.Dispose();
            k.Dispose();
            v.Dispose();
        }
    }
}
