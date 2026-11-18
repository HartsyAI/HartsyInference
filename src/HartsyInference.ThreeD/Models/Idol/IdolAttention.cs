using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>The head layout and attention call that Idol's blocks and its two decoders share: token-major <c>[1, L, heads, headDim]</c>
/// slices of a packed projection, and attention over them returned flat as <c>[1, L, heads·headDim]</c>.</summary>
internal static class IdolAttention
{
    /// <summary>Columns <c>[offset, offset + heads·headDim)</c> of a packed <c>[1, tokens, k·C]</c> tensor as token-major <c>[1, tokens, heads, headDim]</c>.</summary>
    internal static Tensor Slice(IBackend backend, Tensor packed, int offset, int tokens, int heads, int headDim)
    {
        Tensor o = new(new TensorShape(1, tokens, heads, headDim), DType.F32);
        backend.SliceLastDim(o, packed, offset);
        return o;
    }

    /// <summary>Attention of token-major <paramref name="q"/> (<paramref name="queryTokens"/> rows) over <paramref name="k"/> and <paramref name="v"/>
    /// (<paramref name="keyTokens"/> rows), without a mask; consumes the three. The queries and keys are RMS-normalised in every Idol attention, so their scores
    /// are bounded and the backend may run it in F16, as idol-sdk's attention does.</summary>
    internal static Tensor Attend(IBackend backend, Tensor q, Tensor k, Tensor v, int queryTokens, int keyTokens, int heads, int headDim, float scale)
    {
        using Tensor qh = ToHeadMajor(backend, q, queryTokens, heads, headDim);
        using Tensor kh = ToHeadMajor(backend, k, keyTokens, heads, headDim);
        using Tensor vh = ToHeadMajor(backend, v, keyTokens, heads, headDim);
        Tensor att = new(new TensorShape(1, heads, queryTokens, headDim), DType.F32);
        backend.ScaledDotProductAttention(att, qh, kh, vh, null, scale, allowF16: true);
        Tensor flat = new(new TensorShape(1, queryTokens, (long)heads * headDim), DType.F32);
        backend.Permute0213(flat, att, heads, queryTokens, headDim);
        att.Dispose();
        return flat;
    }

    private static Tensor ToHeadMajor(IBackend backend, Tensor tokenMajor, int tokens, int heads, int headDim)
    {
        Tensor o = new(new TensorShape(1, heads, tokens, headDim), DType.F32);
        backend.Permute0213(o, tokenMajor, tokens, heads, headDim);
        tokenMajor.Dispose();
        return o;
    }
}
