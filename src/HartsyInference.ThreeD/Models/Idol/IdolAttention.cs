using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
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

    /// <summary>The layer Idol's blocks and decoders attend through: equal heads, the default scale <c>1/sqrt(headDim)</c>, and the F16
    /// permission, because the queries and keys are RMS-normalised in every Idol attention, so their scores are bounded and the backend
    /// may run it in F16, as idol-sdk's attention does.</summary>
    internal static AttentionLayer Layer(int heads, int headDim) =>
        new(new AttentionLayerConfig { Heads = heads, KvHeads = heads, HeadDim = headDim, AllowF16 = true });

    /// <summary>Attention of token-major <paramref name="q"/> (<paramref name="queryTokens"/> rows) over <paramref name="k"/> and <paramref name="v"/>
    /// (<paramref name="keyTokens"/> rows), without a mask, through <paramref name="layer"/> (made by <see cref="Layer"/>); consumes the three.</summary>
    internal static Tensor Attend(IBackend backend, AttentionLayer layer, Tensor q, Tensor k, Tensor v, int queryTokens, int keyTokens, int heads, int headDim)
    {
        using Tensor qh = ToHeadMajor(backend, q, queryTokens, heads, headDim);
        using Tensor kh = ToHeadMajor(backend, k, keyTokens, heads, headDim);
        using Tensor vh = ToHeadMajor(backend, v, keyTokens, heads, headDim);
        Tensor att = layer.AttendHeadMajorOperands(backend, qh, kh, vh, default, OperandOwnership.None);
        return layer.MergeHeads(backend, att, new TensorShape(1, queryTokens, (long)heads * headDim));   // consumes att
    }

    private static Tensor ToHeadMajor(IBackend backend, Tensor tokenMajor, int tokens, int heads, int headDim)
    {
        Tensor o = new(new TensorShape(1, heads, tokens, headDim), DType.F32);
        backend.Permute0213(o, tokenMajor, tokens, heads, headDim);
        tokenMajor.Dispose();
        return o;
    }
}
