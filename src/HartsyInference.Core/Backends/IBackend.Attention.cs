using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Single-latent sparse attention, indexer, hyper-connection and latent-cache quantization primitives; backends
// without a kernel inherit the throwing defaults.
public partial interface IBackend
{
    /// <summary>Attention where one latent row is both key and value: per token and head, softmax over the scaled
    /// dot products of the indexed rows plus the head's fp32 sink, then the weighted sum of those rows.
    /// See <see cref="SparseLatentAttentionReference.Apply"/> for the exact index and sink rules.</summary>
    /// <param name="output">F32 <c>[tokens,heads,dim]</c>.</param>
    /// <param name="query">F32 <c>[tokens,heads,dim]</c>.</param>
    /// <param name="window">Sliding-window ring with exactly <paramref name="windowSlots"/> rows.</param>
    /// <param name="main">Compressed cache addressed after the ring; may be empty.</param>
    /// <param name="indices">I32 <c>[tokens,k]</c>; -1 is skipped.</param>
    /// <param name="sink">F32 <c>[heads]</c>, joins the softmax denominator only.</param>
    void SparseLatentAttention(Tensor output, Tensor query, in LatentSource window, in LatentSource main,
        Tensor indices, int windowSlots, Tensor sink, float scale) =>
        throw NotSupportedPrimitive(nameof(SparseLatentAttention));

    /// <summary>Indexer scores <c>scale * sum_h relu(q[t,h] . key[n]) * headWeights[t,h]</c>, masked to -infinity at
    /// <c>n >= compressLens[t]</c> and where <paramref name="candidates"/> is zero.</summary>
    /// <param name="scores">F32 <c>[tokens,keys.Rows]</c>.</param>
    /// <param name="query">F32 <c>[tokens,heads,dim]</c>, already rotated and quantize-dequantized.</param>
    /// <param name="headWeights">F32 <c>[tokens,heads]</c>.</param>
    /// <param name="compressLens">I32 <c>[tokens]</c> visible key count per query.</param>
    /// <param name="candidates">Optional U8 <c>[tokens,keys.Rows]</c> allow-mask.</param>
    void IndexerScores(Tensor scores, Tensor query, in LatentSource keys, Tensor headWeights, Tensor compressLens,
        Tensor? candidates, float scale) =>
        throw NotSupportedPrimitive(nameof(IndexerScores));

    /// <summary>Splits mixes <c>[tokens,(2+hc)*hc]</c> into <c>pre[tokens,hc]</c>, <c>post[tokens,hc]</c> and a Sinkhorn-normalized
    /// <c>comb[tokens,hc,hc]</c>; see <see cref="HcReference.SplitSinkhorn"/>.</summary>
    /// <param name="scale">F32 <c>[3]</c> for pre, post and comb.</param>
    /// <param name="bias">F32 <c>[(2+hc)*hc]</c>.</param>
    void HcSplitSinkhorn(Tensor pre, Tensor post, Tensor comb, Tensor mixes, Tensor scale, Tensor bias, int hc,
        int iters, float eps) =>
        throw NotSupportedPrimitive(nameof(HcSplitSinkhorn));

    /// <summary>Collapses the streams of <c>x[tokens,hc,dim]</c> into <c>output[tokens,dim]</c> weighted by <c>pre[tokens,hc]</c>.</summary>
    void HcPreMix(Tensor output, Tensor x, Tensor pre) =>
        throw NotSupportedPrimitive(nameof(HcPreMix));

    /// <summary>Expands <c>x[tokens,dim]</c> to <c>output[tokens,hc,dim]</c> and mixes in <c>residual[tokens,hc,dim]</c>
    /// through <c>post[tokens,hc]</c> and <c>comb[tokens,hc,hc]</c>; <paramref name="output"/> must not alias the residual.</summary>
    void HcPostMix(Tensor output, Tensor x, Tensor residual, Tensor post, Tensor comb) =>
        throw NotSupportedPrimitive(nameof(HcPostMix));

    /// <summary>Quantizes F32 <paramref name="rows"/> <c>[n,dim]</c> into <paramref name="dest"/> row <c>physicalRows[i]</c>
    /// (I32 <c>[n]</c>; negative skips); see <see cref="LatentQuantReference.QuantizeRows"/>.</summary>
    void QuantizeLatentRows(in LatentSource dest, Tensor rows, Tensor physicalRows) =>
        throw NotSupportedPrimitive(nameof(QuantizeLatentRows));

    /// <summary>Replaces F32 values with their quantize-dequantize round trip in <paramref name="encoding"/>,
    /// per group along the last dimension.</summary>
    void ActQuantDequantInPlace(Tensor x, LatentEncoding encoding) =>
        throw NotSupportedPrimitive(nameof(ActQuantDequantInPlace));

    /// <summary>Fills I32 <paramref name="indices"/> with the sliding-window ring slots per query, -1 for empty;
    /// see <see cref="WindowIndicesReference.Apply"/> for the shape and ordering.</summary>
    void BuildWindowIndices(Tensor indices, int windowSize, int seqLen, int startPos) =>
        throw NotSupportedPrimitive(nameof(BuildWindowIndices));

    /// <summary>In-place interleaved rotary on the <c>[dimOffset, dimOffset+rotaryDim)</c> slice of each vector of
    /// <c>x [B,L,D]</c> or <c>[B,L,H,D]</c>, with <c>cos</c>/<c>sin [B,L,rotaryDim/2]</c>; negate sin to invert.</summary>
    void ApplyRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int rotaryDim, int dimOffset) =>
        throw NotSupportedPrimitive(nameof(ApplyRopeInterleaved));
}
