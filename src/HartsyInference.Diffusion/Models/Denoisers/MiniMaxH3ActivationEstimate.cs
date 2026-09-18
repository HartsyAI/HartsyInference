using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Estimates the activation/workspace bytes one <see cref="MiniMaxH3Transformer"/> forward needs at a given
/// sequence length, for use before any weight is allocated — a pre-flight refusal (engine) and a shard-split reserve
/// (engine) both need this from outside <c>HartsyInference.Diffusion</c>, so it lives here rather than as a private
/// method on the pipeline that owns the forward loop, unlike the sibling <c>Estimate*ActivationReserveBytes</c>
/// methods on other model pipelines.</summary>
public static class MiniMaxH3ActivationEstimate
{
    /// <summary>Flat tail for cuBLAS/SDPA workspace and RoPE tables, matching every sibling estimator's fudge term.</summary>
    public const long FudgeBytes = 1024L * 1024 * 1024;

    /// <summary>The floor: activation bytes a forward needs no matter how small a chunk
    /// <see cref="MiniMaxH3ChunkPolicy"/> picks, calibrated to the actual tensor lifetimes in
    /// <see cref="MiniMaxH3Transformer.AttentionChunked"/> and <see cref="MiniMaxH3Transformer.MlpChunked"/> (not
    /// the simpler "everything full-size" over-approximation the two-pass design description suggests): the two
    /// passes peak on different buffers at different times, so the floor takes the worse of them rather than their
    /// sum, and both chunked methods accumulate their per-chunk outputs in an undisposed list until the final
    /// <c>Concat</c> — so a second full-size, hidden-width buffer is briefly live alongside the residual stream even
    /// though no single "attention output" tensor is ever allocated at full size directly. None of this shrinks with
    /// a smaller chunk, so it cannot be rescued by any chunk size or by
    /// weight streaming (weights are not part of this number at all) — the right number for
    /// <see cref="Core.Exceptions.OutOfVramException"/>-style pre-flight refusals.</summary>
    /// <param name="chunkRows">The chunk width the forward will use, from
    /// <see cref="MiniMaxH3ChunkPolicy.ScaledChunkRows"/>. Null keeps the unscaled default.</param>
    /// <param name="sparseAttention">Whether the forward will take <c>AttentionSparse</c>. Defaults to true — the
    /// larger peak — so a caller that does not know the mode cannot under-estimate its way past this check.</param>
    public static long EstimateFloorBytes(
        int seq, MiniMaxH3Config config, DType bodyDType, int? chunkRows = null, bool sparseAttention = true)
    {
        int inner = config.NumAttentionHeads * config.AttentionHeadDim;
        int hidden = config.HiddenSize;
        int ffn = config.FfnHiddenSize;
        long bodyBytes = Math.Max(bodyDType.SizeInBytes, DType.F32.SizeInBytes);

        // h: the block's residual stream, [seq, 1, hidden], always fully resident.
        long residualBytes = (long)seq * hidden * bodyBytes;

        long fullSeqInnerBytes = (long)seq * inner * DType.F32.SizeInBytes;

        int resolvedChunkRows = chunkRows ?? MiniMaxH3ChunkPolicy.DefaultChunkRows;
        bool chunked = resolvedChunkRows < seq;

        // Attention's peak depends on which implementation ForwardNamedBlock picks, and it picks AttentionSparse
        // whenever a sparse session exists — BEFORE it tests seq > chunkRows. There is no chunked sparse path, so
        // a sparse forward keeps its full-sequence q/k/v/gate alongside qkv and the token-major gate it permutes
        // from (8x) at any length. Dense attention does chunk, and then kFull/vFull outlive each chunk in flight,
        // which is the 2x term; unchunked it holds qkv alongside head-major q/k/v (6x).
        // ForwardNamedBlock holds the modulated input for the whole call — it is disposed only after Attention or
        // Mlp returns — so it is a second [seq, hidden] buffer live beside the residual. Modulate can emit fp8, but
        // not on a bf16 checkpoint or with numerics.modulateEmitFp8 off, so reserve F32. Chunked, the kFull/vFull
        // term already covers it; unchunked there is nothing else standing in for it.
        long modulatedBytes = (long)seq * hidden * DType.F32.SizeInBytes;

        long attentionBytes = sparseAttention
            ? 8L * (long)seq * inner * DType.F32.SizeInBytes + modulatedBytes
            : chunked
                ? 2L * fullSeqInnerBytes + (long)resolvedChunkRows * inner * 6L * DType.F32.SizeInBytes
                : 6L * (long)seq * inner * DType.F32.SizeInBytes + modulatedBytes;

        // Mlp chunks in both modes (gateUp + act, 3x its width over whichever row count it runs).
        long mlpBytes = (long)(chunked ? resolvedChunkRows : seq) * ffn * 3L * DType.F32.SizeInBytes
            + (chunked ? 0L : modulatedBytes);

        // The two never run concurrently within a block, so the floor is the worse rather than their sum — summing
        // them false-refused a 39-frame geometry already proven to complete on real hardware (see this class's
        // measured calibration). AttentionChunked's pass 1 used to hold a third full buffer, every chunk's q kept
        // alive for pass 2, which OOMed a 141-frame sharded run on the 12 GB card (at seq=38325 each buffer is
        // 1047.9 MB, exactly the failing allocation); q is now re-projected per chunk, so the term is 2x not 3x.
        long passOneBytes = Math.Max(attentionBytes, mlpBytes);

        long passTwoBytes = 2L * fullSeqInnerBytes + (long)seq * hidden * bodyBytes;

        return residualBytes + Math.Max(passOneBytes, passTwoBytes) + FudgeBytes;
    }
}
