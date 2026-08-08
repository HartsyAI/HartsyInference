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
    /// the simpler "everything full-size" over-approximation the two-pass design description suggests): chunked
    /// attention keeps <c>q</c> chunk-sized throughout (disposed right after its own SDPA call) and only
    /// concatenates <c>k</c>/<c>v</c> to full size, and both chunked methods accumulate their per-chunk outputs in
    /// an undisposed list until the final <c>Concat</c> — so a second full-size, hidden-width buffer is briefly
    /// live alongside the residual stream even though no single "attention output" tensor is ever allocated at full
    /// size directly. None of this shrinks with a smaller chunk, so it cannot be rescued by any chunk size or by
    /// weight streaming (weights are not part of this number at all) — the right number for
    /// <see cref="Core.Exceptions.OutOfVramException"/>-style pre-flight refusals.</summary>
    public static long EstimateFloorBytes(int seq, MiniMaxH3Config config, DType bodyDType)
    {
        int inner = config.NumAttentionHeads * config.AttentionHeadDim;
        int hidden = config.HiddenSize;
        int ffn = config.FfnHiddenSize;
        long bodyBytes = Math.Max(bodyDType.SizeInBytes, DType.F32.SizeInBytes);

        // h: the block's residual stream, [seq, 1, hidden], always fully resident.
        long residualBytes = (long)seq * hidden * bodyBytes;

        // Each chunked method's outChunks list is never trimmed until the final Concat, so by the last iteration it
        // holds a full [seq, hidden] worth of chunk tensors — a second residual-sized buffer, live alongside h.
        long accumulatedOutputBytes = (long)seq * hidden * bodyBytes;

        // kFull and vFull ([1, H, seq, hd] F32) are the only attention buffers AttentionChunked ever concatenates to
        // full size — q stays chunk-sized and is disposed immediately after its own SDPA call.
        long fullKvBytes = (long)seq * inner * 2L * DType.F32.SizeInBytes;

        // One chunk's own transient scratch: either the packed qkv chunk (inner*3 wide) alongside its chunk-sized
        // q/k/v (another inner*3), or the MLP's gateUp+act chunk — the two never run concurrently within a block,
        // and neither scales with seq. Matches MiniMaxH3ChunkPolicy's own per-row rate for the same reason.
        int chunkRows = MiniMaxH3ChunkPolicy.DefaultChunkRows;
        long attnChunkBytes = (long)chunkRows * inner * 6L * DType.F32.SizeInBytes;
        long mlpChunkBytes = (long)chunkRows * ffn * 3L * DType.F32.SizeInBytes;
        long chunkScratchBytes = Math.Max(attnChunkBytes, mlpChunkBytes);

        return residualBytes + accumulatedOutputBytes + fullKvBytes + chunkScratchBytes + FudgeBytes;
    }
}
