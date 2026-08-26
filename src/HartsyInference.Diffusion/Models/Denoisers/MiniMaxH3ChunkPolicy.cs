using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Decides whether one <see cref="MiniMaxH3Transformer"/> forward should route
/// <see cref="MiniMaxH3Transformer.ForwardBlock"/>'s attention/MLP through the chunked path — a first-pass, cheap
/// heuristic. <see cref="MiniMaxH3ActivationEstimate"/> (a later phase) supersedes this with a real activation
/// budget derived from the actual accepted geometry; this policy only needs to keep every geometry that fits
/// unchunked TODAY on the exact-legacy path (so nothing regresses) and start chunking before the unchunked
/// attention/MLP buffers would exceed free VRAM.</summary>
public static class MiniMaxH3ChunkPolicy
{
    /// <summary>Rows to a side once chunking is warranted — small enough to collapse the unchunked ~20 GiB
    /// projection-buffer peak to a few hundred MB, large enough that a 50-block, 30-step generation does not pay
    /// for thousands of tiny kernel launches per chunk. Not tuned against a real budget yet — a fixed constant is
    /// deliberately simple until <see cref="MiniMaxH3ActivationEstimate"/> replaces this whole policy.</summary>
    public const int DefaultChunkRows = 4096;

    /// <summary>Never chunk below this length even when the byte estimate says to — the token refiner's ~300 rows
    /// and any small-frame-count generation must always take the bit-exact legacy path, and a chunk count in the
    /// single digits buys nothing.</summary>
    public const int MinChunkableRows = 8192;

    /// <summary>Returns <see cref="int.MaxValue"/> (never chunk — <see cref="MiniMaxH3Transformer.ForwardBlock"/>'s
    /// <c>seq &gt; chunkRows</c> check then always takes the exact-legacy path) when the unchunked per-block peak
    /// comfortably fits <paramref name="freeBytes"/>, else <see cref="DefaultChunkRows"/>. <c>HARTSY_H3_CHUNK_ROWS</c>
    /// overrides the decision outright (any positive integer) — the CPU backend's <c>GetVramInfo</c> always reports
    /// (0, 0), so this is also how a CPU-only test forces the chunked path to A/B it against the same weights.</summary>
    public static int ResolveChunkRows(int seq, MiniMaxH3Config config, DType bodyDType, long freeBytes)
    {
        if (EngineKnobs.H3ChunkRows.Value is int forced && forced > 0)
        {
            return forced;
        }
        if (seq < MinChunkableRows || freeBytes <= 0)
        {
            return int.MaxValue;
        }

        int inner = config.NumAttentionHeads * config.AttentionHeadDim;
        int ffn = config.FfnHiddenSize;
        int hidden = config.HiddenSize;
        // Attention's peak instant: qkv [seq, inner*3] F32 simultaneously live with head-major q/k/v [1,H,seq,hd]
        // F32 (3× inner) — see MiniMaxH3Transformer.Attention. Mlp's peak: gateUp [seq, ffn*2] F32 simultaneously
        // live with act [seq, ffn] F32. Both run against the same resident h/x, so add that once.
        long attnBytesPerRow = (long)(inner * 3 + inner * 3) * DType.F32.SizeInBytes;
        long mlpBytesPerRow = (long)(ffn * 2 + ffn) * DType.F32.SizeInBytes;
        long residualBytesPerRow = (long)hidden * Math.Max(bodyDType.SizeInBytes, DType.F32.SizeInBytes);
        long peakBytesPerRow = Math.Max(attnBytesPerRow, mlpBytesPerRow) + residualBytesPerRow;

        // Generous safety margin: weights, KV/rope tables, and everything else already resident are not modeled
        // here — this heuristic only distinguishes "clearly fits" from "chunk to be safe", not a tight bound.
        const double safetyFactor = 0.5;
        long unchunkedPeak = seq * peakBytesPerRow;
        return unchunkedPeak <= (long)(freeBytes * safetyFactor) ? int.MaxValue : DefaultChunkRows;
    }
}
