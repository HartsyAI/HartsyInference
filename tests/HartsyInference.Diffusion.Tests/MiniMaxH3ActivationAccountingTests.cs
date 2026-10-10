using Xunit;
using HartsyInference.Cpu;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Arithmetic-only checks on <see cref="MiniMaxH3ActivationEstimate"/>'s buffer accounting. Deliberately
/// untagged: these need no model, GPU, checkpoint or network, and the accounting they pin has already regressed
/// twice — once by charging an unchunked forward for its own buffers, once by reserving the dense peak for a
/// sparse one. Quarantining them behind an opt-in trait is what let the first slip through.</summary>
public sealed class MiniMaxH3ActivationAccountingTests
{

    /// <summary>The released VSA profile takes <c>AttentionSparse</c>, which keeps a full-sequence gate and the
    /// token-major buffer it permutes from alive alongside qkv and head-major q/k/v — an 8x projection peak where
    /// the dense path needs 6x. Charging the dense peak for a sparse forward would approve a near-limit geometry
    /// that then OOMs, which is the dangerous direction for a pre-flight.</summary>
    [Fact]
    public void EstimateFloorBytes_UnchunkedSparseAttention_KeepsTheLargerProjectionPeak()
    {
        MiniMaxH3Config config = new MiniMaxH3Config();
        int seq = MiniMaxH3ChunkPolicy.MinChunkableRows - 1;
        int inner = config.NumAttentionHeads * config.AttentionHeadDim;
        int chunkRows = MiniMaxH3ChunkPolicy.ScratchRows(seq, config, DType.F32, long.MaxValue);

        long dense = MiniMaxH3ActivationEstimate.EstimateFloorBytes(
            seq, config, DType.F32, chunkRows, sparseAttention: false);
        long sparse = MiniMaxH3ActivationEstimate.EstimateFloorBytes(
            seq, config, DType.F32, chunkRows, sparseAttention: true);

        Assert.True(sparse > dense, $"sparse ({sparse}) must reserve more than dense ({dense})");
        Assert.Equal(2L * seq * inner * DType.F32.SizeInBytes, sparse - dense);
    }

    /// <summary>A caller that does not say which attention path it will take must get the larger reservation, so
    /// an omitted argument cannot quietly under-estimate.</summary>
    [Fact]
    public void EstimateFloorBytes_DefaultsToTheSparsePeakWhenTheModeIsUnknown()
    {
        MiniMaxH3Config config = new MiniMaxH3Config();
        int seq = MiniMaxH3ChunkPolicy.MinChunkableRows - 1;
        int chunkRows = MiniMaxH3ChunkPolicy.ScratchRows(seq, config, DType.F32, long.MaxValue);

        Assert.Equal(
            MiniMaxH3ActivationEstimate.EstimateFloorBytes(seq, config, DType.F32, chunkRows, sparseAttention: true),
            MiniMaxH3ActivationEstimate.EstimateFloorBytes(seq, config, DType.F32, chunkRows));
    }

    /// <summary>ForwardNamedBlock disposes the modulated input only after Attention or Mlp returns, so it is live
    /// beside the residual for the whole call. The chunked formula's kFull/vFull term covered it incidentally;
    /// unchunked nothing does, and omitting it under-reserved a [seq, hidden] F32 buffer — about 168 MB just below
    /// the chunking threshold, enough to pass pre-flight and then OOM.</summary>
    [Fact]
    public void EstimateFloorBytes_UnchunkedGeometry_ReservesTheLiveModulatedInput()
    {
        MiniMaxH3Config config = new MiniMaxH3Config();
        int seq = MiniMaxH3ChunkPolicy.MinChunkableRows - 1;
        int inner = config.NumAttentionHeads * config.AttentionHeadDim;
        int chunkRows = MiniMaxH3ChunkPolicy.ScratchRows(seq, config, DType.F32, long.MaxValue);

        long floor = MiniMaxH3ActivationEstimate.EstimateFloorBytes(
            seq, config, DType.F32, chunkRows, sparseAttention: false);

        // h + modulated + qkv + head-major q/k/v, which is what the block actually holds at its attention peak.
        long live = 2L * seq * config.HiddenSize * DType.F32.SizeInBytes
            + 6L * seq * inner * DType.F32.SizeInBytes;
        Assert.True(floor >= live + MiniMaxH3ActivationEstimate.FudgeBytes,
            $"floor ({floor}) must cover the block's live set ({live}) plus the fixed tail");
    }
}
