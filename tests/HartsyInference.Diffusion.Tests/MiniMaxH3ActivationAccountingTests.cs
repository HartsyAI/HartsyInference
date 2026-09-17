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

    /// <summary>A geometry below <see cref="MiniMaxH3ChunkPolicy.MinChunkableRows"/> runs whole, and then
    /// <see cref="MiniMaxH3Transformer.Attention"/>'s qkv and head-major q/k/v ARE the full-sequence buffers the
    /// pass-1 term models — counting both charges one allocation twice. That over-count refused a 90-frame
    /// 512x288 clip by 48 MB on a 24 GB card that had just generated the same geometry.</summary>
    [Fact]
    public void EstimateFloorBytes_UnchunkedGeometry_DoesNotChargeTheFullSequenceBuffersTwice()
    {
        MiniMaxH3Config config = new MiniMaxH3Config();
        int seq = MiniMaxH3ChunkPolicy.MinChunkableRows - 1;
        int inner = config.NumAttentionHeads * config.AttentionHeadDim;
        long residual = (long)seq * config.HiddenSize * DType.F32.SizeInBytes;
        // Unchunked attention's live peak: qkv [seq, inner*3] alongside head-major q/k/v of the same total width.
        long attentionPeak = 2L * seq * inner * 3L * DType.F32.SizeInBytes;
        long ceiling = residual + attentionPeak + MiniMaxH3ActivationEstimate.FudgeBytes;

        long floor = MiniMaxH3ActivationEstimate.EstimateFloorBytes(
            seq, config, DType.F32, MiniMaxH3ChunkPolicy.ScratchRows(seq, config, DType.F32, long.MaxValue),
            sparseAttention: false);

        Assert.True(floor <= ceiling,
            $"an unchunked floor ({floor}) must not exceed what the unchunked forward actually allocates "
            + $"({ceiling}) — charging kFull/vFull on top of a full-sequence scratch term counts them twice");
    }

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

    /// <summary>A sparse forward never chunks its attention: <c>ForwardNamedBlock</c> selects
    /// <c>AttentionSparse</c> before it tests <c>seq &gt; chunkRows</c>, so the full-sequence 8x peak stands at any
    /// length. Sizing a long VSA request by the chunked dense formula reserves far too little and lets a near-limit
    /// generation pass pre-flight and then OOM.</summary>
    [Fact]
    public void EstimateFloorBytes_SparseAttention_KeepsTheFullSequencePeakEvenWhenChunking()
    {
        MiniMaxH3Config config = new MiniMaxH3Config();
        int seq = 40_000;
        int chunkRows = MiniMaxH3ChunkPolicy.DefaultChunkRows;
        Assert.True(chunkRows < seq, "this case only means anything when the geometry would otherwise chunk");

        long sparse = MiniMaxH3ActivationEstimate.EstimateFloorBytes(
            seq, config, DType.F32, chunkRows, sparseAttention: true);
        long dense = MiniMaxH3ActivationEstimate.EstimateFloorBytes(
            seq, config, DType.F32, chunkRows, sparseAttention: false);

        int inner = config.NumAttentionHeads * config.AttentionHeadDim;
        long sparseAttentionPeak = 8L * seq * inner * DType.F32.SizeInBytes;
        Assert.True(sparse >= sparseAttentionPeak,
            $"a chunking-length sparse request must still reserve its full-sequence peak ({sparseAttentionPeak}), got {sparse}");
        Assert.True(sparse > dense, $"sparse ({sparse}) must exceed chunked dense ({dense})");
    }
}
