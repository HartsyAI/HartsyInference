using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Golden arithmetic and structural tests for the backend-independent VSA oracle.</summary>
public sealed unsafe class VideoSparseAttentionReferenceTests
{
    [Fact]
    public void Execute_ComposesExactFineAndUngatedCoarseWithoutSigmoid()
    {
        VideoSparseAttentionPlan plan = new VideoSparseAttentionPlan
        {
            Profile = VideoSparseAttentionProfileKind.FastVideoVsa64V1,
            SequenceLength = 4,
            BlockOffsets = [0, 2, 4],
            SourceIndices = [0, 1, 2, 3],
            SegmentClasses = [0, 1],
            PrefixSinkBlocks = 0,
            KeepFraction = 1f,
        };
        using VideoSparseAttentionReferenceSession session = new VideoSparseAttentionReferenceSession(plan);
        using Tensor query = new Tensor(new TensorShape(1, 1, 4, 1), DType.F32);
        using Tensor key = new Tensor(query.Shape, DType.F32);
        using Tensor value = new Tensor(query.Shape, DType.F32);
        using Tensor gate = new Tensor(query.Shape, DType.F32);
        using Tensor output = new Tensor(query.Shape, DType.F32);
        float* values = (float*)value.DataPointer;
        values[0] = 1f;
        values[1] = 3f;
        values[2] = 5f;
        values[3] = 7f;
        new Span<float>((float*)gate.DataPointer, 4).Fill(1f);

        session.Execute(output, query, key, value, gate);

        float* actual = (float*)output.DataPointer;
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(8f, actual[i], 5);
        }
    }

    [Fact]
    public void Validate_RejectsDuplicateLiveRows()
    {
        VideoSparseAttentionPlan plan = new VideoSparseAttentionPlan
        {
            Profile = VideoSparseAttentionProfileKind.ComfySol64V1,
            SequenceLength = 3,
            BlockOffsets = [0, 2, 3],
            SourceIndices = [0, 1, 1],
            SegmentClasses = [0, 0],
            PrefixSinkBlocks = 1,
        };

        ArgumentException error = Assert.Throws<ArgumentException>(plan.Validate);
        Assert.Contains("more than one block", error.Message, StringComparison.Ordinal);
    }

    /// <summary>FastVideo keeps deterministic lower-index ties, while Comfy uses a strict boundary and forced
    /// neighbours. This fixture makes those two policies observable in the exact branch.</summary>
    [Fact]
    public void Execute_ProfileRoutingTiePoliciesRemainDistinct()
    {
        VideoSparseAttentionPlan BasePlan(VideoSparseAttentionProfileKind profile) => new VideoSparseAttentionPlan
        {
            Profile = profile,
            SequenceLength = 4,
            BlockOffsets = [0, 1, 2, 3, 4],
            SourceIndices = [0, 1, 2, 3],
            SegmentClasses = [0, 0, 1, 1],
            PrefixSinkBlocks = 0,
            KeepFraction = 0.25f,
        };
        using Tensor query = new Tensor(new TensorShape(1, 1, 4, 1), DType.F32);
        using Tensor key = new Tensor(query.Shape, DType.F32);
        using Tensor value = new Tensor(query.Shape, DType.F32);
        using Tensor gate = new Tensor(query.Shape, DType.F32);
        float* values = (float*)value.DataPointer;
        values[0] = 1f;
        values[1] = 10f;
        values[2] = 100f;
        values[3] = 1000f;
        using Tensor fast = new Tensor(query.Shape, DType.F32);
        using Tensor comfy = new Tensor(query.Shape, DType.F32);
        using (VideoSparseAttentionReferenceSession session = new VideoSparseAttentionReferenceSession(
            BasePlan(VideoSparseAttentionProfileKind.FastVideoVsa64V1)))
        {
            session.Execute(fast, query, key, value, gate);
        }
        using (VideoSparseAttentionReferenceSession session = new VideoSparseAttentionReferenceSession(
            BasePlan(VideoSparseAttentionProfileKind.ComfySol64V1)))
        {
            session.Execute(comfy, query, key, value, gate);
        }

        Assert.Equal(1f, ((float*)fast.DataPointer)[3], 5);
        Assert.Equal(550f, ((float*)comfy.DataPointer)[3], 5);
    }

    /// <summary>FastVideo's keep budget is computed from video tiles only. Prefix keys are added as exempt sinks
    /// and therefore cannot consume or replace the retained video tile.</summary>
    [Fact]
    public void Execute_FastVideoPrefixSinksDoNotConsumeVideoBudget()
    {
        VideoSparseAttentionPlan plan = new VideoSparseAttentionPlan
        {
            Profile = VideoSparseAttentionProfileKind.FastVideoVsa64V1,
            SequenceLength = 6,
            BlockOffsets = [0, 1, 2, 3, 4, 5, 6],
            SourceIndices = [0, 1, 2, 3, 4, 5],
            SegmentClasses = [0, 1, 2, 2, 2, 2],
            PrefixSinkBlocks = 2,
            KeepFraction = 0.25f,
        };
        using Tensor query = new Tensor(new TensorShape(1, 1, 6, 1), DType.F32);
        using Tensor key = new Tensor(query.Shape, DType.F32);
        using Tensor value = new Tensor(query.Shape, DType.F32);
        using Tensor gate = new Tensor(query.Shape, DType.F32);
        using Tensor output = new Tensor(query.Shape, DType.F32);
        float* values = (float*)value.DataPointer;
        values[0] = 1f;
        values[1] = 10f;
        values[2] = 100f;
        values[3] = 1000f;
        values[4] = 10000f;
        values[5] = 100000f;

        using VideoSparseAttentionReferenceSession session = new VideoSparseAttentionReferenceSession(plan);
        session.Execute(output, query, key, value, gate);

        Assert.Equal(37f, ((float*)output.DataPointer)[5], 5);
    }

    /// <summary>A full Comfy keep ratio routes every block; its strict boundary must not discard the lowest
    /// ranked block when no block lies below the boundary.</summary>
    [Fact]
    public void Execute_ComfyFullKeepRoutesEveryBlock()
    {
        VideoSparseAttentionPlan plan = new VideoSparseAttentionPlan
        {
            Profile = VideoSparseAttentionProfileKind.ComfySol64V1,
            SequenceLength = 4,
            BlockOffsets = [0, 1, 2, 3, 4],
            SourceIndices = [0, 1, 2, 3],
            SegmentClasses = [2, 2, 2, 2],
            PrefixSinkBlocks = 0,
            KeepFraction = 1f,
        };
        using Tensor query = new Tensor(new TensorShape(1, 1, 4, 1), DType.F32);
        using Tensor key = new Tensor(query.Shape, DType.F32);
        using Tensor value = new Tensor(query.Shape, DType.F32);
        using Tensor gate = new Tensor(query.Shape, DType.F32);
        using Tensor output = new Tensor(query.Shape, DType.F32);
        float* values = (float*)value.DataPointer;
        values[0] = 1f;
        values[1] = 2f;
        values[2] = 3f;
        values[3] = 4f;

        using VideoSparseAttentionReferenceSession session = new VideoSparseAttentionReferenceSession(plan);
        session.Execute(output, query, key, value, gate);

        Assert.Equal(2.5f, ((float*)output.DataPointer)[0], 5);
    }

    /// <summary>The coarse branch uses the query block mean and broadcasts one coarse vector to every row in that
    /// query block; it is not recomputed from each token query.</summary>
    [Fact]
    public void Execute_CoarseAttentionUsesQueryBlockCentroid()
    {
        VideoSparseAttentionPlan plan = new VideoSparseAttentionPlan
        {
            Profile = VideoSparseAttentionProfileKind.FastVideoVsa64V1,
            SequenceLength = 4,
            BlockOffsets = [0, 2, 4],
            SourceIndices = [0, 1, 2, 3],
            SegmentClasses = [0, 1],
            PrefixSinkBlocks = 0,
            KeepFraction = 1f,
        };
        using Tensor query = new Tensor(new TensorShape(1, 1, 4, 1), DType.F32);
        using Tensor key = new Tensor(query.Shape, DType.F32);
        using Tensor value = new Tensor(query.Shape, DType.F32);
        using Tensor zeroGate = new Tensor(query.Shape, DType.F32);
        using Tensor oneGate = new Tensor(query.Shape, DType.F32);
        float* queries = (float*)query.DataPointer;
        float* keys = (float*)key.DataPointer;
        float* values = (float*)value.DataPointer;
        for (int i = 0; i < 4; i++)
        {
            queries[i] = i + 1;
            keys[i] = i + 1;
            values[i] = (i + 1) * 0.5f;
        }
        new Span<float>((float*)oneGate.DataPointer, 4).Fill(1f);
        using Tensor withoutCoarse = new Tensor(query.Shape, DType.F32);
        using Tensor withCoarse = new Tensor(query.Shape, DType.F32);
        using VideoSparseAttentionReferenceSession session = new VideoSparseAttentionReferenceSession(plan);
        session.Execute(withoutCoarse, query, key, value, zeroGate);
        session.Execute(withCoarse, query, key, value, oneGate);

        float* fine = (float*)withoutCoarse.DataPointer;
        float* combined = (float*)withCoarse.DataPointer;
        Assert.Equal(combined[0] - fine[0], combined[1] - fine[1], 5);
        Assert.Equal(combined[2] - fine[2], combined[3] - fine[3], 5);
    }
}
