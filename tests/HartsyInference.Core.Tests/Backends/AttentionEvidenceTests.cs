using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

[Collection("AttentionEvidence")]
public sealed class AttentionEvidenceTests
{
    [Fact]
    public void DisabledObservationDoesNotAllocate()
    {
        Assert.False(AttentionEvidence.Enabled);
        AttentionEvidence.Observed(AttentionEvidenceKind.Repeat, "test");
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10000; i++) AttentionEvidence.Observed(AttentionEvidenceKind.Repeat, "test");
        Assert.Equal(before, GC.GetAllocatedBytesForCurrentThread());
    }

    [Fact]
    public void EnabledObservationDoesNotAllocateAndPreservesNestedContext()
    {
        using AttentionEvidenceSession session = AttentionEvidence.Begin(10002);
        using (AttentionEvidence.Context("outer", 7))
        {
            using (AttentionEvidence.Context("inner", 11))
                AttentionEvidence.Observed(AttentionEvidenceKind.Cast, "inner");
            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 10000; i++) AttentionEvidence.Observed(AttentionEvidenceKind.Repeat, "outer", 128);
            Assert.Equal(before, GC.GetAllocatedBytesForCurrentThread());
        }
        AttentionEvidence.Observed(AttentionEvidenceKind.Cast, "unattributed");
        AttentionEvidenceEvent[] events = session.Snapshot();
        Assert.Equal("inner", events[0].Component);
        Assert.Equal(11, events[0].Layer);
        Assert.Equal("outer", events[1].Component);
        Assert.Equal(7, events[1].Layer);
        Assert.Null(events[^1].Component);
        Assert.Equal(-1, events[^1].Layer);
    }

    [Fact]
    public void OverflowCannotBeReportedAsCompleteEvidence()
    {
        using AttentionEvidenceSession session = AttentionEvidence.Begin(1);
        AttentionEvidence.Observed(AttentionEvidenceKind.Cast, "first");
        AttentionEvidence.Observed(AttentionEvidenceKind.Cast, "second");
        Assert.Equal(1, session.Dropped);
        Assert.Throws<InvalidOperationException>(() => session.Snapshot());
    }

    [Fact]
    public async Task ConcurrentWritersPublishEveryObservation()
    {
        using AttentionEvidenceSession session = AttentionEvidence.Begin(4000);
        Task[] tasks = new Task[4];
        for (int i = 0; i < tasks.Length; i++)
            tasks[i] = Task.Run(() =>
            {
                for (int j = 0; j < 1000; j++) AttentionEvidence.Observed(AttentionEvidenceKind.Workspace, "worker", j);
            });
        await Task.WhenAll(tasks);
        Assert.Equal(4000, session.Snapshot().Length);
        Assert.Equal(0, session.Dropped);
    }

    [Fact]
    public void SessionOwnershipRejectsOverlapAndDisposesIdempotently()
    {
        AttentionEvidenceSession session = AttentionEvidence.Begin(1);
        try { Assert.Throws<InvalidOperationException>(() => AttentionEvidence.Begin(1)); }
        finally { session.Dispose(); }
        session.Dispose();
        Assert.False(AttentionEvidence.Enabled);
        using AttentionEvidenceSession next = AttentionEvidence.Begin(1);
        Assert.True(AttentionEvidence.Enabled);
    }

    [Fact]
    public void CpuExecutionRecordsActualRouteAndLeavesOutputIdentical()
    {
        using CpuBackend backend = new();
        using Tensor q = new(new TensorShape(1, 2, 3, 4), DType.F32);
        using Tensor k = new(new TensorShape(1, 2, 5, 4), DType.F32);
        using Tensor v = new(new TensorShape(1, 2, 5, 4), DType.F32);
        using Tensor baseline = new(q.Shape, DType.F32);
        using Tensor observed = new(q.Shape, DType.F32);
        q.AsSpan<float>().Fill(0.25f);
        k.AsSpan<float>().Fill(-0.5f);
        for (int i = 0; i < v.ElementCount; i++) v.AsSpan<float>()[i] = i / 40f;
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(baseline, q, k, v, null, 0.5f, false);
        backend.Attention(in call);
        using AttentionEvidenceSession session = AttentionEvidence.Begin(8);
        AttentionCall diagnostic = call with { Output = AttentionOperand.HeadMajor(observed) };
        backend.Attention(in diagnostic);
        AttentionEvidenceEvent execution = Assert.Single(session.Snapshot());
        Assert.Equal("cpu.simd", execution.Implementation);
        Assert.Equal(5, execution.ValidKeys);
        Assert.Equal(2, execution.Heads);
        Assert.True(baseline.AsReadOnlySpan<float>().SequenceEqual(observed.AsReadOnlySpan<float>()));
    }
}
