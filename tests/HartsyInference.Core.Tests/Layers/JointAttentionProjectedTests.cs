using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Layers;

/// <summary>The shared tail of a joint attention (concatenation, one attention call, split) over streams the model projected, normed and
/// rotated itself: the same ops <see cref="JointAttention.Attend"/> ends with, and the same ownership.</summary>
public sealed class JointAttentionProjectedTests
{
    private const int Heads = 4, HeadDim = 2, Inner = Heads * HeadDim, ImgRows = 6, TxtRows = 3;

    private static Tensor W(int rows) => new(new TensorShape(rows, Inner), DType.F32);

    private static AttentionLayer Stream(int id) => new(
        new AttentionLayerConfig { Heads = Heads, KvHeads = Heads, HeadDim = HeadDim, QkNorm = AttentionQkNorm.PerHead },
        new AttentionLayerWeights
        {
            Query = W(id * 100 + 1), Key = W(id * 100 + 2), Value = W(id * 100 + 3), Output = W(id * 100 + 4),
            QueryNorm = W(id * 100 + 6), KeyNorm = W(id * 100 + 7), QueryNormEps = 1e-6f, KeyNormEps = 1e-6f,
        });

    private static JointAttention Joint(bool allowF16 = true, params int[] order) =>
        new(new JointAttentionConfig { Heads = Heads, HeadDim = HeadDim, ConcatOrder = order.Length == 0 ? [1, 0] : order, AllowF16 = allowF16 },
            [Stream(1), Stream(2)]);

    private static Tensor[] Operands(int rows, int batch = 1, DType? dtype = null) => [new Tensor(new TensorShape(batch, rows, Heads, HeadDim), dtype ?? DType.F32)];

    private static unsafe bool IsDisposed(Tensor t)
    {
        try
        {
            _ = (nint)t.DataPointer;
            return false;
        }
        catch (ObjectDisposedException)
        {
            return true;
        }
    }

    [Fact]
    public void RunsTheTailAttendEndsWith_InTheConfiguredRowOrder()
    {
        AttentionTraceBackend backend = new();
        Tensor[] q = [Operands(ImgRows)[0], Operands(TxtRows)[0]], k = [Operands(ImgRows)[0], Operands(TxtRows)[0]], v = [Operands(ImgRows)[0], Operands(TxtRows)[0]];
        Tensor?[] attended = [null, null];

        Joint().AttendProjected(backend, q, k, v, attended, default);

        string[] tail =
        [
            "Concat rows 3+6", "Concat rows 3+6", "Concat rows 3+6",
            "Permute0213 9,4,2", "Permute0213 9,4,2", "Permute0213 9,4,2", "Attention Dense heads=4/4 bounded=True",
            "Permute0213 4,9,2", "SliceRows 3@0", "SliceRows 6@3",
        ];
        Assert.Equal(tail, backend.Ops);
        Assert.Equal(new TensorShape(1, ImgRows, Inner), attended[0]!.Shape);
        Assert.Equal(new TensorShape(1, TxtRows, Inner), attended[1]!.Shape);

        AttentionTraceBackend whole = new();
        Span<Tensor?> viaAttend = [null, null];
        Joint().Attend(whole, [new Tensor(new TensorShape(1, ImgRows, Inner), DType.F32), new Tensor(new TensorShape(1, TxtRows, Inner), DType.F32)],
            viaAttend, default);
        Assert.Equal(tail, whole.Ops.Skip(whole.Ops.Count - tail.Length));
    }

    [Fact]
    public void TakesOverAndDisposesTheOperands_AndHandsTheSplitRowsToTheCaller()
    {
        AttentionTraceBackend backend = new();
        Tensor[] q = [Operands(ImgRows)[0], Operands(TxtRows)[0]], k = [Operands(ImgRows)[0], Operands(TxtRows)[0]], v = [Operands(ImgRows)[0], Operands(TxtRows)[0]];
        Tensor?[] attended = [null, null];

        Joint().AttendProjected(backend, q, k, v, attended, default);

        Assert.All(q.Concat(k).Concat(v), t => Assert.True(IsDisposed(t)));
        Assert.All(attended, t => Assert.False(IsDisposed(t!)));
        AttentionCall call = Assert.Single(backend.Calls);
        Assert.True(IsDisposed(call.Query.Storage));
        Assert.Equal(new TensorShape(1, Heads, ImgRows + TxtRows, HeadDim), call.Query.Storage.Shape);
    }

    [Fact]
    public void TheJointContextReachesTheCall_AndTheAssertionFollowsTheBatch()
    {
        AttentionTraceBackend backend = new();
        Tensor bias = new(new TensorShape(1, 1, ImgRows + TxtRows, ImgRows + TxtRows), DType.F32);
        Tensor[] q = [Operands(ImgRows)[0], Operands(TxtRows)[0]], k = [Operands(ImgRows)[0], Operands(TxtRows)[0]], v = [Operands(ImgRows)[0], Operands(TxtRows)[0]];

        Joint(allowF16: false).AttendProjected(backend, q, k, v, new Tensor?[2], new AttentionCallContext { Bias = bias });

        AttentionCall call = Assert.Single(backend.Calls);
        Assert.Same(bias, call.Mask.Bias);
        Assert.False(call.Numerics.LegacyAssertedBounded);
    }

    [Fact]
    public void AFailedAttentionHandsBackNoOutputsAndLeavesNothingUndisposed()
    {
        AttentionTraceBackend backend = new() { FailAttention = true };
        Tensor[] q = [Operands(ImgRows)[0], Operands(TxtRows)[0]], k = [Operands(ImgRows)[0], Operands(TxtRows)[0]], v = [Operands(ImgRows)[0], Operands(TxtRows)[0]];
        Tensor?[] attended = [null, null];

        Assert.Throws<OutOfMemoryException>(() => Joint().AttendProjected(backend, q, k, v, attended, default));

        Assert.All(attended, Assert.Null);
        Assert.All(q.Concat(k).Concat(v), t => Assert.True(IsDisposed(t)));
        Assert.True(IsDisposed(Assert.Single(backend.Calls).Output.Storage));
    }

    [Fact]
    public void MismatchedOperandsAndABatchedF16CallAreRefusedBeforeAnyOp()
    {
        AttentionTraceBackend backend = new();
        Tensor[] one = [Operands(ImgRows)[0]];
        Tensor[] two = [Operands(ImgRows)[0], Operands(TxtRows)[0]];

        Assert.Throws<ArgumentException>(() => Joint().AttendProjected(backend, one, two, two, new Tensor?[2], default));
        Assert.Throws<ArgumentException>(() => Joint().AttendProjected(backend, two, two, two, new Tensor?[1], default));
        Assert.Throws<ArgumentException>(() => Joint().AttendProjected(backend, two, [two[0], Operands(TxtRows + 1)[0]], two, new Tensor?[2], default));
        Tensor[] f16 = [Operands(ImgRows, 2, DType.F16)[0], Operands(TxtRows, 2, DType.F16)[0]];
        Assert.Throws<NotSupportedException>(() => Joint().AttendProjected(backend, f16, f16, f16, new Tensor?[2], default));

        Assert.Empty(backend.Ops);
    }
}
