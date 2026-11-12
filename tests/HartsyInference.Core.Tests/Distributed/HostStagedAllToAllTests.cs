using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Distributed;

/// <summary>The host-staged all-to-all: each sender's destination-major chunks land source-major on every receiver, with the counts matrix
/// deciding the sizes.</summary>
public sealed class HostStagedAllToAllTests
{
    [Fact]
    public void AllToAll_DeliversEachChunkToItsDestination()
    {
        // counts[i, j] = elements rank i sends to rank j; a zero pair exchanges nothing.
        long[,] counts = { { 1, 2, 0 }, { 0, 3, 1 }, { 2, 0, 2 } };
        int ranks = 3;
        Tensor[] send = new Tensor[ranks], recv = new Tensor[ranks];
        for (int i = 0; i < ranks; i++)
        {
            long total = 0;
            for (int j = 0; j < ranks; j++) total += counts[i, j];
            send[i] = new Tensor(new TensorShape(total + 1), DType.F32);
            for (int j = 0; j < ranks; j++)
            {
                long offset = AllToAllLayout.SendOffset(counts, i, j);
                for (int k = 0; k < counts[i, j]; k++) send[i].AsSpan<float>()[(int)(offset + k)] = i * 100 + j * 10 + k;
            }
        }
        for (int j = 0; j < ranks; j++)
        {
            long total = 0;
            for (int i = 0; i < ranks; i++) total += counts[i, j];
            recv[j] = new Tensor(new TensorShape(total + 1), DType.F32);
        }

        using HostStagedComm comm = new(ranks);
        comm.AllToAll(send, recv, counts);

        for (int j = 0; j < ranks; j++)
        {
            for (int i = 0; i < ranks; i++)
            {
                long offset = AllToAllLayout.RecvOffset(counts, i, j);
                for (int k = 0; k < counts[i, j]; k++)
                {
                    Assert.Equal(i * 100 + j * 10 + k, recv[j].AsReadOnlySpan<float>()[(int)(offset + k)]);
                }
            }
        }
        foreach (Tensor t in send) t.Dispose();
        foreach (Tensor t in recv) t.Dispose();
    }

    [Fact]
    public void AllToAll_RefusesAChunkLargerThanItsBuffer()
    {
        long[,] counts = { { 5, 0 }, { 0, 0 } };
        Tensor[] send = [new Tensor(new TensorShape(2), DType.F32), new Tensor(new TensorShape(1), DType.F32)];
        Tensor[] recv = [new Tensor(new TensorShape(5), DType.F32), new Tensor(new TensorShape(1), DType.F32)];
        using HostStagedComm comm = new(2);
        Assert.Throws<ArgumentException>(() => comm.AllToAll(send, recv, counts));
        foreach (Tensor t in send) t.Dispose();
        foreach (Tensor t in recv) t.Dispose();
    }
}
