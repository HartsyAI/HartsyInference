using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Tensor identity of <see cref="ExpertBank"/> and the compact routing readback.</summary>
public sealed class ExpertBankAndRoutingTests
{
    private static ExpertWeights Build(ExpertKey key)
    {
        return new ExpertWeights(
            key,
            new ExpertMatrix(new Tensor(new TensorShape(4), DType.F32)),
            new ExpertMatrix(new Tensor(new TensorShape(4), DType.F32)),
            new ExpertMatrix(new Tensor(new TensorShape(4), DType.F32)));
    }

    [Fact]
    public void Get_ResolvesOncePerExpertAndReturnsTheSameTensorObjects()
    {
        int calls = 0;
        ExpertBank bank = new(3, 8, key => { calls++; return Build(key); });
        ExpertWeights a = bank.Get(5);
        ExpertWeights b = bank.Get(5);
        Assert.Same(a, b);
        Assert.Same(a.W1.Weight, b.W1.Weight);
        Assert.Equal(1, calls);
        Assert.Equal(new ExpertKey(3, 5), a.Key);
        Assert.Equal(1, bank.ResolvedCount);
    }

    [Fact]
    public void Get_RejectsOutOfRangeAndMismatchedKeys()
    {
        ExpertBank bank = new(0, 2, key => Build(new ExpertKey(9, 9)));
        Assert.Throws<ArgumentOutOfRangeException>(() => bank.Get(2));
        Assert.ThrowsAny<Exception>(() => bank.Get(0));
    }

    [Fact]
    public void Weights_ListEachDistinctTensorOnce()
    {
        ExpertWeights weights = Build(new ExpertKey(0, 0));
        Assert.Equal(3, weights.Tensors.Count);
        Assert.Equal(3 * 16, weights.Bytes);
    }

    [Fact]
    public void DistinctKeys_DedupsAndSortsIds()
    {
        ExpertKey[] keys = ExpertRouting.DistinctKeys([5, 2, 5, 9, 2, 0], layer: 4, expertCount: 16);
        Assert.Equal([new ExpertKey(4, 0), new ExpertKey(4, 2), new ExpertKey(4, 5), new ExpertKey(4, 9)], keys);
    }

    [Fact]
    public void DistinctKeys_RejectsIdsOutsideTheExpertRange()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ExpertRouting.DistinctKeys([16], 0, 16));
        Assert.Throws<ArgumentOutOfRangeException>(() => ExpertRouting.DistinctKeys([-1], 0, 16));
    }
}
