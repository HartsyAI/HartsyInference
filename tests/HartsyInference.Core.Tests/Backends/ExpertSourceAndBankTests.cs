using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Expert identity with banks, the source adapter, and cache behavior across two banks sharing a layer number.</summary>
public sealed class ExpertSourceAndBankTests
{
    private const int MatrixElements = 256;
    private const long ExpertBytes = 3 * MatrixElements * 4;

    private static ExpertWeights Weights(ExpertKey key) => new(
        key,
        new ExpertMatrix(new Tensor(new TensorShape(MatrixElements), DType.F32)),
        new ExpertMatrix(new Tensor(new TensorShape(MatrixElements), DType.F32)),
        new ExpertMatrix(new Tensor(new TensorShape(MatrixElements), DType.F32)));

    [Fact]
    public void ExpertKey_DefaultBankIsZeroAndBanksAreDistinct()
    {
        Assert.Equal(new ExpertKey(1, 2), new ExpertKey(1, 2, 0));
        Assert.NotEqual(new ExpertKey(1, 2), new ExpertKey(1, 2, 3));
        Assert.Equal(new ExpertLayerKey(3, 1), new ExpertKey(1, 2, 3).LayerKey);
        Assert.Equal("B3.L1.E2", new ExpertKey(1, 2, 3).ToString());
        Assert.Equal("L1.E2", new ExpertKey(1, 2).ToString());
    }

    [Fact]
    public void SourceBank_ResolvesEachExpertOnceAndCarriesItsBank()
    {
        int calls = 0;
        DelegateExpertSource source = new(ExpertBacking.Pack, key => { calls++; return Weights(key); });
        ExpertBank bank = ExpertBank.FromSource(source, layer: 4, count: 3, bank: 7);

        ExpertWeights first = bank.Get(1);
        Assert.Same(first, bank.Get(1));
        Assert.Equal(new ExpertKey(4, 1, 7), first.Key);
        Assert.Equal(1, calls);
        Assert.Equal(ExpertBacking.Pack, source.Backing);
        Assert.Equal(new ExpertLayerKey(7, 4), bank.LayerKey);
    }

    [Fact]
    public void SourceBank_RejectsAResolverThatReturnsTheWrongKey()
    {
        ExpertBank bank = ExpertBank.FromSource(
            new DelegateExpertSource(ExpertBacking.ResidentHost, _ => Weights(new ExpertKey(0, 0))), layer: 0, count: 2, bank: 0);

        Assert.Throws<InvalidOperationException>(() => bank.Get(1));
    }

    [Fact]
    public void Cache_AcceptsTwoBanksWithTheSameLayerNumberAndKeepsTheirExpertsApart()
    {
        using FakeExpertCache cache = new(4 * ExpertBytes);
        cache.RegisterBank(ExpertBank.FromSource(new DelegateExpertSource(ExpertBacking.ResidentHost, Weights), 0, 4, bank: 0));
        cache.RegisterBank(ExpertBank.FromSource(new DelegateExpertSource(ExpertBacking.ResidentHost, Weights), 0, 4, bank: 1));

        using ExpertLease lease = cache.Acquire([new ExpertKey(0, 1, 0), new ExpertKey(0, 1, 1)]);

        Assert.Equal(2, cache.Stats.ResidentExperts);
        Assert.NotSame(lease.Get(new ExpertKey(0, 1, 0)), lease.Get(new ExpertKey(0, 1, 1)));
    }

    [Fact]
    public void Cache_StillRejectsADifferentBankRegisteredUnderTheSameLayerKey()
    {
        using FakeExpertCache cache = new(4 * ExpertBytes);
        cache.RegisterBank(ExpertBank.FromSource(new DelegateExpertSource(ExpertBacking.ResidentHost, Weights), 0, 4, bank: 1));

        Assert.Throws<InvalidOperationException>(() =>
            cache.RegisterBank(ExpertBank.FromSource(new DelegateExpertSource(ExpertBacking.ResidentHost, Weights), 0, 4, bank: 1)));
    }
}
