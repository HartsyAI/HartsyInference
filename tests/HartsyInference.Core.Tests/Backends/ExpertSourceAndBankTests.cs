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

    [Fact]
    public void Routing_IntoANonZeroBankAcquiresThatBanksExperts_NotBankZeros()
    {
        using FakeExpertCache cache = new(4 * ExpertBytes);
        cache.RegisterBank(ExpertBank.FromSource(new DelegateExpertSource(ExpertBacking.ResidentHost, Weights), 0, 4, bank: 0));
        cache.RegisterBank(ExpertBank.FromSource(new DelegateExpertSource(ExpertBacking.ResidentHost, Weights), 0, 4, bank: 1));

        ExpertKey[] routed = ExpertRouting.DistinctKeys([2], layer: 0, expertCount: 4, bank: 1);
        using ExpertLease lease = cache.Acquire(routed);

        Assert.Equal(1, cache.Stats.ResidentExperts);
        Assert.Same(lease.Get(new ExpertKey(0, 2, 1)), lease.Get(routed[0]));
    }
}
