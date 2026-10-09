using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests.Moe;

/// <summary>Host-resident expert cache: admitted experts are resident, and their tensors are the caller's, never copies.</summary>
public sealed class HostExpertCacheTests
{
    [Fact]
    public void AcquireResident_ReturnsKeysAdmittedByAcquire_WithoutCopying()
    {
        const int experts = 3;
        Tensor[] gate = new Tensor[experts];
        Tensor[] down = new Tensor[experts];
        Tensor[] up = new Tensor[experts];
        for (int i = 0; i < experts; i++)
        {
            gate[i] = new Tensor(new TensorShape(2, 2), DType.F32);
            down[i] = new Tensor(new TensorShape(2, 2), DType.F32);
            up[i] = new Tensor(new TensorShape(2, 2), DType.F32);
        }
        ExpertBank bank = new(0, experts, key => new ExpertWeights(key, new ExpertMatrix(gate[key.Expert]),
            new ExpertMatrix(down[key.Expert]), new ExpertMatrix(up[key.Expert])), 0, ExpertBacking.ResidentHost);
        long budget = experts * 3L * 2 * 2 * sizeof(float);
        HostExpertCache cache = new(budget, [bank]);
        ExpertKey[] keys = [bank.Key(0), bank.Key(1), bank.Key(2)];

        // Admit the experts; the host cache has no upload, so the bank has resolved each one and nothing else happened.
        cache.Acquire(keys).Dispose();
        Assert.Equal(experts, bank.ResolvedCount);

        bool[] resident = new bool[experts];
        Assert.Equal(experts, cache.LookupResident(keys, resident));
        Assert.All(resident, static r => Assert.True(r));

        List<ExpertKey> misses = new(experts);
        using ExpertLease lease = new();
        cache.AcquireResident(keys, misses, lease);
        Assert.Empty(misses);
        Assert.Equal(experts, lease.Count);
        for (int i = 0; i < experts; i++)
        {
            Assert.Same(gate[i], lease[i].W1.Weight);
            Assert.Same(down[i], lease[i].W2.Weight);
            Assert.Same(up[i], lease[i].W3.Weight);
        }
    }

    [Fact]
    public void AcquireResident_ReportsUnadmittedKeysAsMisses()
    {
        Tensor w = new(new TensorShape(2, 2), DType.F32);
        ExpertBank bank = new(0, 2, key => new ExpertWeights(key, new ExpertMatrix(w), new ExpertMatrix(w), new ExpertMatrix(w)), 0);
        HostExpertCache cache = new(1 << 20, [bank]);
        ExpertKey key = bank.Key(1);

        List<ExpertKey> misses = new(2);
        using ExpertLease lease = cache.AcquireResident([key], misses);
        Assert.Equal([key], misses);
        Assert.Empty(lease);
    }
}
