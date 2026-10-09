using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Prefix-cache scoping: one tenant's retained prefix is never handed to another, a different model is a different scope, and the parts of a key cannot run
/// into each other.</summary>
public sealed class PrefixCacheScopeTests
{
    [Fact]
    public void Tenants_Never_Share_A_Key_For_The_Same_Prefix()
    {
        Assert.NotEqual(PrefixCacheScope.Key("alice", "model", "chat-1"), PrefixCacheScope.Key("bob", "model", "chat-1"));
    }

    [Fact]
    public void A_Different_Model_Is_A_Different_Scope()
    {
        Assert.NotEqual(PrefixCacheScope.Key("alice", "model-a", "chat-1"), PrefixCacheScope.Key("alice", "model-b", "chat-1"));
    }

    [Fact]
    public void Length_Prefixes_Keep_Ambiguous_Names_Apart()
    {
        // Without the length prefixes these two would both read "x|y|z|p".
        Assert.NotEqual(PrefixCacheScope.Key("x|y", "z", "p"), PrefixCacheScope.Key("x", "y|z", "p"));
    }

    [Fact]
    public void A_Stored_Prefix_Is_Never_Returned_To_Another_Tenant()
    {
        TransformerConfig cfg = DynamicBatchSchedulerTests.Cfg();
        Dictionary<string, Tensor> w = DynamicBatchSchedulerTests.Weights(cfg);
        using CpuBackend backend = new();
        using GenericTransformer model = new(cfg);
        model.LoadWeights(w, "model");
        GenericTransformerModel adapter = new(model, backend);
        using PagedKvPool pool = new(cfg.NumLayers, cfg.NumKvHeads, cfg.HeadDim, pageSize: 4, maxPages: 16);
        using RetainedSequenceStore store = new(4, 1L << 30);

        // A live cache under alice's key: the store keeps only entries that hold one.
        RetainedSequence sequence = new();
        sequence.Update(adapter.CreateSequenceState(new SequenceStateOptions(8, pool)), [1, 2], bytes: 64);
        string alices = PrefixCacheScope.Key("alice", "model", "chat-1");
        string bobs = PrefixCacheScope.Key("bob", "model", "chat-1");
        store.CheckIn(alices, sequence);

        Assert.Null(store.Checkout(bobs));
        RetainedSequence? back = store.Checkout(alices);
        Assert.NotNull(back);
        Assert.Equal(new[] { 1, 2 }, back.TokenIds);
        back.Dispose();
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
