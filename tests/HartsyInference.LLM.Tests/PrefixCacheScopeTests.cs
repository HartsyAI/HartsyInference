using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Tests.DeepSeekV41;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Prefix-cache scoping: one tenant's retained prefix is never handed to another, a different model is a different scope, the parts of a key cannot run
/// into each other, and a slot holding the V4.1 host yields no key at all.</summary>
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

    [Fact]
    public void A_Slot_Holding_The_V41_Host_Never_Yields_A_Key()
    {
        string directory = Directory.CreateTempSubdirectory("prefix-scope-v41-").FullName;
        try
        {
            DeepSeekV41ModelFixtureCheckpoint.Write(directory);
            DeepSeekV41ModelFixtureCheckpoint.WriteTokenizer(directory);
            using CpuBackend cpu = new();
            using DeepSeekV41TextModel host = HfTextDirectoryLoader.Load(HfCheckpointDirectory.TryProbe(directory)!, cpu);
            TextRequest request = new() { Messages = [], PrefixCacheKey = "chat-1", TenantId = "alice" };

            // A slot holding another model keys the prefix by tenant and loaded path; a request with no prefix key takes no hit anywhere.
            TextDeviceSlot other = new() { LoadedPath = "/models/plain.gguf" };
            Assert.True(PrefixCacheScope.TryKey(other, request, out string? key));
            Assert.Equal(PrefixCacheScope.Key("alice", "/models/plain.gguf", "chat-1"), key);
            Assert.False(PrefixCacheScope.TryKey(other, request with { PrefixCacheKey = null }, out _));

            TextDeviceSlot v41 = new() { LoadedPath = directory, DeepSeekV41 = host };
            Assert.False(PrefixCacheScope.TryKey(v41, request, out string? none));
            Assert.Null(none);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
