using HartsyInference.Core.Backends;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary>Guards the placement no-op contract: an all-defaults config must be indistinguishable from a build
/// without placement support — most importantly a zero-length cache-key contribution, so existing pipeline cache
/// keys stay byte-identical for single-device engines.</summary>
public sealed class PlacementConfigTests
{
    [Fact]
    public void Defaults_AreSingle_WithEmptyCacheKey()
    {
        PlacementConfig config = new();
        Assert.True(config.IsSingle);
        Assert.Equal("", config.CacheKey());
        Assert.True(PlacementConfig.Single.IsSingle);
        Assert.Equal("", PlacementConfig.Single.CacheKey());
    }

    [Fact]
    public void AnyNonDefault_LeavesSingle_AndKeysDistinctly()
    {
        PlacementConfig te = new() { TextEncoderDevice = "cuda:1" };
        PlacementConfig vae = new() { VaeDevice = "cuda:1" };
        PlacementConfig shard = new() { ShardDevices = ["cuda:0", "cuda:1"] };
        PlacementConfig shardRatio = new() { ShardDevices = ["cuda:0", "cuda:1"], ShardRatios = [0.7f, 0.3f] };
        PlacementConfig cfg = new() { CfgParallelDevice = "cuda:1" };
        PlacementConfig dit = new() { ShardDevices = ["cuda:0", "cuda:1"], EnableDitSharding = true };

        PlacementConfig[] all = [te, vae, shard, shardRatio, cfg, dit];
        HashSet<string> keys = new(StringComparer.Ordinal);
        foreach (PlacementConfig config in all)
        {
            Assert.False(config.IsSingle);
            string key = config.CacheKey();
            Assert.NotEqual("", key);
            Assert.True(keys.Add(key), $"cache key collision: '{key}'");
        }
    }

    [Fact]
    public void CacheKey_IsOrderSensitive_ForShardDevices()
    {
        PlacementConfig ab = new() { ShardDevices = ["cuda:0", "cuda:1"] };
        PlacementConfig ba = new() { ShardDevices = ["cuda:1", "cuda:0"] };
        Assert.NotEqual(ab.CacheKey(), ba.CacheKey());
    }
}
