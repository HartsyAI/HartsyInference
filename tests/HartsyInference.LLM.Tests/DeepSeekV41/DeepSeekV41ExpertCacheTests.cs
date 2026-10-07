using HartsyInference.Core.Exceptions;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.ModelAssets.Quant;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

public sealed class DeepSeekV41ExpertCacheTests
{
    private static DeepSeekV41SwigluWeights Tiny() => new(1, 1, [1f], [1f], [1f]);

    [Fact]
    public void Hits_Are_Served_Without_Reloading_And_The_Least_Recently_Used_Expert_Is_Evicted()
    {
        List<int> loads = [];
        DeepSeekV41ExpertCache cache = new(e => { loads.Add(e); return Tiny(); }, 2);
        cache.GetExpert(0);
        cache.GetExpert(1);
        cache.GetExpert(0);   // 0 is now most recent
        cache.GetExpert(2);   // evicts 1
        cache.GetExpert(0);   // still cached
        cache.GetExpert(1);   // reloaded
        Assert.Equal([0, 1, 2, 1], loads);
        Assert.Equal(2, cache.Count);
    }

    [Fact]
    public void Same_Instance_Is_Returned_For_A_Cached_Expert()
    {
        DeepSeekV41ExpertCache cache = new(_ => Tiny(), 4);
        Assert.Same(cache.GetExpert(3), cache.GetExpert(3));
    }

    [Fact]
    public void Rejects_Zero_Capacity()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DeepSeekV41ExpertCache(_ => Tiny(), 0));
    }

    [Fact]
    public void Loader_Refuses_Experts_Whose_Shapes_Do_Not_Match_The_Layer_Widths()
    {
        string dir = Path.Combine(Path.GetTempPath(), "dsv41-expert-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            TinyDeepSeekV41Checkpoint.Write(dir, QuantFlavor.Official);
            using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(dir);
            DeepSeekV41ExpertBank bank = checkpoint.ExpertBank(0);
            // the synthetic experts are [2, 64] logical, nothing like the widths asked for here
            HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => DeepSeekV41ExpertLoader.Load(bank, 0, 8, 8));
            Assert.Contains("experts.0.w1", ex.Message);
        }
        finally
        {
            Directory.Delete(dir, true);
        }
    }
}
