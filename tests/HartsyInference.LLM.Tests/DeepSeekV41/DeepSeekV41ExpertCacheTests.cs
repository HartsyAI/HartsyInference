using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.ModelAssets.Quant;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

public sealed class DeepSeekV41ExpertCacheTests
{
    private static DeepSeekV41SwigluWeights Tiny() => new(1, 1, new[] { 1f }, new[] { 1f }, new[] { 1f });

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

    [Fact]
    public void ReadMatrix_Widens_An_Unquantized_Bf16_Expert_Matrix()
    {
        using Tensor w = new(new TensorShape(2, 3), DType.BF16);
        ushort[] bits = [0x3F80, 0x4000, 0x4040, 0xBF80, 0xC000, 0xC040]; // 1, 2, 3, -1, -2, -3
        bits.CopyTo(w.AsSpan<ushort>());
        Assert.Equal([1f, 2f, 3f, -1f, -2f, -3f], DeepSeekV41ExpertLoader.ReadMatrix("experts.0.w1", w, null, 2, 3));
    }

    [Fact]
    public void ReadMatrix_Refuses_A_Wrong_Shape_Or_A_Wrong_Rank_With_The_Key_Named()
    {
        using Tensor matrix = new(new TensorShape(2, 3), DType.BF16);
        using Tensor vector = new(new TensorShape(6), DType.BF16);
        Assert.Contains("experts.0.w2", Assert.Throws<HartsyInferenceException>(() =>
            DeepSeekV41ExpertLoader.ReadMatrix("experts.0.w2", matrix, null, 3, 2)).Message);
        Assert.Contains("rank 1", Assert.Throws<HartsyInferenceException>(() =>
            DeepSeekV41ExpertLoader.ReadMatrix("experts.0.w3", vector, null, 2, 3)).Message);
    }
}
