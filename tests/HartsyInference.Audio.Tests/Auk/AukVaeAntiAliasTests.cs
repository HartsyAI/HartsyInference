using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Kaiser-sinc golden taps, load semantics and numerics of the anti-aliased SnakeBeta against a naive double-precision reference.</summary>
public sealed class AukVaeAntiAliasTests
{
    [Theory]
    [InlineData(true, 11, 0)]
    public void DownPad_FollowsCausalFlag(bool causal, int left, int right)
    {
        using AntiAliasedSnake act = new(2, 2, 12, causal);
        Assert.Equal(left, act.DownPadLeft);
        Assert.Equal(right, act.DownPadRight);
    }

    [Theory]
    [InlineData(true)]
    public void Forward_MatchesReference_AndPreservesLength(bool causalDown)
    {
        const int Channels = 3;
        Dictionary<string, Tensor> w = new();
        AukVaeTestData.AddAntiAlias(w, new Random(4), "a", Channels);
        using AntiAliasedSnake act = new(Channels, 2, 12, causalDown);
        act.LoadWeights(w, "a");

        Random rng = new(9);
        const int T = 17;
        float[] data = AukVaeTestData.Random(rng, Channels * T, -2, 2);
        using Tensor x = AukVaeTestData.Make(data, 1, Channels, T);
        using Tensor y = act.Forward(new CpuBackend(), x);

        Assert.Equal(T, (int)y.Shape[2]);
        double[][] xr = Enumerable.Range(0, Channels).Select(c => Enumerable.Range(0, T).Select(t => (double)data[c * T + t]).ToArray()).ToArray();
        double[] expected = AukVaeTestData.AntiAlias(w, "a", xr, causalDown).SelectMany(r => r).ToArray();
        Assert.True(AukVaeTestData.MaxAbsDiff(expected, AukVaeTestData.Values(y)) < 1e-4);
    }

    [Fact]
    public void LoadWeights_ExponentiatesSnakeParameters_AndScalesUpFilterByRatio()
    {
        Dictionary<string, Tensor> w = new();
        AukVaeTestData.AddAntiAlias(w, new Random(6), "a", 3);
        using AntiAliasedSnake act = new(3, 2, 12, true);
        act.LoadWeights(w, "a");
        Tensor[] loaded = act.EnumerateWeights().ToArray();
        float[] alpha = AukVaeTestData.Values(w["a.act.alpha"]), up = AukVaeTestData.Values(w["a.upsample.filter"]);
        for (int c = 0; c < 3; c++) Assert.Equal(MathF.Exp(alpha[c]), AukVaeTestData.Values(loaded[0])[c], 1e-6f);
        Assert.Equal(2f * up[3], AukVaeTestData.Values(loaded[2])[12 + 3], 1e-6f);
    }

    [Theory]
    [InlineData("a.act.beta")]
    public void LoadWeights_MissingKey_Throws(string missing)
    {
        Dictionary<string, Tensor> w = new();
        AukVaeTestData.AddAntiAlias(w, new Random(6), "a", 3);
        w.Remove(missing);
        using AntiAliasedSnake act = new(3, 2, 12, true);
        Assert.Throws<KeyNotFoundException>(() => act.LoadWeights(w, "a"));
    }
}
