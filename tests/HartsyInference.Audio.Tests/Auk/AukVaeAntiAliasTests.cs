using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Kaiser-sinc golden taps, load semantics and numerics of the anti-aliased SnakeBeta against a naive double-precision reference.</summary>
public sealed class AukVaeAntiAliasTests
{
    [Fact]
    public void KaiserSincFilter_MatchesCheckpointTaps()
    {
        float[] expected = [0.002f, 0.0094f, -0.0255f, -0.0577f, 0.1286f, 0.4432f, 0.4432f, 0.1286f, -0.0577f, -0.0255f, 0.0094f, 0.002f];
        float[] taps = AntiAliasedSnake.KaiserSincFilter(0.25, 0.3, 12);
        for (int i = 0; i < 12; i++) Assert.Equal(expected[i], taps[i], 3e-4f);
        Assert.Equal(1.0, taps.Sum(x => (double)x), 1e-6);
        for (int i = 0; i < 6; i++) Assert.Equal(taps[i], taps[11 - i], 1e-7f);
    }

    [Theory]
    [InlineData(true, 11, 0)]
    [InlineData(false, 5, 6)]
    public void DownPad_FollowsCausalFlag(bool causal, int left, int right)
    {
        using AntiAliasedSnake act = new(2, 2, 12, causal);
        Assert.Equal(left, act.DownPadLeft);
        Assert.Equal(right, act.DownPadRight);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
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
    public void CausalAndSymmetricDownPads_ProduceDifferentOutputs()
    {
        Dictionary<string, Tensor> w = new();
        AukVaeTestData.AddAntiAlias(w, new Random(1), "a", 2);
        using AntiAliasedSnake causal = new(2, 2, 12, true);
        using AntiAliasedSnake symmetric = new(2, 2, 12, false);
        causal.LoadWeights(w, "a");
        symmetric.LoadWeights(w, "a");
        using Tensor x = AukVaeTestData.Make(AukVaeTestData.Random(new Random(2), 2 * 10, -1, 1), 1, 2, 10);
        using Tensor a = causal.Forward(new CpuBackend(), x);
        using Tensor b = symmetric.Forward(new CpuBackend(), x);
        Assert.True(AukVaeTestData.Values(a).Zip(AukVaeTestData.Values(b), (p, q) => Math.Abs(p - q)).Max() > 1e-3f);
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
    [InlineData("a.upsample.filter")]
    [InlineData("a.downsample.lowpass.filter")]
    public void LoadWeights_MissingKey_Throws(string missing)
    {
        Dictionary<string, Tensor> w = new();
        AukVaeTestData.AddAntiAlias(w, new Random(6), "a", 3);
        w.Remove(missing);
        using AntiAliasedSnake act = new(3, 2, 12, true);
        Assert.Throws<KeyNotFoundException>(() => act.LoadWeights(w, "a"));
    }
}
