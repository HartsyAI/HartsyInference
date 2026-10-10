using HartsyInference.Audio.Models.Vits;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Regression coverage for <see cref="VitsWaveNet.LoadWeights"/>'s new <c>convKeySuffix</c> parameter
/// (added so IndexTTS-2's S2Mel DiT can load the same WaveNet shape from its <c>SConv1d</c>-nested
/// <c>in_layers.0.conv.conv.weight_g</c> keys): confirms the default (empty suffix) still loads VITS's own
/// flat <c>in_layers.0.weight_g</c> naming unchanged, and that a non-empty suffix loads the nested naming
/// instead — proving the generalization didn't disturb the existing VITS call sites.</summary>
public sealed unsafe class VitsWaveNetTests
{
    private const int Hidden = 8;
    private const int Kernel = 3;
    private const int Layers = 2;

    private static Dictionary<string, Tensor> BuildWeights(Random rng, string prefix, string convKeySuffix)
    {
        Dictionary<string, Tensor> w = [];
        for (int i = 0; i < Layers; i++)
        {
            int resSkipCh = i < Layers - 1 ? 2 * Hidden : Hidden;
            w[$"{prefix}.in_layers.{i}{convKeySuffix}.weight"] = Rand(rng, 2 * Hidden, Hidden, Kernel);
            w[$"{prefix}.in_layers.{i}{convKeySuffix}.bias"] = Rand(rng, 2 * Hidden);
            w[$"{prefix}.res_skip_layers.{i}{convKeySuffix}.weight"] = Rand(rng, resSkipCh, Hidden, 1);
            w[$"{prefix}.res_skip_layers.{i}{convKeySuffix}.bias"] = Rand(rng, resSkipCh);
        }
        return w;
    }

    private static Tensor Rand(Random rng, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)((rng.NextDouble() * 2 - 1) * 0.1);
        return t;
    }

    [Fact]
    public void LoadWeights_WithEmptySuffix_LoadsFlatVitsStyleKeys_AndRuns()
    {
        Random rng = new(1);
        Dictionary<string, Tensor> w = BuildWeights(rng, "wn", convKeySuffix: "");
        VitsWaveNet wn = new(Hidden, Kernel, dilationRate: 1, layers: Layers);
        wn.LoadWeights(w, "wn");   // default convKeySuffix: "" — must still work.

        using CpuBackend backend = new();
        Tensor x = Rand(rng, 1, Hidden, 10);
        using Tensor output = wn.Forward(backend, x, 10);
        x.Dispose();

        Assert.Equal(new TensorShape(1, Hidden, 10), output.Shape);
        foreach (float v in output.AsSpan<float>()) Assert.True(float.IsFinite(v));
        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void LoadWeights_WrongSuffix_ThrowsKeyNotFound()
    {
        Random rng = new(3);
        Dictionary<string, Tensor> w = BuildWeights(rng, "wavenet", convKeySuffix: ".conv.conv");
        VitsWaveNet wn = new(Hidden, Kernel, dilationRate: 1, layers: Layers);
        Assert.Throws<KeyNotFoundException>(() => wn.LoadWeights(w, "wavenet"));   // default suffix won't find the nested keys.
        foreach (Tensor t in w.Values) t.Dispose();
    }
}
