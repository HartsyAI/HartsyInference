using HartsyInference.Audio.Models.Vits;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Regression coverage for <see cref="VitsWaveNet.LoadWeights"/>'s new <c>convKeySuffix</c> parameter
/// (added so IndexTTS-2's S2Mel DiT can load the same WaveNet shape from its <c>SConv1d</c>-nested
/// <c>in_layers.0.conv.conv.weight_g</c> keys): confirms the default (empty suffix) still loads VITS's own
/// flat <c>in_layers.0.weight_g</c> naming unchanged, and that a non-empty suffix loads the nested naming
/// instead — proving the generalization didn't disturb the existing VITS call sites.</summary>
public sealed unsafe class VitsWaveNetTests(ITestOutputHelper output)
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

    /// <summary><c>F.pad(mode="reflect")</c> over time, as IndexTTS-2's <c>SConv1d</c> pads its WaveNet convs: the edge
    /// frame is not repeated.</summary>
    [Fact]
    public void ReflectPadTime_MirrorsAroundTheEdgeFrames()
    {
        using CpuBackend backend = new();
        using Tensor x = new(new TensorShape(1, 2, 6), DType.F32);
        Span<float> xs = x.AsSpan<float>();
        for (int c = 0; c < 2; c++) for (int t = 0; t < 6; t++) xs[c * 6 + t] = c * 100 + t;
        using Tensor padded = VitsWaveNet.ReflectPadTime(backend, x, 2, 6, 3, 2);
        int[] want = [3, 2, 1, 0, 1, 2, 3, 4, 5, 4, 3];
        ReadOnlySpan<float> p = padded.AsSpan<float>();
        for (int c = 0; c < 2; c++)
            for (int t = 0; t < want.Length; t++)
                Assert.Equal(c * 100 + want[t], p[c * want.Length + t]);
        Assert.Throws<ArgumentException>(() => VitsWaveNet.ReflectPadTime(backend, x, 2, 6, 6, 0));
    }

    /// <summary>The reflect-padded stack (dilation 2, so each layer pads by more than one frame) on each GPU backend
    /// against the CPU backend.</summary>
    [Theory]
    [Trait("Category", "GpuIntegration")]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void ReflectPaddedForward_MatchesCpu(string kind)
    {
        if (!BackendGate.TryOpen(kind, output.WriteLine, out IBackend? opened)) return;
        using IBackend gpu = opened!;
        const int kernel = 5, t = 37;
        Random rng = new(5);
        Dictionary<string, Tensor> w = [];
        for (int i = 0; i < Layers; i++)
        {
            int resSkipCh = i < Layers - 1 ? 2 * Hidden : Hidden;
            w[$"wn.in_layers.{i}.conv.conv.weight"] = Rand(rng, 2 * Hidden, Hidden, kernel);
            w[$"wn.in_layers.{i}.conv.conv.bias"] = Rand(rng, 2 * Hidden);
            w[$"wn.res_skip_layers.{i}.conv.conv.weight"] = Rand(rng, resSkipCh, Hidden, 1);
            w[$"wn.res_skip_layers.{i}.conv.conv.bias"] = Rand(rng, resSkipCh);
        }
        VitsWaveNet wn = new(Hidden, kernel, dilationRate: 2, layers: Layers, reflectPad: true);
        wn.LoadWeights(w, "wn", convKeySuffix: ".conv.conv");
        using Tensor x = Rand(rng, 1, Hidden, t);
        using CpuBackend cpu = new();
        using Tensor want = wn.Forward(cpu, x, t);
        using Tensor got = wn.Forward(gpu, x, t);
        ReadOnlySpan<float> g = got.AsSpan<float>(), e = want.AsSpan<float>();
        double maxAbs = 0;
        for (int i = 0; i < e.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(g[i] - e[i]));
        output.WriteLine($"{kind}: max |gpu - cpu| {maxAbs:E2}");
        Assert.True(maxAbs < 1e-4, $"{kind} differs from CPU by {maxAbs:E2}");
        foreach (Tensor tensor in w.Values) tensor.Dispose();
    }
}
