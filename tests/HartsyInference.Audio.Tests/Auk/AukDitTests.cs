using Xunit;
using HartsyInference.Audio.Models.Auk;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;

namespace HartsyInference.Audio.Tests.Auk;

/// <summary>AuK DiT on a tiny synthetic config: key/shape validation, forward vs an independent double-precision reference (layout, RoPE positions, noisy-slice extraction, SwiGLU, CFG drops), checkpoint-driven RoPE tables. Real-weight parity is pending.</summary>
public sealed unsafe class AukDitTests
{
    private static readonly AukConfig Cfg = AukDitReference.Tiny;

    private static (AukDit Dit, Dictionary<string, (long[] Shape, float[] Data)> W) Build(int seed = 1, float[]? invFreq = null)
    {
        Dictionary<string, (long[] Shape, float[] Data)> w = AukDitReference.RandomWeights(Cfg, seed, invFreq);
        AukDit dit = new(Cfg);
        dit.LoadWeights(AukDitReference.ToTensors(w));
        return (dit, w);
    }

    private static double[][] Rows(int n, int d, int seed)
    {
        Random rng = new(seed);
        return Enumerable.Range(0, n).Select(_ => Enumerable.Range(0, d).Select(_ => rng.NextDouble() * 2 - 1).ToArray()).ToArray();
    }

    private static Tensor Pack(double[][] rows)
    {
        float[] flat = rows.SelectMany(r => r.Select(v => (float)v)).ToArray();
        return AukDitReference.ToTensor([1, rows.Length, rows[0].Length], flat);
    }

    private static double MaxDiff(Tensor t, double[][] expected)
    {
        float* p = (float*)t.DataPointer;
        Assert.Equal(expected.Length * expected[0].Length, (int)t.ElementCount);
        double m = 0;
        for (int i = 0; i < expected.Length; i++)
            for (int d = 0; d < expected[i].Length; d++)
                m = Math.Max(m, Math.Abs(p[i * expected[i].Length + d] - expected[i][d]));
        return m;
    }

    [Fact]
    public void KeySet_MatchesBaseCheckpointCount()
    {
        // base.json lists 420 tensors, two of which (layer_scale, layer_weights) belong to the thinker fusion.
        Assert.Equal(418, AukDitReference.KeyShapes(AukConfig.Default).Count);
    }

    [Fact]
    public void LoadWeights_MissingOrMisshapenKey_Throws()
    {
        Dictionary<string, (long[] Shape, float[] Data)> w = AukDitReference.RandomWeights(Cfg, 3);
        foreach (string key in w.Keys)
        {
            Dictionary<string, Tensor> missing = AukDitReference.ToTensors(w);
            missing.Remove(key);
            Assert.Throws<InvalidDataException>(() => new AukDit(Cfg).LoadWeights(missing));

            Dictionary<string, Tensor> bad = AukDitReference.ToTensors(w);
            bad[key] = AukDitReference.ToTensor([w[key].Shape[0] + 1], new float[w[key].Shape[0] + 1]);
            Assert.Throws<InvalidDataException>(() => new AukDit(Cfg).LoadWeights(bad));
        }
    }

    [Theory]
    [InlineData(false, false, true)]
    public void Forward_MatchesNaiveReference(bool dropText, bool dropAudio, bool withRef)
    {
        (AukDit dit, Dictionary<string, (long[] Shape, float[] Data)> w) = Build(7);
        double[][] noisy = Rows(5, 8, 11), refl = Rows(3, 8, 12), text = Rows(4, 16, 13);
        CpuBackend backend = new();
        Tensor tn = Pack(noisy), tr = Pack(refl), tt = Pack(text);
        Tensor cond = dit.ProjectText(backend, tt, dropText);
        Tensor v = dit.Forward(backend, tn, withRef ? tr : null, cond, 0.37f, dropAudio);
        Assert.Equal([1, 5, 8], new[] { v.Shape[0], v.Shape[1], v.Shape[2] }.Select(x => (int)x));

        double[][] expected = new AukDitReference(Cfg, w).Forward(noisy, withRef ? refl : null, text, 0.37, dropText, dropAudio);
        Assert.True(MaxDiff(v, expected) < 1e-4, $"max diff {MaxDiff(v, expected)}");
    }

    [Fact]
    public void Rope_TheoryInvFreq_MatchesClosedFormTable()
    {
        float[] inv = AukRope.TheoryInvFreq(16);
        (Tensor cos, Tensor sin) = AukRope.BuildTables(inv, 9);
        float* c = (float*)cos.DataPointer, s = (float*)sin.DataPointer;
        for (int p = 0; p < 9; p++)
            for (int i = 0; i < 8; i++)
            {
                double a = p * Math.Pow(10000.0, -2.0 * i / 16);
                Assert.InRange(c[p * 16 + 2 * i], Math.Cos(a) - 2e-6, Math.Cos(a) + 2e-6);
                Assert.InRange(c[p * 16 + 2 * i + 1], Math.Cos(a) - 2e-6, Math.Cos(a) + 2e-6);
                Assert.InRange(s[p * 16 + 2 * i], Math.Sin(a) - 2e-6, Math.Sin(a) + 2e-6);
                Assert.InRange(s[p * 16 + 2 * i + 1], Math.Sin(a) - 2e-6, Math.Sin(a) + 2e-6);
            }
    }

}
