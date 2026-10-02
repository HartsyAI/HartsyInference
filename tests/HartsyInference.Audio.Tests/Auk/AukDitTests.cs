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

    [Fact]
    public void Forward_BeforeLoad_Throws()
    {
        AukDit dit = new(Cfg);
        Tensor noisy = Pack(Rows(2, 8, 1));
        Assert.Throws<InvalidOperationException>(() => dit.ProjectText(new CpuBackend(), Pack(Rows(2, 16, 2))));
        noisy.Dispose();
    }

    [Theory]
    [InlineData(false, false, true)]
    [InlineData(true, true, true)]
    [InlineData(false, true, true)]
    [InlineData(false, false, false)]
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
    public void Forward_RefLengthChangesPositionsNotOutputShape()
    {
        (AukDit dit, _) = Build(8);
        CpuBackend backend = new();
        Tensor tn = Pack(Rows(4, 8, 1)), tt = Pack(Rows(3, 16, 2));
        Tensor cond = dit.ProjectText(backend, tt);
        Tensor a = dit.Forward(backend, tn, Pack(Rows(2, 8, 3)), cond, 0.5f);
        Tensor b = dit.Forward(backend, tn, Pack(Rows(6, 8, 3)), cond, 0.5f);
        Assert.Equal(a.Shape, b.Shape);
        float* pa = (float*)a.DataPointer, pb = (float*)b.DataPointer;
        double d = 0;
        for (long i = 0; i < a.ElementCount; i++) d += Math.Abs(pa[i] - pb[i]);
        Assert.True(d > 1e-4);
    }

    [Fact]
    public void ZeroProjOut_GivesZeroVelocity()
    {
        Dictionary<string, (long[] Shape, float[] Data)> w = AukDitReference.RandomWeights(Cfg, 5);
        Array.Clear(w["transformer.proj_out.weight"].Data);
        Array.Clear(w["transformer.proj_out.bias"].Data);
        AukDit dit = new(Cfg);
        dit.LoadWeights(AukDitReference.ToTensors(w));
        CpuBackend backend = new();
        Tensor cond = dit.ProjectText(backend, Pack(Rows(3, 16, 2)));
        Tensor v = dit.Forward(backend, Pack(Rows(4, 8, 1)), Pack(Rows(2, 8, 4)), cond, 0.9f);
        float* p = (float*)v.DataPointer;
        for (long i = 0; i < v.ElementCount; i++) Assert.Equal(0f, p[i]);
    }

    [Fact]
    public void ZeroAdaLnBlocks_OnlyHeadActs()
    {
        Dictionary<string, (long[] Shape, float[] Data)> w = AukDitReference.RandomWeights(Cfg, 6);
        foreach (string key in w.Keys.Where(k => k.Contains("attn_norm")))
            Array.Clear(w[key].Data);
        AukDit dit = new(Cfg);
        dit.LoadWeights(AukDitReference.ToTensors(w));
        CpuBackend backend = new();
        double[][] noisy = Rows(4, 8, 1), text = Rows(3, 16, 2), refl = Rows(2, 8, 3);
        Tensor cond = dit.ProjectText(backend, Pack(text));
        Tensor v = dit.Forward(backend, Pack(noisy), Pack(refl), cond, 0.2f);
        Assert.True(MaxDiff(v, new AukDitReference(Cfg, w).Forward(noisy, refl, text, 0.2, false, false)) < 1e-4);
    }

    [Fact]
    public void ProjectText_DropIsZerosAfterNorm_NotProjectionOfZeros()
    {
        (AukDit dit, _) = Build(9);
        CpuBackend backend = new();
        Tensor zeroText = Pack(Rows(3, 16, 1).Select(r => new double[16]).ToArray());
        Tensor dropped = dit.ProjectText(backend, zeroText, drop: true);
        Tensor projected = dit.ProjectText(backend, zeroText, drop: false);
        float* d = (float*)dropped.DataPointer, p = (float*)projected.DataPointer;
        double sum = 0;
        for (long i = 0; i < dropped.ElementCount; i++)
        {
            Assert.Equal(0f, d[i]);
            sum += Math.Abs(p[i]);
        }
        Assert.Equal(new TensorShape(1, 3, Cfg.Dim), dropped.Shape);
        Assert.True(sum > 1e-3);
    }

    [Fact]
    public void SwiGlu_UsesSiluOnFirstHalfTimesSecondHalf()
    {
        AukSwiGluFfn ffn = new();
        Dictionary<string, Tensor> w = new()
        {
            ["f.linear_in.weight"] = AukDitReference.ToTensor([4, 2], [1, 0, 0, 1, 2, 0, 0, 3]),
            ["f.linear_out.weight"] = AukDitReference.ToTensor([2, 2], [1, 0, 0, 1]),
        };
        ffn.Load(w, "f", 2, 2);
        Tensor x = AukDitReference.ToTensor([1, 1, 2], [0.5f, -1.5f]);
        Tensor y = ffn.Forward(new CpuBackend(), x, 1);
        float* p = (float*)y.DataPointer;
        static double Silu(double v) => v / (1 + Math.Exp(-v));
        Assert.Equal(Silu(0.5) * 1.0, p[0], 1e-5);
        Assert.Equal(Silu(-1.5) * -4.5, p[1], 1e-5);
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

    [Fact]
    public void Rope_LoadedInvFreq_DrivesTables()
    {
        float[] inv = [1.0f, 0.75f, 0.5625f, 0.42188f, 0.3164f, 0.2373f, 0.1777f, 0.1333f];
        (AukDit dit, _) = Build(2, inv);
        Assert.Equal(inv, dit.InvFreq.ToArray());
        (Tensor cos, Tensor sin) = dit.GetRopeTables(10);
        Assert.True(cos.Shape[0] >= 10);
        float* c = (float*)cos.DataPointer, s = (float*)sin.DataPointer;
        for (int p = 0; p < 10; p++)
            for (int i = 0; i < 8; i++)
            {
                Assert.InRange(c[p * 16 + 2 * i], Math.Cos(p * (double)inv[i]) - 2e-6, Math.Cos(p * (double)inv[i]) + 2e-6);
                Assert.InRange(s[p * 16 + 2 * i + 1], Math.Sin(p * (double)inv[i]) - 2e-6, Math.Sin(p * (double)inv[i]) + 2e-6);
            }
        double theory = Math.Cos(1 * Math.Pow(10000.0, -2.0 / 16));
        Assert.True(Math.Abs(c[1 * 16 + 2] - theory) > 0.1);
    }

    [Fact]
    [Trait("Category", "SyntheticSmoke")]
    public void Forward_SyntheticSmoke_FiniteAndShaped()
    {
        (AukDit dit, _) = Build(4);
        CpuBackend backend = new();
        Tensor cond = dit.ProjectText(backend, Pack(Rows(6, 16, 2)));
        Tensor v = dit.Forward(backend, Pack(Rows(9, 8, 1)), Pack(Rows(5, 8, 3)), cond, 0.5f);
        Assert.Equal(new TensorShape(1, 9, 8), v.Shape);
        float* p = (float*)v.DataPointer;
        for (long i = 0; i < v.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }
}
