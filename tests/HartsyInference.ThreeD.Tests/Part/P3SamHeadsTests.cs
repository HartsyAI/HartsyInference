using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Part;

namespace HartsyInference.ThreeD.Tests.Part;

/// <summary>Unit lane: the P3-SAM heads evaluated with the prompt-independent first-layer terms precomputed (<see cref="P3SamHeads.Prepare"/>) agree with a
/// direct transcription of upstream's forward (concatenate, then the MLP) on random weights.</summary>
public sealed class P3SamHeadsTests
{
    private const int D = 6, H1 = 8, H2 = 4, Points = 11;

    private static unsafe Tensor Make(Random rng, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
            p[i] = (float)(rng.NextDouble() * 0.6 - 0.3);
        return t;
    }

    internal static Dictionary<string, Tensor> Weights(int seed, int featureDim = D)
    {
        Random rng = new(seed);
        Dictionary<string, Tensor> w = [];
        void Mlp(string name, int input, int hidden, int output)
        {
            w[$"{name}.0.weight"] = Make(rng, hidden, input); w[$"{name}.0.bias"] = Make(rng, hidden);
            w[$"{name}.2.weight"] = Make(rng, hidden, hidden); w[$"{name}.2.bias"] = Make(rng, hidden);
            w[$"{name}.4.weight"] = Make(rng, output, hidden); w[$"{name}.4.bias"] = Make(rng, output);
        }

        for (int i = 1; i <= 3; i++)
        {
            Mlp($"seg_mlp_{i}", featureDim + 6, H1, 1);
            Mlp($"seg_s2_mlp_{i}", H2 + featureDim + 9, H2, 1);
        }

        Mlp("seg_s2_mlp_g", featureDim + 9, H2, H2);
        Mlp("iou_mlp", H2 + featureDim + 9, H2, H2);
        Mlp("iou_mlp_out", H2, H2, 3);
        return w;
    }

    private static unsafe double[] Array64(Tensor t)
    {
        double[] a = new double[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < a.Length; i++)
            a[i] = p[i];
        return a;
    }

    private static double Gelu(double x)
    {
        // Exact GELU through erf(x / sqrt 2) = 2 * Phi(x * sqrt 2) - 1, with Phi from a series (the unit-test scale only needs about 1e-9).
        double z = x / Math.Sqrt(2.0), sum = z, term = z, z2 = z * z;
        for (int k = 1; k < 200; k++)
        {
            term *= -z2 / k;
            sum += term / (2 * k + 1);
        }

        return 0.5 * x * (1.0 + 2.0 / Math.Sqrt(Math.PI) * sum);
    }

    // y = W x + b, W [out, in].
    private static double[] Linear(Tensor w, Tensor b, double[] x)
    {
        int outDim = (int)w.Shape[0], inDim = (int)w.Shape[1];
        double[] wa = Array64(w), ba = Array64(b), y = new double[outDim];
        for (int r = 0; r < outDim; r++)
        {
            y[r] = ba[r];
            for (int c = 0; c < inDim; c++)
                y[r] += wa[r * inDim + c] * x[c];
        }

        return y;
    }

    private static double[] Mlp(Dictionary<string, Tensor> w, string name, double[] x)
    {
        double[] h = Linear(w[$"{name}.0.weight"], w[$"{name}.0.bias"], x).Select(Gelu).ToArray();
        h = Linear(w[$"{name}.2.weight"], w[$"{name}.2.bias"], h).Select(Gelu).ToArray();
        return Linear(w[$"{name}.4.weight"], w[$"{name}.4.bias"], h);
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    [Fact]
    public void Segment_MatchesTheDirectForwardOfUpstream()
    {
        Dictionary<string, Tensor> w = Weights(3);
        Random rng = new(9);
        float[] features = Enumerable.Range(0, Points * D).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] points = Enumerable.Range(0, Points * 3).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] prompt = [0.3f, -0.2f, 0.5f];
        using CpuBackend backend = new();
        P3SamHeads heads = new(w);

        P3SamHeads.Result result = heads.Segment(backend, heads.Prepare(backend, features, points, Points), prompt);

        // Upstream: feats_seg = [feature, point, prompt]; stage 1 -> 3 logits; stage 2 over [feats_seg, logits] with the global max.
        double[][] seg = new double[Points][];
        double[][] stage1 = new double[Points][];
        for (int i = 0; i < Points; i++)
        {
            seg[i] = [.. features.Skip(i * D).Take(D).Select(v => (double)v), .. points.Skip(i * 3).Take(3).Select(v => (double)v), .. prompt.Select(v => (double)v)];
            stage1[i] = [.. Enumerable.Range(1, 3).Select(m => Mlp(w, $"seg_mlp_{m}", seg[i])[0])];
        }

        double[][] g = [.. Enumerable.Range(0, Points).Select(i => Mlp(w, "seg_s2_mlp_g", [.. seg[i], .. stage1[i]]))];
        double[] global = [.. Enumerable.Range(0, H2).Select(c => g.Max(row => row[c]))];
        double[][] stage2 = new double[Points][];
        for (int i = 0; i < Points; i++)
            stage2[i] = [.. Enumerable.Range(1, 3).Select(m => Mlp(w, $"seg_s2_mlp_{m}", [.. global, .. seg[i], .. stage1[i]])[0])];
        double[][] iouRows = [.. Enumerable.Range(0, Points).Select(i => Mlp(w, "iou_mlp", [.. global, .. seg[i], .. stage2[i]]))];
        double[] pooled = [.. Enumerable.Range(0, H2).Select(c => iouRows.Max(row => row[c]))];
        double[] iou = Mlp(w, "iou_mlp_out", pooled).Select(Sigmoid).ToArray();

        for (int m = 0; m < 3; m++)
        {
            Assert.Equal(iou[m], result.Iou[m], 4);
            for (int i = 0; i < Points; i++)
                Assert.Equal(Sigmoid(stage2[i][m]), result.Masks[m][i], 4);
        }
    }

    [Fact]
    public void Constructor_RefusesAMissingOrMisshapenHead()
    {
        Dictionary<string, Tensor> w = Weights(1);
        w.Remove("iou_mlp_out.4.bias");
        Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() => new P3SamHeads(w));
    }
}
