using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.DepthPro2;

namespace HartsyInference.Vision.Tests.DepthPro2;

public sealed class DepthPro2HeadTests
{
    // A tiny head: backbone width 5 with 2 prefix tokens on a 2 x 4 patch grid, pyramid widths 3/4/5/6, decoder width 4. The pyramid is 8 x 16, 4 x 8, 2 x 4
    // and 1 x 2; the decoder ends at 16 x 32 and the head at 32 x 64 (16 pixels per patch).
    private const int Width = 5, Prefix = 2, GridH = 2, GridW = 4, Decoder = 4;
    private static readonly int[] Dims = [3, 4, 5, 6];

    private sealed class Owned(IEnumerable<Tensor> tensors) : IDisposable
    {
        public void Dispose()
        {
            foreach (Tensor t in tensors) t.Dispose();
        }
    }

    private static Tensor Filled(Random rng, double scale, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>())
            v = (float)((rng.NextDouble() - 0.5) * scale);
        return t;
    }

    private static Dictionary<string, Tensor> Weights(Random rng)
    {
        Dictionary<string, Tensor> w = [];
        int[] kernels = [4, 2, 0, 3];
        for (int i = 0; i < 4; i++)
        {
            string p = $"encoder.reassemble.{i}.";
            w[p + "0.weight"] = Filled(rng, 0.8, Dims[i], Width, 1, 1);
            w[p + "0.bias"] = Filled(rng, 0.4, Dims[i]);
            if (kernels[i] > 0)
            {
                w[p + "1.weight"] = Filled(rng, 0.6, Dims[i], Dims[i], kernels[i], kernels[i]);
                w[p + "1.bias"] = Filled(rng, 0.4, Dims[i]);
            }

            w[$"decoder.convs.{i}.weight"] = Filled(rng, 0.5, Decoder, Dims[i], 3, 3);
            string f = $"decoder.fusions.{i}.";
            foreach (string block in i < 3 ? new[] { "resnet1.", "resnet2." } : ["resnet2."])
            {
                w[f + block + "residual.1.weight"] = Filled(rng, 0.5, Decoder, Decoder, 3, 3);
                w[f + block + "residual.1.bias"] = Filled(rng, 0.4, Decoder);
                w[f + block + "residual.3.weight"] = Filled(rng, 0.5, Decoder, Decoder, 3, 3);
                w[f + block + "residual.3.bias"] = Filled(rng, 0.4, Decoder);
            }

            w[f + "deconv.weight"] = Filled(rng, 0.8, Decoder, Decoder, 2, 2);
            w[f + "out_conv.weight"] = Filled(rng, 0.8, Decoder, Decoder, 1, 1);
            w[f + "out_conv.bias"] = Filled(rng, 0.4, Decoder);
        }

        w["head.0.weight"] = Filled(rng, 0.5, 2, Decoder, 3, 3);
        w["head.0.bias"] = Filled(rng, 0.4, 2);
        w["head.1.weight"] = Filled(rng, 0.8, 2, 2, 2, 2);
        w["head.1.bias"] = Filled(rng, 0.4, 2);
        w["head.2.weight"] = Filled(rng, 0.8, 32, 2, 3, 3);
        w["head.2.bias"] = Filled(rng, 0.4, 32);
        w["head.4.weight"] = Filled(rng, 0.5, 1, 32, 1, 1);
        w["head.4.bias"] = Filled(rng, 0.2, 1);
        return w;
    }

    // ---- naive double-precision reference on [c][h][w] arrays ----

    private static double[][][] Alloc(int c, int h, int w)
    {
        double[][][] a = new double[c][][];
        for (int i = 0; i < c; i++)
        {
            a[i] = new double[h][];
            for (int y = 0; y < h; y++) a[i][y] = new double[w];
        }

        return a;
    }

    private static double[][][] Conv(double[][][] x, Tensor weight, Tensor? bias, int stride, int pad)
    {
        int cOut = (int)weight.Shape[0], cIn = (int)weight.Shape[1], k = (int)weight.Shape[2], h = x[0].Length, w = x[0][0].Length;
        int oh = (h + 2 * pad - k) / stride + 1, ow = (w + 2 * pad - k) / stride + 1;
        float[] wt = weight.AsReadOnlySpan<float>().ToArray();
        float[]? bs = bias?.AsReadOnlySpan<float>().ToArray();
        double[][][] y = Alloc(cOut, oh, ow);
        for (int o = 0; o < cOut; o++)
            for (int yy = 0; yy < oh; yy++)
                for (int xx = 0; xx < ow; xx++)
                {
                    double sum = bs is null ? 0 : bs[o];
                    for (int c = 0; c < cIn; c++)
                        for (int ky = 0; ky < k; ky++)
                            for (int kx = 0; kx < k; kx++)
                            {
                                int sy = yy * stride + ky - pad, sx = xx * stride + kx - pad;
                                if (sy >= 0 && sy < h && sx >= 0 && sx < w) sum += wt[((o * cIn + c) * k + ky) * k + kx] * x[c][sy][sx];
                            }

                    y[o][yy][xx] = sum;
                }

        return y;
    }

    private static double[][][] ConvTranspose(double[][][] x, Tensor weight, Tensor? bias)
    {
        int cIn = (int)weight.Shape[0], cOut = (int)weight.Shape[1], k = (int)weight.Shape[2], h = x[0].Length, w = x[0][0].Length;
        float[] wt = weight.AsReadOnlySpan<float>().ToArray();
        float[]? bs = bias?.AsReadOnlySpan<float>().ToArray();
        double[][][] y = Alloc(cOut, k * h, k * w);
        for (int o = 0; o < cOut; o++)
            for (int yy = 0; yy < k * h; yy++)
                for (int xx = 0; xx < k * w; xx++)
                {
                    double sum = bs is null ? 0 : bs[o];
                    for (int c = 0; c < cIn; c++)
                        sum += wt[((c * cOut + o) * k + yy % k) * k + xx % k] * x[c][yy / k][xx / k];
                    y[o][yy][xx] = sum;
                }

        return y;
    }

    private static double[][][] Relu(double[][][] x) => x.Select(p => p.Select(r => r.Select(v => Math.Max(0, v)).ToArray()).ToArray()).ToArray();

    private static double[][][] Add(double[][][] a, double[][][] b) =>
        a.Select((p, c) => p.Select((r, y) => r.Select((v, x) => v + b[c][y][x]).ToArray()).ToArray()).ToArray();

    private static double[][][] Residual(Dictionary<string, Tensor> w, string p, double[][][] x) =>
        Add(x, Conv(Relu(Conv(Relu(x), w[p + "residual.1.weight"], w[p + "residual.1.bias"], 1, 1)), w[p + "residual.3.weight"], w[p + "residual.3.bias"], 1, 1));

    private static double[][][] Reference(Dictionary<string, Tensor> w, float[][] taps)
    {
        double[][][][] levels = new double[4][][][];
        for (int i = 0; i < 4; i++)
        {
            double[][][] grid = Alloc(Width, GridH, GridW);
            for (int n = 0; n < GridH * GridW; n++)
                for (int c = 0; c < Width; c++)
                    grid[c][n / GridW][n % GridW] = taps[i][(Prefix + n) * Width + c];
            string p = $"encoder.reassemble.{i}.";
            double[][][] projected = Conv(grid, w[p + "0.weight"], w[p + "0.bias"], 1, 0);
            levels[i] = i switch
            {
                0 or 1 => ConvTranspose(projected, w[p + "1.weight"], w[p + "1.bias"]),
                2 => projected,
                _ => Conv(projected, w[p + "1.weight"], w[p + "1.bias"], 2, 1),
            };
        }

        double[][][]? features = null;
        for (int i = 3; i >= 0; i--)
        {
            string f = $"decoder.fusions.{i}.";
            double[][][] skip = Conv(levels[i], w[$"decoder.convs.{i}.weight"], null, 1, 1);
            double[][][] x = features is null ? skip : Add(features, Residual(w, f + "resnet1.", skip));
            x = Residual(w, f + "resnet2.", x);
            features = Conv(ConvTranspose(x, w[f + "deconv.weight"], null), w[f + "out_conv.weight"], w[f + "out_conv.bias"], 1, 0);
        }

        double[][][] head = Conv(features!, w["head.0.weight"], w["head.0.bias"], 1, 1);
        head = ConvTranspose(head, w["head.1.weight"], w["head.1.bias"]);
        head = Relu(Conv(head, w["head.2.weight"], w["head.2.bias"], 1, 1));
        return Relu(Conv(head, w["head.4.weight"], w["head.4.bias"], 1, 0));
    }

    [Fact]
    public void Forward_MatchesTheNaiveReference()
    {
        Random rng = new(71);
        Dictionary<string, Tensor> w = Weights(rng);
        using Owned owned = new(w.Values);
        using CpuBackend backend = new();
        DepthPro2Head head = new(Dims, Decoder, Width, w);
        Tensor[] taps = Enumerable.Range(0, 4).Select(_ => Filled(rng, 2.0, 1, Prefix + GridH * GridW, Width)).ToArray();
        using Owned ownedTaps = new(taps);
        double[][][] expected = Reference(w, taps.Select(t => t.AsReadOnlySpan<float>().ToArray()).ToArray());

        List<string> stages = [];
        using Tensor inverse = head.Forward(backend, taps, Prefix, GridH, GridW, (stage, _) => stages.Add(stage));
        Assert.Equal(new TensorShape(1, 1, 16 * GridH, 16 * GridW), inverse.Shape);
        Assert.Equal(["reassemble0", "reassemble1", "reassemble2", "reassemble3", "conv3", "fusion3", "conv2", "fusion2", "conv1", "fusion1", "conv0", "fusion0"],
            stages);

        ReadOnlySpan<float> got = inverse.AsReadOnlySpan<float>();
        double peak = expected[0].SelectMany(r => r).Max(Math.Abs);
        Assert.True(peak > 0, "the synthetic head output is all zero; pick other weights");
        int positives = 0;
        for (int y = 0; y < 16 * GridH; y++)
            for (int x = 0; x < 16 * GridW; x++)
            {
                double want = expected[0][y][x], have = got[y * 16 * GridW + x];
                if (want > 0) positives++;
                Assert.True(Math.Abs(want - have) <= 1e-5 * peak, $"[{y},{x}] expected {want} got {have}");
            }

        Assert.True(positives > 16, "too few positive pixels for the final ReLU to be exercised");
    }

    [Fact]
    public void Constructor_RejectsAMissingMisshapenOrNonF32Weight()
    {
        Random rng = new(72);
        Dictionary<string, Tensor> w = Weights(rng);
        using Owned owned = new(w.Values);
        Dictionary<string, Tensor> broken = new(w);
        broken.Remove("decoder.fusions.1.resnet1.residual.3.bias");
        Assert.Throws<InvalidOperationException>(() => new DepthPro2Head(Dims, Decoder, Width, broken));

        using Tensor wrong = Filled(rng, 1.0, Dims[3], Dims[3], 2, 2);
        broken = new Dictionary<string, Tensor>(w) { ["encoder.reassemble.3.1.weight"] = wrong };
        Assert.Throws<InvalidOperationException>(() => new DepthPro2Head(Dims, Decoder, Width, broken));

        using Tensor half = w["head.4.weight"].CastTo(DType.F16);
        broken = new Dictionary<string, Tensor>(w) { ["head.4.weight"] = half };
        Assert.Throws<InvalidOperationException>(() => new DepthPro2Head(Dims, Decoder, Width, broken));
    }

    [Fact]
    public void Presets_MatchEachRepoConfig()
    {
        // From config.json of apple/DepthPro2-S, -B, -L and -H.
        DepthPro2Preset s = DepthPro2Preset.Small, b = DepthPro2Preset.Base, l = DepthPro2Preset.Large, h = DepthPro2Preset.HugePlus;
        Assert.Equal([2, 5, 8, 11], s.ReassembledBlocks);
        Assert.Equal([5, 7, 9, 11], b.CrossFrameBlocks);
        Assert.Equal([96, 192, 384, 768], b.DimsReassemble);
        Assert.Equal([5, 11, 17, 23], l.ReassembledBlocks);
        Assert.Equal([9, 11, 13, 15, 17, 19, 21, 23], l.CrossFrameBlocks);
        Assert.Equal([7, 15, 23, 31], h.ReassembledBlocks);
        Assert.Equal(11, h.CrossFrameBlocks.Length);
        Assert.Equal([64, 128, 256, 256], DepthPro2Preset.All.Select(p => p.DimDecoder));
        Assert.Equal([384, 768, 1024, 1280], DepthPro2Preset.All.Select(p => p.Backbone.HiddenSize));
        Assert.True(h.Backbone.GatedMlp);
        foreach (DepthPro2Preset p in DepthPro2Preset.All)
        {
            Assert.False(p.Backbone.AttentionQkvBias);
            Assert.False(p.Backbone.HasFinalNorm);
            Assert.True(p.Backbone.RopeNormalizesByLongSide);
            Assert.False(p.Backbone.RoundRopePeriodsToBf16);
            Assert.All(p.ReassembledBlocks.Concat(p.CrossFrameBlocks), block => Assert.InRange(block, 0, p.Backbone.NumLayers - 1));
        }
    }
}
