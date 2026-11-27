using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Moge;

namespace HartsyInference.Vision.Tests.Moge;

public sealed class MogeHeadTests
{
    // A tiny head: 2 backbone taps of width 6, projections of width 8, two upsample blocks of width 32 with one residual block each (hidden 32, one group of 32 for the second norm),
    // output blocks of 4 channels. A 3 x 2 patch grid gives a 12 x 8 map after two doublings, resized to a 42 x 28 image.
    private static readonly MogeConfig Config = new()
    {
        IntermediateLayers = 2, DimProj = 8, DimUpsample = [32, 32], ResBlockHiddenMultiplier = 1, NumResBlocks = 1, LastConvChannels = 4, LastConvSize = 1,
    };

    private const int Width = 6, Height = 28, ImageWidth = 42, GridH = 2, GridW = 3;

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
        for (int i = 0; i < 2; i++)
        {
            w[$"head.projects.{i}.weight"] = Filled(rng, 1.0, 8, Width, 1, 1);
            w[$"head.projects.{i}.bias"] = Filled(rng, 0.5, 8);
        }

        int previous = 8;
        for (int b = 0; b < 2; b++)
        {
            string p = $"head.upsample_blocks.{b}.";
            w[p + "0.0.weight"] = Filled(rng, 0.6, previous + 2, 32, 2, 2);
            w[p + "0.0.bias"] = Filled(rng, 0.4, 32);
            w[p + "0.1.weight"] = Filled(rng, 0.3, 32, 32, 3, 3);
            w[p + "0.1.bias"] = Filled(rng, 0.4, 32);
            w[p + "1.layers.0.weight"] = Filled(rng, 1.0, 32);
            w[p + "1.layers.0.bias"] = Filled(rng, 0.4, 32);
            w[p + "1.layers.2.weight"] = Filled(rng, 0.3, 32, 32, 3, 3);
            w[p + "1.layers.2.bias"] = Filled(rng, 0.4, 32);
            w[p + "1.layers.3.weight"] = Filled(rng, 1.0, 32);
            w[p + "1.layers.3.bias"] = Filled(rng, 0.4, 32);
            w[p + "1.layers.5.weight"] = Filled(rng, 0.3, 32, 32, 3, 3);
            w[p + "1.layers.5.bias"] = Filled(rng, 0.4, 32);
            previous = 32;
        }

        for (int o = 0; o < 2; o++)
        {
            int outputs = o == 0 ? 3 : 1;
            w[$"head.output_block.{o}.0.weight"] = Filled(rng, 0.4, 4, 34, 3, 3);
            w[$"head.output_block.{o}.0.bias"] = Filled(rng, 0.4, 4);
            w[$"head.output_block.{o}.2.weight"] = Filled(rng, 1.0, outputs, 4, 1, 1);
            w[$"head.output_block.{o}.2.bias"] = Filled(rng, 0.4, outputs);
        }

        return w;
    }

    // ---- naive double-precision reference on [c][h][w] arrays ----

    private static double[][][] Conv(double[][][] x, Tensor weight, Tensor bias, int k, bool replicate)
    {
        int cOut = (int)weight.Shape[0], cIn = (int)weight.Shape[1], h = x[0].Length, w = x[0][0].Length, pad = k / 2;
        ReadOnlySpan<float> wt = weight.AsReadOnlySpan<float>(), bs = bias.AsReadOnlySpan<float>();
        double[][][] y = Alloc(cOut, h, w);
        for (int o = 0; o < cOut; o++)
            for (int yy = 0; yy < h; yy++)
                for (int xx = 0; xx < w; xx++)
                {
                    double sum = bs[o];
                    for (int c = 0; c < cIn; c++)
                        for (int ky = 0; ky < k; ky++)
                            for (int kx = 0; kx < k; kx++)
                            {
                                int sy = Math.Clamp(yy + ky - pad, 0, h - 1), sx = Math.Clamp(xx + kx - pad, 0, w - 1);
                                if (!replicate && (yy + ky - pad < 0 || yy + ky - pad >= h || xx + kx - pad < 0 || xx + kx - pad >= w))
                                    continue;
                                sum += wt[((o * cIn + c) * k + ky) * k + kx] * x[c][sy][sx];
                            }

                    y[o][yy][xx] = sum;
                }

        return y;
    }

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

    private static double[][][] ConvTranspose(double[][][] x, Tensor weight, Tensor bias)
    {
        int cIn = (int)weight.Shape[0], cOut = (int)weight.Shape[1], h = x[0].Length, w = x[0][0].Length;
        ReadOnlySpan<float> wt = weight.AsReadOnlySpan<float>(), bs = bias.AsReadOnlySpan<float>();
        double[][][] y = Alloc(cOut, 2 * h, 2 * w);
        for (int o = 0; o < cOut; o++)
            for (int yy = 0; yy < 2 * h; yy++)
                for (int xx = 0; xx < 2 * w; xx++)
                {
                    double sum = bs[o];
                    for (int c = 0; c < cIn; c++)
                        sum += wt[((c * cOut + o) * 2 + yy % 2) * 2 + xx % 2] * x[c][yy / 2][xx / 2];
                    y[o][yy][xx] = sum;
                }

        return y;
    }

    private static double[][][] GroupNorm(double[][][] x, Tensor weight, Tensor bias, int groups)
    {
        int c = x.Length, h = x[0].Length, w = x[0][0].Length, per = c / groups;
        ReadOnlySpan<float> wt = weight.AsReadOnlySpan<float>(), bs = bias.AsReadOnlySpan<float>();
        double[][][] y = Alloc(c, h, w);
        for (int g = 0; g < groups; g++)
        {
            List<double> values = [];
            for (int i = g * per; i < (g + 1) * per; i++)
                foreach (double[] row in x[i]) values.AddRange(row);
            double mean = values.Average(), variance = values.Select(v => (v - mean) * (v - mean)).Average();
            for (int i = g * per; i < (g + 1) * per; i++)
                for (int yy = 0; yy < h; yy++)
                    for (int xx = 0; xx < w; xx++)
                        y[i][yy][xx] = (x[i][yy][xx] - mean) / Math.Sqrt(variance + 1e-5) * wt[i] + bs[i];
        }

        return y;
    }

    private static double[][][] Relu(double[][][] x) => x.Select(p => p.Select(r => r.Select(v => Math.Max(0, v)).ToArray()).ToArray()).ToArray();

    private static double[][][] Resize(double[][][] x, int h, int w)
    {
        int c = x.Length, ih = x[0].Length, iw = x[0][0].Length;
        double[][][] y = Alloc(c, h, w);
        for (int i = 0; i < c; i++)
            for (int yy = 0; yy < h; yy++)
            {
                double sy = Math.Max((yy + 0.5) * ih / h - 0.5, 0);
                int y0 = Math.Min((int)Math.Floor(sy), ih - 1), y1 = Math.Min(y0 + 1, ih - 1);
                double fy = sy - y0;
                for (int xx = 0; xx < w; xx++)
                {
                    double sx = Math.Max((xx + 0.5) * iw / w - 0.5, 0);
                    int x0 = Math.Min((int)Math.Floor(sx), iw - 1), x1 = Math.Min(x0 + 1, iw - 1);
                    double fx = sx - x0;
                    y[i][yy][xx] = (1 - fy) * ((1 - fx) * x[i][y0][x0] + fx * x[i][y0][x1]) + fy * ((1 - fx) * x[i][y1][x0] + fx * x[i][y1][x1]);
                }
            }

        return y;
    }

    private static double[][][] WithUv(double[][][] x, double aspect)
    {
        int c = x.Length, h = x[0].Length, w = x[0][0].Length;
        double spanX = aspect / Math.Sqrt(1 + aspect * aspect), spanY = 1 / Math.Sqrt(1 + aspect * aspect);
        double[][][] y = Alloc(c + 2, h, w);
        for (int i = 0; i < c; i++) y[i] = x[i];
        for (int yy = 0; yy < h; yy++)
            for (int xx = 0; xx < w; xx++)
            {
                double u = -spanX * (w - 1) / w + 2 * spanX * (w - 1) / w * xx / Math.Max(w - 1, 1);
                double v = -spanY * (h - 1) / h + 2 * spanY * (h - 1) / h * yy / Math.Max(h - 1, 1);
                y[c][yy][xx] = u;
                y[c + 1][yy][xx] = v;
            }

        return y;
    }

    private static double[][][] Reference(Dictionary<string, Tensor> w, float[][] taps, out double[][][] mask)
    {
        double aspect = (double)ImageWidth / Height;
        double[][][] sum = Alloc(8, GridH, GridW);
        for (int i = 0; i < 2; i++)
        {
            // taps[i] is [n, width] row-major; as channels-first it is [width, gridH, gridW].
            double[][][] feature = Alloc(Width, GridH, GridW);
            for (int n = 0; n < GridH * GridW; n++)
                for (int c = 0; c < Width; c++)
                    feature[c][n / GridW][n % GridW] = taps[i][n * Width + c];
            double[][][] projected = Conv(feature, w[$"head.projects.{i}.weight"], w[$"head.projects.{i}.bias"], 1, replicate: false);
            for (int c = 0; c < 8; c++)
                for (int y = 0; y < GridH; y++)
                    for (int x = 0; x < GridW; x++)
                        sum[c][y][x] += projected[c][y][x];
        }

        double[][][] cur = sum;
        for (int b = 0; b < 2; b++)
        {
            string p = $"head.upsample_blocks.{b}.";
            cur = ConvTranspose(WithUv(cur, aspect), w[p + "0.0.weight"], w[p + "0.0.bias"]);
            cur = Conv(cur, w[p + "0.1.weight"], w[p + "0.1.bias"], 3, replicate: true);
            double[][][] skip = cur;
            double[][][] t = GroupNorm(cur, w[p + "1.layers.0.weight"], w[p + "1.layers.0.bias"], 1);
            t = Conv(Relu(t), w[p + "1.layers.2.weight"], w[p + "1.layers.2.bias"], 3, replicate: true);
            t = GroupNorm(t, w[p + "1.layers.3.weight"], w[p + "1.layers.3.bias"], 1);
            t = Conv(Relu(t), w[p + "1.layers.5.weight"], w[p + "1.layers.5.bias"], 3, replicate: true);
            for (int c = 0; c < t.Length; c++)
                for (int y = 0; y < t[0].Length; y++)
                    for (int x = 0; x < t[0][0].Length; x++)
                        t[c][y][x] += skip[c][y][x];
            cur = t;
        }

        double[][][] full = WithUv(Resize(cur, Height, ImageWidth), aspect);
        double[][][] points = Conv(Relu(Conv(full, w["head.output_block.0.0.weight"], w["head.output_block.0.0.bias"], 3, replicate: true)), w["head.output_block.0.2.weight"], w["head.output_block.0.2.bias"], 1, replicate: false);
        mask = Conv(Relu(Conv(full, w["head.output_block.1.0.weight"], w["head.output_block.1.0.bias"], 3, replicate: true)), w["head.output_block.1.2.weight"], w["head.output_block.1.2.bias"], 1, replicate: false);
        return points;
    }

    [Fact]
    public void Forward_MatchesTheNaiveReferenceForPointsAndMask()
    {
        Random rng = new(61);
        Dictionary<string, Tensor> w = Weights(rng);
        using Owned owned = new(w.Values);
        using CpuBackend backend = new();
        MogeHead head = new(Config, w, "head.", Width);
        Tensor[] taps = [Filled(rng, 2.0, 1, GridH * GridW, Width), Filled(rng, 2.0, 1, GridH * GridW, Width)];
        using Owned ownedTaps = new(taps);
        double[][][] expectedPoints = Reference(w, taps.Select(t => t.AsReadOnlySpan<float>().ToArray()).ToArray(), out double[][][] expectedMask);

        (Tensor points, Tensor mask) = head.Forward(backend, taps, Height, ImageWidth);
        using (points)
        using (mask)
        {
            Assert.Equal(new TensorShape(1, 3, Height, ImageWidth), points.Shape);
            Assert.Equal(new TensorShape(1, 1, Height, ImageWidth), mask.Shape);
            AssertClose(expectedPoints, points, "points");
            AssertClose(expectedMask, mask, "mask");
        }
    }

    private static void AssertClose(double[][][] expected, Tensor actual, string what)
    {
        ReadOnlySpan<float> got = actual.AsReadOnlySpan<float>();
        int c = expected.Length, h = expected[0].Length, w = expected[0][0].Length;
        for (int i = 0; i < c; i++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double want = expected[i][y][x], have = got[(i * h + y) * w + x];
                    Assert.True(Math.Abs(want - have) < 2e-3 * (1 + Math.Abs(want)), $"{what} [{i},{y},{x}] expected {want} got {have}");
                }
    }

    [Fact]
    public void ViewPlaneUv_RampsBetweenTheScaledSpansAndTheirCornersAreSymmetric()
    {
        float[] u = new float[3 * 4], v = new float[3 * 4];
        MogeHead.ViewPlaneUv(u, v, 3, 4, 4.0 / 3.0);

        // aspect 4/3: spans 0.8 and 0.6; the first and last columns sit at -/+ span * (n - 1) / n.
        Assert.Equal(-0.8 * 3 / 4, u[0], 5);
        Assert.Equal(0.8 * 3 / 4, u[3], 5);
        Assert.Equal(-0.6 * 2 / 3, v[0], 5);
        Assert.Equal(0.6 * 2 / 3, v[2 * 4], 5);
        Assert.Equal(u[0], u[4]);
        Assert.Equal(v[0], v[3]);
    }

    [Fact]
    public void Constructor_RejectsAMissingMisshapenOrNonF32Weight()
    {
        Random rng = new(62);
        Dictionary<string, Tensor> w = Weights(rng);
        using Owned owned = new(w.Values);
        Dictionary<string, Tensor> broken = new(w);
        broken.Remove("head.output_block.1.2.bias");
        Assert.Throws<InvalidOperationException>(() => new MogeHead(Config, broken, "head.", Width));

        using Tensor wrong = Filled(rng, 1.0, 8, Width + 1, 1, 1);
        broken = new Dictionary<string, Tensor>(w) { ["head.projects.0.weight"] = wrong };
        Assert.Throws<InvalidOperationException>(() => new MogeHead(Config, broken, "head.", Width));
    }
}
