using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Tests.Separation;

/// <summary>Random weights in the converter layout for a small TFC-TDF-UNet, plus a plain-loop reference forward written straight from the ONNX op
/// order (no backend ops, no Linear, no depthwise trick), so the network layout and wiring are checked without a real file.</summary>
internal sealed unsafe class KimVocal2Reference : IDisposable
{
    public static readonly int[] Widths = [2, 3, 4];
    public const int DimF = 8;
    public const int Channels = 4;
    public const int Frames = 8;

    public Dictionary<string, Tensor> Weights { get; } = [];
    private readonly Dictionary<string, float[]> _raw = [];

    public KimVocal2Reference(int seed)
    {
        Random rng = new(seed);
        int levels = Widths.Length - 1;
        Put(rng, "first_conv.weight", Widths[0], Channels, 1, 1);
        Put(rng, "first_conv.bias", Widths[0]);
        for (int i = 0; i <= levels; i++)
        {
            string p = i < levels ? $"encoding_blocks.{i}" : "bottleneck_block";
            Block(rng, p, Widths[i], DimF >> i);
            if (i < levels)
            {
                Put(rng, p + ".down.weight", Widths[i + 1], Widths[i], 2, 2);
                Put(rng, p + ".down.bias", Widths[i + 1]);
            }
        }
        for (int i = 0; i < levels; i++)
        {
            int level = levels - 1 - i;
            string p = $"decoding_blocks.{i}";
            Put(rng, p + ".up.weight", Widths[level + 1], Widths[level], 2, 2);
            Put(rng, p + ".up.bias", Widths[level]);
            Block(rng, p, Widths[level], DimF >> level);
        }
        Put(rng, "final_conv.weight", Channels, Widths[0], 1, 1);
        Put(rng, "final_conv.bias", Channels);
    }

    private void Block(Random rng, string p, int width, int freq)
    {
        for (int j = 0; j < 3; j++)
        {
            Put(rng, $"{p}.tfc.{j}.weight", width, width, 3, 3);
            Put(rng, $"{p}.tfc.{j}.bias", width);
        }
        int mid = Math.Max(1, freq / 2);
        Put(rng, $"{p}.tdf.0.weight", mid, freq);
        Put(rng, $"{p}.tdf.1.scale", width, positive: true);
        Put(rng, $"{p}.tdf.1.shift", width);
        Put(rng, $"{p}.tdf.2.weight", freq, mid);
        Put(rng, $"{p}.tdf.3.scale", width, positive: true);
        Put(rng, $"{p}.tdf.3.shift", width);
    }

    private void Put(Random rng, string name, int d0, int d1 = 0, int d2 = 0, int d3 = 0, bool positive = false)
    {
        long[] dims = d3 > 0 ? [d0, d1, d2, d3] : d1 > 0 ? [d0, d1] : [d0];
        long n = 1;
        foreach (long d in dims) n *= d;
        float[] data = new float[n];
        for (long i = 0; i < n; i++) data[i] = positive ? 0.6f + 0.8f * (float)rng.NextDouble() : 0.6f * (float)(rng.NextDouble() - 0.5);
        Tensor t = new(new TensorShape(dims), DType.F32);
        fixed (float* p = data) Buffer.MemoryCopy(p, (void*)t.DataPointer, n * 4, n * 4);
        Weights[name] = t;
        _raw[name] = data;
    }

    /// <summary>Plain-loop forward of <c>[4, DimF, T]</c> (flat, channel-major) following the ONNX node order.</summary>
    public float[] Forward(float[] spec, int time)
    {
        int levels = Widths.Length - 1;
        float[] x = Relu(Conv(spec, Channels, DimF, time, "first_conv", 1, 1, 0));   // [C][F][T]
        x = Transpose(x, Widths[0], DimF, time);                                       // [C][T][F]
        int t = time, f = DimF;
        List<float[]> skips = [];
        for (int i = 0; i < levels; i++)
        {
            x = TfcTdf(x, Widths[i], t, f, $"encoding_blocks.{i}");
            skips.Add(x);
            x = Relu(Conv(x, Widths[i], t, f, $"encoding_blocks.{i}.down", 2, 2, 0));
            t /= 2;
            f /= 2;
        }
        x = TfcTdf(x, Widths[levels], t, f, "bottleneck_block");
        for (int i = 0; i < levels; i++)
        {
            int level = levels - 1 - i;
            x = Relu(ConvTranspose(x, Widths[level + 1], t, f, $"decoding_blocks.{i}.up"));
            t *= 2;
            f *= 2;
            float[] skip = skips[level];
            for (int k = 0; k < x.Length; k++) x[k] *= skip[k];
            x = TfcTdf(x, Widths[level], t, f, $"decoding_blocks.{i}");
        }
        x = Transpose(x, Widths[0], time, DimF);                                        // [C][F][T]
        return Conv(x, Widths[0], DimF, time, "final_conv", 1, 1, 0);
    }

    private float[] TfcTdf(float[] x, int c, int t, int f, string p)
    {
        float[] h = x;
        for (int j = 0; j < 3; j++) h = Relu(Conv(h, c, t, f, $"{p}.tfc.{j}", 3, 1, 1));
        int mid = _raw[$"{p}.tdf.0.weight"].Length / f;
        float[] y = Linear(h, c * t, f, _raw[$"{p}.tdf.0.weight"], mid);
        Affine(y, c, t * mid, _raw[$"{p}.tdf.1.scale"], _raw[$"{p}.tdf.1.shift"]);
        y = Relu(y);
        y = Linear(y, c * t, mid, _raw[$"{p}.tdf.2.weight"], f);
        Affine(y, c, t * f, _raw[$"{p}.tdf.3.scale"], _raw[$"{p}.tdf.3.shift"]);
        y = Relu(y);
        float[] sum = new float[h.Length];
        for (int k = 0; k < sum.Length; k++) sum[k] = h[k] + y[k];
        return sum;
    }

    private static float[] Linear(float[] x, int rows, int inDim, float[] w, int outDim)
    {
        float[] y = new float[rows * outDim];
        for (int r = 0; r < rows; r++)
        {
            for (int o = 0; o < outDim; o++)
            {
                float acc = 0;
                for (int i = 0; i < inDim; i++) acc += x[r * inDim + i] * w[o * inDim + i];
                y[r * outDim + o] = acc;
            }
        }
        return y;
    }

    private static void Affine(float[] y, int c, int perChannel, float[] scale, float[] shift)
    {
        for (int ch = 0; ch < c; ch++)
        {
            for (int i = 0; i < perChannel; i++) y[ch * perChannel + i] = y[ch * perChannel + i] * scale[ch] + shift[ch];
        }
    }

    private static float[] Relu(float[] x)
    {
        float[] y = new float[x.Length];
        for (int i = 0; i < x.Length; i++) y[i] = Math.Max(0f, x[i]);
        return y;
    }

    private static float[] Transpose(float[] x, int c, int a, int b)
    {
        float[] y = new float[x.Length];
        for (int ch = 0; ch < c; ch++)
        {
            for (int i = 0; i < a; i++)
            {
                for (int j = 0; j < b; j++) y[(ch * b + j) * a + i] = x[(ch * a + i) * b + j];
            }
        }
        return y;
    }

    /// <summary>Conv over <c>[cin][h][w]</c> with a square kernel, zero padding and the given stride.</summary>
    private float[] Conv(float[] x, int cin, int h, int w, string name, int k, int stride, int pad)
    {
        float[] wt = _raw[name + ".weight"], bias = _raw[name + ".bias"];
        int cout = bias.Length;
        int oh = (h + 2 * pad - k) / stride + 1, ow = (w + 2 * pad - k) / stride + 1;
        float[] y = new float[cout * oh * ow];
        for (int co = 0; co < cout; co++)
        {
            for (int oy = 0; oy < oh; oy++)
            {
                for (int ox = 0; ox < ow; ox++)
                {
                    float acc = bias[co];
                    for (int ci = 0; ci < cin; ci++)
                    {
                        for (int ky = 0; ky < k; ky++)
                        {
                            for (int kx = 0; kx < k; kx++)
                            {
                                int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
                                if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                                acc += wt[((co * cin + ci) * k + ky) * k + kx] * x[(ci * h + iy) * w + ix];
                            }
                        }
                    }
                    y[(co * oh + oy) * ow + ox] = acc;
                }
            }
        }
        return y;
    }

    private float[] ConvTranspose(float[] x, int cin, int h, int w, string name)
    {
        float[] wt = _raw[name + ".weight"], bias = _raw[name + ".bias"];
        int cout = bias.Length;
        float[] y = new float[cout * 2 * h * 2 * w];
        for (int co = 0; co < cout; co++)
        {
            for (int oy = 0; oy < 2 * h; oy++)
            {
                for (int ox = 0; ox < 2 * w; ox++)
                {
                    float acc = bias[co];
                    for (int ci = 0; ci < cin; ci++)
                        acc += x[(ci * h + oy / 2) * w + ox / 2] * wt[((ci * cout + co) * 2 + oy % 2) * 2 + ox % 2];
                    y[(co * 2 * h + oy) * 2 * w + ox] = acc;
                }
            }
        }
        return y;
    }

    public void Dispose()
    {
        foreach (Tensor t in Weights.Values) t.Dispose();
    }
}
