using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Building blocks shared by the ModifiedDAC encoder and decoder: causal convolutions, Snake residual units,
/// ConvNeXt blocks and the window-limited causal RoPE transformer. Activations are channels-first <c>[1, C, T]</c>
/// tensors for the conv stack and host <c>[T, D]</c> arrays inside the transformer.</summary>
internal static unsafe class DacOps
{
    internal sealed record TransformerLayer(Tensor AttnNorm, Tensor Wqkv, Tensor Wo, Tensor AttnScale,
        Tensor FfnNorm, Tensor W1, Tensor W3, Tensor W2, Tensor FfnScale);

    internal sealed record Transformer(TransformerLayer[] Layers, Tensor FinalNorm, int Heads, int HeadDim,
        int Intermediate, int? Window, float RopeBase, float NormEps);

    internal sealed record ConvNeXt(Tensor DwW, Tensor DwB, Tensor NormW, Tensor NormB, Tensor Pw1W, Tensor Pw1B,
        Tensor Pw2W, Tensor Pw2B, Tensor Gamma);

    internal sealed record Unit(Tensor Alpha1, Tensor Conv1W, Tensor Conv1B, Tensor Alpha2, Tensor Conv2W, Tensor Conv2B);

    internal static Tensor F(IReadOnlyDictionary<string, Tensor> w, string key) => WhisperOps.EnsureF32(w[key]);

    // ---- weight loading --------------------------------------------------------------------------------------

    internal static Transformer LoadTransformer(IReadOnlyDictionary<string, Tensor> w, string prefix, int layers, int heads,
        int headDim, int intermediate, int? window, float ropeBase, float normEps)
    {
        TransformerLayer[] l = new TransformerLayer[layers];
        for (int i = 0; i < layers; i++)
        {
            string p = $"{prefix}.layers.{i}";
            l[i] = new TransformerLayer(F(w, $"{p}.attention_norm.weight"), F(w, $"{p}.attention.wqkv.weight"),
                F(w, $"{p}.attention.wo.weight"), F(w, $"{p}.attention_layer_scale.gamma"), F(w, $"{p}.ffn_norm.weight"),
                F(w, $"{p}.feed_forward.w1.weight"), F(w, $"{p}.feed_forward.w3.weight"), F(w, $"{p}.feed_forward.w2.weight"),
                F(w, $"{p}.ffn_layer_scale.gamma"));
        }
        return new Transformer(l, F(w, $"{prefix}.norm.weight"), heads, headDim, intermediate, window, ropeBase, normEps);
    }

    internal static ConvNeXt LoadConvNeXt(IReadOnlyDictionary<string, Tensor> w, string p) => new(
        F(w, $"{p}.dwconv.conv.weight"), F(w, $"{p}.dwconv.conv.bias"), F(w, $"{p}.norm.weight"), F(w, $"{p}.norm.bias"),
        F(w, $"{p}.pwconv1.weight"), F(w, $"{p}.pwconv1.bias"), F(w, $"{p}.pwconv2.weight"), F(w, $"{p}.pwconv2.bias"),
        F(w, $"{p}.gamma"));

    internal static Unit LoadUnit(IReadOnlyDictionary<string, Tensor> w, string u) => new(F(w, $"{u}.0.alpha"),
        WeightNorm.Compose(w, $"{u}.1.conv"), F(w, $"{u}.1.conv.bias"), F(w, $"{u}.2.alpha"),
        WeightNorm.Compose(w, $"{u}.3.conv"), F(w, $"{u}.3.conv.bias"));

    // ---- transformer (window-limited causal, interleaved RoPE, LayerScale) ------------------------------------

    /// <summary>Runs <paramref name="tr"/> over host <c>[T, D]</c> activations; the window keeps each query's last
    /// <c>Window</c> keys including itself (null = full causal).</summary>
    internal static float[] RunTransformer(IBackend backend, Transformer tr, float[] x, int t, int d)
    {
        int heads = tr.Heads, hd = tr.HeadDim, inter = tr.Intermediate;
        float[,] rope = RopeTable(t, hd, tr.RopeBase);
        foreach (TransformerLayer layer in tr.Layers)
        {
            float[] normed = RmsNorm(backend, x, layer.AttnNorm, t, d, tr.NormEps);
            float[] qkv = Linear(backend, normed, layer.Wqkv, t, d, 3 * d);
            float[] attn = new float[t * d];
            float scale = 1f / MathF.Sqrt(hd);
            for (int hh = 0; hh < heads; hh++)
            {
                float[] q = new float[t * hd], k = new float[t * hd];
                for (int i = 0; i < t; i++)
                    for (int e = 0; e < hd; e++)
                    {
                        q[i * hd + e] = qkv[i * 3 * d + hh * hd + e];
                        k[i * hd + e] = qkv[i * 3 * d + d + hh * hd + e];
                    }
                ApplyRopeInterleaved(q, rope, t, hd); ApplyRopeInterleaved(k, rope, t, hd);
                float[] sc = new float[t];
                for (int i = 0; i < t; i++)
                {
                    int lo = tr.Window is int win ? Math.Max(0, i - win + 1) : 0;
                    float max = float.NegativeInfinity;
                    for (int j = lo; j <= i; j++)
                    {
                        float s = 0f;
                        for (int e = 0; e < hd; e++) s += q[i * hd + e] * k[j * hd + e];
                        sc[j] = s * scale; max = MathF.Max(max, sc[j]);
                    }
                    float sum = 0f;
                    for (int j = lo; j <= i; j++) { sc[j] = MathF.Exp(sc[j] - max); sum += sc[j]; }
                    for (int e = 0; e < hd; e++)
                    {
                        float o = 0f;
                        for (int j = lo; j <= i; j++) o += sc[j] * qkv[j * 3 * d + 2 * d + hh * hd + e];
                        attn[i * d + hh * hd + e] = o / sum;
                    }
                }
            }
            float[] proj = Linear(backend, attn, layer.Wo, t, d, d);
            float* ga = (float*)layer.AttnScale.DataPointer;
            float[] hres = new float[t * d];
            for (int i = 0; i < t; i++) for (int c = 0; c < d; c++) hres[i * d + c] = x[i * d + c] + ga[c] * proj[i * d + c];

            float[] n2 = RmsNorm(backend, hres, layer.FfnNorm, t, d, tr.NormEps);
            float[] a = Linear(backend, n2, layer.W1, t, d, inter), b = Linear(backend, n2, layer.W3, t, d, inter);
            for (int i = 0; i < a.Length; i++) a[i] = a[i] / (1f + MathF.Exp(-a[i])) * b[i];
            float[] ff = Linear(backend, a, layer.W2, t, inter, d);
            float* gf = (float*)layer.FfnScale.DataPointer;
            float[] next = new float[t * d];
            for (int i = 0; i < t; i++) for (int c = 0; c < d; c++) next[i * d + c] = hres[i * d + c] + gf[c] * ff[i * d + c];
            x = next;
        }
        return RmsNorm(backend, x, tr.FinalNorm, t, d, tr.NormEps);
    }

    internal static float[] RmsNorm(IBackend backend, float[] x, Tensor weight, int t, int d, float eps)
    {
        using Tensor inp = FromHost(x, 1, t, d);
        using Tensor outT = new(new TensorShape(1, t, d), DType.F32);
        backend.RmsNorm(outT, inp, weight, eps);
        return ToHost(outT);
    }

    internal static float[] Linear(IBackend backend, float[] x, Tensor weight, int t, int k, int n, Tensor? bias = null)
    {
        using Tensor inp = FromHost(x, 1, t, k);
        using Tensor outT = new(new TensorShape(1, t, n), DType.F32);
        backend.Linear(outT, inp, weight, bias);
        return ToHost(outT);
    }

    private static float[,] RopeTable(int t, int hd, float baseTheta)
    {
        float[,] table = new float[t, hd];   // [pos, 2i] = cos, [pos, 2i+1] = sin
        for (int p = 0; p < t; p++)
            for (int i = 0; i < hd / 2; i++)
            {
                double f = 1.0 / Math.Pow(baseTheta, 2.0 * i / hd), a = p * f;
                table[p, 2 * i] = (float)Math.Cos(a); table[p, 2 * i + 1] = (float)Math.Sin(a);
            }
        return table;
    }

    // fish-speech apply_rotary_emb: adjacent pairs (x[2i], x[2i+1]) rotate by a position-dependent angle.
    private static void ApplyRopeInterleaved(float[] v, float[,] table, int t, int hd)
    {
        for (int p = 0; p < t; p++)
            for (int i = 0; i < hd / 2; i++)
            {
                float c = table[p, 2 * i], s = table[p, 2 * i + 1];
                float x0 = v[p * hd + 2 * i], x1 = v[p * hd + 2 * i + 1];
                v[p * hd + 2 * i] = x0 * c - x1 * s;
                v[p * hd + 2 * i + 1] = x1 * c + x0 * s;
            }
    }

    // ---- conv stack (channels-first) --------------------------------------------------------------------------

    /// <summary>fish-speech <c>CausalConvNet</c>: left-pad <c>effectiveKernel − stride</c>, right-pad so the last window
    /// is complete, no conv padding. Consumes <paramref name="x"/>; returns <c>[1, cOut, frames]</c>.</summary>
    internal static Tensor CausalConv(IBackend backend, Tensor x, Tensor w, Tensor? b, int stride = 1, int dilation = 1, int groups = 1)
    {
        int k = (int)w.Shape[2], cOut = (int)w.Shape[0], len = (int)x.Shape[2];
        int ek = (k - 1) * dilation + 1, pad = ek - stride;
        double frames = (len - ek + pad) / (double)stride + 1;
        int outFrames = (int)Math.Ceiling(frames);
        int extra = (outFrames - 1) * stride + (ek - pad) - len;
        Tensor o = new(new TensorShape(1, cOut, outFrames), DType.F32);
        backend.Conv1d(o, x, w, b, stride, pad, extra, dilation, groups);
        x.Dispose();
        return o;
    }

    /// <summary>fish-speech <c>CausalTransConvNet</c>: transposed conv, then drop the trailing <c>kernel − stride</c>
    /// samples → exactly <c>stride ×</c> the input length.</summary>
    internal static Tensor CausalConvTranspose(IBackend backend, Tensor x, Tensor w, Tensor b, int stride, int kernel)
    {
        int cOut = (int)w.Shape[1], tIn = (int)x.Shape[2], tOut = tIn * stride;
        Tensor o = new(new TensorShape(1, cOut, tOut), DType.F32);
        backend.ConvTranspose1d(o, x, w, b, stride, 0, kernel - stride, 1, 1);
        x.Dispose();
        return o;
    }

    internal static Tensor SnakeOp(IBackend backend, Tensor x, Tensor alpha)
    {
        Tensor o = new(x.Shape, DType.F32);
        backend.Snake(o, x, alpha, null);
        x.Dispose();
        return o;
    }

    /// <summary>DAC residual unit: Snake → causal dilated conv → Snake → causal 1×1 conv, plus the input. Consumes <paramref name="x"/>.</summary>
    internal static Tensor ResidualUnit(IBackend backend, Tensor x, Unit u, int dilation)
    {
        Tensor y = SnakeOp(backend, CloneTensor(x), u.Alpha1);
        y = CausalConv(backend, y, u.Conv1W, u.Conv1B, dilation: dilation);
        y = SnakeOp(backend, y, u.Alpha2);
        y = CausalConv(backend, y, u.Conv2W, u.Conv2B);
        float* xp = (float*)x.DataPointer, yp = (float*)y.DataPointer;
        for (long i = 0; i < y.ElementCount; i++) yp[i] += xp[i];
        x.Dispose();
        return y;
    }

    /// <summary>ConvNeXtBlock: causal depthwise k7 → LayerNorm(1e-6) → Linear(4×) → GELU → Linear → γ → + input.</summary>
    internal static Tensor ConvNeXtBlock(IBackend backend, Tensor x, ConvNeXt b)
    {
        int d = (int)x.Shape[1], t = (int)x.Shape[2];
        Tensor dw = CausalConv(backend, CloneTensor(x), b.DwW, b.DwB, groups: d);
        float[] h = new float[t * d];
        float* dp = (float*)dw.DataPointer;
        for (int c = 0; c < d; c++) for (int j = 0; j < t; j++) h[j * d + c] = dp[c * t + j];
        dw.Dispose();

        using (Tensor inp = FromHost(h, 1, t, d))
        using (Tensor normed = new(new TensorShape(1, t, d), DType.F32))
        {
            backend.LayerNorm(normed, inp, b.NormW, b.NormB, 1e-6f);
            h = ToHost(normed);
        }
        int hidden = (int)b.Pw1W.Shape[0];
        float[] a = Linear(backend, h, b.Pw1W, t, d, hidden, b.Pw1B);
        for (int i = 0; i < a.Length; i++) a[i] = 0.5f * a[i] * (1f + (float)Erf(a[i] / Math.Sqrt(2.0)));   // exact GELU
        float[] o = Linear(backend, a, b.Pw2W, t, hidden, d, b.Pw2B);
        float* g = (float*)b.Gamma.DataPointer;
        float* xp = (float*)x.DataPointer;
        for (int c = 0; c < d; c++) for (int j = 0; j < t; j++) xp[c * t + j] += g[c] * o[j * d + c];
        return x;
    }

    private static double Erf(double x)
    {
        double ax = Math.Abs(x), r;
        if (ax < 2.5)
        {
            double sum = ax, term = ax;
            for (int n = 1; n < 60; n++) { term *= -ax * ax / n; sum += term / (2 * n + 1); }
            r = 2.0 / Math.Sqrt(Math.PI) * sum;
        }
        else
        {
            double f = 0;
            for (int n = 60; n >= 1; n--) f = n / 2.0 / (ax + f);
            r = 1.0 - Math.Exp(-ax * ax) / Math.Sqrt(Math.PI) / (ax + f);
        }
        return x < 0 ? -r : r;
    }

    /// <summary>Channels-first <c>[1, C, T]</c> → host <c>[T, C]</c>.</summary>
    internal static float[] ToTimeMajor(Tensor x)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2];
        float[] h = new float[t * c];
        float* p = (float*)x.DataPointer;
        for (int i = 0; i < c; i++) for (int j = 0; j < t; j++) h[j * c + i] = p[i * t + j];
        return h;
    }

    internal static Tensor FromTimeMajor(float[] h, int t, int c)
    {
        Tensor x = new(new TensorShape(1, c, t), DType.F32);
        float* p = (float*)x.DataPointer;
        for (int i = 0; i < c; i++) for (int j = 0; j < t; j++) p[i * t + j] = h[j * c + i];
        return x;
    }

    internal static Tensor CloneTensor(Tensor x)
    {
        Tensor c = new(x.Shape, DType.F32);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)c.DataPointer, x.ElementCount * 4, x.ElementCount * 4);
        return c;
    }

    internal static Tensor FromHost(float[] v, int b, int t, int d)
    {
        Tensor x = new(new TensorShape(b, t, d), DType.F32);
        v.AsSpan().CopyTo(new Span<float>((void*)x.DataPointer, v.Length));
        return x;
    }

    internal static float[] ToHost(Tensor x) => new ReadOnlySpan<float>((void*)x.DataPointer, (int)x.ElementCount).ToArray();
}
