using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Strict weight access for one HunyuanVideo-1.5 VAE load. Every <see cref="Take"/> marks its key consumed;
/// <see cref="ThrowOnUnconsumed"/> then fails on any checkpoint tensor the layers did not read, so a wrong config or a foreign
/// checkpoint cannot load and produce quietly wrong pixels.</summary>
internal sealed class HunyuanVideo15VaeWeights(IReadOnlyDictionary<string, Tensor> source)
{
    private readonly HashSet<string> _used = new(StringComparer.Ordinal);

    public Tensor Take(string key)
    {
        if (!source.TryGetValue(key, out Tensor? tensor))
            throw new KeyNotFoundException($"HunyuanVideo-1.5 VAE checkpoint has no tensor '{key}'.");
        _used.Add(key);
        return tensor;
    }

    /// <summary>RMS norm: a learnable <c>gamma</c> per channel, no bias (MovieRoll's <c>RMS_norm(bias=False)</c>).</summary>
    public WanRmsNorm Norm(string prefix, float eps)
    {
        Tensor gamma = Take($"{prefix}.gamma");
        WanRmsNorm norm = new((int)gamma.Shape.ElementCount, eps);
        norm.LoadWeights(gamma);
        return norm;
    }

    /// <summary>A conv whose weight is <c>{weightBase}.weight</c> and bias <c>{weightBase}.bias</c>. A kernel of 3 is MovieRoll's
    /// <c>CausalConv3d</c> (causal temporal replicate pad, symmetric replicate pad in H/W); a kernel of 1 is a plain pointwise <c>nn.Conv3d</c>.</summary>
    public CausalConv3d Conv(string weightBase)
    {
        Tensor weight = Take($"{weightBase}.weight");
        Tensor bias = Take($"{weightBase}.bias");
        int k = (int)weight.Shape[2];
        if (k == 3)
            return new CausalConv3d(weight, bias, padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: true, spatialReplicatePad: true);
        if (k == 1)
            return new CausalConv3d(weight, bias);
        throw new InvalidDataException($"HunyuanVideo-1.5 VAE conv '{weightBase}' has kernel {k}; only 1 and 3 exist in this architecture.");
    }

    /// <summary>Throws when the checkpoint holds a tensor no layer read.</summary>
    public void ThrowOnUnconsumed()
    {
        string[] left = source.Keys.Where(key => !_used.Contains(key)).Order(StringComparer.Ordinal).ToArray();
        if (left.Length == 0) return;
        string shown = string.Join(", ", left.Take(8)) + (left.Length > 8 ? $", ... ({left.Length} in total)" : "");
        throw new InvalidDataException(
            $"HunyuanVideo-1.5 VAE checkpoint holds {left.Length} tensor(s) this VAE does not consume: {shown}. "
            + "A layer count or channel config that does not match the checkpoint would load and give wrong output.");
    }
}

/// <summary>Index maps of the HunyuanVideo-1.5 VAE's shuffle resamplers and shortcuts, transcribed from MovieRoll's <c>causal_vae.py</c>
/// (einops patterns and <c>repeat_interleave</c>). Tensors are <c>[1, C, T, H, W]</c> F32. These run on the host: correct, not fast.</summary>
internal static class HunyuanVideo15VaeShuffle
{
    public static (int C, int T, int H, int W) Dims(Tensor x)
    {
        if (x.Shape.Rank != 5 || x.Shape[0] != 1 || x.DType != DType.F32)
            throw new ArgumentException($"HunyuanVideo-1.5 VAE layers take an F32 [1, C, T, H, W] tensor, got {x.Shape} {x.DType}.", nameof(x));
        return ((int)x.Shape[1], (int)x.Shape[2], (int)x.Shape[3], (int)x.Shape[4]);
    }

    public static Tensor Alloc(int c, int t, int h, int w) => new(new TensorShape([1L, c, t, h, w]), DType.F32);

    private static int Idx(int c, int t, int h, int w, int ct, int ch, int cw)
        => (((c * ct) + t) * ch + h) * cw + w;

    private static int Quotient(int value, int divisor, string what)
    {
        if (divisor <= 0 || value % divisor != 0)
            throw new ArgumentException($"{what}: {value} is not divisible by {divisor}.");
        return value / divisor;
    }

    /// <summary>Temporal frame count after a temporal downsample: the first frame stays alone and the rest pair up, so <c>T' = 1 + (T−1)/2</c>.</summary>
    private static int DownFrames(int t)
    {
        if ((t - 1) % 2 != 0) throw new ArgumentException($"A temporal downsample needs 1 + 2n frames, got {t}.");
        return 1 + (t - 1) / 2;
    }

    /// <summary>Main path of a downsample: the conv output <paramref name="h"/> (<c>cout / 8</c> or <c>cout / 4</c> channels) packed into
    /// <paramref name="cout"/> channels, spatially 2× smaller (and temporally halved after the first frame when <paramref name="temporal"/>).</summary>
    public static Tensor PackDownsample(Tensor h, int cout, bool temporal)
    {
        (int co, int t, int hh, int ww) = Dims(h);
        int oh = hh / 2, ow = ww / 2;
        ReadOnlySpan<float> s = h.AsReadOnlySpan<float>();

        if (!temporal)
        {
            if (4 * co != cout) throw new ArgumentException($"spatial downsample: 4·{co} must equal cout {cout}.");
            Tensor o = Alloc(cout, t, oh, ow);
            Span<float> d = o.AsSpan<float>();
            for (int oc = 0; oc < cout; oc++)
                for (int tt = 0; tt < t; tt++)
                    for (int i = 0; i < oh; i++)
                        for (int j = 0; j < ow; j++)
                        {
                            int rr = oc / co, c = oc % co;
                            d[Idx(oc, tt, i, j, t, oh, ow)] = s[Idx(c, tt, 2 * i + rr / 2, 2 * j + rr % 2, t, hh, ww)];
                        }
            return o;
        }

        if (cout != 8 * co) throw new ArgumentException($"temporal downsample: cout {cout} must be 8·{co}.");
        int tOut = DownFrames(t);
        Tensor r = Alloc(cout, tOut, oh, ow);
        Span<float> dst = r.AsSpan<float>();
        for (int oc = 0; oc < cout; oc++)
            for (int u = 0; u < tOut; u++)
                for (int i = 0; i < oh; i++)
                    for (int j = 0; j < ow; j++)
                    {
                        int c, srcT, rr;
                        if (u == 0)
                        {
                            // h_first is the first frame packed as (r2 r3 c) into 4·co channels, then concatenated with itself.
                            int kk = oc % (4 * co);
                            rr = kk / co; c = kk % co; srcT = 0;
                        }
                        else
                        {
                            // h_next packs (r1 r2 r3 c) over frame pairs: output frame u = 1 + f, source frame 1 + 2f + r1.
                            int f = u - 1;
                            int r1 = oc / (4 * co), kk = oc % (4 * co);
                            rr = kk / co; c = kk % co; srcT = 1 + 2 * f + r1;
                        }
                        dst[Idx(oc, u, i, j, tOut, oh, ow)] = s[Idx(c, srcT, 2 * i + rr / 2, 2 * j + rr % 2, t, hh, ww)];
                    }
        return r;
    }

    /// <summary>Shortcut of a downsample: the input <paramref name="x"/> packed the same way and mean-pooled into <paramref name="cout"/> channels.</summary>
    public static Tensor DownsampleShortcut(Tensor x, int cout, bool temporal)
    {
        (int cin, int t, int hh, int ww) = Dims(x);
        int oh = hh / 2, ow = ww / 2;
        ReadOnlySpan<float> s = x.AsReadOnlySpan<float>();

        if (!temporal)
        {
            int gs = Quotient(4 * cin, cout, "spatial downsample shortcut group");
            Tensor o = Alloc(cout, t, oh, ow);
            Span<float> d = o.AsSpan<float>();
            for (int oc = 0; oc < cout; oc++)
                for (int tt = 0; tt < t; tt++)
                    for (int i = 0; i < oh; i++)
                        for (int j = 0; j < ow; j++)
                        {
                            double sum = 0;
                            for (int g = 0; g < gs; g++)
                            {
                                int k = oc * gs + g, rr = k / cin, c = k % cin;
                                sum += s[Idx(c, tt, 2 * i + rr / 2, 2 * j + rr % 2, t, hh, ww)];
                            }
                            d[Idx(oc, tt, i, j, t, oh, ow)] = (float)(sum / gs);
                        }
            return o;
        }

        int tOut = DownFrames(t);
        // Reference: x_first is (r2 r3 c) with 4·cin channels viewed as (cout, (8·cin/cout)//2) = (cout, 4·cin/cout).
        int halfFirst = Quotient(4 * cin, cout, "temporal downsample first-frame group");
        int gsRest = Quotient(8 * cin, cout, "temporal downsample rest group");
        Tensor r = Alloc(cout, tOut, oh, ow);
        Span<float> dst = r.AsSpan<float>();
        for (int oc = 0; oc < cout; oc++)
            for (int u = 0; u < tOut; u++)
                for (int i = 0; i < oh; i++)
                    for (int j = 0; j < ow; j++)
                    {
                        double sum = 0;
                        if (u == 0)
                        {
                            // The first frame is packed (r2 r3 c) into 4·cin channels; the shortcut averages half a group of them.
                            for (int g = 0; g < halfFirst; g++)
                            {
                                int k = oc * halfFirst + g, rr = k / cin, c = k % cin;
                                sum += s[Idx(c, 0, 2 * i + rr / 2, 2 * j + rr % 2, t, hh, ww)];
                            }
                            dst[Idx(oc, u, i, j, tOut, oh, ow)] = (float)(sum / halfFirst);
                        }
                        else
                        {
                            int f = u - 1;
                            for (int g = 0; g < gsRest; g++)
                            {
                                int k = oc * gsRest + g, r1 = k / (4 * cin), kk = k % (4 * cin), rr = kk / cin, c = kk % cin;
                                sum += s[Idx(c, 1 + 2 * f + r1, 2 * i + rr / 2, 2 * j + rr % 2, t, hh, ww)];
                            }
                            dst[Idx(oc, u, i, j, tOut, oh, ow)] = (float)(sum / gsRest);
                        }
                    }
        return r;
    }

    /// <summary>Main path of an upsample: the conv output <paramref name="h"/> (<c>factor · cout</c> channels) unpacked to <paramref name="cout"/>
    /// channels with 2× spatial (and, when <paramref name="temporal"/>, 2× temporal after the first frame) expansion.</summary>
    public static Tensor UnpackUpsample(Tensor h, int cout, bool temporal)
    {
        (int ch, int t, int hh, int ww) = Dims(h);
        int oh = hh * 2, ow = ww * 2;
        ReadOnlySpan<float> s = h.AsReadOnlySpan<float>();

        if (!temporal)
        {
            if (ch != 4 * cout) throw new ArgumentException($"spatial upsample: channels {ch} must be 4·{cout}.");
            Tensor o = Alloc(cout, t, oh, ow);
            Span<float> d = o.AsSpan<float>();
            for (int oc = 0; oc < cout; oc++)
                for (int tt = 0; tt < t; tt++)
                    for (int i = 0; i < oh; i++)
                        for (int j = 0; j < ow; j++)
                        {
                            int rr = (i % 2) * 2 + (j % 2);
                            d[Idx(oc, tt, i, j, t, oh, ow)] = s[Idx(rr * cout + oc, tt, i / 2, j / 2, t, hh, ww)];
                        }
            return o;
        }

        if (ch != 8 * cout) throw new ArgumentException($"temporal upsample: channels {ch} must be 8·{cout}.");
        int tOut = 1 + 2 * (t - 1);
        Tensor r = Alloc(cout, tOut, oh, ow);
        Span<float> dst = r.AsSpan<float>();
        for (int oc = 0; oc < cout; oc++)
            for (int u = 0; u < tOut; u++)
                for (int i = 0; i < oh; i++)
                    for (int j = 0; j < ow; j++)
                    {
                        int rr = (i % 2) * 2 + (j % 2), k, srcT;
                        if (u == 0)
                        {
                            // h_first: the first frame packed (r2 r3 c) with 2·cout channels, keeping the first cout of them.
                            k = rr * (2 * cout) + oc; srcT = 0;
                        }
                        else
                        {
                            // h_next: the remaining frames packed (r1 r2 r3 c); output frame u = 1 + 2f + r1 from source frame 1 + f.
                            int v = u - 1, f = v / 2, r1 = v % 2;
                            k = (r1 * 4 + rr) * cout + oc; srcT = 1 + f;
                        }
                        dst[Idx(oc, u, i, j, tOut, oh, ow)] = s[Idx(k, srcT, i / 2, j / 2, t, hh, ww)];
                    }
        return r;
    }

    /// <summary>Shortcut of an upsample: <paramref name="x"/> channel-repeated and unpacked the way <see cref="UnpackUpsample"/> unpacks the main path.</summary>
    public static Tensor UpsampleShortcut(Tensor x, int cout, bool temporal)
    {
        (int cin, int t, int hh, int ww) = Dims(x);
        int oh = hh * 2, ow = ww * 2;
        ReadOnlySpan<float> s = x.AsReadOnlySpan<float>();

        if (!temporal)
        {
            int rep = Quotient(4 * cout, cin, "spatial upsample shortcut repeat");
            Tensor o = Alloc(cout, t, oh, ow);
            Span<float> d = o.AsSpan<float>();
            for (int oc = 0; oc < cout; oc++)
                for (int tt = 0; tt < t; tt++)
                    for (int i = 0; i < oh; i++)
                        for (int j = 0; j < ow; j++)
                        {
                            int rr = (i % 2) * 2 + (j % 2);
                            int src = (rr * cout + oc) / rep;
                            d[Idx(oc, tt, i, j, t, oh, ow)] = s[Idx(src, tt, i / 2, j / 2, t, hh, ww)];
                        }
            return o;
        }

        int rep1 = Quotient(4 * cout, cin, "temporal upsample first-frame repeat");   // repeats // 2
        int rep2 = Quotient(8 * cout, cin, "temporal upsample rest repeat");
        int cFirst = Quotient(cin, 4, "temporal upsample first-frame pack");
        int cRest = Quotient(cin, 8, "temporal upsample rest pack");
        int tOut = 1 + 2 * (t - 1);
        Tensor r = Alloc(cout, tOut, oh, ow);
        Span<float> dst = r.AsSpan<float>();
        for (int oc = 0; oc < cout; oc++)
            for (int u = 0; u < tOut; u++)
                for (int i = 0; i < oh; i++)
                    for (int j = 0; j < ow; j++)
                    {
                        int rr = (i % 2) * 2 + (j % 2);
                        if (u == 0)
                        {
                            int k = rr * cFirst + oc / rep1;
                            dst[Idx(oc, u, i, j, tOut, oh, ow)] = s[Idx(k, 0, i / 2, j / 2, t, hh, ww)];
                        }
                        else
                        {
                            int v = u - 1, f = v / 2, r1 = v % 2;
                            int k = (r1 * 4 + rr) * cRest + oc / rep2;
                            dst[Idx(oc, u, i, j, tOut, oh, ow)] = s[Idx(k, 1 + f, i / 2, j / 2, t, hh, ww)];
                        }
                    }
        return r;
    }

    /// <summary>Encoder end shortcut: consecutive groups of <c>C / outC</c> channels averaged into <paramref name="outC"/> channels.</summary>
    public static Tensor GroupMean(Tensor h, int outC)
    {
        (int c, int t, int hh, int ww) = Dims(h);
        int gs = Quotient(c, outC, "encoder end shortcut group");
        Tensor o = Alloc(outC, t, hh, ww);
        ReadOnlySpan<float> s = h.AsReadOnlySpan<float>();
        Span<float> d = o.AsSpan<float>();
        int plane = t * hh * ww;
        for (int oc = 0; oc < outC; oc++)
            for (int p = 0; p < plane; p++)
            {
                double sum = 0;
                for (int g = 0; g < gs; g++) sum += s[(oc * gs + g) * plane + p];
                d[oc * plane + p] = (float)(sum / gs);
            }
        return o;
    }

    /// <summary>Decoder start shortcut: each latent channel repeated <c>outC / C</c> times, consecutively.</summary>
    public static Tensor RepeatChannels(Tensor z, int outC)
    {
        (int c, int t, int hh, int ww) = Dims(z);
        int rep = Quotient(outC, c, "decoder start shortcut repeat");
        Tensor o = Alloc(outC, t, hh, ww);
        ReadOnlySpan<float> s = z.AsReadOnlySpan<float>();
        Span<float> d = o.AsSpan<float>();
        int plane = t * hh * ww;
        for (int oc = 0; oc < outC; oc++)
        {
            int src = (oc / rep) * plane, dst = oc * plane;
            for (int p = 0; p < plane; p++) d[dst + p] = s[src + p];
        }
        return o;
    }

    /// <summary>Single-head attention over all <c>T·H·W</c> tokens with MovieRoll's frame-causal mask: a token attends to every token of its own
    /// frame and of earlier frames. Scale is <c>1/√C</c> (SDPA over one head of width C). Quadratic in tokens: host and tiny-config only.</summary>
    public static Tensor FrameCausalAttention(Tensor q, Tensor k, Tensor v)
    {
        (int c, int t, int hh, int ww) = Dims(q);
        int hw = hh * ww, n = t * hw;
        ReadOnlySpan<float> qs = q.AsReadOnlySpan<float>(), ks = k.AsReadOnlySpan<float>(), vs = v.AsReadOnlySpan<float>();
        Tensor o = Alloc(c, t, hh, ww);
        Span<float> d = o.AsSpan<float>();
        float scale = 1f / MathF.Sqrt(c);
        float[] scores = new float[n];
        for (int nq = 0; nq < n; nq++)
        {
            int limit = (nq / hw + 1) * hw;
            float max = float.NegativeInfinity;
            for (int m = 0; m < limit; m++)
            {
                double dot = 0;
                for (int ci = 0; ci < c; ci++) dot += (double)qs[ci * n + nq] * ks[ci * n + m];
                scores[m] = (float)dot * scale;
                if (scores[m] > max) max = scores[m];
            }
            double denom = 0;
            for (int m = 0; m < limit; m++)
            {
                scores[m] = MathF.Exp(scores[m] - max);
                denom += scores[m];
            }
            for (int ci = 0; ci < c; ci++)
            {
                double acc = 0;
                for (int m = 0; m < limit; m++) acc += scores[m] * vs[ci * n + m];
                d[ci * n + nq] = (float)(acc / denom);
            }
        }
        return o;
    }
}

/// <summary>MovieRoll <c>ResnetBlock</c>: RMS norm → SiLU → CausalConv3d(3) → RMS norm → SiLU → CausalConv3d(3), residual via a 1×1×1 conv when channels change.</summary>
internal sealed class HunyuanVideo15VaeResnet
{
    private readonly WanRmsNorm _norm1, _norm2;
    private readonly CausalConv3d _conv1, _conv2;
    private readonly CausalConv3d? _shortcut;

    public HunyuanVideo15VaeResnet(HunyuanVideo15VaeWeights w, string prefix, int cin, int cout, float eps)
    {
        _norm1 = w.Norm($"{prefix}.norm1", eps);
        _conv1 = w.Conv($"{prefix}.conv1.conv");
        _norm2 = w.Norm($"{prefix}.norm2", eps);
        _conv2 = w.Conv($"{prefix}.conv2.conv");
        if (cin != cout) _shortcut = w.Conv($"{prefix}.nin_shortcut");
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor h = Silu(backend, _norm1.Forward(backend, x));
        Tensor c1 = _conv1.Forward(backend, h);
        h.Dispose();
        Tensor h2 = Silu(backend, _norm2.Forward(backend, c1));
        c1.Dispose();
        Tensor c2 = _conv2.Forward(backend, h2);
        h2.Dispose();
        Tensor residual = _shortcut is null ? x : _shortcut.Forward(backend, x);
        Tensor output = new(c2.Shape, DType.F32);
        backend.Add(output, c2, residual);
        c2.Dispose();
        if (_shortcut is not null) residual.Dispose();
        return output;
    }

    internal static Tensor Silu(IBackend backend, Tensor x)
    {
        Tensor y = new(x.Shape, DType.F32);
        backend.Silu(y, x);
        x.Dispose();
        return y;
    }
}

/// <summary>MovieRoll mid-block attention: RMS norm, 1×1×1 q/k/v, frame-causal single-head attention, 1×1×1 output projection, residual.</summary>
internal sealed class HunyuanVideo15VaeMidAttention
{
    private readonly WanRmsNorm _norm;
    private readonly CausalConv3d _q, _k, _v, _proj;

    public HunyuanVideo15VaeMidAttention(HunyuanVideo15VaeWeights w, string prefix, float eps)
    {
        _norm = w.Norm($"{prefix}.norm", eps);
        _q = w.Conv($"{prefix}.q");
        _k = w.Conv($"{prefix}.k");
        _v = w.Conv($"{prefix}.v");
        _proj = w.Conv($"{prefix}.proj_out");
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor hn = _norm.Forward(backend, x);
        Tensor q = _q.Forward(backend, hn), k = _k.Forward(backend, hn), v = _v.Forward(backend, hn);
        hn.Dispose();
        Tensor attn = HunyuanVideo15VaeShuffle.FrameCausalAttention(q, k, v);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor projected = _proj.Forward(backend, attn);
        attn.Dispose();
        Tensor output = new(x.Shape, DType.F32);
        backend.Add(output, x, projected);
        projected.Dispose();
        return output;
    }
}

/// <summary>MovieRoll <c>DownsampleDCAE</c>: CausalConv3d then a shuffle, plus a channel-pooled shortcut of the input.</summary>
internal sealed class HunyuanVideo15VaeDownsample
{
    private readonly CausalConv3d _conv;
    private readonly int _cout;
    private readonly bool _temporal;

    public HunyuanVideo15VaeDownsample(HunyuanVideo15VaeWeights w, string prefix, int cout, bool temporal)
    {
        _conv = w.Conv($"{prefix}.conv.conv");
        _cout = cout;
        _temporal = temporal;
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor h = _conv.Forward(backend, x);
        Tensor main = HunyuanVideo15VaeShuffle.PackDownsample(h, _cout, _temporal);
        h.Dispose();
        Tensor shortcut = HunyuanVideo15VaeShuffle.DownsampleShortcut(x, _cout, _temporal);
        Tensor output = new(main.Shape, DType.F32);
        backend.Add(output, main, shortcut);
        main.Dispose(); shortcut.Dispose();
        return output;
    }
}

/// <summary>MovieRoll <c>UpsampleDCAE</c>: CausalConv3d then an unpacking shuffle, plus a repeated shortcut of the input.</summary>
internal sealed class HunyuanVideo15VaeUpsample
{
    private readonly CausalConv3d _conv;
    private readonly int _cout;
    private readonly bool _temporal;

    public HunyuanVideo15VaeUpsample(HunyuanVideo15VaeWeights w, string prefix, int cout, bool temporal)
    {
        _conv = w.Conv($"{prefix}.conv.conv");
        _cout = cout;
        _temporal = temporal;
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor h = _conv.Forward(backend, x);
        Tensor main = HunyuanVideo15VaeShuffle.UnpackUpsample(h, _cout, _temporal);
        h.Dispose();
        Tensor shortcut = HunyuanVideo15VaeShuffle.UpsampleShortcut(x, _cout, _temporal);
        Tensor output = new(main.Shape, DType.F32);
        backend.Add(output, main, shortcut);
        main.Dispose(); shortcut.Dispose();
        return output;
    }
}
