using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
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

/// <summary>Layout of the HunyuanVideo-1.5 VAE's DCAE resamplers and shortcuts, on the backend. Tensors are <c>[1, C, T, H, W]</c> F32 and
/// batch 1 only. Every DCAE shuffle is one <see cref="PixelShuffles"/> move with time outermost, then height and width, channel innermost
/// (<c>((t·fH + h)·fW + w)·C + c</c>, MovieRoll's <c>(r1 r2 r3 c)</c> packing). A factor of 1 makes the time position irrelevant, so the first-frame
/// branches (time factor 1) use the same order. The shortcut group means (<see cref="GroupMean"/>) are the only host arithmetic: the backend has no
/// grouped reduction, so they read the packed tensors on the host.</summary>
internal static class HunyuanVideo15VaeShuffle
{
    private static readonly ChannelOrder Order = new(ShuffleAxis.Time, ShuffleAxis.Height, ShuffleAxis.Width, ShuffleAxis.Channel);

    /// <summary>The <c>[1, C, T, H, W]</c> extents of an F32 tensor, after checking the rank, the batch and that the element count fits an int index.</summary>
    public static (int C, int T, int H, int W) Dims(Tensor x)
    {
        if (x.Shape.Rank != 5 || x.Shape[0] != 1 || x.DType != DType.F32)
            throw new ArgumentException($"HunyuanVideo-1.5 VAE layers take an F32 [1, C, T, H, W] tensor, got {x.Shape} {x.DType}.", nameof(x));
        RequireIndexable(x.Shape[1], x.Shape[2], x.Shape[3], x.Shape[4]);
        return ((int)x.Shape[1], (int)x.Shape[2], (int)x.Shape[3], (int)x.Shape[4]);
    }

    /// <summary>A new F32 <c>[1, C, T, H, W]</c> tensor. Throws when the element count would not fit the int indices the layers use.</summary>
    public static Tensor Alloc(int c, int t, int h, int w)
    {
        RequireIndexable(c, t, h, w);
        return new(new TensorShape([1L, c, t, h, w]), DType.F32);
    }

    /// <summary>Throws when <c>C·T·H·W</c> exceeds <see cref="int.MaxValue"/>: the span indices here are ints, so a larger clip would truncate
    /// silently. The fix is to chunk the clip, not to widen the indices.</summary>
    private static void RequireIndexable(long c, long t, long h, long w)
    {
        Int128 count = (Int128)c * t * h * w;
        if (count > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(c), $"A [1, {c}, {t}, {h}, {w}] tensor has {count} elements; the VAE indexes with int, so the limit is {int.MaxValue}. Chunk the clip.");
    }

    private static void RequireEvenSpatial(int h, int w, string what)
    {
        if (h % 2 != 0 || w % 2 != 0)
            throw new ArgumentException($"The {what} needs even H and W (a 2x2 spatial shuffle); got H={h}, W={w}.", nameof(h));
    }

    private static int Quotient(int value, int divisor, string what)
    {
        if (divisor <= 0 || value % divisor != 0)
            throw new ArgumentException($"{what}: {value} is not divisible by {divisor}.");
        return value / divisor;
    }

    /// <summary>Frames after a temporal downsample: the first frame stays alone and the rest pair up, <c>T' = 1 + (T−1)/2</c>.</summary>
    private static int DownFrames(int t)
    {
        if ((t - 1) % 2 != 0) throw new ArgumentException($"A temporal downsample needs 1 + 2n frames, got {t}.");
        return 1 + (t - 1) / 2;
    }

    /// <summary>Unpacks <paramref name="packed"/> (<c>channels·fT·fH·fW</c> channels on a small grid) to <c>[1, channels, frames·fT, H·fH, W·fW]</c>.</summary>
    private static Tensor Unshuffle(IBackend backend, Tensor packed, int channels, int fT, int fH, int fW)
    {
        (int c, int frames, int hp, int wp) = Dims(packed);
        if (c != channels * fT * fH * fW)
            throw new ArgumentException($"Unshuffle of {packed.Shape} by {fT}x{fH}x{fW} to {channels} channels needs {channels * fT * fH * fW} packed channels.", nameof(packed));
        Tensor output = Alloc(channels, frames * fT, hp * fH, wp * fW);
        PixelShuffles.DepthToSpace(backend, output, packed, ShuffleSpec.ChannelsFirst(1, frames, hp, wp, channels, fT, fH, fW, Order));
        return output;
    }

    /// <summary>Packs <paramref name="unpacked"/> to <c>[1, C·fT·fH·fW, T/fT, H/fH, W/fW]</c>: the inverse of <see cref="Unshuffle"/>.</summary>
    private static Tensor Shuffle(IBackend backend, Tensor unpacked, int fT, int fH, int fW)
    {
        (int channels, int t, int h, int w) = Dims(unpacked);
        if (t % fT != 0 || h % fH != 0 || w % fW != 0)
            throw new ArgumentException($"Shuffle of {unpacked.Shape} by {fT}x{fH}x{fW} needs each extent divisible by its factor.", nameof(unpacked));
        int frames = t / fT, hp = h / fH, wp = w / fW;
        Tensor packed = Alloc(channels * fT * fH * fW, frames, hp, wp);
        PixelShuffles.SpaceToDepth(backend, packed, unpacked, ShuffleSpec.ChannelsFirst(1, frames, hp, wp, channels, fT, fH, fW, Order));
        return packed;
    }

    /// <summary>Frames <c>[start, start + count)</c> of a <c>[1, C, T, H, W]</c> tensor.</summary>
    private static Tensor Frames(IBackend backend, Tensor x, int start, int count)
    {
        (int c, int _, int h, int w) = Dims(x);
        Tensor output = Alloc(c, count, h, w);
        StridedCopies.SliceAxis(backend, output, x, 2, start, count);
        return output;
    }

    /// <summary>The first <paramref name="count"/> channels of a <c>[1, C, T, H, W]</c> tensor.</summary>
    private static Tensor Channels(IBackend backend, Tensor x, int count)
    {
        (int _, int t, int h, int w) = Dims(x);
        Tensor output = Alloc(count, t, h, w);
        StridedCopies.SliceAxis(backend, output, x, 1, 0, count);
        return output;
    }

    /// <summary>Each channel repeated <paramref name="repeats"/> times in place: channel <c>k</c> of the output reads input channel <c>k / repeats</c>.</summary>
    private static Tensor RepeatChannels(IBackend backend, Tensor x, int repeats)
    {
        (int c, int t, int h, int w) = Dims(x);
        Tensor output = Alloc(c * repeats, t, h, w);
        StridedCopies.RepeatInterleave(backend, output, x, 1, repeats);
        return output;
    }

    /// <summary>Concatenates along <paramref name="dim"/> (1 channels, 2 frames) on the backend.</summary>
    private static Tensor Concat(IBackend backend, int dim, params Tensor[] parts)
    {
        (int c0, int t0, int h0, int w0) = Dims(parts[0]);
        int c = c0, t = t0;
        for (int i = 1; i < parts.Length; i++)
        {
            (int ci, int ti, int hi, int wi) = Dims(parts[i]);
            if (hi != h0 || wi != w0 || (dim == 1 && ti != t0) || (dim == 2 && ci != c0))
                throw new ArgumentException("Concatenated parts disagree outside the concatenation axis.");
            if (dim == 1) c += ci; else t += ti;
        }
        Tensor output = Alloc(c, t, h0, w0);
        backend.Concat(output, parts, dim);
        return output;
    }

    /// <summary>Downsample main path. <paramref name="h"/> is the conv output (<c>cout/8</c> or <c>cout/4</c> channels); the result is <c>cout</c>
    /// channels, half the spatial size, and with <paramref name="temporal"/> the first frame alone then frame pairs packed.</summary>
    public static Tensor PackDownsample(IBackend backend, Tensor h, int cout, bool temporal)
    {
        (int co, int t, int hh, int ww) = Dims(h);
        RequireEvenSpatial(hh, ww, "downsample");
        if (!temporal)
        {
            if (4 * co != cout) throw new ArgumentException($"spatial downsample: 4·{co} must equal cout {cout}.");
            return Shuffle(backend, h, 1, 2, 2);
        }
        if (cout != 8 * co) throw new ArgumentException($"temporal downsample: cout {cout} must be 8·{co}.");
        DownFrames(t);
        // h_first is packed (r2 r3 c) into 4·co channels and concatenated with itself to 8·co, as the reference does.
        Tensor first = Shuffle(backend, Frames(backend, h, 0, 1), 1, 2, 2);
        Tensor firstDup = Concat(backend, 1, first, first);
        if (t == 1) return firstDup;
        Tensor rest = Shuffle(backend, Frames(backend, h, 1, t - 1), 2, 2, 2);
        return Concat(backend, 2, firstDup, rest);
    }

    /// <summary>Downsample shortcut: the input packed the same way, then each group of <c>C/cout</c> packed channels averaged.</summary>
    public static Tensor DownsampleShortcut(IBackend backend, Tensor x, int cout, bool temporal)
    {
        (int cin, int t, int hh, int ww) = Dims(x);
        RequireEvenSpatial(hh, ww, "downsample shortcut");
        if (!temporal)
        {
            Quotient(4 * cin, cout, "spatial downsample shortcut group");
            return GroupMean(Shuffle(backend, x, 1, 2, 2), cout);
        }
        DownFrames(t);
        // The first frame averages half a group, as the reference does: 4·cin packed channels into cout.
        Quotient(4 * cin, cout, "temporal downsample first-frame group");
        Quotient(8 * cin, cout, "temporal downsample rest group");
        Tensor first = GroupMean(Shuffle(backend, Frames(backend, x, 0, 1), 1, 2, 2), cout);
        if (t == 1) return first;
        Tensor rest = GroupMean(Shuffle(backend, Frames(backend, x, 1, t - 1), 2, 2, 2), cout);
        return Concat(backend, 2, first, rest);
    }

    /// <summary>Upsample main path. <paramref name="h"/> has <c>factor·cout</c> channels; the result has <c>cout</c> channels and twice the spatial
    /// size, and with <paramref name="temporal"/> the first frame alone then frame pairs unpacked.</summary>
    public static Tensor UnpackUpsample(IBackend backend, Tensor h, int cout, bool temporal)
    {
        (int ch, int t, int hh, int ww) = Dims(h);
        if (!temporal)
        {
            if (ch != 4 * cout) throw new ArgumentException($"spatial upsample: channels {ch} must be 4·{cout}.");
            return Unshuffle(backend, h, cout, 1, 2, 2);
        }
        if (ch != 8 * cout) throw new ArgumentException($"temporal upsample: channels {ch} must be 8·{cout}.");
        // h_first is packed (r2 r3 c) with 2·cout channels; the first cout are kept.
        Tensor first = Channels(backend, Unshuffle(backend, Frames(backend, h, 0, 1), 2 * cout, 1, 2, 2), cout);
        if (t == 1) return first;
        Tensor rest = Unshuffle(backend, Frames(backend, h, 1, t - 1), cout, 2, 2, 2);
        return Concat(backend, 2, first, rest);
    }

    /// <summary>Upsample shortcut: <paramref name="x"/> unpacked the way <see cref="UnpackUpsample"/> unpacks the main path, then each channel
    /// repeated to <paramref name="cout"/> channels.</summary>
    public static Tensor UpsampleShortcut(IBackend backend, Tensor x, int cout, bool temporal)
    {
        (int cin, int t, int hh, int ww) = Dims(x);
        if (!temporal)
        {
            int rep = Quotient(4 * cout, cin, "spatial upsample shortcut repeat");
            return Unshuffle(backend, RepeatChannels(backend, x, rep), cout, 1, 2, 2);
        }
        int rep1 = Quotient(4 * cout, cin, "temporal upsample first-frame repeat");   // repeats // 2
        int rep2 = Quotient(8 * cout, cin, "temporal upsample rest repeat");
        int cFirst = Quotient(cin, 4, "temporal upsample first-frame pack");
        int cRest = Quotient(cin, 8, "temporal upsample rest pack");
        Tensor first = RepeatChannels(backend, Unshuffle(backend, Frames(backend, x, 0, 1), cFirst, 1, 2, 2), rep1);
        if (t == 1) return first;
        Tensor rest = RepeatChannels(backend, Unshuffle(backend, Frames(backend, x, 1, t - 1), cRest, 2, 2, 2), rep2);
        return Concat(backend, 2, first, rest);
    }

    /// <summary>Encoder end shortcut: consecutive groups of <c>C / outC</c> channels averaged into <paramref name="outC"/> channels. Host arithmetic.</summary>
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

    /// <summary>Decoder start shortcut: the latent with each channel repeated <c>outC / C</c> times, on the backend.</summary>
    public static Tensor RepeatToChannels(IBackend backend, Tensor z, int outC)
        => RepeatChannels(backend, z, Quotient(outC, (int)z.Shape[1], "decoder start shortcut repeat"));

    /// <summary>Single-head attention over all <c>T·H·W</c> tokens with MovieRoll's frame-causal mask: a token attends to every token of its own
    /// frame and of earlier frames. Scale is <c>1/√C</c>. Host arithmetic, quadratic in tokens: tiny-config only.</summary>
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
        Tensor main = HunyuanVideo15VaeShuffle.PackDownsample(backend, h, _cout, _temporal);
        h.Dispose();
        Tensor shortcut = HunyuanVideo15VaeShuffle.DownsampleShortcut(backend, x, _cout, _temporal);
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
        Tensor main = HunyuanVideo15VaeShuffle.UnpackUpsample(backend, h, _cout, _temporal);
        h.Dispose();
        Tensor shortcut = HunyuanVideo15VaeShuffle.UpsampleShortcut(backend, x, _cout, _temporal);
        Tensor output = new(main.Shape, DType.F32);
        backend.Add(output, main, shortcut);
        main.Dispose(); shortcut.Dispose();
        return output;
    }
}
