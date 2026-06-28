using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>ACE-Step's music vocoder (<c>ADaMoSHiFiGANV1</c>, fish-speech lineage): a ConvNeXt-style 1-D backbone
/// (stem k7 + 4 stages of depths [3,3,9,3] / dims [128,256,384,512] with kernel-1 channel transitions, no temporal
/// downsampling) feeding a HiFi-GAN generator head (conv_pre 512→1024 k13 → 7 × [SiLU + ConvTranspose1d] with 4
/// multi-receptive-field ResBlocks each (kernels [3,7,11,13], dilations 1/3/5, outputs averaged) → conv_post k13 →
/// tanh). Total upsample 4·4·2·2·2·2·2 = 512 = the mel hop, so mel <c>[1, 128, T]</c> → mono waveform
/// <c>[1, 1, T·512]</c> @ 44.1 kHz; stereo runs the vocoder once per channel. Weight-norm (<c>weight_g/weight_v</c>)
/// is fused at conversion time — this class consumes plain <c>weight</c> tensors. Numerics validation-pending.</summary>
public sealed unsafe class AdaMosHiFiGanV1
{
    private readonly int[] _depths;
    private readonly int[] _dims;
    private readonly int[] _upsampleRates;
    private readonly int[] _upsampleKernels;
    private readonly int[] _resblockKernels;

    private ChannelLayer[] _channelLayers = [];       // stem + 3 transitions (LN + conv k1)
    private ConvNeXtBlock[][] _stages = [];
    private Tensor? _backboneNormW, _backboneNormB;
    private Tensor? _convPreW, _convPreB;
    private (Tensor W, Tensor? B)[] _ups = [];
    private HifiResBlock[] _resblocks = [];
    private Tensor? _convPostW, _convPostB;

    private sealed class ChannelLayer
    {
        public Tensor? NormW, NormB;       // null for the stem (conv-first then LN)
        public required Tensor ConvW;
        public Tensor? ConvB;
        public Tensor? StemNormW, StemNormB;
    }

    public AdaMosHiFiGanV1(int[]? depths = null, int[]? dims = null,
        int[]? upsampleRates = null, int[]? upsampleKernels = null, int[]? resblockKernels = null)
    {
        _depths = depths ?? [3, 3, 9, 3];
        _dims = dims ?? [128, 256, 384, 512];
        _upsampleRates = upsampleRates ?? [4, 4, 2, 2, 2, 2, 2];
        _upsampleKernels = upsampleKernels ?? [8, 8, 4, 4, 4, 4, 4];
        _resblockKernels = resblockKernels ?? [3, 7, 11, 13];
    }

    /// <summary>Waveform samples per mel frame (product of the upsample rates).</summary>
    public int TotalUpsample => _upsampleRates.Aggregate(1, (a, r) => a * r);

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _channelLayers = new ChannelLayer[_dims.Length];
        for (int i = 0; i < _dims.Length; i++)
        {
            string p = $"backbone.channel_layers.{i}";
            if (i == 0)
            {
                _channelLayers[i] = new ChannelLayer
                {
                    ConvW = w[$"{p}.0.weight"],
                    ConvB = Get(w, $"{p}.0.bias"),
                    StemNormW = GetF32(w, $"{p}.1.weight"),
                    StemNormB = Get(w, $"{p}.1.bias"),
                };
            }
            else
            {
                _channelLayers[i] = new ChannelLayer
                {
                    NormW = GetF32(w, $"{p}.0.weight"),
                    NormB = Get(w, $"{p}.0.bias"),
                    ConvW = w[$"{p}.1.weight"],
                    ConvB = Get(w, $"{p}.1.bias"),
                };
            }
        }
        _stages = new ConvNeXtBlock[_dims.Length][];
        for (int i = 0; i < _dims.Length; i++)
        {
            _stages[i] = new ConvNeXtBlock[_depths[i]];
            for (int j = 0; j < _depths[i]; j++)
            {
                _stages[i][j] = new ConvNeXtBlock();
                _stages[i][j].LoadWeights(w, $"backbone.stages.{i}.{j}");
            }
        }
        _backboneNormW = GetF32(w, "backbone.norm.weight");
        _backboneNormB = Get(w, "backbone.norm.bias");

        _convPreW = w["head.conv_pre.weight"]; _convPreB = Get(w, "head.conv_pre.bias");
        _ups = new (Tensor, Tensor?)[_upsampleRates.Length];
        for (int i = 0; i < _upsampleRates.Length; i++)
            _ups[i] = (w[$"head.ups.{i}.weight"], Get(w, $"head.ups.{i}.bias"));
        _resblocks = new HifiResBlock[_upsampleRates.Length * _resblockKernels.Length];
        for (int i = 0; i < _resblocks.Length; i++)
        {
            _resblocks[i] = new HifiResBlock();
            _resblocks[i].LoadWeights(w, $"head.resblocks.{i}");
        }
        _convPostW = w["head.conv_post.weight"]; _convPostB = Get(w, "head.conv_post.bias");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (ChannelLayer cl in _channelLayers)
            foreach (Tensor? t in new[] { cl.NormW, cl.NormB, cl.ConvW, cl.ConvB, cl.StemNormW, cl.StemNormB })
                if (t is not null) yield return t;
        foreach (ConvNeXtBlock[] stage in _stages)
            foreach (ConvNeXtBlock b in stage)
                foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (Tensor? t in new[] { _backboneNormW, _backboneNormB, _convPreW, _convPreB, _convPostW, _convPostB })
            if (t is not null) yield return t;
        foreach ((Tensor uw, Tensor? ub) in _ups) { yield return uw; if (ub is not null) yield return ub; }
        foreach (HifiResBlock r in _resblocks)
            foreach (Tensor t in r.EnumerateWeights()) yield return t;
    }

    /// <summary>Mono decode: mel <c>[1, 128, T]</c> → waveform <c>[1, 1, T · 512]</c> in [-1, 1].</summary>
    public Tensor Decode(IBackend backend, Tensor mel)
    {
        int t = (int)mel.Shape[2];
        Tensor h = Clone(mel);
        for (int i = 0; i < _dims.Length; i++)
        {
            ChannelLayer cl = _channelLayers[i];
            if (i == 0)
            {
                int k = (int)cl.ConvW.Shape[2];
                Tensor stem = Conv(backend, h, cl.ConvW, cl.ConvB, _dims[0], t, pad: k / 2);
                h.Dispose();
                h = stem;
                LayerNormChannels(h, cl.StemNormW!, cl.StemNormB);
            }
            else
            {
                LayerNormChannels(h, cl.NormW!, cl.NormB);
                Tensor tr = Conv(backend, h, cl.ConvW, cl.ConvB, _dims[i], t, pad: 0);
                h.Dispose();
                h = tr;
            }
            foreach (ConvNeXtBlock b in _stages[i])
            {
                Tensor next = b.Forward(backend, h);
                h.Dispose();
                h = next;
            }
            AceStepDebugDump.Dump($"voc.stage.{i}", h);
        }
        LayerNormChannels(h, _backboneNormW!, _backboneNormB);
        AceStepDebugDump.Dump("voc.backbone_norm", h);

        // HiFi-GAN head.
        int preK = (int)_convPreW!.Shape[2];
        Tensor cur = Conv(backend, h, _convPreW, _convPreB, (int)_convPreW.Shape[0], t, pad: preK / 2);
        h.Dispose();
        AceStepDebugDump.Dump("voc.conv_pre", cur);
        int numKernels = _resblockKernels.Length;
        for (int i = 0; i < _upsampleRates.Length; i++)
        {
            backend.Silu(cur, cur);
            (Tensor uw, Tensor? ub) = _ups[i];
            int rate = _upsampleRates[i], kernel = _upsampleKernels[i];
            int outC = (int)uw.Shape[1];
            int tIn = (int)cur.Shape[2];
            int pad = (kernel - rate) / 2;
            int tOut = (tIn - 1) * rate + kernel - 2 * pad;
            Tensor up = new Tensor(new TensorShape(1, outC, tOut), DType.F32);
            backend.ConvTranspose1d(up, cur, uw, ub, rate, pad, pad, 1, 1);
            cur.Dispose();
            AceStepDebugDump.Dump($"voc.upsample.{i}", up);

            // Multi-receptive-field fusion: average of the per-kernel resblocks.
            Tensor sum = new Tensor(up.Shape, DType.F32);
            for (int j = 0; j < numKernels; j++)
            {
                Tensor r = _resblocks[i * numKernels + j].Forward(backend, up);
                float* sp2 = (float*)sum.DataPointer;
                float* rp = (float*)r.DataPointer;
                long n = sum.Shape.ElementCount;
                for (long e = 0; e < n; e++) sp2[e] += rp[e];
                r.Dispose();
            }
            up.Dispose();
            float invK = 1f / numKernels;
            float* smp = (float*)sum.DataPointer;
            for (long e = 0; e < sum.Shape.ElementCount; e++) smp[e] *= invK;
            cur = sum;
            AceStepDebugDump.Dump($"voc.mrf.{i}", cur);
        }
        backend.Silu(cur, cur);
        int postK = (int)_convPostW!.Shape[2];
        Tensor wav = Conv(backend, cur, _convPostW, _convPostB, 1, (int)cur.Shape[2], pad: postK / 2);
        cur.Dispose();
        float* wp = (float*)wav.DataPointer;
        for (long e = 0; e < wav.Shape.ElementCount; e++) wp[e] = MathF.Tanh(wp[e]);
        AceStepDebugDump.Dump("voc.waveform", wav);
        return wav;
    }

    private static Tensor Conv(IBackend backend, Tensor x, Tensor w, Tensor? b, int outC, int tOut, int pad)
    {
        Tensor o = new Tensor(new TensorShape(1, outC, tOut), DType.F32);
        backend.Conv1d(o, x, w, b, 1, pad, pad, 1, 1);
        return o;
    }

    /// <summary>LayerNorm over the channel axis of <c>[1, C, T]</c> (ConvNeXt "channels_first" norm).</summary>
    internal static void LayerNormChannels(Tensor x, Tensor weight, Tensor? bias, float eps = 1e-6f)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2];
        float* xp = (float*)x.DataPointer;
        float* wp = (float*)weight.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int i = 0; i < t; i++)
        {
            double mean = 0;
            for (int ci = 0; ci < c; ci++) mean += xp[(long)ci * t + i];
            mean /= c;
            double var = 0;
            for (int ci = 0; ci < c; ci++) { double d = xp[(long)ci * t + i] - mean; var += d * d; }
            float inv = 1f / MathF.Sqrt((float)(var / c) + eps);
            for (int ci = 0; ci < c; ci++)
            {
                long off = (long)ci * t + i;
                xp[off] = (float)((xp[off] - mean) * inv) * wp[ci] + (bp is null ? 0f : bp[ci]);
            }
        }
    }

    private static Tensor Clone(Tensor x)
    {
        Tensor o = new Tensor(x.Shape, DType.F32);
        long n = x.Shape.ElementCount;
        Buffer.MemoryCopy((float*)x.DataPointer, (float*)o.DataPointer, n * 4, n * 4);
        return o;
    }

    private static Tensor? Get(IReadOnlyDictionary<string, Tensor> w, string k) => w.TryGetValue(k, out Tensor? t) ? t : null;

    private static Tensor? GetF32(IReadOnlyDictionary<string, Tensor> w, string k) =>
        w.TryGetValue(k, out Tensor? t) ? (t.DType == DType.F32 ? t : t.CastTo(DType.F32)) : null;

    /// <summary>ConvNeXt 1-D block: depthwise k7 → channelwise LayerNorm → pointwise expand → GELU → pointwise
    /// project → γ layer-scale → + residual.</summary>
    private sealed class ConvNeXtBlock
    {
        private Tensor? _dwW, _dwB, _normW, _normB, _pw1W, _pw1B, _pw2W, _pw2B, _gamma;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _dwW = w[$"{p}.dwconv.weight"]; _dwB = Get(w, $"{p}.dwconv.bias");
            _normW = GetF32(w, $"{p}.norm.weight"); _normB = Get(w, $"{p}.norm.bias");
            _pw1W = w[$"{p}.pwconv1.weight"]; _pw1B = Get(w, $"{p}.pwconv1.bias");
            _pw2W = w[$"{p}.pwconv2.weight"]; _pw2B = Get(w, $"{p}.pwconv2.bias");
            _gamma = GetF32(w, $"{p}.gamma");
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor? t in new[] { _dwW, _dwB, _normW, _normB, _pw1W, _pw1B, _pw2W, _pw2B, _gamma })
                if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            int c = (int)x.Shape[1], t = (int)x.Shape[2];
            int k = (int)_dwW!.Shape[2];
            Tensor h = new Tensor(x.Shape, DType.F32);
            backend.Conv1d(h, x, _dwW, _dwB, 1, k / 2, k / 2, 1, c);
            LayerNormChannels(h, _normW!, _normB);

            // Pointwise MLP per time step (token rows [T, C]).
            int hidden = (int)_pw1W!.Shape[0];
            Tensor rows = TransposeToRows(h, c, t);
            h.Dispose();
            Tensor mid = new Tensor(new TensorShape(t, hidden), DType.F32);
            backend.Linear(mid, rows, _pw1W, _pw1B);
            rows.Dispose();
            Tensor act = new Tensor(mid.Shape, DType.F32);
            backend.Gelu(act, mid);
            mid.Dispose();
            Tensor outRows = new Tensor(new TensorShape(t, c), DType.F32);
            backend.Linear(outRows, act, _pw2W!, _pw2B);
            act.Dispose();

            Tensor result = TransposeFromRows(outRows, c, t);
            outRows.Dispose();
            float* rp = (float*)result.DataPointer;
            float* xp = (float*)x.DataPointer;
            float* gp = _gamma is null ? null : (float*)_gamma.DataPointer;
            for (int ci = 0; ci < c; ci++)
            {
                float g = gp is null ? 1f : gp[ci];
                for (int i = 0; i < t; i++)
                {
                    long off = (long)ci * t + i;
                    rp[off] = xp[off] + g * rp[off];
                }
            }
            return result;
        }

        private static Tensor TransposeToRows(Tensor x, int c, int t)
        {
            Tensor o = new Tensor(new TensorShape(t, c), DType.F32);
            float* xp = (float*)x.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int ci = 0; ci < c; ci++)
                for (int i = 0; i < t; i++)
                    op[(long)i * c + ci] = xp[(long)ci * t + i];
            return o;
        }

        private static Tensor TransposeFromRows(Tensor rows, int c, int t)
        {
            Tensor o = new Tensor(new TensorShape(1, c, t), DType.F32);
            float* rp = (float*)rows.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int ci = 0; ci < c; ci++)
                for (int i = 0; i < t; i++)
                    op[(long)ci * t + i] = rp[(long)i * c + ci];
            return o;
        }
    }

    /// <summary>HiFi-GAN ResBlock1: 3 × (SiLU → dilated conv → SiLU → dilation-1 conv) with residuals.</summary>
    private sealed class HifiResBlock
    {
        private readonly List<(Tensor W, Tensor? B, int Dilation)> _convs1 = [];
        private readonly List<(Tensor W, Tensor? B)> _convs2 = [];

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            int[] dilations = [1, 3, 5];
            for (int j = 0; j < dilations.Length; j++)
            {
                _convs1.Add((w[$"{p}.convs1.{j}.weight"], Get(w, $"{p}.convs1.{j}.bias"), dilations[j]));
                _convs2.Add((w[$"{p}.convs2.{j}.weight"], Get(w, $"{p}.convs2.{j}.bias")));
            }
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach ((Tensor cw, Tensor? cb, _) in _convs1) { yield return cw; if (cb is not null) yield return cb; }
            foreach ((Tensor cw, Tensor? cb) in _convs2) { yield return cw; if (cb is not null) yield return cb; }
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            int c = (int)x.Shape[1], t = (int)x.Shape[2];
            Tensor cur = CloneLocal(x);
            for (int j = 0; j < _convs1.Count; j++)
            {
                (Tensor w1, Tensor? b1, int d) = _convs1[j];
                int k = (int)w1.Shape[2];
                Tensor a = new Tensor(cur.Shape, DType.F32);
                backend.Silu(a, cur);
                Tensor h1 = new Tensor(new TensorShape(1, c, t), DType.F32);
                backend.Conv1d(h1, a, w1, b1, 1, d * (k - 1) / 2, d * (k - 1) / 2, d, 1);
                a.Dispose();
                backend.Silu(h1, h1);
                (Tensor w2, Tensor? b2) = _convs2[j];
                Tensor h2 = new Tensor(new TensorShape(1, c, t), DType.F32);
                backend.Conv1d(h2, h1, w2, b2, 1, k / 2, k / 2, 1, 1);
                h1.Dispose();
                float* hp = (float*)h2.DataPointer;
                float* cp = (float*)cur.DataPointer;
                for (long e = 0; e < h2.Shape.ElementCount; e++) hp[e] += cp[e];
                cur.Dispose();
                cur = h2;
            }
            return cur;
        }

        private static Tensor CloneLocal(Tensor x)
        {
            Tensor o = new Tensor(x.Shape, DType.F32);
            long n = x.Shape.ElementCount;
            Buffer.MemoryCopy((float*)x.DataPointer, (float*)o.DataPointer, n * 4, n * 4);
            return o;
        }
    }
}
