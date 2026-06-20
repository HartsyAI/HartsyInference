using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Vits;

/// <summary>VITS HiFi-GAN decoder (<c>Generator</c>): <c>conv_pre</c> → N upsample stages (ConvTranspose1d +
/// multi-receptive-field ResBlock fusion) → <c>conv_post</c> → tanh. Plain HiFi-GAN (no NSF/F0 path — VITS
/// gets pitch from the flow latent). Speaker conditioning deferred (single-speaker). Input is the flow latent
/// <c>[1, inter, T]</c>; output is mono PCM <c>float[]</c>.</summary>
public sealed unsafe class VitsHiFiGan
{
    private const float LeakySlope = 0.1f;
    private readonly VitsConfig _cfg;
    private readonly ResBlock[] _resblocks;
    private readonly int _numKernels;
    private Tensor? _convPreW, _convPreB, _convPostW, _convPostB, _condW, _condB;
    private readonly Tensor?[] _upW, _upB;
    private readonly Tensor?[] _noiseConvW, _noiseConvB;     // NSF source-injection convs (RVC GeneratorNSF), optional

    public VitsHiFiGan(VitsConfig cfg)
    {
        _cfg = cfg;
        _numKernels = cfg.ResBlockKernelSizes.Count;
        int stages = cfg.UpsampleRates.Count;
        _upW = new Tensor?[stages]; _upB = new Tensor?[stages];
        _noiseConvW = new Tensor?[stages]; _noiseConvB = new Tensor?[stages];
        _resblocks = new ResBlock[stages * _numKernels];
        for (int i = 0; i < stages; i++)
        {
            int ch = cfg.UpsampleInitialChannel >> (i + 1);
            for (int k = 0; k < _numKernels; k++)
                _resblocks[i * _numKernels + k] = new ResBlock(ch, cfg.ResBlockKernelSizes[k],
                    cfg.ResBlockDilations[k], cfg.ResBlock == "1");
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "dec")
    {
        _convPreW = VitsWeights.Conv(w, $"{prefix}.conv_pre"); _convPreB = VitsWeights.Bias(w, $"{prefix}.conv_pre");
        _convPostW = VitsWeights.Conv(w, $"{prefix}.conv_post"); _convPostB = VitsWeights.Bias(w, $"{prefix}.conv_post");
        if (w.ContainsKey($"{prefix}.cond.weight") || w.ContainsKey($"{prefix}.cond.weight_g"))
        {
            _condW = VitsWeights.Conv(w, $"{prefix}.cond"); _condB = VitsWeights.Bias(w, $"{prefix}.cond");
        }
        for (int i = 0; i < _upW.Length; i++)
        {
            _upW[i] = VitsWeights.Conv(w, $"{prefix}.ups.{i}"); _upB[i] = VitsWeights.Bias(w, $"{prefix}.ups.{i}");
            if (w.ContainsKey($"{prefix}.noise_convs.{i}.weight") || w.ContainsKey($"{prefix}.noise_convs.{i}.weight_g"))
            {
                _noiseConvW[i] = VitsWeights.Conv(w, $"{prefix}.noise_convs.{i}"); _noiseConvB[i] = VitsWeights.Bias(w, $"{prefix}.noise_convs.{i}");
            }
        }
        for (int i = 0; i < _resblocks.Length; i++) _resblocks[i].LoadWeights(w, $"{prefix}.resblocks.{i}");
    }

    public float[] Forward(IBackend backend, Tensor z, int t, Tensor? g = null, Tensor? harSource = null)
    {
        int ch = _cfg.UpsampleInitialChannel;
        Tensor x = new(new TensorShape(1, ch, t), DType.F32);
        backend.Conv1d(x, z, _convPreW!, _convPreB, 1, 3, 3, 1, 1);     // conv_pre k7

        // Speaker conditioning: x += cond(g), broadcast over time.
        if (g is not null && _condW is not null)
        {
            Tensor gc = new(new TensorShape(1, ch, 1), DType.F32);
            backend.Conv1d(gc, g, _condW, _condB, 1, 0, 0, 1, 1);
            float* xp2 = (float*)x.DataPointer; float* gp = (float*)gc.DataPointer;
            for (int c = 0; c < ch; c++) for (int j = 0; j < t; j++) xp2[(long)c * t + j] += gp[c];
            gc.Dispose();
        }

        int curT = t;
        for (int i = 0; i < _cfg.UpsampleRates.Count; i++)
        {
            LeakyInPlace(x);
            int stride = _cfg.UpsampleRates[i], kernel = _cfg.UpsampleKernelSizes[i];
            int pad = (kernel - stride) / 2;
            int outCh = ch >> 1, outT = (curT - 1) * stride + (kernel - 1) + 1 - 2 * pad;
            Tensor up = new(new TensorShape(1, outCh, outT), DType.F32);
            backend.ConvTranspose1d(up, x, _upW[i]!, _upB[i], stride, pad, pad, 1, 1);
            x.Dispose();

            // MRF: average the num_kernels resblocks.
            Tensor acc = new(new TensorShape(1, outCh, outT), DType.F32);
            float* accP = (float*)acc.DataPointer;
            for (int k = 0; k < _numKernels; k++)
            {
                Tensor rb = _resblocks[i * _numKernels + k].Forward(backend, up, outCh, outT);
                float* rp = (float*)rb.DataPointer;
                for (long n = 0; n < (long)outCh * outT; n++) accP[n] += rp[n];
                rb.Dispose();
            }
            float inv = 1f / _numKernels;
            for (long n = 0; n < (long)outCh * outT; n++) accP[n] *= inv;
            up.Dispose();
            x = acc; ch = outCh; curT = outT;
        }

        LeakyInPlace(x);
        Tensor post = new(new TensorShape(1, 1, curT), DType.F32);
        backend.Conv1d(post, x, _convPostW!, _convPostB, 1, 3, 3, 1, 1);     // conv_post k7 → 1 ch
        x.Dispose();
        float* pp = (float*)post.DataPointer;
        float[] audio = new float[curT];
        for (int j = 0; j < curT; j++) audio[j] = MathF.Tanh(pp[j]);
        post.Dispose();
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_convPreW, _convPreB, _convPostW, _convPostB, _condW, _condB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (Tensor? t in _upW) if (t is not null) yield return t;
        foreach (Tensor? t in _upB) if (t is not null) yield return t;
        foreach (ResBlock r in _resblocks) foreach (Tensor t in r.EnumerateWeights()) yield return t;
    }

    private static void LeakyInPlace(Tensor x)
    {
        float* p = (float*)x.DataPointer;
        for (long i = 0; i < x.ElementCount; i++) if (p[i] < 0) p[i] *= LeakySlope;
    }

    /// <summary>MRF ResBlock — type 1 has two convs (dilated + dilation-1) per residual; type 2 has one.</summary>
    private sealed class ResBlock
    {
        private readonly int _ch, _kernel;
        private readonly int[] _dilations;
        private readonly bool _type1;
        private readonly Tensor?[] _c1W, _c1B, _c2W, _c2B;

        public ResBlock(int ch, int kernel, IReadOnlyList<int> dilations, bool type1)
        {
            _ch = ch; _kernel = kernel; _type1 = type1; _dilations = [.. dilations];
            _c1W = new Tensor?[_dilations.Length]; _c1B = new Tensor?[_dilations.Length];
            _c2W = new Tensor?[_dilations.Length]; _c2B = new Tensor?[_dilations.Length];
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            for (int j = 0; j < _dilations.Length; j++)
            {
                _c1W[j] = VitsWeights.Conv(w, $"{prefix}.convs1.{j}"); _c1B[j] = VitsWeights.Bias(w, $"{prefix}.convs1.{j}");
                if (_type1)
                {
                    _c2W[j] = VitsWeights.Conv(w, $"{prefix}.convs2.{j}"); _c2B[j] = VitsWeights.Bias(w, $"{prefix}.convs2.{j}");
                }
            }
        }

        public Tensor Forward(IBackend backend, Tensor input, int ch, int t)
        {
            Tensor x = Clone(input);
            for (int j = 0; j < _dilations.Length; j++)
            {
                int d = _dilations[j], pad = d * (_kernel - 1) / 2;
                Tensor xt = LeakyConv(backend, x, _c1W[j]!, _c1B[j], ch, t, pad, d);
                if (_type1)
                {
                    int pad2 = (_kernel - 1) / 2;
                    Tensor xt2 = LeakyConv(backend, xt, _c2W[j]!, _c2B[j], ch, t, pad2, 1);
                    xt.Dispose(); xt = xt2;
                }
                float* xp = (float*)x.DataPointer;
                float* tp = (float*)xt.DataPointer;
                for (long n = 0; n < (long)ch * t; n++) xp[n] += tp[n];
                xt.Dispose();
            }
            return x;
        }

        private Tensor LeakyConv(IBackend backend, Tensor x, Tensor w, Tensor? b, int ch, int t, int pad, int dilation)
        {
            Tensor act = new(new TensorShape(1, ch, t), DType.F32);
            float* xp = (float*)x.DataPointer;
            float* ap = (float*)act.DataPointer;
            for (long n = 0; n < (long)ch * t; n++) { float v = xp[n]; ap[n] = v < 0 ? v * LeakySlope : v; }
            Tensor outT = new(new TensorShape(1, ch, t), DType.F32);
            backend.Conv1d(outT, act, w, b, 1, pad, pad, dilation, 1);
            act.Dispose();
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[][] groups = [_c1W, _c1B, _c2W, _c2B];
            foreach (Tensor?[] g in groups) foreach (Tensor? t in g) if (t is not null) yield return t;
        }

        private static Tensor Clone(Tensor t)
        {
            Tensor c = new(t.Shape, DType.F32);
            Buffer.MemoryCopy((void*)t.DataPointer, (void*)c.DataPointer, t.ElementCount * 4, t.ElementCount * 4);
            return c;
        }
    }
}
