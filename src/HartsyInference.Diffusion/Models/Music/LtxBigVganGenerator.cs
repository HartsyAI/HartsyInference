using System.Collections.Generic;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>LTX-2 audio vocoder generator — a BigVGAN-v2 head used twice by <c>LtxAudioVocoder</c> (the main
/// stage-1 generator <c>vocoder.vocoder.*</c> and the bandwidth-extension generator <c>vocoder.bwe_generator.*</c>).
/// Structure mirrors <see cref="AdaMosHiFiGanV1"/>'s HiFi-GAN head but every activation is an anti-aliased
/// SnakeBeta (BigVGAN-v2 "AMP"): <c>conv_in</c> (k7) → N × [<c>ConvTranspose1d</c> upsample (channels halve each
/// stage) → mean of <c>resnets_per_upsample</c> multi-receptive-field <see cref="ResBlock"/>s] →
/// anti-aliased <c>act_out</c> → <c>conv_out</c> (k7) → optional tanh.</summary>
///
/// <remarks>The anti-aliasing (<see cref="AntiAlias"/>) upsamples ×2 with a Kaiser-windowed sinc lowpass, applies
/// SnakeBeta, then downsamples ×2 — killing the harmonics SnakeBeta injects above Nyquist. The lowpass filters are
/// deterministic (a function of ratio/kernel only), so they are <i>computed</i> here and shared rather than loaded
/// from the (identical) checkpoint buffers. SnakeBeta alpha/beta are stored log-scaled (<c>logscale=True</c>) and
/// exponentiated at load so the backend <see cref="IBackend.Snake"/> kernel (which expects linear alpha/beta) is
/// faithful. All tensors are batch-1 <c>[1, C, T]</c>; stereo is carried inside the flattened channel axis.</remarks>
public sealed unsafe class LtxBigVganGenerator : IDisposable
{
    private readonly int _inChannels;
    private readonly int _hiddenChannels;
    private readonly int _outChannels;
    private readonly int[] _upsampleFactors;
    private readonly int[] _upsampleKernels;
    private readonly int[] _resnetKernels;
    private readonly int[][] _resnetDilations;
    private readonly bool _applyTanh;
    private readonly AntiAlias _aa;

    private Tensor? _convInW, _convInB;
    private (Tensor W, Tensor? B)[] _ups = [];
    private ResBlock[] _resnets = [];
    private AntiAliasAct _actOut = null!;
    private Tensor? _convOutW, _convOutB;
    private readonly List<IDisposable> _owned = [];

    /// <param name="applyTanh">Apply a final tanh (<c>final_act_fn == "tanh"</c>). The LTX-2 main and BWE
    /// generators both use <c>final_act_fn == None</c>, so pass <c>false</c> for them.</param>
    public LtxBigVganGenerator(
        int inChannels, int hiddenChannels, int outChannels,
        int[] upsampleFactors, int[] upsampleKernels,
        int[] resnetKernels, int[][] resnetDilations,
        bool applyTanh = false, int antialiasRatio = 2, int antialiasKernelSize = 12)
    {
        if (upsampleFactors.Length != upsampleKernels.Length)
            throw new ArgumentException("upsampleFactors and upsampleKernels must be the same length.");
        if (resnetKernels.Length != resnetDilations.Length)
            throw new ArgumentException("resnetKernels and resnetDilations must be the same length.");
        _inChannels = inChannels;
        _hiddenChannels = hiddenChannels;
        _outChannels = outChannels;
        _upsampleFactors = upsampleFactors;
        _upsampleKernels = upsampleKernels;
        _resnetKernels = resnetKernels;
        _resnetDilations = resnetDilations;
        _applyTanh = applyTanh;
        _aa = new AntiAlias(antialiasRatio, antialiasKernelSize);
        _owned.Add(_aa);
    }

    /// <summary>Waveform samples produced per input frame (product of the upsample factors).</summary>
    public int TotalUpsample => _upsampleFactors.Aggregate(1, (a, r) => a * r);

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _convInW = w[$"{prefix}.conv_in.weight"];
        _convInB = Get(w, $"{prefix}.conv_in.bias");

        int numUps = _upsampleFactors.Length;
        int resPerUp = _resnetKernels.Length;
        _ups = new (Tensor, Tensor?)[numUps];
        _resnets = new ResBlock[numUps * resPerUp];
        int channels = _hiddenChannels;
        for (int i = 0; i < numUps; i++)
        {
            _ups[i] = (w[$"{prefix}.ups.{i}.weight"], Get(w, $"{prefix}.ups.{i}.bias"));
            int outC = channels / 2;
            for (int j = 0; j < resPerUp; j++)
            {
                ResBlock rb = new(outC, _resnetKernels[j], _resnetDilations[j], _aa);
                rb.LoadWeights(w, $"{prefix}.resnets.{i * resPerUp + j}", _owned);
                _resnets[i * resPerUp + j] = rb;
            }
            channels = outC;
        }

        _actOut = new AntiAliasAct(_aa);
        _actOut.LoadWeights(w, $"{prefix}.act_out", _owned);
        _convOutW = w[$"{prefix}.conv_out.weight"];
        _convOutB = Get(w, $"{prefix}.conv_out.bias");
    }

    /// <summary>Decode flattened input frames <c>[1, inChannels, T]</c> → waveform <c>[1, outChannels,
    /// T · TotalUpsample]</c>. Caller owns the returned tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        if ((int)x.Shape[1] != _inChannels)
            throw new ArgumentException($"expected {_inChannels} input channels, got {x.Shape[1]}.");

        Tensor cur = Conv(backend, x, _convInW!, _convInB, _hiddenChannels, padL: 3, padR: 3, dilation: 1);
        int numUps = _upsampleFactors.Length;
        int resPerUp = _resnetKernels.Length;
        int channels = _hiddenChannels;
        for (int i = 0; i < numUps; i++)
        {
            // For snakebeta there is no pre-upsample activation (activations live inside the AMP resblocks).
            (Tensor uw, Tensor? ub) = _ups[i];
            int outC = channels / 2;
            int stride = _upsampleFactors[i], kernel = _upsampleKernels[i];
            int pad = (kernel - stride) / 2;
            int tIn = (int)cur.Shape[2];
            int tOut = (tIn - 1) * stride + (kernel - 1) + 1 - 2 * pad;
            Tensor up = new(new TensorShape(1, outC, tOut), DType.F32);
            backend.ConvTranspose1d(up, cur, uw, ub, stride, pad, pad, 1, 1);
            cur.Dispose();

            // Multi-receptive-field fusion: mean of the per-kernel resblocks.
            Tensor sum = new(up.Shape, DType.F32);
            new Span<float>((float*)sum.DataPointer, (int)sum.Shape.ElementCount).Clear();
            for (int j = 0; j < resPerUp; j++)
            {
                Tensor r = _resnets[i * resPerUp + j].Forward(backend, up);
                AddInPlace(sum, r);
                r.Dispose();
            }
            up.Dispose();
            backend.Scale(sum, sum, 1f / resPerUp);
            cur = sum;
            channels = outC;
        }

        Tensor act = _actOut.Forward(backend, cur);
        cur.Dispose();
        Tensor wav = Conv(backend, act, _convOutW!, _convOutB, _outChannels, padL: 3, padR: 3, dilation: 1);
        act.Dispose();
        if (_applyTanh) backend.Tanh(wav, wav);
        return wav;
    }

    /// <summary>Stride-1 conv with explicit symmetric padding and dilation; allocates the (length-preserving for
    /// matched pad) output.</summary>
    private static Tensor Conv(IBackend backend, Tensor x, Tensor w, Tensor? bias, int outC, int padL, int padR, int dilation)
    {
        int tIn = (int)x.Shape[2];
        int k = (int)w.Shape[2];
        int tOut = tIn + padL + padR - dilation * (k - 1);
        Tensor o = new(new TensorShape(1, outC, tOut), DType.F32);
        backend.Conv1d(o, x, w, bias, 1, padL, padR, dilation, 1);
        return o;
    }

    private static void AddInPlace(Tensor acc, Tensor add)
    {
        float* ap = (float*)acc.DataPointer;
        float* dp = (float*)add.DataPointer;
        long n = acc.Shape.ElementCount;
        for (long e = 0; e < n; e++) ap[e] += dp[e];
    }

    private static Tensor? Get(IReadOnlyDictionary<string, Tensor> w, string k) =>
        w.TryGetValue(k, out Tensor? t) ? t : null;

    public void Dispose()
    {
        foreach (IDisposable d in _owned) d.Dispose();
        _owned.Clear();
    }

    /// <summary>BigVGAN-v2 anti-aliased multi-receptive-field ResBlock: <c>resnets_per_upsample</c> wide, here one
    /// instance per (kernel, dilations) triple. Forward = <c>Σ_j [act1→conv1(dilated)→act2→conv2] + residual</c>.</summary>
    private sealed class ResBlock
    {
        private readonly int _channels;
        private readonly int _kernel;
        private readonly int[] _dilations;
        private readonly AntiAlias _aa;
        private (Tensor W, Tensor? B)[] _convs1 = [];
        private (Tensor W, Tensor? B)[] _convs2 = [];
        private AntiAliasAct[] _acts1 = [];
        private AntiAliasAct[] _acts2 = [];

        public ResBlock(int channels, int kernel, int[] dilations, AntiAlias aa)
        {
            _channels = channels;
            _kernel = kernel;
            _dilations = dilations;
            _aa = aa;
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p, List<IDisposable> owned)
        {
            int n = _dilations.Length;
            _convs1 = new (Tensor, Tensor?)[n];
            _convs2 = new (Tensor, Tensor?)[n];
            _acts1 = new AntiAliasAct[n];
            _acts2 = new AntiAliasAct[n];
            for (int j = 0; j < n; j++)
            {
                _convs1[j] = (w[$"{p}.convs1.{j}.weight"], Get(w, $"{p}.convs1.{j}.bias"));
                _convs2[j] = (w[$"{p}.convs2.{j}.weight"], Get(w, $"{p}.convs2.{j}.bias"));
                _acts1[j] = new AntiAliasAct(_aa);
                _acts1[j].LoadWeights(w, $"{p}.acts1.{j}", owned);
                _acts2[j] = new AntiAliasAct(_aa);
                _acts2[j].LoadWeights(w, $"{p}.acts2.{j}", owned);
            }
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            int t = (int)x.Shape[2];
            Tensor cur = Clone(x);
            for (int j = 0; j < _dilations.Length; j++)
            {
                int d = _dilations[j];
                Tensor a1 = _acts1[j].Forward(backend, cur);
                Tensor h1 = Conv(backend, a1, _convs1[j].W, _convs1[j].B, _channels, d * (_kernel - 1) / 2, d * (_kernel - 1) / 2, d);
                a1.Dispose();
                Tensor a2 = _acts2[j].Forward(backend, h1);
                h1.Dispose();
                Tensor h2 = Conv(backend, a2, _convs2[j].W, _convs2[j].B, _channels, (_kernel - 1) / 2, (_kernel - 1) / 2, 1);
                a2.Dispose();
                AddInPlace(h2, cur);
                cur.Dispose();
                cur = h2;
            }
            return cur;
        }

        private static Tensor Clone(Tensor x)
        {
            Tensor o = new(x.Shape, DType.F32);
            long n = x.Shape.ElementCount;
            Buffer.MemoryCopy((float*)x.DataPointer, (float*)o.DataPointer, n * 4, n * 4);
            return o;
        }
    }

    /// <summary>Anti-aliased activation (BigVGAN-v2 <c>Activation1d</c>): upsample ×ratio with a Kaiser sinc lowpass,
    /// apply SnakeBeta, downsample ×ratio. Net length-preserving. SnakeBeta alpha/beta are loaded exponentiated.</summary>
    private sealed class AntiAliasAct
    {
        private readonly AntiAlias _aa;
        private Tensor _alphaExp = null!, _betaExp = null!;

        public AntiAliasAct(AntiAlias aa) => _aa = aa;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p, List<IDisposable> owned)
        {
            _alphaExp = ExpLoad(w[$"{p}.act.alpha"]);
            _betaExp = ExpLoad(w[$"{p}.act.beta"]);
            owned.Add(_alphaExp);
            owned.Add(_betaExp);
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            Tensor up = _aa.Up(backend, x);
            backend.Snake(up, up, _alphaExp, _betaExp);  // x + sin(αx)² / (β + eps); α,β already exp'd
            Tensor down = _aa.Down(backend, up);
            up.Dispose();
            return down;
        }

        /// <summary>Loads a log-scaled SnakeBeta parameter and exponentiates it (logscale=True) into a rank-1
        /// <c>[C]</c> tensor the <see cref="IBackend.Snake"/> kernel can consume directly.</summary>
        private static Tensor ExpLoad(Tensor src)
        {
            Tensor f = src.DType == DType.F32 ? src : src.CastTo(DType.F32);
            int n = (int)f.ElementCount;
            Tensor o = new(new TensorShape(n), DType.F32);
            float* sp = (float*)f.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int i = 0; i < n; i++) op[i] = MathF.Exp(sp[i]);
            return o;
        }
    }

    /// <summary>Shared resampling kernels + geometry for one (ratio, kernel) anti-alias pair. The Kaiser sinc
    /// lowpass is identical for the up and down passes (same cutoff/half-width/kernel), so a single filter is
    /// computed once and broadcast into per-channel depthwise weights on demand.</summary>
    private sealed class AntiAlias : IDisposable
    {
        private readonly int _ratio;
        private readonly int _kernel;
        private readonly int _upPad, _upCropL, _upCropR;
        private readonly int _downPadL, _downPadR;
        private readonly float[] _filter;
        private readonly Dictionary<int, Tensor> _depthwise = new();

        public AntiAlias(int ratio, int kernel)
        {
            _ratio = ratio;
            _kernel = kernel;
            // UpSample1d (kaiser branch).
            _upPad = kernel / ratio - 1;
            _upCropL = _upPad * ratio + (kernel - ratio) / 2;
            _upCropR = _upPad * ratio + (kernel - ratio + 1) / 2;
            // DownSample1d.
            _downPadL = kernel / 2 + (kernel % 2) - 1;
            _downPadR = kernel / 2;
            _filter = LtxVocoderDsp.KaiserSincFilter(0.5 / ratio, 0.6 / ratio, kernel);
        }

        /// <summary>Per-channel depthwise weight <c>[C, 1, K]</c> (every channel shares the one lowpass kernel).</summary>
        private Tensor Depthwise(int channels)
        {
            if (_depthwise.TryGetValue(channels, out Tensor? cached)) return cached;
            Tensor t = new(new TensorShape(channels, 1, _kernel), DType.F32);
            float* tp = (float*)t.DataPointer;
            for (int c = 0; c < channels; c++)
                for (int k = 0; k < _kernel; k++)
                    tp[(long)c * _kernel + k] = _filter[k];
            _depthwise[channels] = t;
            return t;
        }

        public Tensor Up(IBackend backend, Tensor x)
        {
            int c = (int)x.Shape[1];
            Tensor padded = ReplicatePad(x, _upPad, _upPad);
            int tp = (int)padded.Shape[2];
            int tOut = (tp - 1) * _ratio + (_kernel - 1) + 1;
            Tensor convt = new(new TensorShape(1, c, tOut), DType.F32);
            backend.ConvTranspose1d(convt, padded, Depthwise(c), null, _ratio, 0, 0, 1, c);
            padded.Dispose();
            backend.Scale(convt, convt, _ratio);
            Tensor cropped = Crop(convt, _upCropL, _upCropR);
            convt.Dispose();
            return cropped;
        }

        public Tensor Down(IBackend backend, Tensor x)
        {
            int c = (int)x.Shape[1];
            Tensor padded = ReplicatePad(x, _downPadL, _downPadR);
            int tp = (int)padded.Shape[2];
            int tOut = (tp - _kernel) / _ratio + 1;
            Tensor o = new(new TensorShape(1, c, tOut), DType.F32);
            backend.Conv1d(o, padded, Depthwise(c), null, _ratio, 0, 0, 1, c);
            padded.Dispose();
            return o;
        }

        /// <summary>Edge-replicate ("replicate") padding of <c>[1, C, T]</c> along time.</summary>
        private static Tensor ReplicatePad(Tensor x, int padL, int padR)
        {
            int c = (int)x.Shape[1], t = (int)x.Shape[2];
            int t2 = t + padL + padR;
            Tensor o = new(new TensorShape(1, c, t2), DType.F32);
            float* xp = (float*)x.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int ci = 0; ci < c; ci++)
            {
                float* src = xp + (long)ci * t;
                float* dst = op + (long)ci * t2;
                float left = src[0], right = src[t - 1];
                for (int i = 0; i < padL; i++) dst[i] = left;
                Buffer.MemoryCopy(src, dst + padL, (long)t * 4, (long)t * 4);
                for (int i = 0; i < padR; i++) dst[padL + t + i] = right;
            }
            return o;
        }

        /// <summary>Crops <c>left</c>/<c>right</c> samples off the time axis of <c>[1, C, T]</c>.</summary>
        private static Tensor Crop(Tensor x, int left, int right)
        {
            int c = (int)x.Shape[1], t = (int)x.Shape[2];
            int t2 = t - left - right;
            Tensor o = new(new TensorShape(1, c, t2), DType.F32);
            float* xp = (float*)x.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int ci = 0; ci < c; ci++)
                Buffer.MemoryCopy(xp + (long)ci * t + left, op + (long)ci * t2, (long)t2 * 4, (long)t2 * 4);
            return o;
        }

        public void Dispose()
        {
            foreach (Tensor t in _depthwise.Values) t.Dispose();
            _depthwise.Clear();
        }
    }
}
