using System.Collections.Generic;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Music;

/// <summary>The per-stage activation family used by an <see cref="LtxBigVganGenerator"/>. <see cref="SnakeBeta"/> is
/// the anti-aliased BigVGAN-v2 "AMP" path (the LTX-2.X BWE dual-stage vocoder); <see cref="LeakyRelu"/> is the plain
/// HiFi-GAN path with parameter-free leaky-ReLU activations (the LTX-2 single-stage vocoder).</summary>
public enum LtxVocoderActivation { SnakeBeta, LeakyRelu }

/// <summary>LTX-2 audio vocoder generator — a HiFi-GAN/BigVGAN head. Used by <c>LtxAudioVocoder</c> as the BWE
/// dual-stage's main generator (<c>vocoder.vocoder.*</c>) and bandwidth-extension generator
/// (<c>vocoder.bwe_generator.*</c>), or standalone as the single-stage vocoder (<c>vocoder.*</c>). Real-checkpoint
/// naming (the single-file Lightricks weights): <c>conv_pre</c> (k7) → N × [optional pre-upsample leaky-ReLU →
/// <c>ConvTranspose1d</c> upsample (channels halve each stage) → mean of <c>resnets_per_upsample</c>
/// multi-receptive-field <see cref="ResBlock"/>s under <c>resblocks.*</c>] → <c>act_post</c> → <c>conv_post</c>
/// (k7) → optional tanh. With <see cref="LtxVocoderActivation.SnakeBeta"/> every activation is an anti-aliased
/// SnakeBeta and there is no pre-upsample activation; with <see cref="LtxVocoderActivation.LeakyRelu"/> the
/// activations are parameter-free leaky-ReLUs and a pre-upsample leaky-ReLU is applied each stage.</summary>
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
    private readonly LtxVocoderActivation _activation;
    private readonly float _leakySlope;
    private readonly float _actPostSlope;
    private readonly AntiAlias _aa;

    private Tensor? _convInW, _convInB;
    private (Tensor W, Tensor? B)[] _ups = [];
    private ResBlock[] _resnets = [];
    private AntiAliasAct? _actOut;
    private Tensor? _convOutW, _convOutB;
    private readonly List<IDisposable> _owned = [];

    /// <param name="applyTanh">Apply a final tanh (<c>final_act_fn == "tanh"</c>). The BWE main/extension
    /// generators use <c>final_act_fn == None</c> (pass <c>false</c>); the single-stage vocoder uses tanh.</param>
    /// <param name="activation">The per-stage activation family. <see cref="LtxVocoderActivation.SnakeBeta"/> for the
    /// BWE generators, <see cref="LtxVocoderActivation.LeakyRelu"/> for the single-stage vocoder.</param>
    /// <param name="leakySlope">Negative slope of the pre-upsample and resblock leaky-ReLUs (reference 0.1).</param>
    /// <param name="actPostSlope">Negative slope of the final pre-<c>conv_post</c> leaky-ReLU. The reference uses a
    /// bare <c>nn.LeakyReLU()</c> here (default 0.01), NOT the configured negative slope.</param>
    public LtxBigVganGenerator(
        int inChannels, int hiddenChannels, int outChannels,
        int[] upsampleFactors, int[] upsampleKernels,
        int[] resnetKernels, int[][] resnetDilations,
        bool applyTanh = false, int antialiasRatio = 2, int antialiasKernelSize = 12,
        LtxVocoderActivation activation = LtxVocoderActivation.SnakeBeta,
        float leakySlope = 0.1f, float actPostSlope = 0.01f)
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
        _activation = activation;
        _leakySlope = leakySlope;
        _actPostSlope = actPostSlope;
        _aa = new AntiAlias(antialiasRatio, antialiasKernelSize);
        _owned.Add(_aa);
    }

    /// <summary>Waveform samples produced per input frame (product of the upsample factors).</summary>
    public int TotalUpsample => _upsampleFactors.Aggregate(1, (a, r) => a * r);

    /// <summary>Every loaded weight tensor, for <see cref="IBackend.PreloadWeights"/>/<see cref="IBackend.FreeWeights"/>.
    /// Excludes the anti-alias lowpass filters — those are computed constants, not checkpoint buffers, and are only
    /// lazily materialized on first <see cref="Forward"/>, so nothing useful exists to enumerate before then.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _convInW, _convInB, _convOutW, _convOutB })
        {
            if (t is not null) yield return t;
        }
        foreach ((Tensor w, Tensor? b) in _ups)
        {
            yield return w;
            if (b is not null) yield return b;
        }
        foreach (ResBlock rb in _resnets)
        {
            foreach (Tensor t in rb.EnumerateWeights()) yield return t;
        }
        if (_actOut is not null)
        {
            foreach (Tensor t in _actOut.EnumerateWeights()) yield return t;
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _convInW = w[$"{prefix}.conv_pre.weight"];
        _convInB = Get(w, $"{prefix}.conv_pre.bias");

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
                ResBlock rb = new(outC, _resnetKernels[j], _resnetDilations[j], _aa, _activation, _leakySlope);
                rb.LoadWeights(w, $"{prefix}.resblocks.{i * resPerUp + j}", _owned);
                _resnets[i * resPerUp + j] = rb;
            }
            channels = outC;
        }

        // The single-stage (leaky-ReLU) vocoder's act_post is a parameter-free LeakyReLU; only SnakeBeta loads weights.
        if (_activation == LtxVocoderActivation.SnakeBeta)
        {
            _actOut = new AntiAliasAct(_aa);
            _actOut.LoadWeights(w, $"{prefix}.act_post", _owned);
        }
        _convOutW = w[$"{prefix}.conv_post.weight"];
        _convOutB = Get(w, $"{prefix}.conv_post.bias");
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
            // SnakeBeta puts all activations inside the AMP resblocks; leaky-ReLU applies one before each upsample.
            if (_activation == LtxVocoderActivation.LeakyRelu) backend.LeakyRelu(cur, cur, _leakySlope);
            (Tensor uw, Tensor? ub) = _ups[i];
            int outC = channels / 2;
            int stride = _upsampleFactors[i], kernel = _upsampleKernels[i];
            int pad = (kernel - stride) / 2;
            int tIn = (int)cur.Shape[2];
            int tOut = (tIn - 1) * stride + (kernel - 1) + 1 - 2 * pad;
            Tensor up = new(new TensorShape(1, outC, tOut), DType.F32);
            backend.ConvTranspose1d(up, cur, uw, ub, stride, pad, pad, 1, 1);
            cur.Dispose();

            // Multi-receptive-field fusion: mean of the per-kernel resblocks (accumulated on device).
            Tensor sum = _resnets[i * resPerUp].Forward(backend, up);
            for (int j = 1; j < resPerUp; j++)
            {
                Tensor r = _resnets[i * resPerUp + j].Forward(backend, up);
                backend.Add(sum, sum, r);
                r.Dispose();
            }
            up.Dispose();
            backend.Scale(sum, sum, 1f / resPerUp);
            cur = sum;
            channels = outC;
        }

        Tensor act;
        if (_actOut is not null) { act = _actOut.Forward(backend, cur); }
        else { act = new Tensor(cur.Shape, DType.F32); backend.LeakyRelu(act, cur, _actPostSlope); }  // bare LeakyReLU (0.01)
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
        private readonly LtxVocoderActivation _activation;
        private readonly float _leakySlope;
        private (Tensor W, Tensor? B)[] _convs1 = [];
        private (Tensor W, Tensor? B)[] _convs2 = [];
        private AntiAliasAct[] _acts1 = [];
        private AntiAliasAct[] _acts2 = [];

        public ResBlock(int channels, int kernel, int[] dilations, AntiAlias aa, LtxVocoderActivation activation, float leakySlope)
        {
            _channels = channels;
            _kernel = kernel;
            _dilations = dilations;
            _aa = aa;
            _activation = activation;
            _leakySlope = leakySlope;
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p, List<IDisposable> owned)
        {
            int n = _dilations.Length;
            _convs1 = new (Tensor, Tensor?)[n];
            _convs2 = new (Tensor, Tensor?)[n];
            for (int j = 0; j < n; j++)
            {
                _convs1[j] = (w[$"{p}.convs1.{j}.weight"], Get(w, $"{p}.convs1.{j}.bias"));
                _convs2[j] = (w[$"{p}.convs2.{j}.weight"], Get(w, $"{p}.convs2.{j}.bias"));
            }
            // Leaky-ReLU activations are parameter-free; only SnakeBeta has loadable alpha/beta (+ antialias filters).
            if (_activation != LtxVocoderActivation.SnakeBeta) return;
            _acts1 = new AntiAliasAct[n];
            _acts2 = new AntiAliasAct[n];
            for (int j = 0; j < n; j++)
            {
                _acts1[j] = new AntiAliasAct(_aa);
                _acts1[j].LoadWeights(w, $"{p}.acts1.{j}", owned);
                _acts2[j] = new AntiAliasAct(_aa);
                _acts2[j].LoadWeights(w, $"{p}.acts2.{j}", owned);
            }
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach ((Tensor w, Tensor? b) in _convs1)
            {
                yield return w;
                if (b is not null) yield return b;
            }
            foreach ((Tensor w, Tensor? b) in _convs2)
            {
                yield return w;
                if (b is not null) yield return b;
            }
            foreach (AntiAliasAct a in _acts1)
            {
                foreach (Tensor t in a.EnumerateWeights()) yield return t;
            }
            foreach (AntiAliasAct a in _acts2)
            {
                foreach (Tensor t in a.EnumerateWeights()) yield return t;
            }
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            // The residual chain accumulates on device; `x` is only borrowed (dilations.Length ≥ 1, so the
            // returned tensor is always a fresh conv output and never the caller's input).
            Tensor cur = x;
            for (int j = 0; j < _dilations.Length; j++)
            {
                int d = _dilations[j];
                Tensor a1 = Act(backend, _acts1, j, cur);
                Tensor h1 = Conv(backend, a1, _convs1[j].W, _convs1[j].B, _channels, d * (_kernel - 1) / 2, d * (_kernel - 1) / 2, d);
                a1.Dispose();
                Tensor a2 = Act(backend, _acts2, j, h1);
                h1.Dispose();
                Tensor h2 = Conv(backend, a2, _convs2[j].W, _convs2[j].B, _channels, (_kernel - 1) / 2, (_kernel - 1) / 2, 1);
                a2.Dispose();
                backend.Add(h2, h2, cur);
                if (!ReferenceEquals(cur, x)) cur.Dispose();
                cur = h2;
            }
            return cur;
        }

        /// <summary>Applies the j-th activation out-of-place: anti-aliased SnakeBeta, or a parameter-free leaky-ReLU.</summary>
        private Tensor Act(IBackend backend, AntiAliasAct[] acts, int j, Tensor x)
        {
            if (_activation == LtxVocoderActivation.SnakeBeta) return acts[j].Forward(backend, x);
            Tensor o = new(x.Shape, DType.F32);
            backend.LeakyRelu(o, x, _leakySlope);
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

        public IEnumerable<Tensor> EnumerateWeights()
        {
            yield return _alphaExp;
            yield return _betaExp;
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
            Tensor f = AudioWeightNorm.AsF32(src, out bool owned);
            int n = (int)f.ElementCount;
            Tensor o = new(new TensorShape(n), DType.F32);
            float* sp = (float*)f.DataPointer;
            float* op = (float*)o.DataPointer;
            for (int i = 0; i < n; i++) op[i] = MathF.Exp(sp[i]);
            if (owned) f.Dispose();
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
            Tensor padded = LtxAudioDeviceOps.ReplicatePadTime(backend, x, _upPad, _upPad);
            int tp = (int)padded.Shape[2];
            int tOut = (tp - 1) * _ratio + (_kernel - 1) + 1;
            Tensor convt = new(new TensorShape(1, c, tOut), DType.F32);
            backend.ConvTranspose1d(convt, padded, Depthwise(c), null, _ratio, 0, 0, 1, c);
            padded.Dispose();
            backend.Scale(convt, convt, _ratio);
            Tensor cropped = LtxAudioDeviceOps.CropTime(backend, convt, _upCropL, _upCropR);
            convt.Dispose();
            return cropped;
        }

        public Tensor Down(IBackend backend, Tensor x)
        {
            int c = (int)x.Shape[1];
            Tensor padded = LtxAudioDeviceOps.ReplicatePadTime(backend, x, _downPadL, _downPadR);
            int tp = (int)padded.Shape[2];
            int tOut = (tp - _kernel) / _ratio + 1;
            Tensor o = new(new TensorShape(1, c, tOut), DType.F32);
            backend.Conv1d(o, padded, Depthwise(c), null, _ratio, 0, 0, 1, c);
            padded.Dispose();
            return o;
        }

        public void Dispose()
        {
            foreach (Tensor t in _depthwise.Values) t.Dispose();
            _depthwise.Clear();
        }
    }
}
