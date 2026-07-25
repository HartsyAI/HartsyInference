using HartsyInference.Audio.Dsp;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ResembleEnhance;

/// <summary>Resemble-enhance UnivNet/LVCNet vocoder (upstream <c>enhancer/univnet/univnet.py</c> +
/// <c>lvcnet.py</c> + <c>amp.py</c>): a noise-excited kernel-predictor GAN generator turning the IRMAE-decoded
/// 160-channel acoustic features into a 44.1 kHz waveform. The conditioning is zero-padded by 10 frames; a
/// <c>d_noise 128</c> Gaussian stack is pre-convolved (reflect-pad k7, weight norm) to <c>nc 96</c>; four
/// <c>LVCBlock</c>s upsample by strides <c>[7,5,4,3]</c> (LeakyReLU → weight-norm ConvTranspose1d), run one
/// BigVGAN-style AMP block (3× weight-norm conv → anti-aliased 2× up / SnakeBeta / 2× down → conv, one residual
/// around the chain), then for each dilation in <c>[1,3,9,27]</c> apply LeakyReLU→conv→LeakyReLU and a
/// location-variable convolution whose per-frame kernels/biases come from a kernel-predictor conv net over the
/// conditioning; the 192-channel LVC output gates as <c>x + σ(out[:96])·tanh(out[96:])</c>. A LeakyReLU →
/// reflect-pad weight-norm conv → Tanh head emits PCM, trimmed by the 10 padded frames. Keys under
/// <c>vocoder.*</c> (legacy <c>weight_g</c>/<c>weight_v</c> pairs).</summary>
public sealed unsafe class ResembleUnivNet
{
    private const int Npad = 10;
    private const float Slope = 0.2f;

    private static readonly int[] _strides = [7, 5, 4, 3];
    private static readonly int[] _dilations = [1, 3, 9, 27];
    private static readonly int[] _ampDilations = [1, 3, 5];

    private readonly int _dNoise;
    private readonly int _nc;
    private readonly int _condDim;
    private readonly int _seed;
    private readonly int _hopSize;
    private Tensor? _preW, _preB, _postW, _postB;
    private readonly LvcBlock[] _blocks;

    public ResembleUnivNet(int condDim = 160, int seed = 0, int dNoise = 128, int nc = 96, int kHidden = 64)
    {
        _condDim = condDim;
        _seed = seed;
        _dNoise = dNoise;
        _nc = nc;
        _blocks = new LvcBlock[_strides.Length];
        int hop = 1;
        for (int i = 0; i < _blocks.Length; i++)
        {
            hop *= _strides[i];
            _blocks[i] = new LvcBlock(nc, _strides[i], hop, kHidden);
        }
        _hopSize = hop;
    }

    public void LoadWeights(ResembleWeightReader r, string prefix = "vocoder")
    {
        _preW = r.WeightNormF32($"{prefix}.conv_pre");
        _preB = r.F32($"{prefix}.conv_pre.bias");
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].LoadWeights(r, $"{prefix}.blocks.{i}");
        }
        _postW = r.WeightNormF32($"{prefix}.conv_post.1");
        _postB = r.F32($"{prefix}.conv_post.1.bias");
    }

    /// <summary>Synthesizes a 44.1 kHz waveform from acoustic features <c>[1, condDim, T]</c>; output length is
    /// <c>T * 420</c>.</summary>
    public float[] Forward(IBackend backend, Tensor cond)
    {
        if (cond.Shape.Rank != 3 || cond.Shape[0] != 1 || cond.Shape[1] != _condDim)
        {
            throw new ArgumentException($"cond must be [1, {_condDim}, T]; got {cond.Shape}.", nameof(cond));
        }
        int t = (int)cond.Shape[2];
        int tPad = t + Npad;

        // F.pad(x, (0, npad)): 10 zero frames of conditioning on the right.
        Tensor condPadded = new(new TensorShape(1, _condDim, tPad), DType.F32);
        float* cpp = (float*)condPadded.DataPointer;
        float* cp = (float*)cond.DataPointer;
        for (int c = 0; c < _condDim; c++)
        {
            Buffer.MemoryCopy(cp + (long)c * t, cpp + (long)c * tPad, (long)t * sizeof(float), (long)t * sizeof(float));
        }

        // Noise excitation [1, d_noise, T+npad] → reflect-padded k7 conv_pre → [1, nc, T+npad].
        Tensor noise = GaussianNoise(_dNoise, tPad, _seed);
        Tensor x = ReflectPadConv(backend, noise, _preW!, _preB, _nc, 3);
        noise.Dispose();

        foreach (LvcBlock b in _blocks)
        {
            Tensor n = b.Forward(backend, x, condPadded);
            x.Dispose();
            x = n;
        }
        condPadded.Dispose();

        // conv_post: LeakyReLU → reflect-pad k7 conv (nc → 1) → Tanh, then trim the npad tail.
        Tensor act = new(x.Shape, DType.F32);
        backend.LeakyRelu(act, x, Slope);
        x.Dispose();
        Tensor wav = ReflectPadConv(backend, act, _postW!, _postB, 1, 3);
        act.Dispose();
        Tensor bounded = new(wav.Shape, DType.F32);
        backend.Tanh(bounded, wav);
        wav.Dispose();

        int outLen = t * _hopSize;
        float[] result = new float[outLen];
        new Span<float>((void*)bounded.DataPointer, outLen).CopyTo(result);
        bounded.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_preW, _preB, _postW, _postB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (LvcBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Conv1d with PyTorch <c>padding_mode="reflect"</c> semantics: reflect-pads by <paramref name="pad"/>
    /// manually, then runs the backend conv unpadded.</summary>
    private static Tensor ReflectPadConv(IBackend backend, Tensor x, Tensor w, Tensor? b, int outCh, int pad)
    {
        int ch = (int)x.Shape[1], t = (int)x.Shape[2];
        Tensor padded = PadEdges(x, ch, t, pad, pad, replicate: false);
        Tensor outT = new(new TensorShape(1, outCh, t), DType.F32);
        backend.Conv1d(outT, padded, w, b, 1, 0, 0, 1, 1);
        padded.Dispose();
        return outT;
    }

    /// <summary>Pads a <c>[1, C, T]</c> tensor on both ends — replicate (edge value) or reflect.</summary>
    private static Tensor PadEdges(Tensor x, int ch, int t, int padL, int padR, bool replicate)
    {
        Tensor outT = new(new TensorShape(1, ch, t + padL + padR), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        int tOut = t + padL + padR;
        for (int c = 0; c < ch; c++)
        {
            long src = (long)c * t;
            long dst = (long)c * tOut;
            for (int j = 0; j < padL; j++)
            {
                int idx = replicate ? 0 : padL - j;
                op[dst + j] = xp[src + idx];
            }
            Buffer.MemoryCopy(xp + src, op + dst + padL, (long)t * sizeof(float), (long)t * sizeof(float));
            for (int j = 0; j < padR; j++)
            {
                int idx = replicate ? t - 1 : t - 2 - j;
                op[dst + padL + t + j] = xp[src + idx];
            }
        }
        return outT;
    }

    private static Tensor GaussianNoise(int channels, int t, int seed)
    {
        Tensor x = new(new TensorShape(1, channels, t), DType.F32);
        float* p = (float*)x.DataPointer;
        uint rng = DeterministicRng.Seed(seed);
        long n = (long)channels * t;
        for (long i = 0; i < n; i++) p[i] = DeterministicRng.NextGaussian(ref rng);
        return x;
    }

    /// <summary>One LVCNet block: LeakyReLU → weight-norm ConvTranspose1d upsample, an AMP block, then per
    /// dilation a LeakyReLU→conv→LeakyReLU stack whose output goes through the location-variable convolution
    /// (kernel-size 3, dilation 1, per-cond-frame kernels) and gates the running signal.</summary>
    private sealed class LvcBlock
    {
        private readonly int _nc, _stride, _hop;
        private Tensor? _convtW, _convtB;
        private readonly Tensor?[] _convW, _convB;
        private readonly AmpBlock _amp;
        private readonly KernelPredictor _predictor;

        public LvcBlock(int nc, int stride, int hop, int kHidden)
        {
            _nc = nc;
            _stride = stride;
            _hop = hop;
            _convW = new Tensor?[_dilations.Length];
            _convB = new Tensor?[_dilations.Length];
            _amp = new AmpBlock(nc);
            _predictor = new KernelPredictor(nc, _dilations.Length, kHidden);
        }

        public void LoadWeights(ResembleWeightReader r, string p)
        {
            _convtW = r.WeightNormF32($"{p}.convt_pre.1");
            _convtB = r.F32($"{p}.convt_pre.1.bias");
            _amp.LoadWeights(r, $"{p}.amp_block");
            for (int d = 0; d < _dilations.Length; d++)
            {
                _convW[d] = r.WeightNormF32($"{p}.conv_blocks.{d}.1");
                _convB[d] = r.F32($"{p}.conv_blocks.{d}.1.bias");
            }
            _predictor.LoadWeights(r, $"{p}.kernel_predictor");
        }

        public Tensor Forward(IBackend backend, Tensor x, Tensor cond)
        {
            int t = (int)x.Shape[2];
            // convt_pre: LeakyReLU FIRST, then ConvTranspose1d(k=2s, stride s, pad s/2+s%2, output_padding s%2)
            // — an exact ×stride upsample.
            Tensor act = new(x.Shape, DType.F32);
            backend.LeakyRelu(act, x, Slope);
            int torchPad = _stride / 2 + _stride % 2;
            int outPad = _stride % 2;
            int upT = t * _stride;
            Tensor up = new(new TensorShape(1, _nc, upT), DType.F32);
            backend.ConvTranspose1d(up, act, _convtW!, _convtB, _stride, torchPad, torchPad - outPad, 1, 1);
            act.Dispose();

            Tensor signal = _amp.Forward(backend, up);
            up.Dispose();

            (Tensor kernels, Tensor biases) = _predictor.Predict(backend, cond);

            for (int d = 0; d < _dilations.Length; d++)
            {
                // conv_blocks[d]: LeakyReLU → weight-norm conv (k3, dilation, same-pad) → LeakyReLU.
                Tensor a1 = new(signal.Shape, DType.F32);
                backend.LeakyRelu(a1, signal, Slope);
                int pad = _dilations[d];
                Tensor conv = new(signal.Shape, DType.F32);
                backend.Conv1d(conv, a1, _convW[d]!, _convB[d], 1, pad, pad, _dilations[d], 1);
                a1.Dispose();
                Tensor a2 = new(signal.Shape, DType.F32);
                backend.LeakyRelu(a2, conv, Slope);
                conv.Dispose();

                Tensor gated = ApplyLvcGate(signal, a2, kernels, biases, d);
                a2.Dispose();
                signal.Dispose();
                signal = gated;
            }
            kernels.Dispose();
            biases.Dispose();
            return signal;
        }

        /// <summary>Location-variable convolution + gated activation: for layer <paramref name="di"/>, each
        /// output sample at position <c>t</c> convolves <paramref name="features"/> (k=3, dilation 1, same-pad)
        /// with the kernel predicted for cond frame <c>t / hop</c>, then
        /// <c>x + σ(out[:nc])·tanh(out[nc:])</c>. Kernel layout follows the upstream view
        /// <c>(layers, C_in, C_out=2nc, k, frames)</c>.</summary>
        private Tensor ApplyLvcGate(Tensor x, Tensor features, Tensor kernels, Tensor biases, int di)
        {
            int nc = _nc, k = 3, hop = _hop;
            int upT = (int)x.Shape[2];
            int frames = (int)kernels.Shape[2];
            int outCh = 2 * nc;

            Tensor outT = new(x.Shape, DType.F32);
            float* xp = (float*)x.DataPointer;
            float* fp = (float*)features.DataPointer;
            float* op = (float*)outT.DataPointer;
            float* kp = (float*)kernels.DataPointer;
            float* bp = (float*)biases.DataPointer;

            // kernels [layers, nc, 2nc, k, frames] — channel-flat index ((l·nc + ic)·2nc + oc)·k + kk, per frame.
            long kLayerStride = (long)nc * outCh * k * frames;
            long kBase = (long)di * kLayerStride;
            long bBase = (long)di * outCh * frames;

            System.Threading.Tasks.Parallel.For(0, frames, frame =>
            {
                int tStart = frame * hop;
                int tEnd = Math.Min(tStart + hop, upT);
                Span<float> acc = stackalloc float[outCh];
                for (int tpos = tStart; tpos < tEnd; tpos++)
                {
                    for (int oc = 0; oc < outCh; oc++)
                    {
                        acc[oc] = bp[bBase + (long)oc * frames + frame];
                    }
                    for (int ic = 0; ic < nc; ic++)
                    {
                        long kIc = kBase + (long)ic * outCh * k * frames;
                        long fRow = (long)ic * upT;
                        for (int kk = 0; kk < k; kk++)
                        {
                            int it = tpos + kk - 1;
                            if (it < 0 || it >= upT) continue;
                            float v = fp[fRow + it];
                            long kOff = kIc + (long)kk * frames + frame;
                            for (int oc = 0; oc < outCh; oc++)
                            {
                                acc[oc] += v * kp[kOff + (long)oc * k * frames];
                            }
                        }
                    }
                    for (int oc = 0; oc < nc; oc++)
                    {
                        float sig = 1f / (1f + MathF.Exp(-acc[oc]));
                        float tan = MathF.Tanh(acc[nc + oc]);
                        op[(long)oc * upT + tpos] = xp[(long)oc * upT + tpos] + sig * tan;
                    }
                }
            });
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] own = [_convtW, _convtB];
            foreach (Tensor? t in own) if (t is not null) yield return t;
            foreach (Tensor? t in _convW) if (t is not null) yield return t;
            foreach (Tensor? t in _convB) if (t is not null) yield return t;
            foreach (Tensor t in _amp.EnumerateWeights()) yield return t;
            foreach (Tensor t in _predictor.EnumerateWeights()) yield return t;
        }
    }

    /// <summary>BigVGAN-style AMP block: three (weight-norm conv k3 dil 1/3/5 → anti-aliased SnakeBeta → conv k3)
    /// stages chained, with ONE residual around the whole chain (upstream <c>AMPBlock.forward</c>).</summary>
    private sealed class AmpBlock
    {
        private readonly int _ch;
        private readonly Tensor?[] _aW, _aB, _bW, _bB, _logAlpha, _logBeta, _upFilter, _downFilter;

        public AmpBlock(int ch)
        {
            _ch = ch;
            int n = _ampDilations.Length;
            _aW = new Tensor?[n];
            _aB = new Tensor?[n];
            _bW = new Tensor?[n];
            _bB = new Tensor?[n];
            _logAlpha = new Tensor?[n];
            _logBeta = new Tensor?[n];
            _upFilter = new Tensor?[n];
            _downFilter = new Tensor?[n];
        }

        public void LoadWeights(ResembleWeightReader r, string p)
        {
            for (int i = 0; i < _ampDilations.Length; i++)
            {
                _aW[i] = r.WeightNormF32($"{p}.{i}.0");
                _aB[i] = r.F32($"{p}.{i}.0.bias");
                _logAlpha[i] = r.F32($"{p}.{i}.1.act.log_alpha");
                _logBeta[i] = r.F32($"{p}.{i}.1.act.log_beta");
                _upFilter[i] = ExpandDepthwise(r.F32($"{p}.{i}.1.upsample.filter"), _ch);
                _downFilter[i] = ExpandDepthwise(r.F32($"{p}.{i}.1.downsample.lowpass.filter"), _ch);
                _bW[i] = r.WeightNormF32($"{p}.{i}.2");
                _bB[i] = r.F32($"{p}.{i}.2.bias");
            }
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            int t = (int)x.Shape[2];
            Tensor h = x;
            bool owned = false;
            for (int i = 0; i < _ampDilations.Length; i++)
            {
                int pad = _ampDilations[i];
                Tensor c1 = new(new TensorShape(1, _ch, t), DType.F32);
                backend.Conv1d(c1, h, _aW[i]!, _aB[i], 1, pad, pad, _ampDilations[i], 1);
                if (owned) h.Dispose();
                Tensor snaked = UpActDown(backend, c1, i);
                c1.Dispose();
                Tensor c2 = new(new TensorShape(1, _ch, t), DType.F32);
                backend.Conv1d(c2, snaked, _bW[i]!, _bB[i], 1, 1, 1, 1, 1);
                snaked.Dispose();
                h = c2;
                owned = true;
            }
            Tensor outT = new(x.Shape, DType.F32);
            backend.Add(outT, x, h);
            h.Dispose();
            return outT;
        }

        /// <summary>Anti-aliased activation: kaiser-sinc 2× upsample (replicate-pad depthwise transposed conv)
        /// → SnakeBeta → 2× lowpass downsample (replicate-pad depthwise strided conv). Filter taps come from the
        /// checkpoint buffers (kernel 12, ratio 2).</summary>
        private Tensor UpActDown(IBackend backend, Tensor x, int i)
        {
            int t = (int)x.Shape[2];
            // UpSample1d(ratio 2, kernel 12): replicate-pad 5, ConvTranspose1d stride 2, trim 15/15, scale ×2.
            Tensor padded = PadEdges(x, _ch, t, 5, 5, replicate: true);
            Tensor up = new(new TensorShape(1, _ch, 2 * t), DType.F32);
            backend.ConvTranspose1d(up, padded, _upFilter[i]!, null, 2, 15, 15, 1, _ch);
            padded.Dispose();
            float* upp = (float*)up.DataPointer;
            long upN = (long)_ch * 2 * t;
            for (long n = 0; n < upN; n++) upp[n] *= 2f;

            // SnakeBeta: x + (1/β)·sin²(αx), per-channel α/β = clamp(exp(log), 1e-2, 50).
            float* la = (float*)_logAlpha[i]!.DataPointer;
            float* lb = (float*)_logBeta[i]!.DataPointer;
            int upT = 2 * t;
            for (int c = 0; c < _ch; c++)
            {
                float alpha = Math.Clamp(MathF.Exp(la[c]), 1e-2f, 50f);
                float invBeta = 1f / Math.Clamp(MathF.Exp(lb[c]), 1e-2f, 50f);
                long off = (long)c * upT;
                for (int j = 0; j < upT; j++)
                {
                    float s = MathF.Sin(upp[off + j] * alpha);
                    upp[off + j] += invBeta * s * s;
                }
            }

            // DownSample1d(ratio 2, kernel 12): replicate-pad (5, 6), depthwise conv stride 2.
            Tensor downPadded = PadEdges(up, _ch, upT, 5, 6, replicate: true);
            up.Dispose();
            Tensor down = new(new TensorShape(1, _ch, t), DType.F32);
            backend.Conv1d(down, downPadded, _downFilter[i]!, null, 2, 0, 0, 1, _ch);
            downPadded.Dispose();
            return down;
        }

        /// <summary>Expands a shared <c>[1,1,K]</c> filter buffer to the depthwise <c>[C,1,K]</c> layout.</summary>
        private static Tensor ExpandDepthwise(Tensor filter, int ch)
        {
            int k = (int)filter.Shape[filter.Shape.Rank - 1];
            Tensor outT = new(new TensorShape(ch, 1, k), DType.F32);
            float* fp = (float*)filter.DataPointer;
            float* op = (float*)outT.DataPointer;
            for (int c = 0; c < ch; c++)
            {
                for (int j = 0; j < k; j++)
                {
                    op[(long)c * k + j] = fp[j];
                }
            }
            filter.Dispose();
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[][] all = [_aW, _aB, _bW, _bB, _logAlpha, _logBeta, _upFilter, _downFilter];
            foreach (Tensor?[] arr in all) foreach (Tensor? t in arr) if (t is not null) yield return t;
        }
    }

    /// <summary>The LVC kernel predictor: weight-norm conv (k5) + LeakyReLU, three residual double-conv blocks,
    /// then <c>kernel_conv</c>/<c>bias_conv</c> (k3) emitting per-cond-frame LVC kernels and biases.</summary>
    private sealed class KernelPredictor
    {
        private const int ResidualBlocks = 3;

        private readonly int _nc, _nLayers, _kernelSize, _kHidden;
        private Tensor? _inW, _inB, _kW, _kB, _bW, _bB;
        private readonly Tensor?[] _r1W, _r1B, _r2W, _r2B;

        public KernelPredictor(int nc, int nLayers, int kHidden)
        {
            _nc = nc;
            _nLayers = nLayers;
            _kernelSize = 3;
            _kHidden = kHidden;
            _r1W = new Tensor?[ResidualBlocks];
            _r1B = new Tensor?[ResidualBlocks];
            _r2W = new Tensor?[ResidualBlocks];
            _r2B = new Tensor?[ResidualBlocks];
        }

        public void LoadWeights(ResembleWeightReader r, string p)
        {
            _inW = r.WeightNormF32($"{p}.input_conv.0");
            _inB = r.F32($"{p}.input_conv.0.bias");
            for (int i = 0; i < ResidualBlocks; i++)
            {
                _r1W[i] = r.WeightNormF32($"{p}.residual_convs.{i}.1");
                _r1B[i] = r.F32($"{p}.residual_convs.{i}.1.bias");
                _r2W[i] = r.WeightNormF32($"{p}.residual_convs.{i}.3");
                _r2B[i] = r.F32($"{p}.residual_convs.{i}.3.bias");
            }
            _kW = r.WeightNormF32($"{p}.kernel_conv");
            _kB = r.F32($"{p}.kernel_conv.bias");
            _bW = r.WeightNormF32($"{p}.bias_conv");
            _bB = r.F32($"{p}.bias_conv.bias");
        }

        /// <summary>Returns per-cond-frame LVC kernels and biases as channel-major <c>[1, C, frames]</c> buffers
        /// whose flat layout equals the upstream views <c>(layers, C_in, C_out=2nc, k, F)</c> and
        /// <c>(layers, C_out, F)</c>.</summary>
        public (Tensor Kernels, Tensor Biases) Predict(IBackend backend, Tensor cond)
        {
            int t = (int)cond.Shape[2];
            // input_conv: condDim → kHidden, k5 pad 2 → LeakyReLU(0.2).
            Tensor hid = new(new TensorShape(1, _kHidden, t), DType.F32);
            backend.Conv1d(hid, cond, _inW!, _inB, 1, 2, 2, 1, 1);
            Tensor c = new(hid.Shape, DType.F32);
            backend.LeakyRelu(c, hid, Slope);
            hid.Dispose();

            for (int i = 0; i < ResidualBlocks; i++)
            {
                Tensor c1 = new(c.Shape, DType.F32);
                backend.Conv1d(c1, c, _r1W[i]!, _r1B[i], 1, 1, 1, 1, 1);
                Tensor a1 = new(c.Shape, DType.F32);
                backend.LeakyRelu(a1, c1, Slope);
                c1.Dispose();
                Tensor c2 = new(c.Shape, DType.F32);
                backend.Conv1d(c2, a1, _r2W[i]!, _r2B[i], 1, 1, 1, 1, 1);
                a1.Dispose();
                Tensor a2 = new(c.Shape, DType.F32);
                backend.LeakyRelu(a2, c2, Slope);
                c2.Dispose();
                Tensor sum = new(c.Shape, DType.F32);
                backend.Add(sum, c, a2);
                a2.Dispose();
                c.Dispose();
                c = sum;
            }

            // The channel-major [1, C, t] buffers already ARE the row-major upstream views
            // (kernels: [layers, C_in, C_out, k, frames]; biases: [layers, C_out, frames]) — no copy needed.
            int kOut = _nLayers * _nc * (2 * _nc) * _kernelSize;
            Tensor kernels = new(new TensorShape(1, kOut, t), DType.F32);
            backend.Conv1d(kernels, c, _kW!, _kB, 1, 1, 1, 1, 1);
            int bOut = _nLayers * 2 * _nc;
            Tensor biases = new(new TensorShape(1, bOut, t), DType.F32);
            backend.Conv1d(biases, c, _bW!, _bB, 1, 1, 1, 1, 1);
            c.Dispose();

            return (kernels, biases);
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] own = [_inW, _inB, _kW, _kB, _bW, _bB];
            foreach (Tensor? t in own) if (t is not null) yield return t;
            Tensor?[][] res = [_r1W, _r1B, _r2W, _r2B];
            foreach (Tensor?[] arr in res) foreach (Tensor? t in arr) if (t is not null) yield return t;
        }
    }
}
