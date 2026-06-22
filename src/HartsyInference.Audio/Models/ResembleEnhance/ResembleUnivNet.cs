using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ResembleEnhance;

/// <summary>Resemble-enhance UnivNet (LVCNet) vocoder: a noise-excited, kernel-predictor GAN generator that
/// turns a 128-bin mel into a 44.1 kHz waveform. A Gaussian noise channel stack (<c>d_noise 128</c>) is
/// pre-convolved to <c>nc 96</c>, then four LVCBlocks upsample by strides <c>[7,5,4,3]</c> (= 420×) using
/// Location-Variable Convolution: a kernel-predictor MLP conditioned on the mel emits per-location conv
/// kernels and biases, which a gated (tanh·sigmoid) location-variable convolution applies at dilations
/// <c>[1,3,9,27]</c>. A post-conv + Tanh produces the waveform. Weights live under <c>vocoder.*</c>.</summary>
public sealed unsafe class ResembleUnivNet
{
    private static readonly int[] _strides = [7, 5, 4, 3];
    private static readonly int[] _dilations = [1, 3, 9, 27];

    private readonly int _dNoise;
    private readonly int _nc;
    private readonly int _condExtra;
    private readonly int _melDim;
    private readonly int _seed;
    private Tensor? _preW, _preB, _postW, _postB;
    private readonly LvcBlock[] _blocks;

    public ResembleUnivNet(int melDim = 128, int kernelSize = 3, int seed = 0,
        int dNoise = 128, int nc = 96, int condExtra = 32, int kHidden = 64)
    {
        _melDim = melDim;
        _seed = seed;
        _dNoise = dNoise;
        _nc = nc;
        _condExtra = condExtra;
        _blocks = new LvcBlock[_strides.Length];
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i] = new LvcBlock(nc, melDim + condExtra, _strides[i], _dilations, kernelSize, kHidden);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "vocoder")
    {
        _preW = WhisperOps.EnsureF32(w[$"{prefix}.conv_pre.weight"]); _preB = Bias(w, $"{prefix}.conv_pre.bias");
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"{prefix}.res_stack.{i}");
        _postW = WhisperOps.EnsureF32(w[$"{prefix}.conv_post.weight"]); _postB = Bias(w, $"{prefix}.conv_post.bias");
    }

    /// <summary>Synthesizes a 44.1 kHz waveform from a conditioning mel <c>[1, mels, T]</c>.</summary>
    public float[] Forward(IBackend backend, Tensor mel)
    {
        if (mel.Shape.Rank != 3 || mel.Shape[0] != 1) throw new ArgumentException($"mel must be [1, mels, T]; got {mel.Shape}.", nameof(mel));
        int t = (int)mel.Shape[2];

        // Mel → [mel + extra] conditioning by zero-padding the extra channels (the predictor reads all of it).
        Tensor cond = BuildCond(mel, _melDim, _condExtra, t);

        // Noise excitation [1, d_noise, T] → conv_pre → [1, nc, T].
        Tensor noise = GaussianNoise(_dNoise, t, _seed);
        Tensor x = new(new TensorShape(1, _nc, t), DType.F32);
        backend.Conv1d(x, noise, _preW!, _preB, 1, 3, 3, 1, 1);
        noise.Dispose();

        foreach (LvcBlock b in _blocks)
        {
            Tensor n = b.Forward(backend, x, cond);
            x.Dispose();
            x = n;
        }
        cond.Dispose();

        // conv_post (nc → 1, k7) → Tanh.
        int outLen = (int)x.Shape[2];
        Tensor wav = new(new TensorShape(1, 1, outLen), DType.F32);
        backend.Conv1d(wav, x, _postW!, _postB, 1, 3, 3, 1, 1);
        x.Dispose();
        Tensor act = new(wav.Shape, DType.F32);
        backend.Tanh(act, wav);
        wav.Dispose();

        float[] result = new float[outLen];
        new Span<float>((void*)act.DataPointer, outLen).CopyTo(result);
        act.Dispose();
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_preW, _preB, _postW, _postB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (LvcBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    private static Tensor BuildCond(Tensor mel, int melDim, int extra, int t)
    {
        int total = melDim + extra;
        Tensor cond = new(new TensorShape(1, total, t), DType.F32);
        float* cp = (float*)cond.DataPointer;
        float* mp = (float*)mel.DataPointer;
        long melBytes = (long)melDim * t * sizeof(float);
        Buffer.MemoryCopy(mp, cp, melBytes, melBytes);
        // The trailing `extra` channels stay zeroed (NativeMemory alloc is zero-initialized).
        return cond;
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

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;

    /// <summary>One LVCNet block: a ConvTranspose1d upsample, a kernel-predictor MLP over the mel, then a
    /// stack of gated Location-Variable Convolutions at increasing dilations on the upsampled signal.</summary>
    private sealed class LvcBlock
    {
        private readonly int _nc, _stride, _kernelSize;
        private readonly int[] _dilations;
        private Tensor? _convtW, _convtB;
        private readonly KernelPredictor _predictor;

        public LvcBlock(int nc, int condDim, int stride, int[] dilations, int kernelSize, int kHidden)
        {
            _nc = nc; _stride = stride; _kernelSize = kernelSize; _dilations = dilations;
            _predictor = new KernelPredictor(nc, condDim, dilations.Length, kernelSize, kHidden);
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _convtW = WhisperOps.EnsureF32(w[$"{p}.convt_pre.weight"]); _convtB = Bias(w, $"{p}.convt_pre.bias");
            _predictor.LoadWeights(w, $"{p}.kernel_predictor");
        }

        public Tensor Forward(IBackend backend, Tensor x, Tensor cond)
        {
            int t = (int)x.Shape[2];
            // ConvTranspose1d upsample by `stride` (k = 2*stride, pad = stride/2, causal-trim trailing).
            int k = 2 * _stride;
            int padL = _stride / 2;
            int padR = k - _stride - padL;
            int upT = (t - 1) * _stride + k - padL - padR;
            Tensor up = new(new TensorShape(1, _nc, upT), DType.F32);
            backend.ConvTranspose1d(up, x, _convtW!, _convtB, _stride, padL, padR, 1, 1);
            Tensor act = new(up.Shape, DType.F32);
            backend.LeakyRelu(act, up, 0.2f);
            up.Dispose();

            // Kernel predictor reads the conditioning mel → per-(dilation) kernels + biases.
            (Tensor kernels, Tensor biases) = _predictor.Predict(backend, cond);

            // Gated location-variable convolution at each dilation, applied sequentially.
            Tensor signal = act;
            for (int d = 0; d < _dilations.Length; d++)
            {
                Tensor next = ApplyLvc(signal, kernels, biases, d, _dilations[d]);
                signal.Dispose();
                signal = next;
            }
            kernels.Dispose();
            biases.Dispose();
            return signal;
        }

        /// <summary>Location-variable convolution for dilation index <paramref name="di"/>: each output time step
        /// uses a kernel interpolated from the predicted per-frame kernels, splits into a tanh and a sigmoid
        /// half, and gates them; the result is residually added to the input signal.</summary>
        private Tensor ApplyLvc(Tensor signal, Tensor kernels, Tensor biases, int di, int dilation)
        {
            int nc = _nc, k = _kernelSize;
            int upT = (int)signal.Shape[2];
            int frames = (int)kernels.Shape[3];      // [nDil, 2*nc, nc*k, frames]
            int hopUp = upT / frames;                // samples-per-cond-frame after upsampling
            if (hopUp < 1) hopUp = 1;

            Tensor outT = new(signal.Shape, DType.F32);
            float* sp = (float*)signal.DataPointer;
            float* op = (float*)outT.DataPointer;
            float* kp = (float*)kernels.DataPointer;
            float* bp = (float*)biases.DataPointer;

            // Strides into the kernel tensor [nDil, 2*nc, nc*k, frames].
            long kDilStride = (long)(2 * nc) * (nc * k) * frames;
            long kOutStride = (long)(nc * k) * frames;
            long kBase = (long)di * kDilStride;
            // Bias tensor [nDil, 2*nc, frames].
            long bDilStride = (long)(2 * nc) * frames;
            long bBase = (long)di * bDilStride;

            int half = nc;          // 2*nc output channels → split into two nc halves.

            for (int oc = 0; oc < half; oc++)
            {
                long kCosOut = kBase + (long)oc * kOutStride;
                long kSinOut = kBase + (long)(half + oc) * kOutStride;
                long bCos = bBase + (long)oc * frames;
                long bSin = bBase + (long)(half + oc) * frames;
                for (int tpos = 0; tpos < upT; tpos++)
                {
                    int frame = tpos / hopUp;
                    if (frame >= frames) frame = frames - 1;
                    float accA = bp[bCos + frame];
                    float accB = bp[bSin + frame];
                    for (int ic = 0; ic < nc; ic++)
                    {
                        long kicA = kCosOut + (long)ic * k * frames;
                        long kicB = kSinOut + (long)ic * k * frames;
                        long inBase = (long)ic * upT;
                        for (int kk = 0; kk < k; kk++)
                        {
                            int it = tpos + (kk - (k - 1) / 2) * dilation;
                            if (it < 0 || it >= upT) continue;
                            float v = sp[inBase + it];
                            accA += v * kp[kicA + (long)kk * frames + frame];
                            accB += v * kp[kicB + (long)kk * frames + frame];
                        }
                    }
                    float gated = MathF.Tanh(accA) * (1f / (1f + MathF.Exp(-accB)));
                    op[(long)oc * upT + tpos] = sp[(long)oc * upT + tpos] + gated;
                }
            }
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] own = [_convtW, _convtB];
            foreach (Tensor? t in own) if (t is not null) yield return t;
            foreach (Tensor t in _predictor.EnumerateWeights()) yield return t;
        }
    }

    /// <summary>The LVC kernel predictor: a small conv MLP over the conditioning mel that emits, per cond
    /// frame, the convolution kernels and biases used by every location-variable conv in the block.</summary>
    private sealed class KernelPredictor
    {
        private readonly int _nc, _nDil, _kernelSize, _kHidden;
        private Tensor? _inW, _inB, _kW, _kB, _bW, _bB;

        public KernelPredictor(int nc, int condDim, int nDil, int kernelSize, int kHidden = 64)
        {
            _nc = nc; _nDil = nDil; _kernelSize = kernelSize; _kHidden = kHidden;
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _inW = WhisperOps.EnsureF32(w[$"{p}.input_conv.0.weight"]); _inB = Bias(w, $"{p}.input_conv.0.bias");
            _kW = WhisperOps.EnsureF32(w[$"{p}.kernel_conv.weight"]); _kB = Bias(w, $"{p}.kernel_conv.bias");
            _bW = WhisperOps.EnsureF32(w[$"{p}.bias_conv.weight"]); _bB = Bias(w, $"{p}.bias_conv.bias");
        }

        /// <summary>Returns (kernels <c>[nDil, 2*nc, nc*k, frames]</c>, biases <c>[1, nDil, 2*nc, frames]</c>).</summary>
        public (Tensor Kernels, Tensor Biases) Predict(IBackend backend, Tensor cond)
        {
            int t = (int)cond.Shape[2];
            // input_conv: condDim → kHidden, k3 pad1 → LeakyReLU.
            Tensor hid = new(new TensorShape(1, _kHidden, t), DType.F32);
            backend.Conv1d(hid, cond, _inW!, _inB, 1, 1, 1, 1, 1);
            Tensor act = new(hid.Shape, DType.F32);
            backend.LeakyRelu(act, hid, 0.2f);
            hid.Dispose();

            // kernel_conv: kHidden → nDil*2*nc*nc*k, k3 pad1. Output is reshaped to the 5D kernel layout.
            int kOut = _nDil * 2 * _nc * (_nc * _kernelSize);
            Tensor kFlat = new(new TensorShape(1, kOut, t), DType.F32);
            backend.Conv1d(kFlat, act, _kW!, _kB, 1, 1, 1, 1, 1);

            int bOut = _nDil * 2 * _nc;
            Tensor bFlat = new(new TensorShape(1, bOut, t), DType.F32);
            backend.Conv1d(bFlat, act, _bW!, _bB, 1, 1, 1, 1, 1);
            act.Dispose();

            // [1, kOut, t] is already the row-major [nDil, 2*nc, nc*k, frames] layout (channel-major over t).
            Tensor kernels = new(new TensorShape(_nDil, 2 * _nc, _nc * _kernelSize, t), DType.F32);
            long kBytes = (long)kOut * t * sizeof(float);
            Buffer.MemoryCopy((void*)kFlat.DataPointer, (void*)kernels.DataPointer, kBytes, kBytes);
            kFlat.Dispose();

            Tensor biases = new(new TensorShape(1, _nDil, 2 * _nc, t), DType.F32);
            long bBytes = (long)bOut * t * sizeof(float);
            Buffer.MemoryCopy((void*)bFlat.DataPointer, (void*)biases.DataPointer, bBytes, bBytes);
            bFlat.Dispose();

            return (kernels, biases);
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] own = [_inW, _inB, _kW, _kB, _bW, _bB];
            foreach (Tensor? t in own) if (t is not null) yield return t;
        }
    }
}
