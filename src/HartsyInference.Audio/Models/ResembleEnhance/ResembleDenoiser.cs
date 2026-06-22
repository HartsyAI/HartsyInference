using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ResembleEnhance;

/// <summary>Resemble-enhance 2D-STFT denoiser: a conv UNet over the complex STFT decomposed as
/// <c>[mag, cos(phase), sin(phase)]</c>. Its own front-end uses <c>n_fft 1680 / hop 420</c>. The UNet
/// (hidden <c>16→32→64→128→256</c>, 4 down + 2 mid + 4 up, 3×3 Conv2d, GroupNorm(dim/16), GELU,
/// skip-concat) emits 3 channels which become a sigmoid magnitude mask and tanh phase residuals; the
/// masked magnitude and rotated phase are reassembled and inverse-STFT'd back to PCM. Weights live under
/// <c>denoiser.net.*</c>. Runs as both the standalone denoise stage and the enhancer's mel pre-conditioner.</summary>
public sealed unsafe class ResembleDenoiser
{
    private const int NFft = 1680;
    private const int Hop = 420;
    private const int NumBins = NFft / 2 + 1;

    private static readonly int[] _channels = [16, 32, 64, 128, 256];

    private Tensor? _inW, _inB, _outW, _outB;
    private readonly DownBlock[] _down;
    private readonly MidBlock[] _mid;
    private readonly UpBlock[] _up;
    private readonly float _eps;

    public ResembleDenoiser(float eps = 1e-5f)
    {
        _eps = eps;
        // Encoder: 16→32, 32→64, 64→128, 128→256 (each halves F,T via a stride-2 conv).
        _down = new DownBlock[4];
        for (int i = 0; i < 4; i++) _down[i] = new DownBlock(_channels[i], _channels[i + 1], eps);
        // Two mid blocks at 256 (no resample).
        _mid = new MidBlock[2];
        for (int i = 0; i < 2; i++) _mid[i] = new MidBlock(_channels[4], eps);
        // Decoder: each up block upsamples to the symmetric encoder skip's channel count, concats it,
        // then fuses to the next-smaller width. up[i] consumes skips[3-i] (= channels[4-i]) → channels[3-i].
        _up = new UpBlock[4];
        for (int i = 0; i < 4; i++) _up[i] = new UpBlock(_channels[4 - i], _channels[3 - i], eps);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "denoiser.net")
    {
        _inW = WhisperOps.EnsureF32(w[$"{prefix}.in_conv.weight"]); _inB = Bias(w, $"{prefix}.in_conv.bias");
        for (int i = 0; i < _down.Length; i++) _down[i].LoadWeights(w, $"{prefix}.down.{i}");
        for (int i = 0; i < _mid.Length; i++) _mid[i].LoadWeights(w, $"{prefix}.mid.{i}");
        for (int i = 0; i < _up.Length; i++) _up[i].LoadWeights(w, $"{prefix}.up.{i}");
        _outW = WhisperOps.EnsureF32(w[$"{prefix}.out_conv.weight"]); _outB = Bias(w, $"{prefix}.out_conv.bias");
    }

    /// <summary>Denoises a 44.1 kHz mono PCM clip: forward STFT → UNet mask/phase → inverse STFT → PCM.</summary>
    public float[] Denoise(IBackend backend, float[] pcm44k)
    {
        if (pcm44k is null || pcm44k.Length == 0) throw new ArgumentException("pcm44k must be non-empty.", nameof(pcm44k));

        int frames = 1 + pcm44k.Length / Hop;
        // [mag, cos, sin] per (bin, frame). Layout is channel-first [1, 3, F, T].
        float[] real = new float[(long)frames * NumBins];
        float[] imag = new float[(long)frames * NumBins];
        Tensor spec = new(new TensorShape(1, 3, NumBins, frames), DType.F32);
        float* sp = (float*)spec.DataPointer;
        long plane = (long)NumBins * frames;
        AnalyzeStft(pcm44k, frames, real, imag, sp, plane);

        // UNet over the [3, F, T] spectrogram → 3 output channels.
        Tensor masked = RunUnet(backend, spec, frames);
        spec.Dispose();

        // Decompose outputs → sigmoid mask, tanh phase residual cos/sin → rotate phase, apply mask.
        // real/imag are frame-major [f*NumBins + k] (the iSTFT layout); the UNet output is channel-first
        // NCHW [c*plane + k*frames + f], so the two indexings differ and must be mapped per (k, f).
        float* mp = (float*)masked.DataPointer;
        for (int k = 0; k < NumBins; k++)
            for (int f = 0; f < frames; f++)
            {
                long ri = (long)f * NumBins + k;
                long ci = (long)k * frames + f;
                float mag = MathF.Sqrt(real[ri] * real[ri] + imag[ri] * imag[ri]);
                float invMag = mag > 1e-9f ? 1f / mag : 0f;
                float cos = real[ri] * invMag;
                float sin = imag[ri] * invMag;

                float mask = 1f / (1f + MathF.Exp(-mp[ci]));               // channel 0
                float cosRes = MathF.Tanh(mp[plane + ci]);                 // channel 1
                float sinRes = MathF.Tanh(mp[2 * plane + ci]);            // channel 2

                // Rotate the unit phasor by the predicted residual.
                float sepCos = cos * cosRes - sin * sinRes;
                float sepSin = sin * cosRes + cos * sinRes;
                float outMag = mag * mask;
                real[ri] = outMag * sepCos;
                imag[ri] = outMag * sepSin;
            }
        masked.Dispose();

        return IStft.Apply(real, imag, frames, NFft, Hop);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_inW, _inB, _outW, _outB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (DownBlock d in _down) foreach (Tensor t in d.EnumerateWeights()) yield return t;
        foreach (MidBlock m in _mid) foreach (Tensor t in m.EnumerateWeights()) yield return t;
        foreach (UpBlock u in _up) foreach (Tensor t in u.EnumerateWeights()) yield return t;
    }

    private Tensor RunUnet(IBackend backend, Tensor spec, int frames)
    {
        // in_conv: 3→16, 3×3, pad 1.
        Tensor x = new(new TensorShape(1, _channels[0], NumBins, frames), DType.F32);
        backend.Conv2D(x, spec, _inW!, _inB, 1, 1, 1, 1);

        // Encoder with stored skips (the pre-downsample feature of each level).
        Tensor[] skips = new Tensor[_down.Length];
        for (int i = 0; i < _down.Length; i++)
        {
            (Tensor down, Tensor skip) = _down[i].Forward(backend, x);
            skips[i] = skip;
            x.Dispose();
            x = down;
        }

        foreach (MidBlock m in _mid)
        {
            Tensor n = m.Forward(backend, x);
            x.Dispose();
            x = n;
        }

        // Decoder: upsample, concat the matching skip, fuse.
        for (int i = 0; i < _up.Length; i++)
        {
            Tensor skip = skips[_down.Length - 1 - i];
            Tensor n = _up[i].Forward(backend, x, skip);
            x.Dispose();
            skip.Dispose();
            x = n;
        }

        // out_conv: 16→3, 3×3, pad 1.
        Tensor outT = new(new TensorShape(1, 3, NumBins, frames), DType.F32);
        backend.Conv2D(outT, x, _outW!, _outB, 1, 1, 1, 1);
        x.Dispose();
        return outT;
    }

    private static void AnalyzeStft(float[] pcm, int frames, float[] real, float[] imag, float* spec, long plane)
    {
        float[] window = HannWindow.Get(NFft);
        float[] frame = new float[NFft];
        float[] re = new float[NumBins];
        float[] im = new float[NumBins];
        int half = NFft / 2;
        for (int f = 0; f < frames; f++)
        {
            // center=True: the frame is centered at f*hop, reflect-padded at the clip edges.
            int center = f * Hop;
            for (int i = 0; i < NFft; i++)
            {
                int idx = center - half + i;
                float s = SampleReflect(pcm, idx);
                frame[i] = s * window[i];
            }
            Fft.RealTransform(frame, re, im, NFft);
            long rowOff = (long)f * NumBins;
            for (int k = 0; k < NumBins; k++)
            {
                real[rowOff + k] = re[k];
                imag[rowOff + k] = im[k];
                float mag = MathF.Sqrt(re[k] * re[k] + im[k] * im[k]);
                float invMag = mag > 1e-9f ? 1f / mag : 0f;
                long pos = (long)k * frames + f;
                spec[pos] = mag;                       // channel 0: magnitude
                spec[plane + pos] = re[k] * invMag;    // channel 1: cos(phase)
                spec[2 * plane + pos] = im[k] * invMag; // channel 2: sin(phase)
            }
        }
    }

    private static float SampleReflect(float[] pcm, int idx)
    {
        int n = pcm.Length;
        if (n == 1) return pcm[0];
        if (idx < 0) idx = -idx;
        int period = 2 * (n - 1);
        idx %= period;
        if (idx >= n) idx = period - idx;
        return pcm[idx];
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;

    private static int Groups(int ch) => Math.Max(1, ch / 16);

    /// <summary>Two 3×3 conv + GroupNorm + GELU at the input resolution, then a stride-2 conv downsample.
    /// Returns (downsampled, pre-downsample skip).</summary>
    private sealed class DownBlock
    {
        private readonly int _inCh, _outCh;
        private readonly float _eps;
        private Tensor? _c1W, _c1B, _n1W, _n1B, _c2W, _c2B, _n2W, _n2B, _dsW, _dsB;

        public DownBlock(int inCh, int outCh, float eps) { _inCh = inCh; _outCh = outCh; _eps = eps; }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _c1W = WhisperOps.EnsureF32(w[$"{p}.conv1.weight"]); _c1B = Bias(w, $"{p}.conv1.bias");
            _n1W = WhisperOps.EnsureF32(w[$"{p}.norm1.weight"]); _n1B = WhisperOps.EnsureF32(w[$"{p}.norm1.bias"]);
            _c2W = WhisperOps.EnsureF32(w[$"{p}.conv2.weight"]); _c2B = Bias(w, $"{p}.conv2.bias");
            _n2W = WhisperOps.EnsureF32(w[$"{p}.norm2.weight"]); _n2B = WhisperOps.EnsureF32(w[$"{p}.norm2.bias"]);
            _dsW = WhisperOps.EnsureF32(w[$"{p}.downsample.weight"]); _dsB = Bias(w, $"{p}.downsample.bias");
        }

        public (Tensor Down, Tensor Skip) Forward(IBackend backend, Tensor x)
        {
            int f = (int)x.Shape[2], t = (int)x.Shape[3];
            Tensor skip = ConvNormAct(backend, x, _inCh, _outCh, _c1W!, _c1B, _n1W!, _n1B!, f, t, _eps);
            Tensor refined = ConvNormAct(backend, skip, _outCh, _outCh, _c2W!, _c2B, _n2W!, _n2B!, f, t, _eps);
            // Stride-2 conv → halves F and T (Upsample(0.5)).
            int fo = (f + 1) / 2, to = (t + 1) / 2;
            Tensor down = new(new TensorShape(1, _outCh, fo, to), DType.F32);
            backend.Conv2D(down, refined, _dsW!, _dsB, 2, 2, 1, 1);
            refined.Dispose();
            return (down, skip);
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] a = [_c1W, _c1B, _n1W, _n1B, _c2W, _c2B, _n2W, _n2B, _dsW, _dsB];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    /// <summary>Two 3×3 conv + GroupNorm + GELU at the bottleneck resolution with a residual add.</summary>
    private sealed class MidBlock
    {
        private readonly int _ch;
        private readonly float _eps;
        private Tensor? _c1W, _c1B, _n1W, _n1B, _c2W, _c2B, _n2W, _n2B;

        public MidBlock(int ch, float eps) { _ch = ch; _eps = eps; }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _c1W = WhisperOps.EnsureF32(w[$"{p}.conv1.weight"]); _c1B = Bias(w, $"{p}.conv1.bias");
            _n1W = WhisperOps.EnsureF32(w[$"{p}.norm1.weight"]); _n1B = WhisperOps.EnsureF32(w[$"{p}.norm1.bias"]);
            _c2W = WhisperOps.EnsureF32(w[$"{p}.conv2.weight"]); _c2B = Bias(w, $"{p}.conv2.bias");
            _n2W = WhisperOps.EnsureF32(w[$"{p}.norm2.weight"]); _n2B = WhisperOps.EnsureF32(w[$"{p}.norm2.bias"]);
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            int f = (int)x.Shape[2], t = (int)x.Shape[3];
            Tensor h1 = ConvNormAct(backend, x, _ch, _ch, _c1W!, _c1B, _n1W!, _n1B!, f, t, _eps);
            Tensor h2 = ConvNormAct(backend, h1, _ch, _ch, _c2W!, _c2B, _n2W!, _n2B!, f, t, _eps);
            h1.Dispose();
            Tensor outT = new(x.Shape, DType.F32);
            backend.Add(outT, x, h2);
            h2.Dispose();
            return outT;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] a = [_c1W, _c1B, _n1W, _n1B, _c2W, _c2B, _n2W, _n2B];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    /// <summary>ConvTranspose2d upsample (Upsample(2)), skip-concat, then two 3×3 conv + GroupNorm + GELU.</summary>
    private sealed class UpBlock
    {
        private readonly int _inCh, _outCh;
        private readonly float _eps;
        private Tensor? _usW, _usB, _c1W, _c1B, _n1W, _n1B, _c2W, _c2B, _n2W, _n2B;

        public UpBlock(int inCh, int outCh, float eps) { _inCh = inCh; _outCh = outCh; _eps = eps; }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _usW = WhisperOps.EnsureF32(w[$"{p}.upsample.weight"]); _usB = Bias(w, $"{p}.upsample.bias");
            _c1W = WhisperOps.EnsureF32(w[$"{p}.conv1.weight"]); _c1B = Bias(w, $"{p}.conv1.bias");
            _n1W = WhisperOps.EnsureF32(w[$"{p}.norm1.weight"]); _n1B = WhisperOps.EnsureF32(w[$"{p}.norm1.bias"]);
            _c2W = WhisperOps.EnsureF32(w[$"{p}.conv2.weight"]); _c2B = Bias(w, $"{p}.conv2.bias");
            _n2W = WhisperOps.EnsureF32(w[$"{p}.norm2.weight"]); _n2B = WhisperOps.EnsureF32(w[$"{p}.norm2.bias"]);
        }

        public Tensor Forward(IBackend backend, Tensor x, Tensor skip)
        {
            int sf = (int)skip.Shape[2], st = (int)skip.Shape[3];
            // ConvTranspose2d k4 s2 p1 doubles dims; upsample to the skip's channel count, crop to its (F,T).
            Tensor up = new(new TensorShape(1, _inCh, sf, st), DType.F32);
            Tensor upFull = new(new TensorShape(1, _inCh, 2 * (int)x.Shape[2], 2 * (int)x.Shape[3]), DType.F32);
            backend.ConvTranspose2d(upFull, x, _usW!, _usB, 2, 2, 1, 1);
            CropTo(upFull, up, _inCh, sf, st);
            upFull.Dispose();

            // Concat upsampled (inCh) with skip (inCh) → 2*inCh, then fuse to outCh.
            Tensor cat = new(new TensorShape(1, 2 * _inCh, sf, st), DType.F32);
            ReadOnlySpan<Tensor> parts = [up, skip];
            backend.Concat(cat, parts, 1);
            up.Dispose();

            Tensor h1 = ConvNormAct(backend, cat, 2 * _inCh, _outCh, _c1W!, _c1B, _n1W!, _n1B!, sf, st, _eps);
            cat.Dispose();
            Tensor h2 = ConvNormAct(backend, h1, _outCh, _outCh, _c2W!, _c2B, _n2W!, _n2B!, sf, st, _eps);
            h1.Dispose();
            return h2;
        }

        private static void CropTo(Tensor src, Tensor dst, int ch, int f, int t)
        {
            int sf = (int)src.Shape[2], st = (int)src.Shape[3];
            float* sp = (float*)src.DataPointer;
            float* dp = (float*)dst.DataPointer;
            for (int c = 0; c < ch; c++)
                for (int y = 0; y < f; y++)
                    for (int x = 0; x < t; x++)
                        dp[((long)c * f + y) * t + x] = sp[((long)c * sf + y) * st + x];
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] a = [_usW, _usB, _c1W, _c1B, _n1W, _n1B, _c2W, _c2B, _n2W, _n2B];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    private static Tensor ConvNormAct(IBackend backend, Tensor x, int inCh, int outCh,
        Tensor cW, Tensor? cB, Tensor nW, Tensor nB, int f, int t, float eps)
    {
        Tensor c = new(new TensorShape(1, outCh, f, t), DType.F32);
        backend.Conv2D(c, x, cW, cB, 1, 1, 1, 1);
        Tensor n = new(c.Shape, DType.F32);
        backend.GroupNorm(n, c, nW, nB, Groups(outCh), eps);
        c.Dispose();
        Tensor g = new(n.Shape, DType.F32);
        backend.Gelu(g, n);
        n.Dispose();
        return g;
    }
}
