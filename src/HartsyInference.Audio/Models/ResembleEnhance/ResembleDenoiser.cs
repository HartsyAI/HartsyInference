using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ResembleEnhance;

/// <summary>Resemble-enhance 2D-STFT denoiser (upstream <c>denoiser/unet.py</c> + <c>denoiser/denoiser.py</c>):
/// a UNet over the complex STFT (<c>n_fft 1680 / hop 420</c>, last frame dropped) decomposed as
/// <c>[mag, cos(phase), sin(phase)]</c>. Layout: <c>input_proj</c> (3→16 3×3), 4 <c>encoder_blocks</c>
/// (pre_conv 3×3 doubling channels + two PreactResBlocks, then parameter-free nearest ×0.5 downsample; the
/// pre-downsample tensor is the skip), 2 <c>middle_blocks</c> at 256, 4 <c>decoder_blocks</c> (nearest ×2
/// upsample, skip ADD, pre_conv halving channels + two PreactResBlocks), and a <c>head</c>
/// (3×3 conv → GELU → 1×1 conv → 3 channels). Each PreactResBlock is
/// <c>x + Conv(GELU(GN(Conv(GELU(GN(x))))))</c> with <c>GroupNorm(dim/16)</c>. The 3 output channels become a
/// sigmoid magnitude mask and tanh phase-residual real/imag; the masked magnitude and rotated phase are
/// inverse-STFT'd back to PCM. Weights live under <c>denoiser.net.*</c>.</summary>
public sealed unsafe class ResembleDenoiser
{
    private const int NFft = 1680;
    private const int Hop = 420;
    private const int NumBins = NFft / 2 + 1;
    private const int HiddenDim = 16;
    private const int NumBlocks = 4;
    private const int NumMiddleBlocks = 2;
    private const float Eps = 1e-7f;

    private Tensor? _inW, _inB, _head0W, _head0B, _head2W, _head2B;
    private readonly UNetBlock[] _encoder;
    private readonly UNetBlock[] _middle;
    private readonly UNetBlock[] _decoder;

    public ResembleDenoiser()
    {
        _encoder = new UNetBlock[NumBlocks];
        for (int i = 0; i < NumBlocks; i++)
        {
            _encoder[i] = new UNetBlock(HiddenDim << i, HiddenDim << (i + 1));
        }
        _middle = new UNetBlock[NumMiddleBlocks];
        for (int i = 0; i < NumMiddleBlocks; i++)
        {
            _middle[i] = new UNetBlock(HiddenDim << NumBlocks, HiddenDim << NumBlocks);
        }
        _decoder = new UNetBlock[NumBlocks];
        for (int i = 0; i < NumBlocks; i++)
        {
            int level = NumBlocks - 1 - i;
            _decoder[i] = new UNetBlock(HiddenDim << (level + 1), HiddenDim << level);
        }
    }

    public void LoadWeights(ResembleWeightReader r, string prefix = "denoiser.net")
    {
        _inW = r.F32($"{prefix}.input_proj.weight");
        _inB = r.F32($"{prefix}.input_proj.bias");
        for (int i = 0; i < _encoder.Length; i++)
        {
            _encoder[i].LoadWeights(r, $"{prefix}.encoder_blocks.{i}");
        }
        for (int i = 0; i < _middle.Length; i++)
        {
            _middle[i].LoadWeights(r, $"{prefix}.middle_blocks.{i}");
        }
        for (int i = 0; i < _decoder.Length; i++)
        {
            _decoder[i].LoadWeights(r, $"{prefix}.decoder_blocks.{i}");
        }
        _head0W = r.F32($"{prefix}.head.0.weight");
        _head0B = r.F32($"{prefix}.head.0.bias");
        _head2W = r.F32($"{prefix}.head.2.weight");
        _head2B = r.F32($"{prefix}.head.2.bias");
    }

    /// <summary>Denoises a 44.1 kHz mono clip. Mirrors upstream <c>Denoiser.forward</c>: the input is peak-
    /// normalized internally and the output stays in that normalized domain (the caller rescales), trimmed or
    /// zero-padded to the input length.</summary>
    public float[] Denoise(IBackend backend, float[] pcm44k)
    {
        if (pcm44k is null || pcm44k.Length == 0)
        {
            throw new ArgumentException("pcm44k must be non-empty.", nameof(pcm44k));
        }

        // _normalize: x / (max|x| + 1e-7).
        float absMax = 0f;
        for (int i = 0; i < pcm44k.Length; i++)
        {
            float a = MathF.Abs(pcm44k[i]);
            if (a > absMax) absMax = a;
        }
        float invPeak = 1f / (absMax + 1e-7f);
        float[] pcm = new float[pcm44k.Length];
        for (int i = 0; i < pcm.Length; i++)
        {
            pcm[i] = pcm44k[i] * invPeak;
        }

        // torch.stft(center=True) gives 1 + T/hop frames; upstream drops the last one.
        int frames = pcm.Length / Hop;
        if (frames < 1)
        {
            throw new ArgumentException($"Clip too short for the denoiser: {pcm44k.Length} samples < one {Hop}-sample hop.", nameof(pcm44k));
        }
        float[] real = new float[(long)(frames + 1) * NumBins];
        float[] imag = new float[(long)(frames + 1) * NumBins];
        Tensor spec = new(new TensorShape(1, 3, NumBins, frames), DType.F32);
        float* sp = (float*)spec.DataPointer;
        long plane = (long)NumBins * frames;
        AnalyzeStft(pcm, frames, real, imag, sp, plane);

        Tensor masked = RunUnet(backend, spec, frames);
        spec.Dispose();

        // Decompose outputs → sigmoid mask, tanh real/imag → unit phase residual → rotate phase, apply mask.
        // real/imag are frame-major [f*NumBins + k] (the iSTFT layout); the UNet output is channel-first
        // NCHW [c*plane + k*frames + f], so the two indexings differ and must be mapped per (k, f).
        float* mp = (float*)masked.DataPointer;
        for (int k = 0; k < NumBins; k++)
        {
            for (int f = 0; f < frames; f++)
            {
                long ri = (long)f * NumBins + k;
                long ci = (long)k * frames + f;
                float mag = MathF.Sqrt(real[ri] * real[ri] + imag[ri] * imag[ri]);
                float invMag = 1f / (mag + Eps);
                float cos = real[ri] * invMag;
                float sin = imag[ri] * invMag;

                float mask = 1f / (1f + MathF.Exp(-mp[ci]));                // channel 0 → sigmoid mag mask
                float resReal = MathF.Tanh(mp[plane + ci]);                 // channel 1 → tanh real
                float resImag = MathF.Tanh(mp[2 * plane + ci]);             // channel 2 → tanh imag
                // _magphase on the tanh pair yields the unit phase-rotation residual.
                float resMag = MathF.Sqrt(resReal * resReal + resImag * resImag + Eps);
                float cosRes = resReal / resMag;
                float sinRes = resImag / resMag;

                float sepCos = cos * cosRes - sin * sinRes;
                float sepSin = sin * cosRes + cos * sinRes;
                float outMag = mag * mask;
                if (outMag < 0f) outMag = 0f;
                real[ri] = outMag * sepCos;
                imag[ri] = outMag * sepSin;
            }
        }
        masked.Dispose();

        // Upstream re-appends the dropped frame with replicate padding before the iSTFT.
        long lastOff = (long)(frames - 1) * NumBins;
        long padOff = (long)frames * NumBins;
        for (int k = 0; k < NumBins; k++)
        {
            real[padOff + k] = real[lastOff + k];
            imag[padOff + k] = imag[lastOff + k];
        }
        float[] wav = IStft.Apply(real, imag, frames + 1, NFft, Hop);

        // o padded (or trimmed) back to the input length.
        float[] outPcm = new float[pcm.Length];
        int copy = Math.Min(wav.Length, outPcm.Length);
        Array.Copy(wav, outPcm, copy);
        return outPcm;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_inW, _inB, _head0W, _head0B, _head2W, _head2B];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (UNetBlock b in _encoder) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (UNetBlock b in _middle) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (UNetBlock b in _decoder) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    private Tensor RunUnet(IBackend backend, Tensor spec, int frames)
    {
        // pad_to_fit: zero-pad F and T up to multiples of 2^num_blocks so every ×0.5/×2 resample is exact.
        int scale = 1 << NumBlocks;
        int fPad = (scale - NumBins % scale) % scale;
        int tPad = (scale - frames % scale) % scale;
        int fFit = NumBins + fPad, tFit = frames + tPad;
        Tensor padded = PadZeros(spec, 3, NumBins, frames, fFit, tFit);

        Tensor x = new(new TensorShape(1, HiddenDim, fFit, tFit), DType.F32);
        backend.Conv2D(x, padded, _inW!, _inB, 1, 1, 1, 1);
        padded.Dispose();

        Tensor[] skips = new Tensor[NumBlocks];
        for (int i = 0; i < _encoder.Length; i++)
        {
            (Tensor down, Tensor skip) = _encoder[i].ForwardEncoder(backend, x);
            skips[i] = skip;
            x.Dispose();
            x = down;
        }

        foreach (UNetBlock m in _middle)
        {
            Tensor n = m.ForwardMiddle(backend, x);
            x.Dispose();
            x = n;
        }

        for (int i = 0; i < _decoder.Length; i++)
        {
            Tensor skip = skips[NumBlocks - 1 - i];
            Tensor n = _decoder[i].ForwardDecoder(backend, x, skip);
            x.Dispose();
            skip.Dispose();
            x = n;
        }

        // head: 3×3 conv (16→16) → GELU → 1×1 conv (16→3).
        Tensor h = new(new TensorShape(1, HiddenDim, fFit, tFit), DType.F32);
        backend.Conv2D(h, x, _head0W!, _head0B, 1, 1, 1, 1);
        x.Dispose();
        Tensor g = new(h.Shape, DType.F32);
        backend.GeluErf(g, h);
        h.Dispose();
        Tensor outFit = new(new TensorShape(1, 3, fFit, tFit), DType.F32);
        backend.Conv2D(outFit, g, _head2W!, _head2B, 1, 1, 0, 0);
        g.Dispose();

        Tensor outT = CropTo(outFit, 3, fFit, tFit, NumBins, frames);
        outFit.Dispose();
        return outT;
    }

    private static Tensor PadZeros(Tensor src, int ch, int f, int t, int fOut, int tOut)
    {
        Tensor dst = new(new TensorShape(1, ch, fOut, tOut), DType.F32);
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)dst.DataPointer;
        // Tensor memory is zero-initialized; copy the valid region row by row.
        for (int c = 0; c < ch; c++)
        {
            for (int y = 0; y < f; y++)
            {
                long srcOff = ((long)c * f + y) * t;
                long dstOff = ((long)c * fOut + y) * tOut;
                Buffer.MemoryCopy(sp + srcOff, dp + dstOff, (long)t * sizeof(float), (long)t * sizeof(float));
            }
        }
        return dst;
    }

    private static Tensor CropTo(Tensor src, int ch, int fIn, int tIn, int fOut, int tOut)
    {
        Tensor dst = new(new TensorShape(1, ch, fOut, tOut), DType.F32);
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)dst.DataPointer;
        for (int c = 0; c < ch; c++)
        {
            for (int y = 0; y < fOut; y++)
            {
                long srcOff = ((long)c * fIn + y) * tIn;
                long dstOff = ((long)c * fOut + y) * tOut;
                Buffer.MemoryCopy(sp + srcOff, dp + dstOff, (long)tOut * sizeof(float), (long)tOut * sizeof(float));
            }
        }
        return dst;
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
                float invMag = 1f / (mag + Eps);
                long pos = (long)k * frames + f;
                spec[pos] = mag;                        // channel 0: magnitude
                spec[plane + pos] = re[k] * invMag;     // channel 1: cos(phase)
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

    /// <summary>Upstream <c>UNetBlock</c>: <c>pre_conv</c> + two PreactResBlocks, with the resample role decided
    /// by position — encoder blocks nearest-downsample ×0.5 after (returning the pre-downsample skip), decoder
    /// blocks nearest-upsample ×2 and ADD the encoder skip before <c>pre_conv</c>, middle blocks do neither.</summary>
    private sealed class UNetBlock
    {
        private readonly int _inCh, _outCh;
        private Tensor? _preW, _preB;
        private readonly PreactResBlock _res1, _res2;

        public UNetBlock(int inCh, int outCh)
        {
            _inCh = inCh;
            _outCh = outCh;
            _res1 = new PreactResBlock(outCh);
            _res2 = new PreactResBlock(outCh);
        }

        public void LoadWeights(ResembleWeightReader r, string p)
        {
            _preW = r.F32($"{p}.pre_conv.weight");
            _preB = r.F32($"{p}.pre_conv.bias");
            _res1.LoadWeights(r, $"{p}.res_block1");
            _res2.LoadWeights(r, $"{p}.res_block2");
        }

        public (Tensor Down, Tensor Skip) ForwardEncoder(IBackend backend, Tensor x)
        {
            Tensor skip = Body(backend, x);
            int f = (int)skip.Shape[2], t = (int)skip.Shape[3];
            // nn.Upsample(scale_factor=0.5, mode='nearest'): out[i] = in[2i].
            int fo = f / 2, to = t / 2;
            Tensor down = new(new TensorShape(1, _outCh, fo, to), DType.F32);
            float* sp = (float*)skip.DataPointer;
            float* dp = (float*)down.DataPointer;
            for (int c = 0; c < _outCh; c++)
            {
                for (int y = 0; y < fo; y++)
                {
                    for (int z = 0; z < to; z++)
                    {
                        dp[((long)c * fo + y) * to + z] = sp[((long)c * f + 2 * y) * t + 2 * z];
                    }
                }
            }
            return (down, skip);
        }

        public Tensor ForwardMiddle(IBackend backend, Tensor x) => Body(backend, x);

        public Tensor ForwardDecoder(IBackend backend, Tensor x, Tensor skip)
        {
            int fo = (int)skip.Shape[2], to = (int)skip.Shape[3];
            Tensor up = new(new TensorShape(1, _inCh, fo, to), DType.F32);
            backend.UpsampleNearest2D(up, x, 2, 2);
            Tensor sum = new(up.Shape, DType.F32);
            backend.Add(sum, up, skip);
            up.Dispose();
            Tensor outT = Body(backend, sum);
            sum.Dispose();
            return outT;
        }

        private Tensor Body(IBackend backend, Tensor x)
        {
            int f = (int)x.Shape[2], t = (int)x.Shape[3];
            Tensor pre = new(new TensorShape(1, _outCh, f, t), DType.F32);
            backend.Conv2D(pre, x, _preW!, _preB, 1, 1, 1, 1);
            Tensor r1 = _res1.Forward(backend, pre);
            pre.Dispose();
            Tensor r2 = _res2.Forward(backend, r1);
            r1.Dispose();
            return r2;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] own = [_preW, _preB];
            foreach (Tensor? t in own) if (t is not null) yield return t;
            foreach (Tensor t in _res1.EnumerateWeights()) yield return t;
            foreach (Tensor t in _res2.EnumerateWeights()) yield return t;
        }
    }

    /// <summary>Upstream <c>PreactResBlock</c>: <c>x + Conv3×3(GELU(GN(Conv3×3(GELU(GN(x))))))</c> with
    /// <c>GroupNorm(dim/16, dim)</c>. Sequential indices in the checkpoint: 0=GN, 2=Conv, 3=GN, 5=Conv.</summary>
    private sealed class PreactResBlock
    {
        private readonly int _ch;
        private Tensor? _n1W, _n1B, _c1W, _c1B, _n2W, _n2B, _c2W, _c2B;

        public PreactResBlock(int ch) => _ch = ch;

        public void LoadWeights(ResembleWeightReader r, string p)
        {
            _n1W = r.F32($"{p}.0.weight");
            _n1B = r.F32($"{p}.0.bias");
            _c1W = r.F32($"{p}.2.weight");
            _c1B = r.F32($"{p}.2.bias");
            _n2W = r.F32($"{p}.3.weight");
            _n2B = r.F32($"{p}.3.bias");
            _c2W = r.F32($"{p}.5.weight");
            _c2B = r.F32($"{p}.5.bias");
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            int groups = Math.Max(1, _ch / 16);
            Tensor h1 = NormActConv(backend, x, _n1W!, _n1B!, _c1W!, _c1B, groups);
            Tensor h2 = NormActConv(backend, h1, _n2W!, _n2B!, _c2W!, _c2B, groups);
            h1.Dispose();
            Tensor outT = new(x.Shape, DType.F32);
            backend.Add(outT, x, h2);
            h2.Dispose();
            return outT;
        }

        private static Tensor NormActConv(IBackend backend, Tensor x, Tensor nW, Tensor nB, Tensor cW, Tensor? cB, int groups)
        {
            Tensor n = new(x.Shape, DType.F32);
            backend.GroupNorm(n, x, nW, nB, groups, 1e-5f);
            Tensor g = new(x.Shape, DType.F32);
            backend.GeluErf(g, n);
            n.Dispose();
            Tensor c = new(x.Shape, DType.F32);
            backend.Conv2D(c, g, cW, cB, 1, 1, 1, 1);
            g.Dispose();
            return c;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] a = [_n1W, _n1B, _c1W, _c1B, _n2W, _n2B, _c2W, _c2B];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }
}
