using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>Firefly HiFi-GAN generator (Fish-Speech 1.5). Same MRF topology as the VITS generator
/// (<c>conv_pre</c> → N <c>ConvTranspose1d</c> upsample stages each fused with <c>num_kernels</c> ResBlock1
/// branches → <c>conv_post</c> → tanh) but with <b>SiLU</b> activations throughout instead of LeakyReLU — the one
/// architectural divergence that prevents reusing <see cref="Vits.VitsHiFiGan"/>. Input is the firefly quantizer's
/// <c>[1, inputChannels, T]</c> decoder feature; output is mono 44.1 kHz PCM with a final <c>tanh</c>.</summary>
public sealed unsafe class FireflySiluGenerator
{
    private readonly int _initChannels;
    private readonly int[] _upRates;
    private readonly int[] _upKernels;
    private readonly int[] _resKernels;
    private readonly int[][] _resDilations;
    private readonly int _numKernels;

    private Tensor? _convPreW, _convPreB, _convPostW, _convPostB;
    private readonly Tensor?[] _upW, _upB;
    private readonly ResBlock[] _resblocks;

    public FireflySiluGenerator(int initChannels, int[] upRates, int[] upKernels, int[] resKernels, int[][] resDilations)
    {
        _initChannels = initChannels;
        _upRates = upRates;
        _upKernels = upKernels;
        _resKernels = resKernels;
        _resDilations = resDilations;
        _numKernels = resKernels.Length;
        int stages = upRates.Length;
        _upW = new Tensor?[stages]; _upB = new Tensor?[stages];
        _resblocks = new ResBlock[stages * _numKernels];
        for (int i = 0; i < stages; i++)
        {
            int ch = initChannels >> (i + 1);
            for (int k = 0; k < _numKernels; k++)
                _resblocks[i * _numKernels + k] = new ResBlock(ch, resKernels[k], resDilations[k]);
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _convPreW = WhisperOps.EnsureF32(w[$"{prefix}.conv_pre.weight"]);
        _convPreB = w.TryGetValue($"{prefix}.conv_pre.bias", out Tensor? cpb) ? WhisperOps.EnsureF32(cpb) : null;
        _convPostW = WhisperOps.EnsureF32(w[$"{prefix}.conv_post.weight"]);
        _convPostB = w.TryGetValue($"{prefix}.conv_post.bias", out Tensor? cqb) ? WhisperOps.EnsureF32(cqb) : null;
        for (int i = 0; i < _upW.Length; i++)
        {
            _upW[i] = WhisperOps.EnsureF32(w[$"{prefix}.ups.{i}.weight"]);
            _upB[i] = w.TryGetValue($"{prefix}.ups.{i}.bias", out Tensor? ub) ? WhisperOps.EnsureF32(ub) : null;
        }
        for (int i = 0; i < _resblocks.Length; i++) _resblocks[i].LoadWeights(w, $"{prefix}.resblocks.{i}");
    }

    public float[] Forward(IBackend backend, Tensor z, int t)
    {
        int ch = _initChannels;
        int preK = (int)_convPreW!.Shape[2];
        int prePad = (preK - 1) / 2;
        int preT = t + 2 * prePad - (preK - 1);
        Tensor x = new(new TensorShape(1, ch, preT), DType.F32);
        backend.Conv1d(x, z, _convPreW!, _convPreB, 1, prePad, prePad, 1, 1);

        int curT = preT;
        for (int i = 0; i < _upRates.Length; i++)
        {
            SiluInPlace(backend, x);
            int stride = _upRates[i], kernel = _upKernels[i];
            int pad = (kernel - stride) / 2;
            int outCh = ch >> 1, outT = (curT - 1) * stride + (kernel - 1) + 1 - 2 * pad;
            Tensor up = new(new TensorShape(1, outCh, outT), DType.F32);
            backend.ConvTranspose1d(up, x, _upW[i]!, _upB[i], stride, pad, pad, 1, 1);
            x.Dispose();

            Tensor acc = new(new TensorShape(1, outCh, outT), DType.F32);
            float* accP = (float*)acc.DataPointer;
            for (long n = 0; n < acc.ElementCount; n++) accP[n] = 0f;
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

        SiluInPlace(backend, x);
        int postK = (int)_convPostW!.Shape[2];
        int postPad = (postK - 1) / 2;
        int postT = curT + 2 * postPad - (postK - 1);
        Tensor post = new(new TensorShape(1, 1, postT), DType.F32);
        backend.Conv1d(post, x, _convPostW!, _convPostB, 1, postPad, postPad, 1, 1);
        x.Dispose();
        float* pp = (float*)post.DataPointer;
        float[] audio = new float[postT];
        for (int j = 0; j < postT; j++) audio[j] = MathF.Tanh(pp[j]);
        post.Dispose();
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_convPreW, _convPreB, _convPostW, _convPostB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (Tensor? t in _upW) if (t is not null) yield return t;
        foreach (Tensor? t in _upB) if (t is not null) yield return t;
        foreach (ResBlock r in _resblocks) foreach (Tensor t in r.EnumerateWeights()) yield return t;
    }

    private static void SiluInPlace(IBackend backend, Tensor x)
    {
        backend.Silu(x, x);
    }

    /// <summary>HiFi-GAN ResBlock1 with SiLU activations — three dilated branches, each <c>SiLU → dilated Conv1d
    /// → SiLU → Conv1d</c> added back as a residual.</summary>
    private sealed class ResBlock
    {
        private readonly int _ch, _kernel;
        private readonly int[] _dilations;
        private readonly Tensor?[] _c1W, _c1B, _c2W, _c2B;

        public ResBlock(int ch, int kernel, int[] dilations)
        {
            _ch = ch; _kernel = kernel; _dilations = dilations;
            _c1W = new Tensor?[dilations.Length]; _c1B = new Tensor?[dilations.Length];
            _c2W = new Tensor?[dilations.Length]; _c2B = new Tensor?[dilations.Length];
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            for (int j = 0; j < _dilations.Length; j++)
            {
                _c1W[j] = WhisperOps.EnsureF32(w[$"{prefix}.convs1.{j}.weight"]);
                _c1B[j] = w.TryGetValue($"{prefix}.convs1.{j}.bias", out Tensor? b1) ? WhisperOps.EnsureF32(b1) : null;
                _c2W[j] = WhisperOps.EnsureF32(w[$"{prefix}.convs2.{j}.weight"]);
                _c2B[j] = w.TryGetValue($"{prefix}.convs2.{j}.bias", out Tensor? b2) ? WhisperOps.EnsureF32(b2) : null;
            }
        }

        public Tensor Forward(IBackend backend, Tensor input, int ch, int t)
        {
            Tensor x = new(input.Shape, DType.F32);
            Buffer.MemoryCopy((void*)input.DataPointer, (void*)x.DataPointer, input.ElementCount * 4, input.ElementCount * 4);
            for (int j = 0; j < _dilations.Length; j++)
            {
                int d = _dilations[j], pad = d * (_kernel - 1) / 2;
                Tensor a1 = new(x.Shape, DType.F32);
                backend.Silu(a1, x);
                Tensor c1 = new(new TensorShape(1, ch, t), DType.F32);
                backend.Conv1d(c1, a1, _c1W[j]!, _c1B[j], 1, pad, pad, d, 1);
                a1.Dispose();
                Tensor a2 = new(c1.Shape, DType.F32);
                backend.Silu(a2, c1);
                c1.Dispose();
                int pad2 = (_kernel - 1) / 2;
                Tensor c2 = new(new TensorShape(1, ch, t), DType.F32);
                backend.Conv1d(c2, a2, _c2W[j]!, _c2B[j], 1, pad2, pad2, 1, 1);
                a2.Dispose();
                float* xp = (float*)x.DataPointer; float* c2p = (float*)c2.DataPointer;
                for (long n = 0; n < (long)ch * t; n++) xp[n] += c2p[n];
                c2.Dispose();
            }
            return x;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[][] groups = [_c1W, _c1B, _c2W, _c2B];
            foreach (Tensor?[] g in groups) foreach (Tensor? t in g) if (t is not null) yield return t;
        }
    }
}
