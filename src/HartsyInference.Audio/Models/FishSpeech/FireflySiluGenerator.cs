using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Audio.Models.Whisper;

namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>Firefly HiFi-GAN generator (Fish-Speech 1.5, fish-speech <c>firefly.HiFiGANGenerator</c>).
/// <code>
///   x = conv_pre(z)                                    # FishConvNet k=13, causal, weight-norm
///   for i in 0..num_upsamples:
///       x = silu(x); x = ups[i](x)                     # FishTransConvNet, weight-norm
///       x = resblocks[i](x)                            # ParallelBlock = mean of 3 ResBlock1 (k=3/7/11)
///   x = silu(x); x = conv_post(x); x = tanh(x)         # FishConvNet k=13, causal, weight-norm
/// </code>
/// Every conv is a <c>FishConvNet</c> (causal constant left-pad) or <c>FishTransConvNet</c> (transpose + right-trim),
/// all weight-normed (<c>...conv.parametrizations.weight.original0/1</c>), with <b>SiLU</b> activations and a final
/// <c>tanh</c>. Input is the firefly quantizer's <c>[1, numMels, T]</c> feature; output is mono 44.1 kHz PCM.</summary>
public sealed unsafe class FireflySiluGenerator
{
    private readonly int _initChannels;
    private readonly int _numMels;
    private readonly int _preK;
    private readonly int _postK;
    private readonly int[] _upRates;
    private readonly int[] _upKernels;
    private readonly int[] _resKernels;
    private readonly int[][] _resDilations;
    private readonly int _numKernels;

    private Tensor? _convPreW, _convPreB, _convPostW, _convPostB;
    private readonly Tensor?[] _upW, _upB;
    private readonly ResBlock1[][] _resblocks;     // [stage][kernelBranch]

    public FireflySiluGenerator(int initChannels, int[] upRates, int[] upKernels, int[] resKernels, int[][] resDilations,
        int numMels = 512, int preConvKernelSize = 13, int postConvKernelSize = 13)
    {
        _initChannels = initChannels;
        _numMels = numMels;
        _preK = preConvKernelSize;
        _postK = postConvKernelSize;
        _upRates = upRates;
        _upKernels = upKernels;
        _resKernels = resKernels;
        _resDilations = resDilations;
        _numKernels = resKernels.Length;
        int stages = upRates.Length;
        _upW = new Tensor?[stages]; _upB = new Tensor?[stages];
        _resblocks = new ResBlock1[stages][];
        for (int i = 0; i < stages; i++)
        {
            int ch = initChannels >> (i + 1);
            _resblocks[i] = new ResBlock1[_numKernels];
            for (int k = 0; k < _numKernels; k++)
                _resblocks[i][k] = new ResBlock1(ch, resKernels[k], resDilations[k]);
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _convPreW = FireflyOps.LoadConvWeight(w, $"{prefix}.conv_pre.conv");
        _convPreB = WhisperOps.EnsureF32(w[$"{prefix}.conv_pre.conv.bias"]);
        _convPostW = FireflyOps.LoadConvWeight(w, $"{prefix}.conv_post.conv");
        _convPostB = WhisperOps.EnsureF32(w[$"{prefix}.conv_post.conv.bias"]);
        for (int i = 0; i < _upW.Length; i++)
        {
            _upW[i] = FireflyOps.LoadTransConvWeight(w, $"{prefix}.ups.{i}.conv");
            _upB[i] = WhisperOps.EnsureF32(w[$"{prefix}.ups.{i}.conv.bias"]);
            for (int k = 0; k < _numKernels; k++)
                _resblocks[i][k].LoadWeights(w, $"{prefix}.resblocks.{i}.blocks.{k}");
        }
    }

    /// <summary>Optional per-stage activation hook for parity debugging (key = stage name). Not used in production.</summary>
    public Action<string, Tensor>? DebugHook { get; set; }

    public float[] Forward(IBackend backend, Tensor z, int t)
    {
        // conv_pre: FishConvNet k=preK, causal.
        int ch = _initChannels;
        Tensor x = FireflyOps.CausalConv(backend, z, _convPreW!, _convPreB, _numMels, t, ch, _preK, dilation: 1, groups: 1);
        int curT = t;
        DebugHook?.Invoke("h_conv_pre", x);

        for (int i = 0; i < _upRates.Length; i++)
        {
            backend.Silu(x, x);
            int stride = _upRates[i], kernel = _upKernels[i];
            int outCh = ch >> 1;
            Tensor up = FireflyOps.TransConv(backend, x, _upW[i]!, _upB[i], ch, curT, outCh, kernel, stride);
            x.Dispose();
            int outT = (int)up.Shape[2];
            DebugHook?.Invoke($"h_up_{i}", up);

            // ParallelBlock: mean of the per-kernel ResBlock1 outputs.
            Tensor acc = new(new TensorShape(1, outCh, outT), DType.F32);
            float* accP = (float*)acc.DataPointer;
            for (long n = 0; n < acc.ElementCount; n++) accP[n] = 0f;
            for (int k = 0; k < _numKernels; k++)
            {
                Tensor rb = _resblocks[i][k].Forward(backend, up, outCh, outT);
                float* rp = (float*)rb.DataPointer;
                for (long n = 0; n < (long)outCh * outT; n++) accP[n] += rp[n];
                rb.Dispose();
            }
            float inv = 1f / _numKernels;
            for (long n = 0; n < (long)outCh * outT; n++) accP[n] *= inv;
            up.Dispose();
            x = acc; ch = outCh; curT = outT;
            DebugHook?.Invoke($"h_res_{i}", x);
        }

        backend.Silu(x, x);
        // conv_post: FishConvNet k=postK, causal → 1 channel.
        Tensor post = FireflyOps.CausalConv(backend, x, _convPostW!, _convPostB, ch, curT, 1, _postK, dilation: 1, groups: 1);
        x.Dispose();
        float* pp = (float*)post.DataPointer;
        float[] audio = new float[curT];
        for (int j = 0; j < curT; j++) audio[j] = MathF.Tanh(pp[j]);
        post.Dispose();
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_convPreW, _convPreB, _convPostW, _convPostB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (Tensor? t in _upW) if (t is not null) yield return t;
        foreach (Tensor? t in _upB) if (t is not null) yield return t;
        foreach (ResBlock1[] stage in _resblocks)
            foreach (ResBlock1 r in stage)
                foreach (Tensor t in r.EnumerateWeights()) yield return t;
    }

    /// <summary>Firefly <c>ResBlock1</c>: three branches, each <c>SiLU → FishConvNet(dilation[j]) → SiLU →
    /// FishConvNet(dilation[j])</c> added back as a residual. Both convs in a branch share the branch dilation
    /// (fish-speech sets <c>convs2</c> dilation = <c>convs1</c> dilation, unlike vanilla HiFi-GAN).</summary>
    private sealed class ResBlock1
    {
        private readonly int _ch, _kernel;
        private readonly int[] _dilations;
        private readonly Tensor?[] _c1W, _c1B, _c2W, _c2B;

        public ResBlock1(int ch, int kernel, int[] dilations)
        {
            _ch = ch; _kernel = kernel; _dilations = dilations;
            _c1W = new Tensor?[dilations.Length]; _c1B = new Tensor?[dilations.Length];
            _c2W = new Tensor?[dilations.Length]; _c2B = new Tensor?[dilations.Length];
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            for (int j = 0; j < _dilations.Length; j++)
            {
                _c1W[j] = FireflyOps.LoadConvWeight(w, $"{prefix}.convs1.{j}.conv");
                _c1B[j] = WhisperOps.EnsureF32(w[$"{prefix}.convs1.{j}.conv.bias"]);
                _c2W[j] = FireflyOps.LoadConvWeight(w, $"{prefix}.convs2.{j}.conv");
                _c2B[j] = WhisperOps.EnsureF32(w[$"{prefix}.convs2.{j}.conv.bias"]);
            }
        }

        public Tensor Forward(IBackend backend, Tensor input, int ch, int t)
        {
            Tensor x = new(input.Shape, DType.F32);
            Buffer.MemoryCopy((void*)input.DataPointer, (void*)x.DataPointer, input.ElementCount * 4, input.ElementCount * 4);
            for (int j = 0; j < _dilations.Length; j++)
            {
                int d = _dilations[j];
                Tensor a1 = new(x.Shape, DType.F32);
                backend.Silu(a1, x);
                Tensor c1 = FireflyOps.CausalConv(backend, a1, _c1W[j]!, _c1B[j], ch, t, ch, _kernel, dilation: d, groups: 1);
                a1.Dispose();
                Tensor a2 = new(c1.Shape, DType.F32);
                backend.Silu(a2, c1);
                c1.Dispose();
                Tensor c2 = FireflyOps.CausalConv(backend, a2, _c2W[j]!, _c2B[j], ch, t, ch, _kernel, dilation: d, groups: 1);
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
