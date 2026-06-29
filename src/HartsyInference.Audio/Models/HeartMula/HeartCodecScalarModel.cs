using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.HeartMula;

/// <summary>HeartCodec SQ-Codec waveform decoder (<c>sq_codec.ScalarModel.decode</c>). A causal conv stack that
/// lifts the 128-D latent (out_channels/2 after the estimator-output reshape) to the 48 kHz mono waveform at
/// 1920× the latent frame rate (∏ upsample factors [5,4,4,4,3] = 960, ×2 from the PostProcessor repeat):
/// <list type="number">
///   <item><b>round_func9</b> scalar quant: <c>round(9·x)/9</c>.</item>
///   <item><b>decoder[0]</b> look-ahead Conv1d 128→2048, k=5, <b>non-causal</b> (symmetric pad 2).</item>
///   <item><b>decoder[1..5]</b> ResDecoderBlock: causal ConvTranspose1d upsample + 5 dilated (1,3,5,7,9)
///   causal ResidualUnits (PReLU(conv1 k7) → PReLU(conv2 k1) → +x).</item>
///   <item><b>decoder[6]</b> PostProcessor: repeat each frame 2× then causal Conv1d k7 + PReLU.</item>
///   <item><b>decoder[7]</b> final Conv1d 64→1, k=7, causal.</item>
/// </list>
/// All convs are weight-normed (<c>parametrizations.weight.original0/1</c>, fused at load).</summary>
public sealed unsafe class HeartCodecScalarModel
{
    private static readonly int[] UpFactors = [5, 4, 4, 4, 3];
    private static readonly int[] UpKernels = [10, 8, 8, 8, 6];
    private static readonly int[] ResDilations = [1, 3, 5, 7, 9];
    private const int NumSamples = 2;       // PostProcessor repeat factor
    private const int LatentDim = 128;      // out_channels / 2

    // decoder[0] look-ahead conv.
    private Tensor? _d0W, _d0B;
    // 5 ResDecoderBlocks.
    private readonly UpBlock[] _blocks = new UpBlock[5];
    // PostProcessor.
    private Tensor? _ppW, _ppB, _ppAct;
    // final conv.
    private Tensor? _outW, _outB;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _d0W = WeightNormFusion.Compose(w, $"{prefix}.decoder.0");
        _d0B = WhisperOps.EnsureF32(w[$"{prefix}.decoder.0.bias"]);
        for (int i = 0; i < 5; i++)
        {
            _blocks[i] = new UpBlock();
            _blocks[i].Load(w, $"{prefix}.decoder.{i + 1}");
        }
        // decoder.6 = PostProcessor.
        _ppW = WeightNormFusion.Compose(w, $"{prefix}.decoder.6.conv");
        _ppB = WhisperOps.EnsureF32(w[$"{prefix}.decoder.6.conv.bias"]);
        _ppAct = WhisperOps.EnsureF32(w[$"{prefix}.decoder.6.activation.weight"]);
        _outW = WeightNormFusion.Compose(w, $"{prefix}.decoder.7");
        _outB = WhisperOps.EnsureF32(w[$"{prefix}.decoder.7.bias"]);
    }

    /// <summary>Decodes a latent <c>[B, 128, L]</c> into a waveform <c>[B, L·1920]</c>.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        int b = (int)latent.Shape[0];
        int t = (int)latent.Shape[2];

        // round_func9 scalar quant.
        Tensor x = new(latent.Shape, DType.F32);
        float* lp = (float*)latent.DataPointer; float* xp = (float*)x.DataPointer;
        long n = latent.ElementCount;
        for (long i = 0; i < n; i++) xp[i] = MathF.Round(9f * lp[i]) / 9f;

        // decoder[0]: non-causal Conv1d k5, pad 2 (symmetric).
        int d0Out = (int)_d0W!.Shape[0];
        Tensor cur = new(new TensorShape(b, d0Out, t), DType.F32);
        backend.Conv1d(cur, x, _d0W!, _d0B, 1, 2, 2, 1, 1);
        x.Dispose();
        int curLen = t;

        // 5 ResDecoderBlocks.
        for (int i = 0; i < 5; i++)
        {
            Tensor nb = _blocks[i].Forward(backend, cur, b, curLen, UpFactors[i], UpKernels[i]);
            cur.Dispose();
            cur = nb;
            curLen *= UpFactors[i];
        }

        // PostProcessor: repeat 2x (frame-major) then causal conv k7 + PReLU.
        int ppCh = (int)_ppW!.Shape[1];          // in channels (=64)
        int repLen = curLen * NumSamples;
        Tensor rep = new(new TensorShape(b, ppCh, repLen), DType.F32);
        float* cp = (float*)cur.DataPointer; float* rp = (float*)rep.DataPointer;
        // upstream: transpose to [B,T,C], repeat(1,1,num_samples).view(B,-1,C), transpose back.
        // Net effect: each time-step is duplicated NumSamples times consecutively.
        for (int bi = 0; bi < b; bi++)
            for (int c = 0; c < ppCh; c++)
                for (int ti = 0; ti < curLen; ti++)
                {
                    float val = cp[((long)bi * ppCh + c) * curLen + ti];
                    for (int s = 0; s < NumSamples; s++)
                        rp[((long)bi * ppCh + c) * repLen + ti * NumSamples + s] = val;
                }
        cur.Dispose();
        curLen = repLen;
        int ppOut = (int)_ppW!.Shape[0];
        Tensor pp = new(new TensorShape(b, ppOut, curLen), DType.F32);
        int ppK = (int)_ppW!.Shape[2];
        backend.Conv1d(pp, rep, _ppW!, _ppB, 1, ppK - 1, 0, 1, 1);   // causal
        rep.Dispose();
        Prelu(pp, _ppAct!, b, ppOut, curLen);

        // final conv: 64→1, k7, causal.
        int outK = (int)_outW!.Shape[2];
        Tensor wav = new(new TensorShape(b, 1, curLen), DType.F32);
        backend.Conv1d(wav, pp, _outW!, _outB, 1, outK - 1, 0, 1, 1);
        pp.Dispose();
        return wav;   // [B, 1, curLen]
    }

    private static unsafe void Prelu(Tensor x, Tensor alpha, int b, int c, int t)
    {
        float* xp = (float*)x.DataPointer;
        float* ap = (float*)alpha.DataPointer;
        bool single = alpha.ElementCount == 1;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                float a = single ? ap[0] : ap[ci];
                long off = ((long)bi * c + ci) * t;
                for (int ti = 0; ti < t; ti++)
                {
                    float v = xp[off + ti];
                    if (v < 0) xp[off + ti] = a * v;
                }
            }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_d0W, _d0B, _ppW, _ppB, _ppAct, _outW, _outB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (UpBlock bl in _blocks) if (bl is not null) foreach (Tensor t in bl.Weights()) yield return t;
    }

    /// <summary>ResDecoderBlock: causal ConvTranspose1d upsample (no activation) then 5 dilated ResidualUnits.</summary>
    private sealed class UpBlock
    {
        private Tensor? _upW, _upB;
        private readonly ResUnit[] _res = new ResUnit[5];

        public void Load(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _upW = WeightNormFusion.Compose(w, $"{p}.up_conv.layer");
            _upB = WhisperOps.EnsureF32(w[$"{p}.up_conv.layer.bias"]);
            for (int i = 0; i < 5; i++) { _res[i] = new ResUnit(); _res[i].Load(w, $"{p}.convs.{i}"); }
        }

        public Tensor Forward(IBackend backend, Tensor x, int b, int t, int stride, int k)
        {
            int outCh = (int)_upW!.Shape[1];   // ConvTranspose weight [C_in, C_out, K]
            int outLen = t * stride;           // causal: padRight = K - stride removes trailing pad
            Tensor up = new(new TensorShape(b, outCh, outLen), DType.F32);
            backend.ConvTranspose1d(up, x, _upW!, _upB, stride, 0, k - stride, 1, 1);
            Tensor cur = up;
            for (int i = 0; i < 5; i++)
            {
                Tensor nb = _res[i].Forward(backend, cur, b, outCh, outLen, ResDilations[i]);
                cur.Dispose();
                cur = nb;
            }
            return cur;
        }

        public IEnumerable<Tensor> Weights()
        {
            if (_upW is not null) yield return _upW;
            if (_upB is not null) yield return _upB;
            foreach (ResUnit r in _res) foreach (Tensor t in r.Weights()) yield return t;
        }
    }

    /// <summary>ResidualUnit: PReLU(conv1 k7 dilated causal) → PReLU(conv2 k1 causal) → + input.</summary>
    private sealed class ResUnit
    {
        private Tensor? _c1W, _c1B, _c2W, _c2B, _a1, _a2;

        public void Load(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _c1W = WeightNormFusion.Compose(w, $"{p}.conv1");
            _c1B = WhisperOps.EnsureF32(w[$"{p}.conv1.bias"]);
            _c2W = WeightNormFusion.Compose(w, $"{p}.conv2");
            _c2B = WhisperOps.EnsureF32(w[$"{p}.conv2.bias"]);
            _a1 = WhisperOps.EnsureF32(w[$"{p}.activation1.weight"]);
            _a2 = WhisperOps.EnsureF32(w[$"{p}.activation2.weight"]);
        }

        public Tensor Forward(IBackend backend, Tensor x, int b, int c, int t, int dilation)
        {
            int k1 = (int)_c1W!.Shape[2];        // 7
            Tensor h = new(new TensorShape(b, c, t), DType.F32);
            backend.Conv1d(h, x, _c1W!, _c1B, 1, dilation * (k1 - 1), 0, dilation, 1);   // causal dilated
            Prelu(h, _a1!, b, c, t);
            Tensor h2 = new(new TensorShape(b, c, t), DType.F32);
            backend.Conv1d(h2, h, _c2W!, _c2B, 1, 0, 0, 1, 1);   // k1, causal pad 0 (left_padding = 1*(1-1)=0)
            h.Dispose();
            Prelu(h2, _a2!, b, c, t);
            // residual add.
            float* hp = (float*)h2.DataPointer; float* xp = (float*)x.DataPointer;
            long n = (long)b * c * t;
            for (long i = 0; i < n; i++) hp[i] += xp[i];
            return h2;
        }

        public IEnumerable<Tensor> Weights()
        {
            Tensor?[] ws = [_c1W, _c1B, _c2W, _c2B, _a1, _a2];
            foreach (Tensor? t in ws) if (t is not null) yield return t;
        }
    }
}
