using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.HeartMula;

/// <summary>HeartCodec SQ-Codec waveform decoder (<c>sq_codec.ScalarModel.decode</c>) — a causal conv stack that lifts the 128-D latent to the 48 kHz mono waveform at 1920x the latent frame rate (∏ upsample factors [5,4,4,4,3] = 960, ×2 from the PostProcessor repeat).</summary>
// round_func9 scalar quant: round(9·x)/9. decoder[0]: look-ahead Conv1d 128→2048, k=5, non-causal
// (symmetric pad 2). decoder[1..5]: ResDecoderBlock — causal ConvTranspose1d upsample + 5 dilated
// (1,3,5,7,9) causal ResidualUnits (PReLU(conv1 k7) → PReLU(conv2 k1) → +x). decoder[6]: PostProcessor —
// repeat each frame 2x then causal Conv1d k7 + PReLU. decoder[7]: final Conv1d 64→1, k=7, causal.
// All convs are weight-normed (parametrizations.weight.original0/1, fused at load).
public sealed unsafe class HeartCodecScalarModel
{
    private static readonly int[] UpFactors = [5, 4, 4, 4, 3];
    private static readonly int[] UpKernels = [10, 8, 8, 8, 6];
    private static readonly int[] ResDilations = [1, 3, 5, 7, 9];
    private const int NumSamples = 2;       // PostProcessor repeat factor
    private const int LatentDim = 128;      // out_channels / 2

    // Chunked decode: the conv stack is causal except decoder[0] (k5, symmetric pad 2), so a waveform sample
    // depends on at most ~44 latent frames of left context (per block the 5 dilated k7 ResidualUnits reach back
    // 6·(1+3+5+7+9)=150 samples at that block's rate — 150/5=30 latent frames at block 1, 150/20=7.5 at block 2,
    // … — plus ≤2 frames per ConvTranspose and 2 for decoder[0]) and exactly 2 frames of right context
    // (decoder[0]'s look-ahead). Decoding in ChunkFrames-sized chunks re-fed with ChunkCtxLeft/Right frames of
    // real context and keeping only the interior therefore reproduces the monolithic decode exactly while
    // bounding peak activation memory to O(chunk) instead of O(L) — the old whole-window decode peaked at
    // ~1.9 GB of host+device activations per 29.76 s window (the Swarm OOM driver).
    private const int ChunkFrames = 256;
    private const int ChunkCtxLeft = 64;
    private const int ChunkCtxRight = 2;
    private static readonly int SamplesPerFrame = ComputeSamplesPerFrame();
    private static readonly int ChunkOverride = ReadChunkOverride();

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

    /// <summary>Decodes a latent <c>[B, 128, L]</c> into a waveform <c>[B, 1, L·1920]</c>; long latents decode in bounded overlapping chunks (see the chunk constants above) and still match the monolithic decode exactly.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        int b = (int)latent.Shape[0];
        int t = (int)latent.Shape[2];
        int chunk = ChunkOverride;
        if (chunk <= 0 || t <= chunk + ChunkCtxLeft + ChunkCtxRight) return DecodeCore(backend, latent);

        Tensor wav = new(new TensorShape(b, 1, t * SamplesPerFrame), DType.F32);
        float* wp = (float*)wav.DataPointer;
        // Every chunk feeds the same fixed-length window (edge chunks just carry extra context), so all chunks
        // share one set of conv shapes — one cuDNN plan build instead of one per distinct edge length.
        int fedLen = chunk + ChunkCtxLeft + ChunkCtxRight;
        for (int start = 0; start < t; start += chunk)
        {
            int end = Math.Min(start + chunk, t);
            int ctxStart = Math.Clamp(start - ChunkCtxLeft, 0, t - fedLen);
            int ctxEnd = ctxStart + fedLen;
            Tensor part = SliceTime(latent, b, t, ctxStart, ctxEnd - ctxStart);
            Tensor partWav = DecodeCore(backend, part);
            part.Dispose();
            float* pp = (float*)partWav.DataPointer;   // syncs only this chunk's waveform to host
            long chunkLen = partWav.Shape[2];
            long keepOff = (long)(start - ctxStart) * SamplesPerFrame;
            long keepLen = (long)(end - start) * SamplesPerFrame;
            for (int bi = 0; bi < b; bi++)
            {
                Buffer.MemoryCopy(pp + bi * chunkLen + keepOff,
                    wp + ((long)bi * t + start) * SamplesPerFrame, keepLen * 4, keepLen * 4);
            }
            partWav.Dispose();
        }
        return wav;   // [B, 1, L·1920]
    }

    private Tensor DecodeCore(IBackend backend, Tensor latent)
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
        // Net effect: each time-step is duplicated NumSamples times consecutively (GPU-resident nearest repeat).
        int ppCh = (int)_ppW!.Shape[1];          // in channels (=64)
        int repLen = curLen * NumSamples;
        Tensor rep = new(new TensorShape(b, ppCh, repLen), DType.F32);
        backend.RepeatTime(rep, cur, NumSamples);
        cur.Dispose();
        curLen = repLen;
        int ppOut = (int)_ppW!.Shape[0];
        Tensor pp = new(new TensorShape(b, ppOut, curLen), DType.F32);
        int ppK = (int)_ppW!.Shape[2];
        backend.Conv1d(pp, rep, _ppW!, _ppB, 1, ppK - 1, 0, 1, 1);   // causal
        rep.Dispose();
        backend.Prelu(pp, pp, _ppAct!);

        // final conv: 64→1, k7, causal.
        int outK = (int)_outW!.Shape[2];
        Tensor wav = new(new TensorShape(b, 1, curLen), DType.F32);
        backend.Conv1d(wav, pp, _outW!, _outB, 1, outK - 1, 0, 1, 1);
        pp.Dispose();
        return wav;   // [B, 1, curLen]
    }

    // Host copy of latent[:, :, t0 .. t0+len) → [B, C, len]; source rows are B·C with row stride t.
    private static Tensor SliceTime(Tensor latent, int b, int t, int t0, int len)
    {
        int c = (int)latent.Shape[1];
        Tensor outp = new(new TensorShape(b, c, len), DType.F32);
        float* src = (float*)latent.DataPointer;
        float* dst = (float*)outp.DataPointer;
        long rows = (long)b * c;
        for (long r = 0; r < rows; r++)
            Buffer.MemoryCopy(src + r * t + t0, dst + r * len, (long)len * 4, (long)len * 4);
        return outp;
    }

    private static int ComputeSamplesPerFrame()
    {
        int perFrame = NumSamples;
        foreach (int factor in UpFactors) perFrame *= factor;
        return perFrame;
    }

    // HARTSY_HEARTCODEC_SCALAR_CHUNK overrides the decode chunk length in latent frames; 0 (or negative)
    // restores the monolithic whole-latent decode.
    private static int ReadChunkOverride()
    {
        string? env = Environment.GetEnvironmentVariable("HARTSY_HEARTCODEC_SCALAR_CHUNK");
        if (env is not null && int.TryParse(env, out int value)) return value;
        return ChunkFrames;
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
            backend.Prelu(h, h, _a1!);
            Tensor h2 = new(new TensorShape(b, c, t), DType.F32);
            backend.Conv1d(h2, h, _c2W!, _c2B, 1, 0, 0, 1, 1);   // k1, causal pad 0 (left_padding = 1*(1-1)=0)
            h.Dispose();
            backend.Prelu(h2, h2, _a2!);
            backend.Add(h2, h2, x);   // residual (GPU-resident)
            return h2;
        }

        public IEnumerable<Tensor> Weights()
        {
            Tensor?[] ws = [_c1W, _c1B, _c2W, _c2B, _a1, _a2];
            foreach (Tensor? t in ws) if (t is not null) yield return t;
        }
    }
}
