using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.HeartMula;

/// <summary>HeartCodec decoder (48 kHz, 12.5 Hz, 8-codebook RVQ). The real architecture (upstream
/// <c>heartlib.heartcodec</c>) is a <b>flow-matching</b> codec, not a WaveNet:
/// <list type="number">
///   <item><see cref="HeartCodecRvq"/> — ResidualVQ decode: code grid <c>[8,T]</c> → summed quantized vectors
///   → <c>project_out</c> → <c>[1,T,512]</c>.</item>
///   <item><c>cond_feature_emb</c> (Linear 512→512) + nearest 2× upsample → conditioning <c>mu [1,2T,512]</c>.</item>
///   <item><see cref="HeartCodecEstimator"/> velocity net integrated by a fixed Euler CFM ODE (10 steps,
///   guidance 1.25, classifier-free with a zeroed-conditioning negative branch) from Gaussian noise to the
///   codec latent <c>[1,2T,256]</c>.</item>
///   <item>reshape <c>[1,2T,256]→[2,2T,128]</c> (the 256-D latent is two 128-D channels) then
///   <see cref="HeartCodecScalarModel"/> decodes each to a 48 kHz waveform (1920× per frame), yielding a
///   2-channel (stereo) output.</item>
/// </list>
///
/// <para>Keys (HeartCodec checkpoint, prefix usually <c>""</c>): <c>flow_matching.vq_embed.*</c>,
/// <c>flow_matching.cond_feature_emb.*</c>, <c>flow_matching.estimator.*</c>, <c>scalar_model.*</c>.</para></summary>
public sealed unsafe class HeartCodecDecoder : IDisposable
{
    private const int CondDim = 512;
    private const int LatentDim = 256;    // estimator out_channels
    private const int NumSteps = 10;
    private const float Guidance = 1.25f;

    private readonly HeartMulaConfig _cfg;
    private readonly HeartCodecRvq _rvq;
    private readonly HeartCodecEstimator _estimator;
    private readonly HeartCodecScalarModel _scalar;
    private Tensor? _condW, _condB;       // cond_feature_emb Linear [512,512]
    private int _disposed;

    public HeartCodecDecoder(HeartMulaConfig cfg)
    {
        _cfg = cfg;
        _rvq = new HeartCodecRvq(cfg.CodecNumQuantizers, cfg.CodecCodebookSize, cfg.CodecCodebookDim, CondDim);
        _estimator = new HeartCodecEstimator();
        _scalar = new HeartCodecScalarModel();
    }

    public int SampleRate => _cfg.Lm.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _rvq.LoadWeights(w, $"{p}flow_matching.vq_embed");
        _condW = WhisperOps.EnsureF32(w[$"{p}flow_matching.cond_feature_emb.weight"]);
        _condB = WhisperOps.EnsureF32(w[$"{p}flow_matching.cond_feature_emb.bias"]);
        _estimator.LoadWeights(w, $"{p}flow_matching.estimator");
        _scalar.LoadWeights(w, $"{p}scalar_model");
    }

    /// <summary>Decodes an 8-codebook grid <c>[NumCodebooks, T]</c> into a mono 48 kHz waveform (the two codec
    /// channels averaged). Use <see cref="DecodeStereo"/> for the raw 2-channel output.</summary>
    public float[] Decode(IBackend backend, int[,] codes, int seed)
    {
        float[][] stereo = DecodeStereo(backend, codes, seed);
        if (stereo.Length == 0 || stereo[0].Length == 0) return [];
        int n = stereo[0].Length;
        float[] mono = new float[n];
        for (int i = 0; i < n; i++) mono[i] = 0.5f * (stereo[0][i] + stereo[1][i]);
        return mono;
    }

    /// <summary>Full flow-matching decode of a code grid <c>[8,T]</c> → 2-channel waveform <c>float[2][]</c>.</summary>
    public float[][] DecodeStereo(IBackend backend, int[,] codes, int seed)
    {
        ThrowIfDisposed();
        if (_condW is null) throw new InvalidOperationException("HeartCodecDecoder weights not loaded.");
        int t = codes.GetLength(1);
        if (t <= 0) return [[], []];

        // 1. RVQ decode → [1,T,512]; cond_feature_emb Linear; nearest 2x → mu [1,2T,512].
        Tensor rvq = _rvq.Decode(backend, codes, t);
        Tensor cond = WhisperOps.ProjectLinear(backend, rvq, _condW!, _condB, 1, t, CondDim, CondDim);
        rvq.Dispose();
        int t2 = 2 * t;
        Tensor mu = NearestUpsample2x(cond, t, CondDim);
        cond.Dispose();

        // 2. CFM Euler with CFG → latent [1, 2T, 256].
        Tensor latent = SolveEulerCfg(backend, mu, t2, seed);
        mu.Dispose();

        // 3. reshape [1,2T,256] → [2,2T,128] (channels), scalar-decode each.
        Tensor chan = ReshapeToChannels(latent, t2);   // [2,128,2T] (already transposed to [B,C,L])
        latent.Dispose();
        Tensor wav = _scalar.Decode(backend, chan);     // [2,1,L]
        chan.Dispose();

        int samples = (int)wav.Shape[2];
        float[][] outp = [new float[samples], new float[samples]];
        float* wp = (float*)wav.DataPointer;
        for (int ch = 0; ch < 2; ch++)
            for (int i = 0; i < samples; i++)
                outp[ch][i] = wp[(long)ch * 1 * samples + i];
        wav.Dispose();
        return outp;
    }

    // ── components exposed for parity testing ──
    public Tensor RvqDecode(IBackend backend, int[,] codes, int t) => _rvq.Decode(backend, codes, t);

    public Tensor CondEmb(IBackend backend, Tensor rvqOut, int t)
    {
        Tensor cond = WhisperOps.ProjectLinear(backend, rvqOut, _condW!, _condB, 1, t, CondDim, CondDim);
        Tensor mu = NearestUpsample2x(cond, t, CondDim);
        cond.Dispose();
        return mu;
    }

    public Tensor EstimatorForward(IBackend backend, Tensor input, float[] timestep) =>
        _estimator.Forward(backend, input, timestep);

    public Tensor ScalarDecode(IBackend backend, Tensor latentBCL) => _scalar.Decode(backend, latentBCL);

    /// <summary>Fixed-Euler OT-CFM ODE (upstream <c>FlowMatching.solve_euler</c>) with classifier-free guidance.
    /// Integrates from Gaussian noise (t=0) to the codec latent (t=1) in <see cref="NumSteps"/> steps. The
    /// in-context region is empty (start segment), so the noise-blend term and incontext branch are zero.</summary>
    public Tensor SolveEulerCfg(IBackend backend, Tensor mu, int t2, int seed)
    {
        // initial latent noise x [1, 2T, 256].
        Tensor x = GaussianNoise(1, t2, LatentDim, seed);

        // Estimator input = cat(x, incontext(0), mu) over the feature dim → [B, 2T, 256+256+512=1024].
        // CFG batches it ×2 with mu replaced by zeros on the negative branch.
        int inCh = 2 * LatentDim + CondDim;
        float dt = 1f / NumSteps;
        float tcur = 0f;
        float* xp = (float*)x.DataPointer;
        float* mup = (float*)mu.DataPointer;

        for (int step = 0; step < NumSteps; step++)
        {
            // Build CFG input [2, 2T, 1024]: row0 = uncond (mu=0), row1 = cond (mu).
            Tensor input = new(new TensorShape(2, t2, inCh), DType.F32);
            float* ip = (float*)input.DataPointer;
            for (int ti = 0; ti < t2; ti++)
            {
                long xo = (long)ti * LatentDim;
                long muo = (long)ti * CondDim;
                // uncond row (batch 0).
                long b0 = ((long)0 * t2 + ti) * inCh;
                for (int c = 0; c < LatentDim; c++) ip[b0 + c] = xp[xo + c];           // x
                for (int c = 0; c < LatentDim; c++) ip[b0 + LatentDim + c] = 0f;       // incontext
                for (int c = 0; c < CondDim; c++) ip[b0 + 2 * LatentDim + c] = 0f;     // mu zeroed
                // cond row (batch 1).
                long b1 = ((long)1 * t2 + ti) * inCh;
                for (int c = 0; c < LatentDim; c++) ip[b1 + c] = xp[xo + c];
                for (int c = 0; c < LatentDim; c++) ip[b1 + LatentDim + c] = 0f;
                for (int c = 0; c < CondDim; c++) ip[b1 + 2 * LatentDim + c] = mup[muo + c];
            }

            Tensor dphi = _estimator.Forward(backend, input, [tcur, tcur]);   // [2, 2T, 256]
            input.Dispose();
            float* dp = (float*)dphi.DataPointer;
            // CFG: v = uncond + guidance * (cond - uncond); x += dt * v.
            for (int ti = 0; ti < t2; ti++)
            {
                long un = ((long)0 * t2 + ti) * LatentDim;
                long co = ((long)1 * t2 + ti) * LatentDim;
                long xo = (long)ti * LatentDim;
                for (int c = 0; c < LatentDim; c++)
                {
                    float u = dp[un + c], cc = dp[co + c];
                    float v = u + Guidance * (cc - u);
                    xp[xo + c] += dt * v;
                }
            }
            dphi.Dispose();
            tcur += dt;
        }
        return x;   // [1, 2T, 256]
    }

    // nearest-neighbor 2x upsample over the time axis of [1, t, dim] → [1, 2t, dim].
    private static Tensor NearestUpsample2x(Tensor x, int t, int dim)
    {
        Tensor outp = new(new TensorShape(1, 2 * t, dim), DType.F32);
        float* sp = (float*)x.DataPointer; float* dp = (float*)outp.DataPointer;
        for (int ti = 0; ti < t; ti++)
        {
            long src = (long)ti * dim;
            for (int r = 0; r < 2; r++)
            {
                long dst = (long)(ti * 2 + r) * dim;
                for (int c = 0; c < dim; c++) dp[dst + c] = sp[src + c];
            }
        }
        return outp;
    }

    // [1, 2T, 256] → reshape [1,2T,2,128] → permute [1,2,2T,128] → [2,2T,128] → transpose to [2,128,2T].
    private static Tensor ReshapeToChannels(Tensor latent, int t2)
    {
        int half = LatentDim / 2;   // 128
        Tensor outp = new(new TensorShape(2, half, t2), DType.F32);
        float* lp = (float*)latent.DataPointer;   // [1, 2T, 256]
        float* op = (float*)outp.DataPointer;     // [2, 128, 2T]
        for (int ti = 0; ti < t2; ti++)
            for (int ch = 0; ch < 2; ch++)
                for (int c = 0; c < half; c++)
                {
                    float val = lp[(long)ti * LatentDim + ch * half + c];
                    op[((long)ch * half + c) * t2 + ti] = val;
                }
        return outp;
    }

    private static Tensor GaussianNoise(int b, int t, int dim, int seed)
    {
        Tensor x = new(new TensorShape(b, t, dim), DType.F32);
        float* xp = (float*)x.DataPointer;
        long n = (long)b * t * dim;
        Random rng = new(seed);
        for (long i = 0; i < n; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            xp[i] = (float)(r * Math.Cos(2.0 * Math.PI * u2));
            if (i + 1 < n) xp[i + 1] = (float)(r * Math.Sin(2.0 * Math.PI * u2));
        }
        return x;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _rvq.EnumerateWeights()) yield return t;
        if (_condW is not null) yield return _condW;
        if (_condB is not null) yield return _condB;
        foreach (Tensor t in _estimator.EnumerateWeights()) yield return t;
        foreach (Tensor t in _scalar.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(HeartCodecDecoder));
    }
}
