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

    // Upstream detokenize constants (duration = 29.76 s fixed decode window).
    private const double SegmentSeconds = 29.76;
    private const int MinCodes = 372;      // int(29.76 * 12.5) — codes per window
    private const int HopCodes = 320;      // 372 // 93 * 80
    private const int OvlpCodes = 52;      // 372 - 320
    private const int OvlpLatent = 104;    // 52 * 2 — latent frames carried as in-context
    private const int SegLatent = 744;     // int(29.76 * 25) — latent frames per window

    /// <summary>Full upstream <c>detokenize</c>: codes are tiled up to the fixed 372-frame window (the DiT is
    /// trained at latent length 744), long grids decode as overlapping windows whose last 104 latent frames seed
    /// the next window's in-context region, waveforms crossfade over the overlap, and the result is truncated to
    /// the true code duration. → 2-channel waveform <c>float[2][]</c>.</summary>
    public float[][] DecodeStereo(IBackend backend, int[,] codes, int seed)
    {
        ThrowIfDisposed();
        if (_condW is null) throw new InvalidOperationException("HeartCodecDecoder weights not loaded.");
        int t = codes.GetLength(1);
        if (t <= 0) return [[], []];
        long targetLen = (long)(t / 12.5 * SampleRate);

        // Code tiling to the fixed window grid (cyclic repeat, as upstream's cat-doubling+truncate).
        int[,] work = codes;
        if (work.GetLength(1) < MinCodes) work = TileOrTruncate(work, MinCodes);
        int len = work.GetLength(1);
        if ((len - OvlpLatent) % HopCodes > 0)   // upstream quirk: modulo uses the latent overlap (104)
        {
            int lenCodes = (int)Math.Ceiling((len - OvlpCodes) / (double)HopCodes) * HopCodes + OvlpCodes;
            if (lenCodes != len) work = TileOrTruncate(work, lenCodes);
            len = lenCodes;
        }

        int minWav = (int)(SegmentSeconds * SampleRate);      // 1428480 samples per window
        int hopWav = minWav / 93 * 80;                        // 1228800
        int ovlpWav = minWav - hopWav;                        // 199680

        float[][]? output = null;
        Tensor? prevLatent = null;
        int windowIdx = 0;
        try
        {
            for (int sinx = 0; sinx + HopCodes <= len; sinx += HopCodes, windowIdx++)
            {
                int wLen = Math.Min(MinCodes, len - sinx);    // python slice clamps; only differs on upstream's quirk lens
                Tensor mu = BuildMu(backend, work, sinx, wLen);
                int t2 = 2 * wLen;

                Tensor? incontext = null;
                int inLen = 0;
                if (windowIdx > 0 && OvlpLatent > 0 && prevLatent is not null)
                {
                    // in-context = last 104 latent frames of the previous window, rest zeros.
                    incontext = new Tensor(new TensorShape(1, t2, LatentDim), DType.F32);
                    float* icp = (float*)incontext.DataPointer;
                    float* plp = (float*)prevLatent.DataPointer;
                    long tail = (long)(SegLatent - OvlpLatent) * LatentDim;
                    Buffer.MemoryCopy(plp + tail, icp, (long)t2 * LatentDim * 4, (long)OvlpLatent * LatentDim * 4);
                    inLen = OvlpLatent;
                }

                Tensor noise = GaussianNoise(1, t2, LatentDim, unchecked(seed + 7919 * windowIdx));
                Tensor latent = SolveEulerCfg(backend, mu, t2, noise, incontext, inLen);
                mu.Dispose(); noise.Dispose(); incontext?.Dispose();

                float[][] segWav = ScalarDecodeStereo(backend, latent, t2);
                prevLatent?.Dispose();
                prevLatent = latent;

                // truncate the segment to the fixed window length, then crossfade-append.
                int segLen = Math.Min(segWav[0].Length, minWav);
                if (output is null)
                {
                    output = [new float[segLen], new float[segLen]];
                    for (int ch = 0; ch < 2; ch++) Array.Copy(segWav[ch], output[ch], segLen);
                }
                else
                {
                    output = CrossfadeAppend(output, segWav, segLen, ovlpWav);
                }
            }
        }
        finally
        {
            prevLatent?.Dispose();
        }

        if (output is null) return [[], []];
        int finalLen = (int)Math.Min(targetLen, output[0].Length);
        float[][] outp = [new float[finalLen], new float[finalLen]];
        for (int ch = 0; ch < 2; ch++) Array.Copy(output[ch], outp[ch], finalLen);
        return outp;
    }

    /// <summary>Single-segment decode with caller-supplied init noise <c>[2T,256]</c> (no tiling/windowing) —
    /// the deterministic path the parity test compares against the Python oracle.</summary>
    public float[][] DecodeSegmentStereo(IBackend backend, int[,] codes, Tensor initNoise)
    {
        ThrowIfDisposed();
        if (_condW is null) throw new InvalidOperationException("HeartCodecDecoder weights not loaded.");
        int t = codes.GetLength(1);
        int t2 = 2 * t;
        Tensor mu = BuildMu(backend, codes, 0, t);
        Tensor noise = new(new TensorShape(1, t2, LatentDim), DType.F32);
        Buffer.MemoryCopy((void*)initNoise.DataPointer, (void*)noise.DataPointer, (long)t2 * LatentDim * 4, (long)t2 * LatentDim * 4);
        Tensor latent = SolveEulerCfg(backend, mu, t2, noise, incontext: null, incontextLen: 0);
        mu.Dispose(); noise.Dispose();
        float[][] outp = ScalarDecodeStereo(backend, latent, t2);
        latent.Dispose();
        return outp;
    }

    // RVQ decode + cond_feature_emb + nearest 2x on a window [sinx, sinx+wLen) of the code grid → mu [1,2w,512].
    private Tensor BuildMu(IBackend backend, int[,] codes, int sinx, int wLen)
    {
        int q = codes.GetLength(0);
        int[,] win = codes;
        if (sinx != 0 || wLen != codes.GetLength(1))
        {
            win = new int[q, wLen];
            for (int i = 0; i < q; i++) for (int j = 0; j < wLen; j++) win[i, j] = codes[i, sinx + j];
        }
        Tensor rvq = _rvq.Decode(backend, win, wLen);
        Tensor cond = WhisperOps.ProjectLinear(backend, rvq, _condW!, _condB, 1, wLen, CondDim, CondDim);
        rvq.Dispose();
        Tensor mu = NearestUpsample2x(backend, cond, wLen, CondDim);
        cond.Dispose();
        return mu;
    }

    // [1,2T,256] latent → ScalarModel per channel → float[2][2T*1920]. Channels decode sequentially:
    // the upsampling stack's activations peak at ~[1,64,2T·1920] floats, so batch-1 halves host memory.
    private float[][] ScalarDecodeStereo(IBackend backend, Tensor latent, int t2)
    {
        Tensor chan = ReshapeToChannels(backend, latent, t2);   // [2,128,2T]
        int half = LatentDim / 2;
        float[][] outp = new float[2][];
        for (int ch = 0; ch < 2; ch++)
        {
            Tensor one = new(new TensorShape(1, half, t2), DType.F32);
            backend.SliceRows(one, chan, ch * half);   // channel ch = rows [ch*128 .. ) of [2,128,2T]
            Tensor wav = _scalar.Decode(backend, one);   // [1,1,L]
            one.Dispose();
            int samples = (int)wav.Shape[2];
            outp[ch] = new float[samples];
            float* wp = (float*)wav.DataPointer;
            for (int i = 0; i < samples; i++) outp[ch][i] = wp[i];
            wav.Dispose();
        }
        chan.Dispose();
        return outp;
    }

    // Cyclic repeat/truncate of the code grid to wantLen frames (upstream cat-doubling + slice).
    private static int[,] TileOrTruncate(int[,] src, int wantLen)
    {
        int q = src.GetLength(0), t = src.GetLength(1);
        int[,] outp = new int[q, wantLen];
        for (int i = 0; i < q; i++) for (int j = 0; j < wantLen; j++) outp[i, j] = src[i, j % t];
        return outp;
    }

    // Linear-ramp overlap-add (upstream ov_win), then append the non-overlapped remainder.
    private static float[][] CrossfadeAppend(float[][] output, float[][] seg, int segLen, int ovlp)
    {
        int oldLen = output[0].Length;
        int ov = Math.Min(ovlp, Math.Min(oldLen, segLen));
        int newLen = oldLen + segLen - ov;
        float[][] merged = [new float[newLen], new float[newLen]];
        for (int ch = 0; ch < 2; ch++)
        {
            Array.Copy(output[ch], merged[ch], oldLen);
            for (int i = 0; i < ov; i++)
            {
                float up = ov <= 1 ? 1f : i / (float)(ov - 1);   // np.linspace(0,1,ovlp)
                merged[ch][oldLen - ov + i] = output[ch][oldLen - ov + i] * (1f - up) + seg[ch][i] * up;
            }
            Array.Copy(seg[ch], ov, merged[ch], oldLen, segLen - ov);
        }
        return merged;
    }

    // ── components exposed for parity testing ──
    public Tensor RvqDecode(IBackend backend, int[,] codes, int t) => _rvq.Decode(backend, codes, t);

    public Tensor CondEmb(IBackend backend, Tensor rvqOut, int t)
    {
        Tensor cond = WhisperOps.ProjectLinear(backend, rvqOut, _condW!, _condB, 1, t, CondDim, CondDim);
        Tensor mu = NearestUpsample2x(backend, cond, t, CondDim);
        cond.Dispose();
        return mu;
    }

    public Tensor EstimatorForward(IBackend backend, Tensor input, float[] timestep) =>
        _estimator.Forward(backend, input, timestep);

    public Tensor ScalarDecode(IBackend backend, Tensor latentBCL) => _scalar.Decode(backend, latentBCL);

    /// <summary>Fixed-Euler OT-CFM ODE with self-seeded noise and no in-context region (start segment).</summary>
    public Tensor SolveEulerCfg(IBackend backend, Tensor mu, int t2, int seed)
    {
        Tensor noise = GaussianNoise(1, t2, LatentDim, seed);
        Tensor x = SolveEulerCfg(backend, mu, t2, noise, incontext: null, incontextLen: 0);
        noise.Dispose();
        return x;
    }

    /// <summary>Fixed-Euler OT-CFM ODE (upstream <c>FlowMatching.solve_euler</c>) with classifier-free guidance.
    /// Integrates from <paramref name="initNoise"/> (t=0) to the codec latent (t=1) in <see cref="NumSteps"/> steps.
    /// When <paramref name="incontextLen"/> &gt; 0 the first rows are re-blended each step
    /// (<c>x = (1-(1-1e-6)t)·noise + t·incontext</c>) and pinned to the in-context latents at the end.</summary>
    public Tensor SolveEulerCfg(IBackend backend, Tensor mu, int t2, Tensor initNoise, Tensor? incontext, int incontextLen)
    {
        // x starts as a copy of the init noise; the original is kept for the per-step in-context blend.
        Tensor x = new(new TensorShape(1, t2, LatentDim), DType.F32);
        Buffer.MemoryCopy((void*)initNoise.DataPointer, (void*)x.DataPointer, (long)t2 * LatentDim * 4, (long)t2 * LatentDim * 4);

        // Estimator input = cat(x, incontext, mu) over the feature dim → [B, 2T, 256+256+512=1024].
        // CFG batches it ×2 with mu replaced by zeros on the negative branch (incontext kept on both).
        float dt = 1f / NumSteps;
        float tcur = 0f;

        // incontext (or zeros when absent) and a zeroed conditioning block, shared across steps (GPU-resident).
        Tensor ic = incontext ?? Zeros(backend, 1, t2, LatentDim);
        Tensor zeroCond = Zeros(backend, 1, t2, CondDim);

        for (int step = 0; step < NumSteps; step++)
        {
            // Progressive noise→in-context blend of the pinned region: x[:icLen] = a·noise + tcur·incontext.
            if (incontextLen > 0)
                x = BlendHead(backend, x, initNoise, incontext!, incontextLen, t2, 1f - (1f - 1e-6f) * tcur, tcur);

            // Build CFG input [2, 2T, 1024]: row0 = uncond (mu=0), row1 = cond (mu).
            Tensor uncondRow = new(new TensorShape(1, t2, 2 * LatentDim + CondDim), DType.F32);
            Tensor condRow = new(new TensorShape(1, t2, 2 * LatentDim + CondDim), DType.F32);
            backend.Concat(uncondRow, [x, ic, zeroCond], 2);
            backend.Concat(condRow, [x, ic, mu], 2);
            Tensor input = new(new TensorShape(2, t2, 2 * LatentDim + CondDim), DType.F32);
            backend.Concat(input, [uncondRow, condRow], 0);
            uncondRow.Dispose(); condRow.Dispose();

            Tensor dphi = _estimator.Forward(backend, input, [tcur, tcur]);   // [2, 2T, 256]
            input.Dispose();
            // CFG: v = uncond + guidance·(cond − uncond); x += dt·v.
            Tensor uncond = new(new TensorShape(1, t2, LatentDim), DType.F32);
            Tensor cond = new(new TensorShape(1, t2, LatentDim), DType.F32);
            backend.SliceRows(uncond, dphi, 0);      // dphi batch 0
            backend.SliceRows(cond, dphi, t2);       // dphi batch 1
            dphi.Dispose();
            backend.CfgEulerStep(x, cond, uncond, Guidance, dt);   // pos=cond, neg=uncond
            uncond.Dispose(); cond.Dispose();
            tcur += dt;
        }

        // Pin the in-context region to the previous window's latents (upstream post-solve assignment).
        if (incontextLen > 0)
            x = PinHead(backend, x, incontext!, incontextLen, t2);

        if (incontext is null) ic.Dispose();   // only the zeros stand-in is ours to free
        zeroCond.Dispose();
        return x;   // [1, 2T, 256]
    }

    private static Tensor Zeros(IBackend backend, int b, int t, int dim)
    {
        Tensor z = new(new TensorShape(b, t, dim), DType.F32);
        backend.Fill(z, 0f);
        return z;
    }

    // x[:icLen] = a·noise[:icLen] + tcur·incontext[:icLen], rest unchanged. Consumes x, returns a new tensor.
    private static Tensor BlendHead(IBackend backend, Tensor x, Tensor noise, Tensor incontext, int icLen, int t2, float a, float tcur)
    {
        Tensor head = new(new TensorShape(1, icLen, LatentDim), DType.F32);
        Tensor icHead = new(new TensorShape(1, icLen, LatentDim), DType.F32);
        backend.SliceRows(head, noise, 0);
        backend.SliceRows(icHead, incontext, 0);
        backend.Scale(head, head, a);
        backend.Scale(icHead, icHead, tcur);
        backend.Add(head, head, icHead);
        icHead.Dispose();
        Tensor tail = new(new TensorShape(1, t2 - icLen, LatentDim), DType.F32);
        backend.SliceRows(tail, x, icLen);
        x.Dispose();
        Tensor merged = new(new TensorShape(1, t2, LatentDim), DType.F32);
        backend.Concat(merged, [head, tail], 1);
        head.Dispose(); tail.Dispose();
        return merged;
    }

    // x[:icLen] = incontext[:icLen], rest unchanged. Consumes x, returns a new tensor.
    private static Tensor PinHead(IBackend backend, Tensor x, Tensor incontext, int icLen, int t2)
    {
        Tensor head = new(new TensorShape(1, icLen, LatentDim), DType.F32);
        backend.SliceRows(head, incontext, 0);
        Tensor tail = new(new TensorShape(1, t2 - icLen, LatentDim), DType.F32);
        backend.SliceRows(tail, x, icLen);
        x.Dispose();
        Tensor merged = new(new TensorShape(1, t2, LatentDim), DType.F32);
        backend.Concat(merged, [head, tail], 1);
        head.Dispose(); tail.Dispose();
        return merged;
    }

    // nearest-neighbor 2x upsample over the time axis of [1, t, dim] → [1, 2t, dim] (GPU-resident:
    // transpose to channels-first, frame-repeat, transpose back — each output frame duplicated consecutively).
    private static Tensor NearestUpsample2x(IBackend backend, Tensor x, int t, int dim)
    {
        Tensor xc = new(new TensorShape(1, dim, t), DType.F32);
        backend.Transpose2D(xc, x, t, dim);
        Tensor rep = new(new TensorShape(1, dim, 2 * t), DType.F32);
        backend.RepeatTime(rep, xc, 2);
        xc.Dispose();
        Tensor outp = new(new TensorShape(1, 2 * t, dim), DType.F32);
        backend.Transpose2D(outp, rep, dim, 2 * t);
        rep.Dispose();
        return outp;
    }

    // [1, 2T, 256] → view [1,2T,2,128] → Permute0213 [1,2,2T,128] (= [2,2T,128]) → Transpose2D → [2,128,2T].
    private static Tensor ReshapeToChannels(IBackend backend, Tensor latent, int t2)
    {
        int half = LatentDim / 2;   // 128
        Tensor perm = new(new TensorShape(1, 2, t2, half), DType.F32);
        backend.Permute0213(perm, latent, t2, 2, half);   // latent viewed as [1,2T,2,128]
        Tensor outp = new(new TensorShape(2, half, t2), DType.F32);
        backend.Transpose2D(outp, perm, t2, half);         // [2,2T,128] → [2,128,2T]
        perm.Dispose();
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
