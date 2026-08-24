using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.F5Tts;
using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Preprocessing;
using F5SwaySamplingScheduler = HartsyInference.Audio.Models.F5Tts.F5SwaySamplingScheduler;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>F5-TTS zero-shot voice-clone pipeline. Takes a reference audio mel + the
/// text that was spoken in it + the target text we want to generate, runs flow-matching
/// Euler under <see cref="F5SwaySamplingScheduler"/>, and vocodes the result through
/// <see cref="Vocos"/>.
///
/// <para>Pipeline shape (matches <c>F5TTS.api.F5TTS.infer</c> in the reference repo):
/// <code>
///   ref_mel    = mel_extract(ref_audio_wav)                  # [1, 100, T_ref]
///   target_T   = T_ref * len(target_text) / len(ref_text)    # duration heuristic
///   T_total    = T_ref + target_T
///   cond_mel   = concat([ref_mel, zeros(T_target)], dim=T)   # [1, 100, T_total]
///   x          = randn(target_mel_shape)                     # starting noise on target
///   text_ids   = encode(ref_text + " " + target_text)        # joint chars
///
///   for (t, dt) in zip(sigmas[:-1], deltas):
///     v_cond  = dit(x_cond_audio_text)
///     v_uncond = dit(x_drop_audio_drop_text)
///     v       = v_cond + cfg_scale * (v_cond - v_uncond)
///     x       = x + dt * v        # update ONLY the target portion
///
///   # x[T_ref:] is the generated mel
///   audio = vocos(x[T_ref:])
/// </code></para>
///
/// <para><b>Status:</b> numerically validated against the upstream F5-TTS. The DiT velocity is bit-exact
/// (corr 1.0) and the full flow-matching sample loop matches <c>CFM.sample</c> to corr 1.0 on the generated
/// mel; Vocos is corr 0.9999. Key fixes for parity: the ConvNeXt text stem masks the filler tail, the
/// timestep sinusoid is scaled by 1000, the text stem uses exact-erf GELU while the DiT FFN uses tanh GELU,
/// and CFG is cond-anchored (<c>pred + (pred − null)·cfg</c>) with the reference region replaced only at the
/// end (no per-step anchoring).</para></summary>
public sealed class F5TtsPipeline : IAudioPipeline, IDisposable
{
    private readonly F5TtsConfig _cfg;
    private readonly F5Dit _dit;
    private readonly Vocos _vocos;
    private readonly F5Tokenizer _tokenizer;
    private readonly SafeTensorsLoader _ditLoader;
    private readonly SafeTensorsLoader _vocosLoader;
    private int _disposed;

    /// <summary>Model name (HF repo id).</summary>
    public string ModelName { get; }

    private F5TtsPipeline(string modelName, F5TtsConfig cfg, F5Dit dit, Vocos vocos, F5Tokenizer tok, SafeTensorsLoader ditLoader, SafeTensorsLoader vocosLoader)
    {
        ModelName = modelName;
        _cfg = cfg;
        _dit = dit;
        _vocos = vocos;
        _tokenizer = tok;
        _ditLoader = ditLoader;
        _vocosLoader = vocosLoader;
    }

    /// <summary>Loads the F5-TTS v1 Base pipeline (DiT + Vocos vocoder + char tokenizer).
    /// Both checkpoints are auto-downloaded into the HartsyInference cache on first use.</summary>
    public static async Task<F5TtsPipeline> LoadAsync(F5TtsConfig? cfg = null, CancellationToken ct = default)
    {
        F5TtsConfig resolved = cfg ?? F5TtsConfig.V1Base;

        // DiT: SWivid/F5-TTS, file F5TTS_v1_Base/model_1250000.safetensors
        string repoDir = AudioModelCache.GetRepoDirectory("SWivid/F5-TTS", "tts");
        Task<string> ditPath = AudioModelCache.GetAsync("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors", category: "tts", ct: ct);
        Task<string> vocabPath = AudioModelCache.GetAsync("SWivid/F5-TTS", "F5TTS_v1_Base/vocab.txt", category: "tts", ct: ct);

        // Vocos: lucasnewman/vocos-mel-24khz (safetensors)
        Task<string> vocosPath = AudioModelCache.GetAsync("lucasnewman/vocos-mel-24khz", "model.safetensors", category: "tts", ct: ct);

        await Task.WhenAll(ditPath, vocabPath, vocosPath).ConfigureAwait(false);

        SafeTensorsLoader ditLoader = new();
        ditLoader.Load(ditPath.Result);
        F5Dit dit = new(resolved);
        dit.LoadWeights(ditLoader.GetAllTensors());

        SafeTensorsLoader vocosLoader = new();
        vocosLoader.Load(vocosPath.Result);
        Vocos vocos = new(VocosConfig.Mel24k);
        vocos.LoadWeights(vocosLoader.GetAllTensors());

        // The tokenizer expects vocab.txt in a known directory — copy it into the F5-TTS
        // base of the cache so the tokenizer can find it without bypassing the cache layer.
        string baseCacheDir = AudioModelCache.GetRepoDirectory("SWivid/F5-TTS", "tts");
        string vocabDest = Path.Combine(baseCacheDir, "vocab.txt");
        if (!File.Exists(vocabDest)) File.Copy(vocabPath.Result, vocabDest, overwrite: true);
        F5Tokenizer tok = new(baseCacheDir);

        return new F5TtsPipeline("SWivid/F5-TTS", resolved, dit, vocos, tok, ditLoader, vocosLoader);
    }

    /// <summary>Generates audio for <paramref name="targetText"/> in the voice of
    /// <paramref name="refMel"/>. <paramref name="refMel"/> is a precomputed log-mel of
    /// the reference WAV (shape <c>[1, 100, T_ref]</c>, natural-log, no normalization —
    /// matches Vocos's own feature extractor). <paramref name="refText"/> is the text
    /// spoken in the reference audio (used to align the joint character sequence).
    ///
    /// <para>The sample loop is numerically validated against <c>CFM.sample</c> (generated mel corr 1.0). The
    /// produced audio is at the vocoder's native scale; F5's reference pipeline additionally rescales it to the
    /// reference clip's loudness (out of scope here).</para></summary>
    public float[] Generate(IBackend backend, Tensor refMel, string refText, string targetText, F5TtsOptions? options = null)
    {
        // Upstream forces ref_text to end with ". " so the joint char stream (ref_text + gen_text) aligns at the
        // boundary and len(ref_text) in the duration ratio is correct; without it the tempo drifts.
        string refNorm = NormalizeRefText(refText);
        bool profG = Environment.GetEnvironmentVariable("HARTSY_F5_PROFILE") == "1";
        System.Diagnostics.Stopwatch gsw = System.Diagnostics.Stopwatch.StartNew();
        Tensor mel = GenerateMel(backend, refMel, _tokenizer.Encode(refNorm), _tokenizer.Encode(targetText), options);
        if (profG) { backend.Sync(); HartsyInference.Core.Logging.Logs.Info($"[F5 time] DiT sample loop: {gsw.Elapsed.TotalMilliseconds:0}ms"); gsw.Restart(); }
        // DEBUG: dump the pre-vocoder mel [1,melDim,T] to localize DiT-vs-vocoder (F5_DUMP_MEL=path).
        string? melPath = Environment.GetEnvironmentVariable("F5_DUMP_MEL");
        if (melPath is not null)
        {
            int md = (int)mel.Shape[1], mt = (int)mel.Shape[2];
            long n = (long)md * mt;
            double s = 0, s2 = 0, mn = double.MaxValue, mx = double.MinValue;
            unsafe
            {
                float* mp = (float*)mel.DataPointer;
                for (long i = 0; i < n; i++) { float v = mp[i]; s += v; s2 += (double)v * v; mn = Math.Min(mn, v); mx = Math.Max(mx, v); }
            }
            HartsyInference.Core.Logging.Logs.Info($"[F5 mel] shape [{md},{mt}] mean {s / n:F4} std {Math.Sqrt(s2 / n - (s / n) * (s / n)):F4} min {mn:F3} max {mx:F3}");
        }
        float[] audio = _vocos.Forward(backend, mel);
        if (profG) { backend.Sync(); HartsyInference.Core.Logging.Logs.Info($"[F5 time] Vocos vocode({mel.Shape[2]} frames → {audio.Length} smp): {gsw.Elapsed.TotalMilliseconds:0}ms"); }
        mel.Dispose();
        return audio;
    }

    /// <summary>Generates audio directly from a raw reference waveform, owning the F5/Vocos mel front-end so
    /// callers cannot mis-configure it (HTK mel scale, no area norm, center padding, magnitude spectrum,
    /// natural-log clamp 1e-5 — see <see cref="MelSpectrogramExtractor.F5VocosConfig"/>). The reference is
    /// resampled to 24 kHz and RMS-normalized to 0.1 exactly as upstream <c>preprocess_ref_audio_text</c>.</summary>
    public float[] GenerateFromAudio(IBackend backend, ReadOnlySpan<float> refAudio, int sampleRate,
        string refText, string targetText, F5TtsOptions? options = null)
    {
        ThrowIfDisposed();
        bool prof = Environment.GetEnvironmentVariable("HARTSY_F5_PROFILE") == "1";
        System.Diagnostics.Stopwatch _secSw = System.Diagnostics.Stopwatch.StartNew();
        MelSpectrogramExtractor.Config melCfg = MelSpectrogramExtractor.F5VocosConfig();
        int sr = melCfg.SampleRate;
        float[] mono24k = sampleRate == sr ? refAudio.ToArray() : Resampler.Create(sampleRate, sr).Resample(refAudio);

        // Upstream: if rms < target_rms (0.1), scale the reference up so quiet clips don't under-drive the DiT.
        const float targetRms = 0.1f;
        double sumSq = 0.0;
        for (int i = 0; i < mono24k.Length; i++) sumSq += (double)mono24k[i] * mono24k[i];
        float rms = mono24k.Length > 0 ? (float)Math.Sqrt(sumSq / mono24k.Length) : 0f;
        if (rms > 1e-6f && rms < targetRms)
        {
            float g = targetRms / rms;
            for (int i = 0; i < mono24k.Length; i++) mono24k[i] *= g;
        }

        // Trim leading/trailing silence so tRef reflects actual speech. Otherwise pauses/silence inflate the
        // reference length, which over-estimates the target duration (target = ref_frames·target_len/ref_len)
        // and lets the model pad the output start with reference continuation ("reference bleed") — mirrors
        // upstream preprocess_ref_audio_text, which clips the reference to its non-silent span.
        mono24k = TrimEdgeSilence(mono24k, sr);

        MelSpectrogramExtractor extractor = new(melCfg);
        float[,] melData = extractor.Compute(mono24k);
        int tRef = melData.GetLength(1);
        Tensor refMel = new(new TensorShape(1, _cfg.MelDim, tRef), DType.F32);
        unsafe
        {
            float* mp = (float*)refMel.DataPointer;
            for (int d = 0; d < _cfg.MelDim; d++)
                for (int t = 0; t < tRef; t++) mp[d * tRef + t] = melData[d, t];
        }
        if (prof) { backend.Sync(); HartsyInference.Core.Logging.Logs.Info($"[F5 time] mel-extract(ref {mono24k.Length} smp → {tRef} frames): {_secSw.Elapsed.TotalMilliseconds:0}ms"); }
        float[] audio = Generate(backend, refMel, refText, targetText, options);
        refMel.Dispose();
        return audio;
    }

    /// <summary>Trims leading/trailing silence from the reference waveform using windowed RMS, keeping a small
    /// margin so onsets/offsets aren't clipped. Conservative — only removes clearly-silent edge regions
    /// (window RMS below a fraction of the loudest window); never touches interior audio.</summary>
    private static float[] TrimEdgeSilence(float[] x, int sr)
    {
        int win = Math.Max(1, sr / 50);          // 20 ms windows
        int nWin = x.Length / win;
        if (nWin < 4) return x;                    // too short to bother
        float[] rms = new float[nWin];
        float maxRms = 0f;
        for (int wi = 0; wi < nWin; wi++)
        {
            double s = 0; int off = wi * win;
            for (int i = 0; i < win; i++) { float v = x[off + i]; s += (double)v * v; }
            rms[wi] = (float)Math.Sqrt(s / win);
            if (rms[wi] > maxRms) maxRms = rms[wi];
        }
        if (maxRms < 1e-4f) return x;              // effectively silent — leave as-is
        float thresh = 0.08f * maxRms;             // silence = below 8% of the loudest window
        int first = 0; while (first < nWin && rms[first] < thresh) first++;
        int last = nWin - 1; while (last > first && rms[last] < thresh) last--;
        if (first == 0 && last == nWin - 1) return x;   // no edge silence
        int margin = sr / 20;                      // 50 ms keepout on each side
        int start = Math.Max(0, first * win - margin);
        int end = Math.Min(x.Length, (last + 1) * win + margin);
        int len = end - start;
        if (len <= 0 || len >= x.Length) return x;
        float[] outp = new float[len];
        Array.Copy(x, start, outp, 0, len);
        return outp;
    }

    private static string NormalizeRefText(string refText)
    {
        string s = (refText ?? string.Empty).Trim();
        if (s.Length == 0) return s;
        if (s.EndsWith(". ", StringComparison.Ordinal) || s.EndsWith("。", StringComparison.Ordinal)) return s;
        return s.EndsWith('.') ? s + " " : s + ". ";
    }

    /// <summary>Runs the flow-matching sample loop and returns the generated target mel <c>[1, mel_dim, T_tgt]</c>
    /// (pre-vocoder). Mirrors <c>CFM.sample</c>: the reference region evolves freely under the fixed masked
    /// conditioning and is sliced off at the end (so no per-step anchoring is needed). <paramref name="initialNoise"/>
    /// injects a deterministic <c>[1, mel_dim, T_total]</c> noise tensor (else a seeded Gaussian is built).</summary>
    public Tensor GenerateMel(IBackend backend, Tensor refMel, int[] refIds, int[] targetIds,
        F5TtsOptions? options = null, Tensor? initialNoise = null)
    {
        ThrowIfDisposed();
        F5TtsOptions opts = options ?? new F5TtsOptions();
        if (Environment.GetEnvironmentVariable("F5_DUMP_MEL") is not null)
        {
            long rn = refMel.ElementCount; double rs = 0, rs2 = 0, rmn = double.MaxValue, rmx = double.MinValue;
            unsafe { float* rp = (float*)refMel.DataPointer; for (long i = 0; i < rn; i++) { float v = rp[i]; rs += v; rs2 += (double)v * v; rmn = Math.Min(rmn, v); rmx = Math.Max(rmx, v); } }
            HartsyInference.Core.Logging.Logs.Info($"[F5 refmel] shape [{refMel.Shape[1]},{refMel.Shape[2]}] mean {rs / rn:F4} std {Math.Sqrt(rs2 / rn - (rs / rn) * (rs / rn)):F4} min {rmn:F3} max {rmx:F3}");
            // per-mel-bin mean (100 values) → localize which bins differ from F5 (filterbank shape).
            int rmd = (int)refMel.Shape[1], rmt = (int)refMel.Shape[2];
            double[] binMean = new double[rmd];
            unsafe { float* rp = (float*)refMel.DataPointer; for (int d = 0; d < rmd; d++) { double a = 0; for (int t = 0; t < rmt; t++) a += rp[(long)d * rmt + t]; binMean[d] = a / rmt; } }
            File.WriteAllText("/tmp/f5_ours_binmean.txt", string.Join(" ", binMean.Select(x => x.ToString("F4"))));
        }

        int tRef = (int)refMel.Shape[2];

        // Duration heuristic: target frames = ref_frames * len(target_text) / len(ref_text) / speed.
        int tTarget = (int)Math.Round((double)tRef * targetIds.Length / Math.Max(1, refIds.Length) / opts.Speed);
        if (tTarget < 1) tTarget = 1;
        int tTotal = initialNoise is not null ? (int)initialNoise.Shape[2] : tRef + tTarget;

        // step_cond: [ref_mel, zeros] along time — the masked conditioning held fixed across all steps.
        Tensor condMel = ConcatRefAndZeros(refMel, tTotal - tRef, _cfg.MelDim);
        Tensor x = initialNoise ?? BuildNoise(_cfg.MelDim, tTotal, opts.Seed);

        int[] jointIds = new int[refIds.Length + targetIds.Length];
        Array.Copy(refIds, jointIds, refIds.Length);
        Array.Copy(targetIds, 0, jointIds, refIds.Length, targetIds.Length);

        // Pin the DiT weights VRAM-resident for the whole sample loop (32 forwards for 16 steps × CFG). Without
        // this the CUDA path streams every weight from host each forward — the audio stack's lazy auto-promote
        // loses its headroom gate to the transient pool, so weights never promote (no-op on CPU).
        backend.PreloadWeights(_dit.EnumerateWeights());
        _dit.ResetTextCache();   // text embedding is cached across steps; clear any prior generation's entry
        try
        {
            F5SwaySamplingScheduler sched = new(opts.Steps, opts.SwayCoef);
            for (int step = 0; step < sched.Steps; step++)
            {
                float t = sched.Timesteps[step];
                float dt = sched.Deltas[step];
                Tensor vCond = _dit.Forward(backend, x, condMel, jointIds, t, dropAudioCond: false, dropText: false);
                Tensor vUncond = _dit.Forward(backend, x, condMel, jointIds, t, dropAudioCond: true, dropText: true);
                ApplyCfgAndStep(x, vCond, vUncond, opts.CfgStrength, dt);
                vCond.Dispose();
                vUncond.Dispose();
            }

            Tensor targetMel = SliceTargetPortion(x, tRef, tTotal);
            if (!ReferenceEquals(x, initialNoise)) x.Dispose();
            condMel.Dispose();
            return targetMel;
        }
        finally { backend.FreeWeights(_dit.EnumerateWeights()); }
    }

    private static unsafe Tensor ConcatRefAndZeros(Tensor refMel, int tTarget, int melDim)
    {
        int tRef = (int)refMel.Shape[2];
        int tTotal = tRef + tTarget;
        Tensor outT = new(new TensorShape(1, melDim, tTotal), DType.F32);
        float* op = (float*)outT.DataPointer;
        float* rp = (float*)refMel.DataPointer;
        for (int d = 0; d < melDim; d++)
        {
            for (int t = 0; t < tRef; t++) op[d * tTotal + t] = rp[d * tRef + t];
            for (int t = tRef; t < tTotal; t++) op[d * tTotal + t] = 0f;
        }
        return outT;
    }

    private static unsafe Tensor BuildNoise(int melDim, int tTotal, ulong seed)
    {
        Tensor noise = new(new TensorShape(1, melDim, tTotal), DType.F32);
        float* p = (float*)noise.DataPointer;
        // Mulberry32-based deterministic Gaussian via Box-Muller. Good enough for inference.
        uint state = (uint)(seed ^ (seed >> 32));
        long n = melDim * tTotal;
        for (long i = 0; i < n; i += 2)
        {
            uint a = NextRandom(ref state);
            uint b = NextRandom(ref state);
            double u1 = (a + 1.0) / 4294967296.0;  // (0, 1]
            double u2 = (b + 1.0) / 4294967296.0;
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            p[i] = (float)(r * Math.Cos(theta));
            if (i + 1 < n) p[i + 1] = (float)(r * Math.Sin(theta));
        }
        return noise;
    }

    private static uint NextRandom(ref uint state)
    {
        state += 0x6D2B79F5u;
        uint z = state;
        z = (z ^ (z >> 15)) * (z | 1);
        z ^= z + (z ^ (z >> 7)) * (z | 61);
        return z ^ (z >> 14);
    }

    private static unsafe void ApplyCfgAndStep(Tensor x, Tensor vCond, Tensor vUncond, float cfg, float dt)
    {
        // vCond/vUncond are [1, T, mel_dim] (channels-last); x is [1, mel_dim, T] (channels-first).
        // F5's CFG is cond-anchored: v = pred + (pred - null) * cfg  (NOT the standard uncond + cfg*(cond-uncond)).
        int t = (int)x.Shape[2];
        int melDim = (int)x.Shape[1];
        float* xp = (float*)x.DataPointer;
        float* cp = (float*)vCond.DataPointer;
        float* up = (float*)vUncond.DataPointer;
        for (int tt = 0; tt < t; tt++)
            for (int d = 0; d < melDim; d++)
            {
                float v = cp[tt * melDim + d] + cfg * (cp[tt * melDim + d] - up[tt * melDim + d]);
                xp[d * t + tt] += dt * v;
            }
    }

    private static unsafe Tensor SliceTargetPortion(Tensor x, int tRef, int tTotal)
    {
        int melDim = (int)x.Shape[1];
        int tTarget = tTotal - tRef;
        Tensor outT = new(new TensorShape(1, melDim, tTarget), DType.F32);
        float* op = (float*)outT.DataPointer;
        float* xp = (float*)x.DataPointer;
        for (int d = 0; d < melDim; d++)
            for (int tt = 0; tt < tTarget; tt++)
                op[d * tTarget + tt] = xp[d * tTotal + tRef + tt];
        return outT;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(F5TtsPipeline));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _dit.Dispose();
            _vocos.Dispose();
            _tokenizer.Dispose();
            _ditLoader.Dispose();
            _vocosLoader.Dispose();
        }
    }
}

/// <summary>F5-TTS generation knobs. Default values match the reference repo.</summary>
public sealed record F5TtsOptions
{
    /// <summary>NFE (function evaluations). 32 is the F5-TTS default.</summary>
    public int Steps { get; init; } = 32;

    /// <summary>Sway sampling coefficient. -1.0 default (biases toward noise end).</summary>
    public float SwayCoef { get; init; } = -1.0f;

    /// <summary>CFG strength for the cond-anchored formula <c>v = v_cond + cfg * (v_cond - v_uncond)</c> (see
    /// <see cref="F5TtsPipeline.ApplyCfgAndStep"/>); F5-TTS upstream default is <c>cfg_strength=2.0</c>.</summary>
    public float CfgStrength { get; init; } = 2.0f;

    /// <summary>Speech speed multiplier. 1.0 = natural. Larger = faster (shorter target duration).</summary>
    public float Speed { get; init; } = 1.0f;

    /// <summary>RNG seed for initial noise sample.</summary>
    public ulong Seed { get; init; } = 42;
}
