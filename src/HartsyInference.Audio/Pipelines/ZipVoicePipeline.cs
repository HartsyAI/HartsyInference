using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Vocoders;
using HartsyInference.Audio.Models.ZipVoice;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Audio.Phonemizer.Espeak;

namespace HartsyInference.Audio.Pipelines;

/// <summary>ZipVoice (k2-fsa/ZipVoice) zero-shot voice-clone pipeline: flow-matching Euler sampling under a
/// <c>ZipformerModel</c> (fm_decoder + text_encoder), vocoded via the shared <see cref="Vocos"/> port
/// (same <c>vocos-mel-24khz</c> family F5-TTS already uses in this engine). English-only (see
/// <see cref="ZipVoiceTokenizer"/>); does not implement upstream's long-text chunking — very long text may
/// exceed the model's effectively-trained span.
///
/// <para>Pipeline shape (matches <c>ZipVoice.sample</c> + <c>infer_zipvoice.py:generate_sentence</c>):
/// <code>
/// promptMel   = mel_extract(ref_wav) * feat_scale             # [1, T_ref, 100], channels-last
/// jointTokens = tokenize(ref_text) + tokenize(target_text)
/// targetT     = ceil(T_ref / len(promptTokens) * len(targetTokens) / speed)
/// totalT      = T_ref + targetT
/// textHidden  = text_encoder(embed(jointTokens + [pad]))
/// textCond    = gather(textHidden, avg_upsample_index(len(jointTokens), totalT))
/// speechCond  = [promptMel, zeros(targetT)] with everything past T_ref zeroed
/// x           = randn(1, totalT, 100)
/// for (t, dt) in euler_schedule(steps, t_shift):
///   vCond   = fm_decoder([x, textCond, speechCond], t)
///   vUncond = fm_decoder([x, 0, speechCond_or_0], t)            # CFG, t-dependent branch
///   x      += dt * ((1+g)*vCond - g*vUncond)
/// mel = x[:, T_ref:, :] / feat_scale                             # generated region only
/// audio = vocos(mel.transpose)
/// </code></para></summary>
public sealed class ZipVoicePipeline : IAudioPipeline, IDisposable
{
    private readonly ZipVoiceConfig _cfg;
    private readonly ZipVoiceModel _model;
    private readonly Vocos _vocos;
    private readonly ZipVoiceTokenizer _tokenizer;
    private readonly EspeakPhonemizer _phonemizer;
    private readonly SafeTensorsLoader _modelLoader;
    private readonly SafeTensorsLoader _vocosLoader;
    private int _disposed;

    public string ModelName => "k2-fsa/ZipVoice";

    private ZipVoicePipeline(ZipVoiceConfig cfg, ZipVoiceModel model, Vocos vocos, ZipVoiceTokenizer tokenizer,
        EspeakPhonemizer phonemizer, SafeTensorsLoader modelLoader, SafeTensorsLoader vocosLoader)
    {
        _cfg = cfg; _model = model; _vocos = vocos; _tokenizer = tokenizer; _phonemizer = phonemizer;
        _modelLoader = modelLoader; _vocosLoader = vocosLoader;
    }

    public static async Task<ZipVoicePipeline> LoadAsync(ZipVoiceConfig? cfg = null, CancellationToken ct = default)
    {
        ZipVoiceConfig resolved = cfg ?? ZipVoiceConfig.Default;

        Task<string> modelPath = AudioModelCache.GetAsync("k2-fsa/ZipVoice", "zipvoice/model.safetensors", ct: ct);
        Task<string> tokensPath = AudioModelCache.GetAsync("k2-fsa/ZipVoice", "zipvoice/tokens.txt", ct: ct);
        Task<string> vocosPath = AudioModelCache.GetAsync("lucasnewman/vocos-mel-24khz", "model.safetensors", ct: ct);
        await Task.WhenAll(modelPath, tokensPath, vocosPath).ConfigureAwait(false);

        SafeTensorsLoader modelLoader = new();
        modelLoader.Load(modelPath.Result);
        ZipVoiceModel model = new(resolved);
        model.LoadWeights(modelLoader.GetAllTensors());

        SafeTensorsLoader vocosLoader = new();
        vocosLoader.Load(vocosPath.Result);
        Vocos vocos = new(VocosConfig.Mel24k);
        vocos.LoadWeights(vocosLoader.GetAllTensors());

        ZipVoiceTokenizer tokenizer = ZipVoiceTokenizer.FromFile(tokensPath.Result);
        EspeakPhonemizer phonemizer = EspeakPhonemizer.FromCache("en");

        return new ZipVoicePipeline(resolved, model, vocos, tokenizer, phonemizer, modelLoader, vocosLoader);
    }

    /// <summary>Clones the voice in <paramref name="refAudio"/> (any sample rate, resampled to 24 kHz) reading
    /// <paramref name="refText"/>, to speak <paramref name="targetText"/>.</summary>
    public float[] GenerateFromAudio(IBackend backend, ReadOnlySpan<float> refAudio, int sampleRate,
        string refText, string targetText, ZipVoiceOptions? options = null)
    {
        ThrowIfDisposed();
        ZipVoiceOptions opts = options ?? new ZipVoiceOptions();

        MelSpectrogramExtractor.Config melCfg = MelSpectrogramExtractor.F5VocosConfig();
        int sr = melCfg.SampleRate;
        float[] mono24k = sampleRate == sr ? refAudio.ToArray() : Resampler.Create(sampleRate, sr).Resample(refAudio);

        const float targetRms = 0.1f;
        float promptRms = Rms(mono24k);
        if (promptRms > 1e-6f && promptRms < targetRms)
        {
            float g = targetRms / promptRms;
            for (int i = 0; i < mono24k.Length; i++) mono24k[i] *= g;
        }
        mono24k = TrimEdgeSilence(mono24k, sr);

        MelSpectrogramExtractor extractor = new(melCfg);
        float[,] melData = extractor.Compute(mono24k);   // [nMels, T], natural-log
        int tRef = melData.GetLength(1);
        Tensor promptMel = ToChannelsLastScaled(melData, tRef, _cfg.FeatDim, _cfg.FeatScale);

        int[] promptTokens = _tokenizer.Encode(refText, _phonemizer);
        int[] targetTokens = _tokenizer.Encode(targetText, _phonemizer);
        if (promptTokens.Length == 0 || targetTokens.Length == 0)
            throw new InvalidOperationException("ZipVoice: reference or target text phonemized to zero tokens.");

        int[] joint = new int[promptTokens.Length + targetTokens.Length];
        Array.Copy(promptTokens, joint, promptTokens.Length);
        Array.Copy(targetTokens, 0, joint, promptTokens.Length, targetTokens.Length);

        int targetFrames = ZipVoiceModel.PredictTargetFrames(tRef, promptTokens.Length, targetTokens.Length, opts.Speed);
        int totalFrames = tRef + targetFrames;

        Tensor textHidden = _model.EmbedAndEncodeText(backend, joint, _tokenizer.PadId);
        int[] frameToTokenIndex = ZipVoiceModel.BuildFrameToTokenIndex(joint.Length, totalFrames);
        Tensor textCondition = _model.GatherTextCondition(textHidden, frameToTokenIndex);
        textHidden.Dispose();

        if (Environment.GetEnvironmentVariable("HARTSY_ZIPVOICE_DEBUG") == "1")
        {
            int avg = joint.Length > 0 ? totalFrames / joint.Length : 0;
            int impliedPromptBoundary = avg * promptTokens.Length;
            HartsyInference.Core.Logging.Logs.Info(
                $"[ZipVoice debug] tRef={tRef} promptTokens.Length={promptTokens.Length} targetTokens.Length={targetTokens.Length} "
                + $"joint.Length={joint.Length} targetFrames={targetFrames} totalFrames={totalFrames} avg={avg} "
                + $"impliedPromptBoundary(avg*promptTokenCount)={impliedPromptBoundary} delta(impliedPromptBoundary-tRef)={impliedPromptBoundary - tRef}");
            HartsyInference.Core.Logging.Logs.Info(
                $"[ZipVoice debug] refText phonemized ipa=\"{_phonemizer.PhonemizeToIpa(ZipVoiceTokenizer.NormalizePunctuation(refText), "en-us", preservePunctuation: true)}\"");
            HartsyInference.Core.Logging.Logs.Info(
                $"[ZipVoice debug] targetText phonemized ipa=\"{_phonemizer.PhonemizeToIpa(ZipVoiceTokenizer.NormalizePunctuation(targetText), "en-us", preservePunctuation: true)}\"");
            // Where does frameToTokenIndex cross from a prompt-token index (< promptTokens.Length) to a
            // target-token index (>= promptTokens.Length)? Compare against the real audio boundary tRef.
            int firstTargetTokenFrame = Array.FindIndex(frameToTokenIndex, idx => idx >= promptTokens.Length);
            HartsyInference.Core.Logging.Logs.Info(
                $"[ZipVoice debug] firstFrameWithTargetTokenConditioning={firstTargetTokenFrame} (vs real tRef={tRef}); "
                + $"frames [tRef, firstFrameWithTargetTokenConditioning) still get PROMPT-token text conditioning "
                + $"despite speechCondition being zero there = {Math.Max(0, firstTargetTokenFrame - tRef)} frames of drift");
        }

        Tensor speechCondition = BuildSpeechCondition(promptMel, tRef, totalFrames, _cfg.FeatDim);
        promptMel.Dispose();

        backend.PreloadWeights(_model.EnumerateWeights());
        Tensor x1;
        try
        {
            x1 = RunEulerSampling(backend, textCondition, speechCondition, totalFrames, opts);
        }
        finally { backend.FreeWeights(_model.EnumerateWeights()); }
        textCondition.Dispose();
        speechCondition.Dispose();

        Tensor generated = SliceFrames(x1, tRef, targetFrames, _cfg.FeatDim, 1.0f / _cfg.FeatScale);
        x1.Dispose();

        Tensor melChannelsFirst = new(new TensorShape(1, _cfg.FeatDim, targetFrames), DType.F32);
        backend.Transpose2D(melChannelsFirst, generated, targetFrames, _cfg.FeatDim);
        generated.Dispose();

        float[] audio = _vocos.Forward(backend, melChannelsFirst);
        melChannelsFirst.Dispose();

        if (promptRms > 1e-6f && promptRms < targetRms)
        {
            float g = promptRms / targetRms;
            for (int i = 0; i < audio.Length; i++) audio[i] *= g;
        }
        return audio;
    }

    private Tensor RunEulerSampling(IBackend backend, Tensor textCondition, Tensor speechCondition, int totalFrames, ZipVoiceOptions opts)
    {
        int featDim = _cfg.FeatDim;
        Tensor x = BuildNoise(featDim, totalFrames, opts.Seed);
        Tensor zeroText = new(new TensorShape(1, totalFrames, featDim), DType.F32);   // zero-initialized
        Tensor zeroSpeech = new(new TensorShape(1, totalFrames, featDim), DType.F32);

        float[] timesteps = GetTimeSteps(opts.Steps, opts.TShift);
        for (int step = 0; step < opts.Steps; step++)
        {
            float t = timesteps[step];
            float dt = timesteps[step + 1] - timesteps[step];

            Tensor vCond = _model.ForwardVelocity(backend, x, textCondition, speechCondition, totalFrames, t);
            Tensor xNext;
            if (opts.GuidanceScale != 0f)
            {
                bool speechUncondZeroed = t > 0.5f;
                Tensor speechForUncond = speechUncondZeroed ? zeroSpeech : speechCondition;
                float g = speechUncondZeroed ? opts.GuidanceScale : opts.GuidanceScale * 2f;
                Tensor vUncond = _model.ForwardVelocity(backend, x, zeroText, speechForUncond, totalFrames, t);
                xNext = ApplyCfgAndStep(backend, x, vCond, vUncond, g, dt);
                vUncond.Dispose();
            }
            else
            {
                xNext = StepDevice(backend, x, vCond, dt);
            }
            vCond.Dispose();
            x.Dispose();
            x = xNext;
        }
        zeroText.Dispose();
        zeroSpeech.Dispose();
        return x;
    }

    /// <summary><c>timesteps = t_shift·linspace(0,1,steps+1) / (1 + (t_shift-1)·linspace(0,1,steps+1))</c>.</summary>
    private static float[] GetTimeSteps(int steps, float tShift)
    {
        float[] ts = new float[steps + 1];
        for (int i = 0; i <= steps; i++)
        {
            float lin = (float)i / steps;
            ts[i] = tShift * lin / (1f + (tShift - 1f) * lin);
        }
        return ts;
    }

    /// <summary><c>x + dt·((1+g)·vCond − g·vUncond)</c>, fully GPU-resident (chained <see cref="IBackend.Scale"/>/
    /// <see cref="IBackend.Add"/>) — was a host <c>DataPointer</c> loop that read/wrote the loop-carried latent
    /// every one of the 16 sampling steps, forcing a device↔host round trip right before the next step's
    /// backbone forward pass needed <c>x</c> back on-device.</summary>
    private static Tensor ApplyCfgAndStep(IBackend backend, Tensor x, Tensor vCond, Tensor vUncond, float g, float dt)
    {
        Tensor scaledCond = new(vCond.Shape, DType.F32);
        backend.Scale(scaledCond, vCond, 1f + g);
        Tensor scaledUncond = new(vUncond.Shape, DType.F32);
        backend.Scale(scaledUncond, vUncond, -g);
        Tensor combined = new(x.Shape, DType.F32);
        backend.Add(combined, scaledCond, scaledUncond);
        scaledCond.Dispose(); scaledUncond.Dispose();

        Tensor scaledStep = new(x.Shape, DType.F32);
        backend.Scale(scaledStep, combined, dt);
        combined.Dispose();
        Tensor xNext = new(x.Shape, DType.F32);
        backend.Add(xNext, x, scaledStep);
        scaledStep.Dispose();
        return xNext;
    }

    /// <summary><c>x + dt·v</c>, GPU-resident — the no-CFG counterpart of <see cref="ApplyCfgAndStep"/>.</summary>
    private static Tensor StepDevice(IBackend backend, Tensor x, Tensor v, float dt)
    {
        Tensor scaledV = new(x.Shape, DType.F32);
        backend.Scale(scaledV, v, dt);
        Tensor xNext = new(x.Shape, DType.F32);
        backend.Add(xNext, x, scaledV);
        scaledV.Dispose();
        return xNext;
    }

    private static unsafe Tensor ToChannelsLastScaled(float[,] melData, int t, int dim, float scale)
    {
        Tensor output = new(new TensorShape(1, t, dim), DType.F32);
        float* op = (float*)output.DataPointer;
        for (int d = 0; d < dim; d++)
            for (int tt = 0; tt < t; tt++)
                op[(long)tt * dim + d] = melData[d, tt] * scale;
        return output;
    }

    /// <summary>Zero-pads the reference mel to <paramref name="totalFrames"/>; the prompt-length prefix keeps
    /// the reference mel verbatim, everything from <paramref name="tRef"/> onward is zero (the region to be
    /// generated).</summary>
    private static unsafe Tensor BuildSpeechCondition(Tensor promptMel, int tRef, int totalFrames, int dim)
    {
        Tensor output = new(new TensorShape(1, totalFrames, dim), DType.F32);
        float* op = (float*)output.DataPointer;
        float* pp = (float*)promptMel.DataPointer;
        long promptBytes = (long)tRef * dim * sizeof(float);
        Buffer.MemoryCopy(pp, op, promptBytes, promptBytes);
        // Remaining [tRef, totalFrames) is already zero from the Tensor allocator.
        return output;
    }

    private static unsafe Tensor SliceFrames(Tensor x, int start, int count, int dim, float scale)
    {
        Tensor output = new(new TensorShape(1, count, dim), DType.F32);
        float* op = (float*)output.DataPointer;
        float* xp = (float*)x.DataPointer;
        long n = (long)count * dim;
        long srcOff = (long)start * dim;
        for (long i = 0; i < n; i++) op[i] = xp[srcOff + i] * scale;
        return output;
    }

    private static unsafe Tensor BuildNoise(int dim, int t, ulong seed)
    {
        Tensor noise = new(new TensorShape(1, t, dim), DType.F32);
        float* p = (float*)noise.DataPointer;
        uint state = (uint)(seed ^ (seed >> 32));
        long n = (long)t * dim;
        for (long i = 0; i < n; i += 2)
        {
            uint a = NextRandom(ref state);
            uint b = NextRandom(ref state);
            double u1 = (a + 1.0) / 4294967296.0;
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

    private static float Rms(ReadOnlySpan<float> x)
    {
        double sumSq = 0.0;
        for (int i = 0; i < x.Length; i++) sumSq += (double)x[i] * x[i];
        return x.Length > 0 ? (float)Math.Sqrt(sumSq / x.Length) : 0f;
    }

    private static float[] TrimEdgeSilence(float[] x, int sr)
    {
        int win = Math.Max(1, sr / 50);
        int nWin = x.Length / win;
        if (nWin < 4) return x;
        float[] rms = new float[nWin];
        float maxRms = 0f;
        for (int wi = 0; wi < nWin; wi++)
        {
            double s = 0; int off = wi * win;
            for (int i = 0; i < win; i++) { float v = x[off + i]; s += (double)v * v; }
            rms[wi] = (float)Math.Sqrt(s / win);
            if (rms[wi] > maxRms) maxRms = rms[wi];
        }
        if (maxRms < 1e-4f) return x;
        float thresh = 0.08f * maxRms;
        int first = 0; while (first < nWin && rms[first] < thresh) first++;
        int last = nWin - 1; while (last > first && rms[last] < thresh) last--;
        if (first == 0 && last == nWin - 1) return x;
        int margin = sr / 20;
        int start = Math.Max(0, first * win - margin);
        int end = Math.Min(x.Length, (last + 1) * win + margin);
        int len = end - start;
        if (len <= 0 || len >= x.Length) return x;
        float[] outp = new float[len];
        Array.Copy(x, start, outp, 0, len);
        return outp;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(ZipVoicePipeline));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _model.Dispose();
            _vocos.Dispose();
            _modelLoader.Dispose();
            _vocosLoader.Dispose();
        }
    }
}

/// <summary>ZipVoice generation knobs. Defaults match the base (non-distilled) <c>zipvoice</c> checkpoint's
/// CLI defaults (<c>zipvoice_distill</c> would use Steps=8, GuidanceScale=3.0 — not exposed here since only
/// the base checkpoint is wired up).</summary>
public sealed record ZipVoiceOptions
{
    public int Steps { get; init; } = ZipVoiceConfig.BaseSampling.NumSteps;
    public float GuidanceScale { get; init; } = ZipVoiceConfig.BaseSampling.GuidanceScale;
    public float TShift { get; init; } = ZipVoiceConfig.BaseSampling.TShift;
    public float Speed { get; init; } = 1.0f;
    public ulong Seed { get; init; } = 42;
}
