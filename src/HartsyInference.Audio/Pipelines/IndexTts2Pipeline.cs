using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Codecs;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Audio.Models.LengthRegulation;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Pipelines;

/// <summary>IndexTTS-2 emotion-controllable voice-cloning pipeline, both released checkpoint families
/// (<see cref="IndexTts2Version"/>). Call graph (ground-truthed against the real <c>infer_v2_5.py</c> and
/// <c>infer_v2.py</c>'s <c>infer_generator</c>, read in full):
/// <code>
///   spkCondEmb   = w2vBert(audio16k).hidden[17] normalized          # [1,Tw,1024] — ALSO doubles as the emo
///                                                                   # reference when none is given
///   refMel       = mel(audio22k)                                    # [1,80,Tref], reflect-padded (n_fft-hop)/2
///   style        = CAM++(kaldiFbank(audio16k) - mean)                # [1,192] — S2Mel style in BOTH versions
///   promptCond   = lengthRegulator(spkCondEmb, ylens=Tref)           # [1,Tref,512]
///   emoVec       = merge_emovec(ComputeEmoVec(spkCondEmb), ComputeEmoVec(emoCondEmb), alpha)  # or the explicit
///                  8-dim-vector / QwenEmotion-text lookup path, see ResolveEmotion
///   speakerCond  = 2.5: spk_emb_proj(style)                          # [1,1,1280]
///                  2.0: conformerPerceiver(spkCondEmb)               # [1,32,1280]
///   codes        = gpt.Generate(speakerCond, emoVec, textIds, langId) # AR sample semantic-codec codes
///   S_infer      = 2.5: semanticCodec.Decode(codes)                   # [1,2T,1024] (EnhancedCodec, 2x upsample)
///                  2.0: vq2emb(codes) + gpt_layer(gpt.SecondPass(...))# [1,T,1024] (MaskGCT RepCodec, no resample)
///   cond         = lengthRegulator(S_infer, ylens=len*1.72[*durationFactor])  # [1,Tgen,512]
///   vcTarget     = CFM.Solve(mu=cat([promptCond,cond]), spk=style, cond=zeroPadded(refMel), promptLen=Tref)
///   wav          = bigVgan(vcTarget[:, :, Tref:])                      # 22050 Hz
/// </code>
/// The two versions differ only in the speaker-conditioning path, the semantic codec / S2Mel handoff and the text
/// tokenizer; everything else is shared and byte-for-byte the same code path.</summary>
public sealed class IndexTts2Pipeline : IDisposable
{
    private const int DefaultMaxTextTokensPerSegment = 120;
    private const float TailFadeMs = 20.0f;

    private readonly IndexTts2Config _cfg;
    private readonly Func<string, int[]> _encode;
    private readonly IDisposable? _tokenizerOwner;
    private readonly IndexTts2SemanticFeatures _semanticFeatures;
    private readonly CamPlusSpeakerEncoder _camplus;
    private readonly VocosFactorizedCodec _semanticCodec;
    private readonly IndexTts2T2sDecoder _gpt;
    private readonly InterpolateLengthRegulator _lengthRegulator;
    private readonly IndexTts2Dit _dit;
    private readonly ConditionalCfm _cfm;
    private readonly IndexTts2BigVganGenerator _bigVgan;
    private readonly IndexTts2EmotionVectorLookup? _emoLookup;
    private readonly string? _qwenEmoDir;
    private readonly IBackend? _qwenBackend;
    private readonly object _qwenLock = new();
    private IndexTts2QwenEmotion? _qwenEmotion;
    private readonly MelSpectrogramExtractor _refMelExtractor = new(MelSpectrogramExtractor.IndexTts2RefMelConfig());
    private readonly List<IDisposable> _loaders;
    private readonly List<Tensor> _convertedWeights;
    private int _disposed;

    public string ModelName => _cfg.Version == IndexTts2Version.V2_0 ? "indextts2-2.0" : "indextts2-2.5";

    private IndexTts2Pipeline(IndexTts2Config cfg, Func<string, int[]> encode, IDisposable? tokenizerOwner, IndexTts2SemanticFeatures semanticFeatures,
        CamPlusSpeakerEncoder camplus, VocosFactorizedCodec semanticCodec, IndexTts2T2sDecoder gpt,
        InterpolateLengthRegulator lengthRegulator, IndexTts2Dit dit, IndexTts2BigVganGenerator bigVgan,
        IndexTts2EmotionVectorLookup? emoLookup, string? qwenEmoDir, IBackend? qwenBackend,
        List<IDisposable> loaders, List<Tensor> convertedWeights)
    {
        _cfg = cfg;
        _encode = encode;
        _tokenizerOwner = tokenizerOwner;
        _semanticFeatures = semanticFeatures;
        _camplus = camplus;
        _semanticCodec = semanticCodec;
        _gpt = gpt;
        _lengthRegulator = lengthRegulator;
        _dit = dit;
        _cfm = new ConditionalCfm(dit, cfg.S2MelDit.InChannels);
        _bigVgan = bigVgan;
        _emoLookup = emoLookup;
        _qwenEmoDir = qwenEmoDir;
        _qwenBackend = qwenBackend;
        _loaders = loaders;
        _convertedWeights = convertedWeights;
    }

    /// <summary>Loads IndexTTS-2 from already-downloaded checkpoint files. <paramref name="cfg"/> selects the
    /// version (default <see cref="IndexTts2Config.V2_5"/>); the file arguments then mean: <paramref name="tokenizerPath"/>
    /// = the tiktoken rank file (2.5) or SentencePiece <c>bpe.model</c> (2.0); <paramref name="codecPath"/> = the
    /// bundled <c>codec.pth</c> (2.5) or <c>amphion/MaskGCT</c>'s <c>semantic_codec/model.safetensors</c> (2.0).
    /// <paramref name="feat1Path"/>/
    /// <paramref name="feat2Path"/> (the explicit emotion-vector lookup banks) and <paramref name="qwenEmoDir"/>
    /// (the free-text emotion classifier) are optional — omitting either simply disables that emotion mode
    /// (<see cref="IndexTts2Options.EmoVector"/> / <see cref="IndexTts2Options.UseEmoText"/> throw if used
    /// without the matching data loaded), matching the real reference's own <c>use_qwen_emo</c> opt-in.
    /// <paramref name="qwenBackend"/> is required only when <paramref name="qwenEmoDir"/> is given — the
    /// classifier's own <see cref="TextGenerationPipeline"/> needs a backend at construction time, unlike
    /// every other component here which defers to whatever backend <see cref="Synthesize"/> is called with
    /// (Audio has no dependency on any concrete backend package, so the caller supplies one). The classifier
    /// (a 0.6B LLM, ~2.4 GB once widened to F32) is loaded lazily on the first <see cref="IndexTts2Options.UseEmoText"/>
    /// request, so a pipeline that never uses text emotion pays nothing for it; the directory's files are checked up front.</summary>
    public static async Task<IndexTts2Pipeline> LoadAsync(
        string tokenizerPath, string gptPath, string s2melPath, string codecPath,
        string w2vBertSafetensorsPath, string w2vStatsPath, string campplusPath, string bigVganPath,
        string? feat1Path = null, string? feat2Path = null, string? qwenEmoDir = null, IBackend? qwenBackend = null,
        IndexTts2Config? cfg = null, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        IndexTts2Config resolved = cfg ?? IndexTts2Config.V2_5;

        IDisposable? tokenizerOwner = null;
        Func<string, int[]> encode;
        IndexTts2SemanticFeatures? semanticFeatures = null;
        CamPlusSpeakerEncoder? camplus = null;
        VocosFactorizedCodec? semanticCodec = null;
        IndexTts2T2sDecoder? gpt = null;
        InterpolateLengthRegulator? lengthRegulator = null;
        IndexTts2Dit? dit = null;
        IndexTts2BigVganGenerator? bigVgan = null;
        IndexTts2EmotionVectorLookup? emoLookup = null;
        List<IDisposable> loaders = [];
        List<Tensor> converted = [];
        try
        {
            if (resolved.Version == IndexTts2Version.V2_0)
            {
                IndexTtsTokenizer spm = new(tokenizerPath);
                tokenizerOwner = spm;
                encode = spm.Encode;
            }
            else
            {
                encode = new IndexTts2TiktokenTokenizer(tokenizerPath).Encode;
            }

            PytorchPickleLoader gptLoader = new();
            gptLoader.Load(gptPath, recursiveFlatten: true);
            loaders.Add(gptLoader);
            Dictionary<string, Tensor> gptWeights = IndexTtsPipeline.ToF32(gptLoader.GetAllTensors(), converted);

            gpt = new IndexTts2T2sDecoder(resolved.Gpt, resolved.NumberTextTokens, resolved.MaxMelTokens);
            gpt.LoadWeights(gptWeights);
            IndexTts2SpeakerConditioning expectedMode = resolved.Version == IndexTts2Version.V2_0
                ? IndexTts2SpeakerConditioning.ConformerPerceiver
                : IndexTts2SpeakerConditioning.Campplus;
            if (gpt.Mode != expectedMode)
                throw new InvalidDataException($"gpt.pth is a {gpt.Mode} checkpoint but the config is IndexTTS-{(resolved.Version == IndexTts2Version.V2_0 ? "2.0" : "2.5")} ({expectedMode}) — wrong checkpoint for this version.");

            SafeTensorsLoader w2vLoader = new();
            w2vLoader.Load(w2vBertSafetensorsPath);
            loaders.Add(w2vLoader);
            Dictionary<string, Tensor> w2vWeights = IndexTtsPipeline.ToF32(w2vLoader.GetAllTensors(), converted);

            PytorchPickleLoader statsLoader = new();
            statsLoader.Load(w2vStatsPath, recursiveFlatten: true);
            loaders.Add(statsLoader);
            Dictionary<string, Tensor> statsWeights = statsLoader.GetAllTensors();

            semanticFeatures = new IndexTts2SemanticFeatures();
            semanticFeatures.LoadWeights(w2vWeights, "", statsWeights["mean"], statsWeights["var"]);

            PytorchPickleLoader campplusLoader = new();
            campplusLoader.Load(campplusPath, recursiveFlatten: true);
            loaders.Add(campplusLoader);
            Dictionary<string, Tensor> campplusWeights = IndexTtsPipeline.ToF32(campplusLoader.GetAllTensors(), converted);
            camplus = new CamPlusSpeakerEncoder();
            camplus.LoadWeights(campplusWeights);

            semanticCodec = new VocosFactorizedCodec(resolved.SemanticCodec);
            if (resolved.Version == IndexTts2Version.V2_0)
            {
                // MaskGCT's RepCodec: a plain safetensors with no module prefix; only Quantize + vq2emb are ever
                // called, so the decoder half stays unloaded.
                SafeTensorsLoader codecLoader = new();
                codecLoader.Load(codecPath);
                loaders.Add(codecLoader);
                Dictionary<string, Tensor> codecWeights = IndexTtsPipeline.ToF32(codecLoader.GetAllTensors(), converted);
                semanticCodec.LoadWeights(codecWeights, prefix: "", loadDecoder: false);
            }
            else
            {
                PytorchPickleLoader codecLoader = new();
                codecLoader.Load(codecPath, recursiveFlatten: true);
                loaders.Add(codecLoader);
                Dictionary<string, Tensor> codecWeights = IndexTtsPipeline.ToF32(codecLoader.GetAllTensors(), converted);
                semanticCodec.LoadWeights(codecWeights, prefix: "model", loadDecoder: true);
            }

            lengthRegulator = new InterpolateLengthRegulator(resolved.LengthRegulatorChannels, resolved.LengthRegulatorInChannels, resolved.LengthRegulatorNumStages);

            PytorchPickleLoader s2melLoader = new();
            s2melLoader.Load(s2melPath, recursiveFlatten: true);
            loaders.Add(s2melLoader);
            Dictionary<string, Tensor> s2melWeights = IndexTtsPipeline.ToF32(s2melLoader.GetAllTensors(), converted);
            lengthRegulator.LoadWeights(s2melWeights, "net.length_regulator");
            if (resolved.Version == IndexTts2Version.V2_0) gpt.LoadGptLayer(s2melWeights, "net.gpt_layer");
            dit = new IndexTts2Dit(resolved.S2MelDit);
            dit.LoadWeights(s2melWeights, "net.cfm.estimator");

            PytorchPickleLoader bigVganLoader = new();
            bigVganLoader.Load(bigVganPath, recursiveFlatten: true);
            loaders.Add(bigVganLoader);
            Dictionary<string, Tensor> bigVganWeights = IndexTtsPipeline.ToF32(bigVganLoader.GetAllTensors(), converted);
            bigVgan = new IndexTts2BigVganGenerator(resolved.BigVgan);
            bigVgan.LoadWeights(bigVganWeights, "generator");

            if (feat1Path is not null && feat2Path is not null)
            {
                // The loaders own the tensors' backing storage (an F32 tensor is returned as-is, not copied), so
                // they stay alive for the pipeline's lifetime — disposing them here would free the banks the
                // lookup still reads.
                PytorchPickleLoader feat1Loader = new();
                feat1Loader.Load(feat1Path, recursiveFlatten: true);
                loaders.Add(feat1Loader);
                PytorchPickleLoader feat2Loader = new();
                feat2Loader.Load(feat2Path, recursiveFlatten: true);
                loaders.Add(feat2Loader);
                emoLookup = new IndexTts2EmotionVectorLookup(feat1Loader.GetAllTensors()["data"], feat2Loader.GetAllTensors()["data"]);
            }

            if (qwenEmoDir is not null)
            {
                if (qwenBackend is null)
                    throw new ArgumentException("qwenBackend is required when qwenEmoDir is given.", nameof(qwenBackend));
                foreach (string name in QwenFiles)
                {
                    string path = Path.Combine(qwenEmoDir, name);
                    if (!File.Exists(path)) throw new FileNotFoundException($"QwenEmotion file missing: {path}", path);
                }
            }

            return new IndexTts2Pipeline(resolved, encode, tokenizerOwner, semanticFeatures, camplus, semanticCodec, gpt,
                lengthRegulator, dit, bigVgan, emoLookup, qwenEmoDir, qwenBackend, loaders, converted);
        }
        catch
        {
            foreach (Tensor t in converted) t.Dispose();
            tokenizerOwner?.Dispose();
            emoLookup?.Dispose();
            bigVgan?.Dispose();
            dit?.Dispose();
            gpt?.Dispose();
            camplus?.Dispose();
            semanticCodec?.Dispose();
            semanticFeatures?.Dispose();
            foreach (IDisposable l in loaders) l.Dispose();
            throw;
        }
    }

    /// <summary>Clones the voice in <paramref name="referenceAudioMono"/> speaking <paramref name="text"/> with
    /// the emotion <paramref name="options"/> selects. Returns 22050 Hz mono PCM float samples in [-1, 1].</summary>
    public float[] Synthesize(IBackend backend, string text, float[] referenceAudioMono, int referenceSampleRate, IndexTts2Options? options = null)
    {
        ThrowIfDisposed();
        IndexTts2Options opts = options ?? new IndexTts2Options();

        List<int[]> segments = IndexTtsPipeline.SplitIntoTokenSegments(text, DefaultMaxTextTokensPerSegment, EncodeNormalized);
        if (segments.Count == 0) throw new ArgumentException("Text produced zero tokens after tokenization.", nameof(text));

        float[] audio16k = S3GenReference.Resample(referenceAudioMono, referenceSampleRate, 16_000);
        float[] audio22k = S3GenReference.Resample(referenceAudioMono, referenceSampleRate, 22_050);

        Tensor spkCondEmb = _semanticFeatures.Forward(backend, audio16k);
        int tSpk = (int)spkCondEmb.Shape[1];
        Tensor refMel = ComputeRefMel(audio22k);
        int tRef = (int)refMel.Shape[2];
        Tensor style = S3GenReference.SpeakerEmbedding(backend, _camplus, audio16k);
        Tensor promptCondition = _lengthRegulator.Forward(backend, spkCondEmb, tSpk, tRef);
        // 2.5: spk_emb_proj(CAM++ style). 2.0: Conformer+Perceiver over the w2v-bert feature. CAM++ style still
        // feeds S2Mel in both versions.
        Tensor speakerConditioning = _cfg.Version == IndexTts2Version.V2_0
            ? _gpt.ComputeSpeakerConditioningConformerPerceiver(backend, spkCondEmb, tSpk)
            : _gpt.ComputeSpeakerConditioning(backend, style);

        try
        {
            (Tensor emoVec, float[]? normalizedWeights, bool ownEmoVec) = ResolveEmotion(backend, opts, text, spkCondEmb, tSpk, audio16k, style);
            try
            {
                if (segments.Count == 1)
                    return FadeOutTail(SynthesizeSegment(backend, speakerConditioning, emoVec, segments[0], opts, promptCondition, refMel, tRef, style, opts.Seed), _cfg.SampleRate);

                List<float[]> pieces = new(segments.Count);
                for (int i = 0; i < segments.Count; i++)
                {
                    ulong segSeed = unchecked(opts.Seed + (ulong)i);
                    pieces.Add(SynthesizeSegment(backend, speakerConditioning, emoVec, segments[i], opts, promptCondition, refMel, tRef, style, segSeed));
                }
                int total = 0;
                foreach (float[] p in pieces) total += p.Length;
                float[] joined = new float[total];
                int offset = 0;
                foreach (float[] p in pieces) { p.CopyTo(joined, offset); offset += p.Length; }
                return FadeOutTail(joined, _cfg.SampleRate);
            }
            finally
            {
                if (ownEmoVec) emoVec.Dispose();
            }
        }
        finally
        {
            speakerConditioning.Dispose();
            promptCondition.Dispose();
            style.Dispose();
            refMel.Dispose();
            spkCondEmb.Dispose();
        }
    }

    /// <summary>Real emotion resolution (<c>infer_generator</c>'s own branching, read in full): an explicit
    /// vector or QwenEmotion-text guidance forces the audio emotion-reference off and routes through
    /// <see cref="IndexTts2EmotionVectorLookup"/>, blended with the natural <c>merge_emovec</c> result by the
    /// vector's own (post-normalization) leftover weight (<c>emovec_mat + (1-sum(weights))*emovec</c>);
    /// otherwise an explicit or implicit (defaults to the speaker's own clip, alpha forced to 1) audio
    /// reference goes straight through <c>merge_emovec</c>.</summary>
    private (Tensor EmoVec, float[]? NormalizedWeights, bool Owned) ResolveEmotion(IBackend backend, IndexTts2Options opts, string text,
        Tensor spkCondEmb, int tSpk, float[] speakerAudio16k, Tensor style)
    {
        float[]? emoVector = opts.EmoVector;
        if (opts.UseEmoText)
        {
            emoVector = GetQwenEmotion().Infer(opts.EmoText ?? text);
        }

        if (emoVector is not null)
        {
            // Real infer_generator: an explicit/text vector can't be alpha-mixed with an audio reference, so alpha
            // scales the vector itself (clamped to [0,1], truncated to 4 decimals) instead.
            float scale = Math.Clamp(opts.EmoAlpha, 0f, 1f);
            if (scale != 1f)
            {
                float[] scaled = new float[emoVector.Length];
                for (int i = 0; i < scaled.Length; i++) scaled[i] = (int)(emoVector[i] * scale * 10000f) / 10000f;
                emoVector = scaled;
            }
        }

        Tensor baseVec = _gpt.ComputeEmoVec(backend, spkCondEmb, tSpk);
        try
        {
            if (emoVector is not null)
            {
                if (_emoLookup is null)
                    throw new InvalidOperationException("IndexTts2Options.EmoVector/UseEmoText requires the pipeline to be loaded with feat1/feat2 (pass feat1Path/feat2Path to LoadAsync).");
                uint rngState = DeterministicRng.Seed(unchecked((int)opts.Seed));
                Tensor emovecMat = _emoLookup.ComputeEmoVecMat(emoVector, style, opts.UseRandomEmoExemplar, ref rngState);
                float[] normalized = IndexTts2EmotionVectorLookup.NormalizeEmoVec(emoVector, applyBias: true);
                float leftover = 1f - normalized.Sum();

                // emo_audio_prompt defaults to the speaker's own clip when no external one is given; alpha is
                // forced to 1.0, so merge_emovec(spk, spk, alpha=1) collapses to base_vec itself.
                Tensor naturalEmovec = baseVec;
                Tensor finalVec = AddScaled(emovecMat, naturalEmovec, leftover);
                emovecMat.Dispose();
                return (finalVec, normalized, true);
            }

            Tensor emoCondEmb = opts.EmoAudioReference is { } emoRef
                ? _semanticFeatures.Forward(backend, S3GenReference.Resample(emoRef.Audio, emoRef.SampleRate, 16_000))
                : spkCondEmb;
            float alpha = opts.EmoAudioReference is null ? 1.0f : opts.EmoAlpha;
            int tEmo = (int)emoCondEmb.Shape[1];
            Tensor emoVecFromAudio = _gpt.ComputeEmoVec(backend, emoCondEmb, tEmo);
            if (!ReferenceEquals(emoCondEmb, spkCondEmb)) emoCondEmb.Dispose();

            Tensor merged = HartsyInference.Audio.Models.IndexTts2.IndexTts2T2sDecoder.MergeEmoVec(baseVec, emoVecFromAudio, alpha);
            emoVecFromAudio.Dispose();
            return (merged, null, true);
        }
        finally
        {
            baseVec.Dispose();
        }
    }

    private static readonly string[] QwenFiles = ["config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja"];

    /// <summary>Loads the QwenEmotion classifier on first use (see <see cref="LoadAsync"/>'s remarks).</summary>
    private IndexTts2QwenEmotion GetQwenEmotion()
    {
        if (_qwenEmoDir is null || _qwenBackend is null)
            throw new InvalidOperationException("IndexTts2Options.UseEmoText requires the pipeline to be loaded with a QwenEmotion classifier (pass qwenEmoDir to LoadAsync).");
        lock (_qwenLock)
        {
            if (_qwenEmotion is not null) return _qwenEmotion;

            using System.Text.Json.JsonDocument configDoc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(_qwenEmoDir, "config.json")));
            TransformerConfig qwenCfg = Qwen3HfConfigReader.FromHuggingFace(configDoc.RootElement);

            SafeTensorsLoader qwenLoader = new();
            qwenLoader.Load(Path.Combine(_qwenEmoDir, "model.safetensors"));
            _loaders.Add(qwenLoader);
            Dictionary<string, Tensor> qwenWeights = IndexTtsPipeline.ToF32(qwenLoader.GetAllTensors(), _convertedWeights);

            GenericTransformer transformer = new(qwenCfg);
            transformer.LoadWeights(qwenWeights, "model");
            _loaders.Add(transformer);

            using FileStream tokenizerStream = File.OpenRead(Path.Combine(_qwenEmoDir, "tokenizer.json"));
            GgufTokenizer qwenTokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerStream);
            JinjaChatTemplate template = new(File.ReadAllText(Path.Combine(_qwenEmoDir, "chat_template.jinja")));

            TextGenerationPipeline textPipeline = new(transformer, qwenTokenizer, _qwenBackend, template);
            return _qwenEmotion = new IndexTts2QwenEmotion(textPipeline);
        }
    }

    /// <summary><c>a + scale*b</c>, elementwise, both <c>[1, hidden]</c>.</summary>
    private static unsafe Tensor AddScaled(Tensor a, Tensor b, float scale)
    {
        Tensor result = new(a.Shape, DType.F32);
        float* ap = (float*)a.DataPointer, bp = (float*)b.DataPointer, rp = (float*)result.DataPointer;
        long n = a.ElementCount;
        for (long i = 0; i < n; i++) rp[i] = ap[i] + scale * bp[i];
        return result;
    }

    private unsafe float[] SynthesizeSegment(IBackend backend, Tensor speakerConditioning, Tensor emoVec, int[] textIds,
        IndexTts2Options opts, Tensor promptCondition, Tensor refMel, int tRef, Tensor style, ulong seed)
    {
        uint rngState = DeterministicRng.Seed(unchecked((int)seed));
        bool v20 = _cfg.Version == IndexTts2Version.V2_0;
        int? langId = v20 ? null : IndexTts2TiktokenTokenizer.LangToToken(opts.Language);
        int maxMel = opts.MaxMelTokens ?? 1_500;
        IndexTtsOptions gptOpts = new()
        {
            Temperature = opts.Temperature,
            TopK = opts.TopK,
            TopP = opts.TopP,
            RepetitionPenalty = opts.RepetitionPenalty,
            MaxMelTokens = maxMel,
        };
        int[] codes = _gpt.Generate(backend, speakerConditioning, emoVec, textIds, langId, gptOpts, ref rngState);

        Tensor sInfer;
        if (v20)
        {
            // infer_v2.py: S_infer = vq2emb(codes) + gpt_layer(second-pass latent), target = (code_lens * 1.72).long().
            using Tensor latent = _gpt.ComputeSecondPassLatent(backend, speakerConditioning, emoVec, textIds, codes);
            sInfer = _semanticCodec.VqToEmbedding(backend, codes);
            AddInPlace(sInfer, latent);
        }
        else
        {
            sInfer = _semanticCodec.Decode(backend, codes);
        }
        int tSInfer = (int)sInfer.Shape[1];
        int targetLen = Math.Max(1, (int)(tSInfer * _cfg.ContentLengthRatio * opts.DurationFactor));
        Tensor cond = _lengthRegulator.Forward(backend, sInfer, tSInfer, targetLen);
        sInfer.Dispose();

        int catLen = tRef + targetLen;
        Tensor mu = ConcatTime(promptCondition, cond, tRef, targetLen, _cfg.LengthRegulatorChannels);
        cond.Dispose();

        Tensor promptX = new(new TensorShape(1, _cfg.S2MelDit.InChannels, catLen), DType.F32);
        CopyRefMelPrefix(promptX, refMel, tRef, catLen);

        Tensor vcTarget = _cfm.Solve(backend, mu, style, promptX, _cfg.DiffusionSteps, _cfg.InferenceCfgRate, unchecked((int)seed), promptLen: tRef);
        mu.Dispose();
        promptX.Dispose();

        Tensor genOnly = SliceTime(vcTarget, tRef, targetLen);
        vcTarget.Dispose();

        Tensor wave = _bigVgan.Forward(backend, genOnly, targetLen);
        genOnly.Dispose();

        int n = (int)wave.ElementCount;
        float[] pcm = new float[n];
        float* wp = (float*)wave.DataPointer;
        for (int i = 0; i < n; i++) pcm[i] = Math.Clamp(wp[i], -1f, 1f);
        wave.Dispose();
        return pcm;
    }

    /// <summary><c>a += b</c>, elementwise, same shape.</summary>
    private static unsafe void AddInPlace(Tensor a, Tensor b)
    {
        float* ap = (float*)a.DataPointer, bp = (float*)b.DataPointer;
        long n = a.ElementCount;
        for (long i = 0; i < n; i++) ap[i] += bp[i];
    }

    private static unsafe void CopyRefMelPrefix(Tensor promptX, Tensor refMel, int tRef, int catLen)
    {
        int ch = (int)refMel.Shape[1];
        float* dp = (float*)promptX.DataPointer;
        float* sp = (float*)refMel.DataPointer;
        for (int c = 0; c < ch; c++)
            Buffer.MemoryCopy(sp + (long)c * tRef, dp + (long)c * catLen, (long)tRef * 4, (long)tRef * 4);
    }

    private static unsafe Tensor ConcatTime(Tensor a, Tensor b, int aLen, int bLen, int channels)
    {
        Tensor result = new(new TensorShape(1, aLen + bLen, channels), DType.F32);
        float* ap = (float*)a.DataPointer, bp = (float*)b.DataPointer, rp = (float*)result.DataPointer;
        Buffer.MemoryCopy(ap, rp, (long)aLen * channels * 4, (long)aLen * channels * 4);
        Buffer.MemoryCopy(bp, rp + (long)aLen * channels, (long)bLen * channels * 4, (long)bLen * channels * 4);
        return result;
    }

    private static unsafe Tensor SliceTime(Tensor x, int startFrame, int frameCount)
    {
        int channels = (int)x.Shape[1], totalT = (int)x.Shape[2];
        Tensor result = new(new TensorShape(1, channels, frameCount), DType.F32);
        float* sp = (float*)x.DataPointer, dp = (float*)result.DataPointer;
        for (int c = 0; c < channels; c++)
            Buffer.MemoryCopy(sp + (long)c * totalT + startFrame, dp + (long)c * frameCount, (long)frameCount * 4, (long)frameCount * 4);
        return result;
    }

    /// <summary>Ramps the final <paramref name="fadeMs"/> down with a raised cosine — identical logic to
    /// <see cref="IndexTtsPipeline"/>'s own fade, duplicated rather than shared since it's a tiny (~10 line),
    /// self-contained DSP helper and the two pipelines otherwise share nothing else at this layer.</summary>
    private static float[] FadeOutTail(float[] pcm, int sampleRate, float fadeMs = TailFadeMs)
    {
        if (pcm.Length == 0 || fadeMs <= 0f) return pcm;
        int n = Math.Min((int)(sampleRate * fadeMs / 1000.0), pcm.Length);
        if (n <= 1) return pcm;
        int start = pcm.Length - n;
        for (int i = 0; i < n; i++)
        {
            float ramp = 0.5f * (1f + MathF.Cos(i / (float)(n - 1) * MathF.PI));
            pcm[start + i] *= ramp;
        }
        return pcm;
    }

    /// <summary>Real <c>mel_fn(audio_22k)</c>: reflect-pad by <c>(n_fft-hop)/2</c>, then the IndexTTS-2 mel
    /// preset (center=False). Returns channel-first <c>[1, 80, T]</c>.</summary>
    private unsafe Tensor ComputeRefMel(float[] audio22k)
    {
        MelSpectrogramExtractor.Config mcfg = _refMelExtractor.Configuration;
        int pad = (mcfg.NFft - mcfg.HopLength) / 2;
        float[] padded = pad > 0 && audio22k.Length > pad ? SignalPadding.Reflect(audio22k, pad) : audio22k;
        float[,] mel = _refMelExtractor.Compute(padded);
        int nMels = mel.GetLength(0), t = mel.GetLength(1);
        Tensor outT = new(new TensorShape(1, nMels, t), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int m = 0; m < nMels; m++)
            for (int f = 0; f < t; f++)
                op[(long)m * t + f] = mel[m, f];
        return outT;
    }

    private int[] EncodeNormalized(string segment) => _encode(IndexTtsTextNormalizer.InjectCjkBoundaries(segment));

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _semanticFeatures.EnumerateWeights()) yield return t;
        foreach (Tensor t in _camplus.EnumerateWeights()) yield return t;
        foreach (Tensor t in _semanticCodec.EnumerateWeights()) yield return t;
        foreach (Tensor t in _gpt.EnumerateWeights()) yield return t;
        foreach (Tensor t in _lengthRegulator.EnumerateWeights()) yield return t;
        foreach (Tensor t in _dit.EnumerateWeights()) yield return t;
        foreach (Tensor t in _bigVgan.EnumerateWeights()) yield return t;
        if (_emoLookup is not null) foreach (Tensor t in _emoLookup.EnumerateWeights()) yield return t;
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(IndexTts2Pipeline));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _semanticFeatures.Dispose();
        _camplus.Dispose();
        _semanticCodec.Dispose();
        _gpt.Dispose();
        _dit.Dispose();
        _bigVgan.Dispose();
        _tokenizerOwner?.Dispose();
        _emoLookup?.Dispose();
        foreach (IDisposable l in _loaders) l.Dispose();
        foreach (Tensor t in _convertedWeights) t.Dispose();
        GC.SuppressFinalize(this);
    }
}
