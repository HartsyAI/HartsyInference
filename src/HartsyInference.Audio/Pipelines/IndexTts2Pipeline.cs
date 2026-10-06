using System.Diagnostics;
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
    private readonly IndexTtsTokenizer? _spm;
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
    private readonly object _ownedLock = new();
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
        _spm = tokenizerOwner as IndexTtsTokenizer;
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

    private const double MaxReferenceSeconds = 15.0;

    /// <summary>Zero crossings of the near-ideal resampler used where the reference's own resampling is librosa's
    /// <c>soxr_hq</c> (any rate → 22.05 kHz / 16 kHz straight from the file).</summary>
    private const int PrecisionFilterWidth = 64;

    /// <summary>Reproduces the reference's clip resampling chain: <c>librosa.load</c> brings the file to 22.05 kHz
    /// (soxr high quality — approximated by a long sinc), and the 16 kHz copy is
    /// <c>torchaudio.transforms.Resample(22050, 16000)</c> OF THAT 22.05 kHz signal (torchaudio defaults, reproduced
    /// exactly by <see cref="SincResampler"/>).</summary>
    private static (float[] Audio16k, float[] Audio22k) ResampleReference(ReadOnlySpan<float> clip, int sampleRate)
    {
        float[] audio22k = sampleRate == 22_050 ? clip.ToArray() : SincResampler.Resample(clip, sampleRate, 22_050, PrecisionFilterWidth);
        float[] audio16k = SincResampler.Resample(audio22k, 22_050, 16_000);
        return (audio16k, audio22k);
    }

    /// <summary>Runs the reference clip through every front-end stage that depends on nothing else and returns the
    /// reusable <see cref="IndexTts2Reference"/> (the reference's <c>cache_*</c> fields). Like the reference, only the
    /// first 15 seconds of the clip are used.</summary>
    public IndexTts2Reference PrepareReference(IBackend backend, float[] referenceAudioMono, int referenceSampleRate, IndexTts2Timings? timings = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(referenceAudioMono);
        if (referenceAudioMono.Length == 0) throw new ArgumentException("The reference clip is empty.", nameof(referenceAudioMono));
        Stopwatch sw = Stopwatch.StartNew();

        int cap = (int)Math.Min(int.MaxValue, MaxReferenceSeconds * referenceSampleRate);
        ReadOnlySpan<float> clip = referenceAudioMono.AsSpan(0, Math.Min(cap, referenceAudioMono.Length));
        (float[] audio16k, float[] audio22k) = ResampleReference(clip, referenceSampleRate);

        Tensor? spkCondEmb = null, refMel = null, style = null, promptCondition = null, speakerConditioning = null, baseEmoVec = null;
        try
        {
            spkCondEmb = _semanticFeatures.Forward(backend, audio16k);
            int tSpk = (int)spkCondEmb.Shape[1];
            refMel = ComputeRefMel(audio22k);
            int tRef = (int)refMel.Shape[2];
            style = S3GenReference.SpeakerEmbedding(backend, _camplus, audio16k);
            promptCondition = ComputePromptCondition(backend, spkCondEmb, tSpk, tRef);
            // 2.5: spk_emb_proj(CAM++ style). 2.0: Conformer+Perceiver over the w2v-bert feature. CAM++ style still
            // feeds S2Mel in both versions.
            speakerConditioning = _cfg.Version == IndexTts2Version.V2_0
                ? _gpt.ComputeSpeakerConditioningConformerPerceiver(backend, spkCondEmb, tSpk)
                : _gpt.ComputeSpeakerConditioning(backend, style);
            baseEmoVec = _gpt.ComputeEmoVec(backend, spkCondEmb, tSpk);

            IndexTts2Reference reference = new(this, spkCondEmb, tSpk, refMel, tRef, style, promptCondition, speakerConditioning, baseEmoVec, audio16k);
            if (timings is not null) timings.ReferenceMs += sw.Elapsed.TotalMilliseconds;
            return reference;
        }
        catch
        {
            spkCondEmb?.Dispose();
            refMel?.Dispose();
            style?.Dispose();
            promptCondition?.Dispose();
            speakerConditioning?.Dispose();
            baseEmoVec?.Dispose();
            throw;
        }
    }

    /// <summary>The S2Mel prompt condition, <c>length_regulator(content, ylens=Tref)</c>. The two checkpoints feed it
    /// different content: 2.0 (<c>infer_v2.py</c>) the semantic codec's quantized embedding of the reference
    /// (<c>_, S_ref = semantic_codec.quantize(spk_cond_emb)</c>), 2.5 (<c>infer_v2_5.py</c>, where <c>S_ref</c> is
    /// commented out) the w2v-bert feature itself.</summary>
    private Tensor ComputePromptCondition(IBackend backend, Tensor spkCondEmb, int tSpk, int tRef)
    {
        if (_cfg.Version != IndexTts2Version.V2_0) return _lengthRegulator.Forward(backend, spkCondEmb, tSpk, tRef);
        (int[] _, Tensor sRef) = _semanticCodec.Quantize(backend, spkCondEmb, tSpk);
        using (sRef) return _lengthRegulator.Forward(backend, sRef, (int)sRef.Shape[1], tRef);
    }

    /// <summary>Clones the voice in <paramref name="referenceAudioMono"/> speaking <paramref name="text"/> with
    /// the emotion <paramref name="options"/> selects. Returns 22050 Hz mono PCM float samples in [-1, 1].</summary>
    public float[] Synthesize(IBackend backend, string text, float[] referenceAudioMono, int referenceSampleRate, IndexTts2Options? options = null)
    {
        using IndexTts2Reference reference = PrepareReference(backend, referenceAudioMono, referenceSampleRate, options?.Timings);
        return Synthesize(backend, text, reference, options);
    }

    /// <summary>Synthesizes <paramref name="text"/> against an already-prepared reference (see
    /// <see cref="PrepareReference"/>) — the fast path for repeated generations with one voice.</summary>
    public float[] Synthesize(IBackend backend, string text, IndexTts2Reference reference, IndexTts2Options? options = null)
    {
        List<float[]> chunks = [];
        int total = 0;
        foreach (float[] chunk in SynthesizeStream(backend, text, reference, options))
        {
            chunks.Add(chunk);
            total += chunk.Length;
        }
        float[] joined = new float[total];
        int offset = 0;
        foreach (float[] c in chunks) { c.CopyTo(joined, offset); offset += c.Length; }
        return joined;
    }

    /// <summary>Streams the synthesis one text segment at a time: each yielded chunk is that segment's audio (22050 Hz
    /// mono) followed by <see cref="IndexTts2Options.IntervalSilenceMs"/> of silence when more segments follow, and the
    /// final chunk carries the tail fade — concatenating the chunks is exactly what <c>Synthesize</c> returns. The
    /// first chunk is ready after only the first segment has been generated; shrink it with
    /// <see cref="IndexTts2Options.QuickStreamingTokens"/>. Arguments are validated immediately; generation starts on
    /// the first <c>MoveNext</c>.</summary>
    public IEnumerable<float[]> SynthesizeStream(IBackend backend, string text, IndexTts2Reference reference, IndexTts2Options? options = null, CancellationToken ct = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(reference);
        reference.ThrowIfDisposed();
        if (!ReferenceEquals(reference.Owner, this)) throw new ArgumentException("The reference was prepared by a different pipeline.", nameof(reference));
        IndexTts2Options opts = options ?? new IndexTts2Options();
        List<int[]> segments = SegmentText(text, opts);
        if (segments.Count == 0) throw new ArgumentException("Text produced zero tokens after tokenization.", nameof(text));
        return StreamSegments(backend, text, reference, opts, segments, ct);
    }

    private IEnumerable<float[]> StreamSegments(IBackend backend, string text, IndexTts2Reference reference, IndexTts2Options opts,
        List<int[]> segments, CancellationToken ct)
    {
        IndexTts2Timings timings = opts.Timings ?? new IndexTts2Timings();
        Stopwatch wall = Stopwatch.StartNew();
        timings.SegmentCount += segments.Count;

        (Tensor emoVec, bool ownEmoVec) = ResolveEmotion(backend, opts, text, reference, timings);
        try
        {
            int silenceSamples = Math.Max(0, (int)(_cfg.SampleRate * (long)opts.IntervalSilenceMs / 1000L));
            for (int i = 0; i < segments.Count; i++)
            {
                ct.ThrowIfCancellationRequested();
                ulong segSeed = unchecked(opts.Seed + (ulong)i);
                float[] pcm = SynthesizeSegment(backend, reference, emoVec, segments[i], opts, segSeed, timings);
                timings.AudioSeconds += pcm.Length / (double)_cfg.SampleRate;
                if (i == 0) timings.FirstAudioMs = wall.Elapsed.TotalMilliseconds;

                bool last = i == segments.Count - 1;
                if (last)
                {
                    pcm = FadeOutTail(pcm, _cfg.SampleRate);
                }
                else if (silenceSamples > 0)
                {
                    float[] padded = new float[pcm.Length + silenceSamples];
                    pcm.CopyTo(padded, 0);
                    pcm = padded;
                }
                yield return pcm;
            }
        }
        finally
        {
            if (ownEmoVec) emoVec.Dispose();
        }
    }

    /// <summary>The text's segments as token-id arrays. 2.0 follows the reference tokenizer exactly (whole-text
    /// tokenization, then <see cref="IndexTts2TextSegmenter"/>); 2.5 uses the sentence-level splitter.</summary>
    private List<int[]> SegmentText(string text, IndexTts2Options opts)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (_spm is null) return IndexTtsPipeline.SplitIntoTokenSegments(text, DefaultMaxTextTokensPerSegment, EncodeNormalized);

        string prepared = text.Trim().Length == 1 ? text : IndexTtsTextNormalizer.InjectCjkBoundaries(IndexTts2TextNormalizer.Normalize(text));
        if (prepared.Length == 0) return [];
        IReadOnlyList<(string Piece, int Id)> pieces = _spm.EncodeToPieces(prepared);
        IndexTts2TextSegmenter.Token[] tokens = [.. pieces.Select(static p => new IndexTts2TextSegmenter.Token(p.Piece, p.Id))];
        List<IndexTts2TextSegmenter.Token[]> split = IndexTts2TextSegmenter.Split(tokens, DefaultMaxTextTokensPerSegment, opts.QuickStreamingTokens);
        return [.. split.Select(static seg => seg.Select(static t => t.Id).ToArray())];
    }

    /// <summary>Real emotion resolution (<c>infer_generator</c>'s own branching, read in full): an explicit
    /// vector or QwenEmotion-text guidance forces the audio emotion-reference off and routes through
    /// <see cref="IndexTts2EmotionVectorLookup"/>, blended with the natural <c>merge_emovec</c> result by the
    /// vector's own leftover weight (<c>emovec_mat + (1-sum(weights))*emovec</c>, weights exactly as given unless
    /// <see cref="IndexTts2Options.NormalizeEmoVector"/>); otherwise an explicit or implicit (defaults to the
    /// speaker's own clip, alpha forced to 1) audio reference goes straight through <c>merge_emovec</c>.</summary>
    private (Tensor EmoVec, bool Owned) ResolveEmotion(IBackend backend, IndexTts2Options opts, string text, IndexTts2Reference reference, IndexTts2Timings timings)
    {
        Stopwatch sw = Stopwatch.StartNew();
        try
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
                if (opts.NormalizeEmoVector) emoVector = IndexTts2EmotionVectorLookup.NormalizeEmoVec(emoVector, applyBias: true);

                if (_emoLookup is null)
                    throw new InvalidOperationException("IndexTts2Options.EmoVector/UseEmoText requires the pipeline to be loaded with feat1/feat2 (pass feat1Path/feat2Path to LoadAsync).");
                uint rngState = DeterministicRng.Seed(unchecked((int)opts.Seed));
                using Tensor emovecMat = _emoLookup.ComputeEmoVecMat(emoVector, reference.Style, opts.UseRandomEmoExemplar, ref rngState);
                float leftover = 1f - emoVector.Sum();

                // emo_audio_prompt defaults to the speaker's own clip when no external one is given; alpha is
                // forced to 1.0, so merge_emovec(spk, spk, alpha=1) collapses to the speaker's base vector itself.
                return (AddScaled(emovecMat, reference.BaseEmoVec, leftover), true);
            }

            if (opts.EmoAudioReference is not { } emoRef)
            {
                // Speaker's own clip as the emotion reference, alpha forced to 1: merge_emovec(base, base, 1) == base.
                return (reference.BaseEmoVec, false);
            }

            // reference: _load_and_cut_audio(emo_audio_prompt, 15, sr=16000)
            int emoCap = (int)Math.Min(int.MaxValue, MaxReferenceSeconds * emoRef.SampleRate);
            float[] emoAudio = SincResampler.Resample(emoRef.Audio.AsSpan(0, Math.Min(emoCap, emoRef.Audio.Length)), emoRef.SampleRate, 16_000, PrecisionFilterWidth);
            using Tensor emoCondEmb = _semanticFeatures.Forward(backend, emoAudio);
            using Tensor emoVecFromAudio = _gpt.ComputeEmoVec(backend, emoCondEmb, (int)emoCondEmb.Shape[1]);
            return (HartsyInference.Audio.Models.IndexTts2.IndexTts2T2sDecoder.MergeEmoVec(reference.BaseEmoVec, emoVecFromAudio, opts.EmoAlpha), true);
        }
        finally
        {
            timings.EmotionMs += sw.Elapsed.TotalMilliseconds;
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

            // Built into locals and committed to the pipeline's lists only once everything loaded, so a failed
            // attempt releases what it allocated instead of leaking it until Dispose (and a retry starts clean).
            List<Tensor> converted = [];
            SafeTensorsLoader? qwenLoader = null;
            GenericTransformer? transformer = null;
            try
            {
                qwenLoader = new SafeTensorsLoader();
                qwenLoader.Load(Path.Combine(_qwenEmoDir, "model.safetensors"));
                Dictionary<string, Tensor> qwenWeights = IndexTtsPipeline.ToF32(qwenLoader.GetAllTensors(), converted);

                transformer = new GenericTransformer(qwenCfg);
                transformer.LoadWeights(qwenWeights, "model");

                using FileStream tokenizerStream = File.OpenRead(Path.Combine(_qwenEmoDir, "tokenizer.json"));
                GgufTokenizer qwenTokenizer = HfTokenizerJson.LoadByteLevelBpe(tokenizerStream);
                JinjaChatTemplate template = new(File.ReadAllText(Path.Combine(_qwenEmoDir, "chat_template.jinja")));

                TextGenerationPipeline textPipeline = new(transformer, qwenTokenizer, _qwenBackend, template);
                IndexTts2QwenEmotion loaded = new(textPipeline);
                lock (_ownedLock)
                {
                    _loaders.Add(qwenLoader);
                    _loaders.Add(transformer);
                    _convertedWeights.AddRange(converted);
                }
                return _qwenEmotion = loaded;
            }
            catch
            {
                transformer?.Dispose();
                qwenLoader?.Dispose();
                foreach (Tensor t in converted) t.Dispose();
                throw;
            }
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

    private float[] SynthesizeSegment(IBackend backend, IndexTts2Reference reference, Tensor emoVec, int[] textIds,
        IndexTts2Options opts, ulong seed, IndexTts2Timings timings)
    {
        Stopwatch sw = Stopwatch.StartNew();
        int[] codes = GenerateCodes(backend, reference, emoVec, textIds, opts, seed);
        timings.GptMs += sw.Elapsed.TotalMilliseconds;
        timings.CodeCount += codes.Length;

        sw.Restart();
        using Tensor cond = SemanticCondition(backend, reference, emoVec, textIds, codes, opts, out int targetLen);
        timings.SemanticMs += sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        using Tensor mel = SolveMel(backend, reference, cond, targetLen, opts, seed);
        timings.FlowMs += sw.Elapsed.TotalMilliseconds;

        sw.Restart();
        float[] pcm = Vocode(backend, mel, targetLen);
        timings.VocoderMs += sw.Elapsed.TotalMilliseconds;
        return pcm;
    }

    /// <summary>Stage 1: autoregressive sampling of the segment's semantic-codec codes.</summary>
    internal int[] GenerateCodes(IBackend backend, IndexTts2Reference reference, Tensor emoVec, int[] textIds, IndexTts2Options opts, ulong seed)
    {
        uint rngState = DeterministicRng.Seed(unchecked((int)seed));
        int? langId = _cfg.Version == IndexTts2Version.V2_0 ? null : IndexTts2TiktokenTokenizer.LangToToken(opts.Language);
        IndexTtsOptions gptOpts = new()
        {
            Temperature = opts.Temperature,
            TopK = opts.TopK,
            TopP = opts.TopP,
            RepetitionPenalty = opts.RepetitionPenalty,
            MaxMelTokens = opts.MaxMelTokens ?? 1_500,
        };
        return _gpt.Generate(backend, reference.SpeakerConditioning, emoVec, textIds, langId, gptOpts, ref rngState);
    }

    /// <summary>Stage 2: codes → the length-regulated S2Mel content condition <c>[1, targetLen, 512]</c>
    /// (2.0: <c>vq2emb(codes) + gpt_layer(second-pass latent)</c>, 2.5: the codec's own <c>Decode</c>), concatenated
    /// after the reference's prompt condition.</summary>
    internal Tensor SemanticCondition(IBackend backend, IndexTts2Reference reference, Tensor emoVec, int[] textIds, int[] codes, IndexTts2Options opts, out int targetLen)
    {
        Tensor sInfer;
        if (_cfg.Version == IndexTts2Version.V2_0)
        {
            // infer_v2.py: S_infer = vq2emb(codes) + gpt_layer(second-pass latent), target = (code_lens * 1.72).long().
            using Tensor latent = _gpt.ComputeSecondPassLatent(backend, reference.SpeakerConditioning, emoVec, textIds, codes);
            sInfer = _semanticCodec.VqToEmbedding(backend, codes);
            AddInPlace(sInfer, latent);
        }
        else
        {
            sInfer = _semanticCodec.Decode(backend, codes);
        }
        using (sInfer)
        {
            int tSInfer = (int)sInfer.Shape[1];
            targetLen = Math.Max(1, (int)(tSInfer * _cfg.ContentLengthRatio * opts.DurationFactor));
            return _lengthRegulator.Forward(backend, sInfer, tSInfer, targetLen);
        }
    }

    /// <summary>Stage 3: the 25-step CFG flow-matching solve; returns the generated mel only
    /// (<c>[1, 80, targetLen]</c>, the reference-prompt frames sliced off).</summary>
    internal Tensor SolveMel(IBackend backend, IndexTts2Reference reference, Tensor cond, int targetLen, IndexTts2Options opts, ulong seed)
    {
        int tRef = reference.TRef;
        int catLen = tRef + targetLen;
        using Tensor mu = ConcatTime(reference.PromptCondition, cond, tRef, targetLen, _cfg.LengthRegulatorChannels);
        using Tensor promptX = new(new TensorShape(1, _cfg.S2MelDit.InChannels, catLen), DType.F32);
        CopyRefMelPrefix(promptX, reference.RefMel, tRef, catLen);

        using Tensor vcTarget = _cfm.Solve(backend, mu, reference.Style, promptX, _cfg.DiffusionSteps, _cfg.InferenceCfgRate, unchecked((int)seed), promptLen: tRef);
        return SliceTime(vcTarget, tRef, targetLen);
    }

    /// <summary>Stage 4: BigVGAN-22k, mel → PCM clamped to [-1, 1].</summary>
    internal unsafe float[] Vocode(IBackend backend, Tensor mel, int melLen)
    {
        using Tensor wave = _bigVgan.Forward(backend, mel, melLen);
        int n = (int)wave.ElementCount;
        float[] pcm = new float[n];
        float* wp = (float*)wave.DataPointer;
        for (int i = 0; i < n; i++) pcm[i] = Math.Clamp(wp[i], -1f, 1f);
        return pcm;
    }

    // Component access for the stage-level parity tests.
    internal IndexTts2Dit Dit => _dit;
    internal IndexTts2SemanticFeatures SemanticFeatures => _semanticFeatures;
    internal IndexTts2T2sDecoder Gpt => _gpt;
    internal VocosFactorizedCodec SemanticCodec => _semanticCodec;
    internal InterpolateLengthRegulator LengthRegulator => _lengthRegulator;
    internal CamPlusSpeakerEncoder CamPlus => _camplus;

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
    internal unsafe Tensor ComputeRefMel(float[] audio22k)
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
        lock (_qwenLock)
        lock (_ownedLock)
        {
            foreach (IDisposable l in _loaders) l.Dispose();
            foreach (Tensor t in _convertedWeights) t.Dispose();
        }
        GC.SuppressFinalize(this);
    }
}
