using System.Text.RegularExpressions;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.IndexTts;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Pipelines;

/// <summary>IndexTTS-1.5 zero-shot voice-cloning pipeline. Call graph (matches the reference <c>infer()</c>,
/// ground-truthed against the real checkpoint and upstream source — see
/// <c>docs/Research/INDEX_TTS_ARCHITECTURE.md</c> for the full trail):
/// <code>
///   refMel   = mel(referenceAudio)                                   # [1, T_ref, 100], reused for both paths below
///   prefix   = Perceiver(Conformer(refMel))                          # [1, 32, 1280], no position embedding
///   textIds  = tokenizer.Encode(CjkBoundaries(text))
///   latent   = T2sDecoder.Generate(prefix, textIds)                  # AR sample mel codes, keep final_norm'd hidden states
///   wav      = BigVgan(latent, refMel)                               # ECAPA d-vector + GPT-latent conditioning
/// </code>
/// No codec decode step: the DVAE checkpoint (<c>dvae.pth</c>) is confirmed unused at inference by the reference
/// <c>infer.py</c> (the whole load block is commented out upstream) — BigVGAN conditions on the GPT's own latent
/// hidden states directly, not a decoded mel.</summary>
public sealed class IndexTtsPipeline : IDisposable
{
    private readonly IndexTtsConfig _cfg;
    private readonly IndexTtsTokenizer _tokenizer;
    private readonly IndexTtsSpeakerEncoder _speakerEncoder;
    private readonly IndexTtsT2sDecoder _t2s;
    private readonly IndexTtsBigVganGenerator _bigVgan;
    private readonly MelSpectrogramExtractor _melExtractor;
    private readonly PytorchPickleLoader _gptLoader;
    private readonly PytorchPickleLoader _bigVganLoader;
    private readonly List<Tensor> _convertedWeights;
    private int _disposed;

    public string ModelName => "indextts-1.5";

    private IndexTtsPipeline(IndexTtsConfig cfg, IndexTtsTokenizer tokenizer, IndexTtsSpeakerEncoder speakerEncoder,
        IndexTtsT2sDecoder t2s, IndexTtsBigVganGenerator bigVgan, PytorchPickleLoader gptLoader, PytorchPickleLoader bigVganLoader,
        List<Tensor> convertedWeights)
    {
        _cfg = cfg;
        _tokenizer = tokenizer;
        _speakerEncoder = speakerEncoder;
        _t2s = t2s;
        _bigVgan = bigVgan;
        _gptLoader = gptLoader;
        _bigVganLoader = bigVganLoader;
        _convertedWeights = convertedWeights;
        // IndexTTS's own MelSpectrogramFeatures matches F5VocosConfig's torchaudio MelSpectrogram parameterization
        // exactly (24kHz, n_fft 1024, hop 256, 100 mels, HTK scale, magnitude spectrum, center padding, natural
        // log) EXCEPT the log floor: the reference calls safe_log(mel, clip_val=1e-7), not F5/Vocos's 1e-5 —
        // verified against the real index-tts source (utils/common.py's safe_log, utils/feature_extractors.py).
        _melExtractor = new MelSpectrogramExtractor(MelSpectrogramExtractor.F5VocosConfig() with { LogFloor = 1e-7f });
    }

    /// <summary>Loads IndexTTS-1.5 from already-downloaded checkpoint files. <c>dvae.pth</c> is deliberately not
    /// a parameter here — confirmed unused in the real inference path, and the Engine layer no longer downloads
    /// it at all.</summary>
    public static Task<IndexTtsPipeline> LoadAsync(string tokenizerPath, string gptPath, string bigVganPath,
        IndexTtsConfig? cfg, CancellationToken ct = default)
    {
        ct.ThrowIfCancellationRequested();
        IndexTtsConfig resolved = cfg ?? IndexTtsConfig.V1_5;

        IndexTtsTokenizer? tokenizer = null;
        PytorchPickleLoader? gptLoader = null;
        IndexTtsSpeakerEncoder? speakerEncoder = null;
        IndexTtsT2sDecoder? t2s = null;
        PytorchPickleLoader? bigVganLoader = null;
        IndexTtsBigVganGenerator? bigVgan = null;
        List<Tensor> converted = [];
        try
        {
            tokenizer = new IndexTtsTokenizer(tokenizerPath);

            // The loaders must outlive this method: whatever's still natively F32 passes through WhisperOps.
            // EnsureF32 unchanged, so the loader is that tensor's only owner. But the real gpt.pth/bigvgan_*.pth
            // are entirely BF16 (verified directly), so EVERY EnsureF32 call below actually allocates a new F32
            // tensor the loader does NOT own -- converting the whole dictionary to F32 here, once, and tracking
            // every genuinely-new conversion in `converted` means every downstream EnsureF32 call (scattered
            // across GptBlock, the Conformer/Perceiver encoder, ECAPA-TDNN, BigVGAN's resblocks, ...) becomes a
            // guaranteed no-op passthrough again, restoring the single-owner assumption all of that code was
            // written against instead of requiring each of those classes to track its own conversions. Both
            // loaders are still kept as fields for the (now BF16-only) originals, disposed only in
            // IndexTtsPipeline.Dispose() on the success path -- or right here, alongside every other component
            // already constructed, if a later step throws.
            gptLoader = new PytorchPickleLoader();
            gptLoader.Load(gptPath, recursiveFlatten: true);
            Dictionary<string, Tensor> gptWeights = ToF32(gptLoader.GetAllTensors(), converted);

            speakerEncoder = new IndexTtsSpeakerEncoder(resolved.ConditioningEncoder, resolved.Gpt.Hidden);
            speakerEncoder.LoadWeights(gptWeights, "model.conditioning_encoder", "model.perceiver_encoder");

            t2s = new IndexTtsT2sDecoder(resolved.Gpt, resolved.MaxMelTokens);
            t2s.LoadWeights(gptWeights);

            bigVganLoader = new PytorchPickleLoader();
            bigVganLoader.Load(bigVganPath, recursiveFlatten: true);
            Dictionary<string, Tensor> bigVganWeights = ToF32(bigVganLoader.GetAllTensors(), converted);
            bigVgan = new IndexTtsBigVganGenerator(resolved.BigVgan);
            bigVgan.LoadWeights(bigVganWeights, "generator", "generator.speaker_encoder");

            return Task.FromResult(new IndexTtsPipeline(resolved, tokenizer, speakerEncoder, t2s, bigVgan, gptLoader, bigVganLoader, converted));
        }
        catch
        {
            foreach (Tensor t in converted) t.Dispose();
            bigVgan?.Dispose();
            t2s?.Dispose();
            speakerEncoder?.Dispose();
            bigVganLoader?.Dispose();
            gptLoader?.Dispose();
            tokenizer?.Dispose();
            throw;
        }
    }

    /// <summary>Converts every floating-point tensor in <paramref name="raw"/> to F32, appending any tensor that
    /// was genuinely newly-allocated (i.e. not already F32) to <paramref name="converted"/> so the caller can
    /// dispose it later — <see cref="WhisperOps.EnsureF32"/>'s passthrough-when-already-F32 contract means a
    /// reference check is enough to tell the two cases apart. Non-floating-point tensors (e.g. a GPT-2-style
    /// checkpoint's integer position-id or boolean causal-mask buffers) pass through untouched: nothing downstream
    /// reads them as weights, and <see cref="WhisperOps.EnsureF32"/> only knows how to cast floating-point
    /// source dtypes.</summary>
    private static Dictionary<string, Tensor> ToF32(IReadOnlyDictionary<string, Tensor> raw, List<Tensor> converted)
    {
        Dictionary<string, Tensor> result = new(raw.Count);
        foreach ((string key, Tensor tensor) in raw)
        {
            if (!tensor.DType.IsFloatingPoint)
            {
                result[key] = tensor;
                continue;
            }
            Tensor f32 = WhisperOps.EnsureF32(tensor);
            if (!ReferenceEquals(f32, tensor)) converted.Add(f32);
            result[key] = f32;
        }
        return result;
    }

    /// <summary>Default per-segment text budget, matching the reference CLI's own <c>infer()</c> default
    /// (<c>max_text_tokens_per_segment=120</c>) — chosen upstream for pacing/quality, well under
    /// <see cref="IndexTtsConfig.MaxTextTokens"/> (600, the GPT's actual trained text-position-table size).</summary>
    private const int DefaultMaxTextTokensPerSegment = 120;

    /// <summary>Tail fade-out length, matching the reference's own <c>TAIL_FADE_MS</c> (<c>utils/common.py</c>).</summary>
    private const float TailFadeMs = 20.0f;

    /// <summary>Clones the voice in <paramref name="referenceAudioMono"/> speaking <paramref name="text"/>.
    /// Returns 24 kHz mono PCM float samples in [-1, 1].</summary>
    /// <remarks>Text longer than <see cref="DefaultMaxTextTokensPerSegment"/> tokens is split into sentence-bounded
    /// segments and synthesized one at a time against the same reference voice, then concatenated — matching the
    /// reference <c>infer()</c>'s own chunk-and-concatenate strategy. Without this, text that fits under
    /// <see cref="IndexTtsConfig.MaxTextTokens"/> but is long enough that <c>condLen + textLen</c> leaves little of
    /// the GPT's 1,402-token block size for mel generation would silently stop speaking mid-utterance once the
    /// per-call generation cap is hit, with no error. This is a simplified port (sentence-then-comma-then-hard-token
    /// splitting) of the reference's more intricate recursive <c>split_segments_by_token</c>, not a byte-exact
    /// match — the goal is avoiding silent truncation, not reproducing its exact segment boundaries.</remarks>
    public float[] Synthesize(IBackend backend, string text, float[] referenceAudioMono, int referenceSampleRate, IndexTtsOptions? options = null)
    {
        ThrowIfDisposed();
        IndexTtsOptions opts = options ?? new IndexTtsOptions();

        // Validate + segment text (cheap, no tensor allocation) before computing the reference mel, so a rejected
        // prompt never leaves an undisposed Tensor behind.
        List<int[]> segments = SplitIntoTokenSegments(text, DefaultMaxTextTokensPerSegment);
        if (segments.Count == 0) throw new ArgumentException("Text produced zero tokens after tokenization.", nameof(text));
        int totalTokens = 0;
        foreach (int[] seg in segments) totalTokens += seg.Length;
        if (totalTokens > _cfg.MaxTextTokens)
            throw new ArgumentException($"Text has {totalTokens} tokens, exceeding IndexTTS-1.5's max_text_tokens ({_cfg.MaxTextTokens}).", nameof(text));

        float[] refAt24k = referenceSampleRate == _cfg.SampleRate
            ? referenceAudioMono
            : Resampler.Create(referenceSampleRate, _cfg.SampleRate).Resample(referenceAudioMono);
        (Tensor refMel, int refMelLen) = ComputeMelChannelsLast(refAt24k);
        Tensor prefix = _speakerEncoder.Forward(backend, refMel, refMelLen);

        try
        {
            if (segments.Count == 1)
                return FadeOutTail(SynthesizeSegment(backend, prefix, refMel, refMelLen, segments[0], opts, opts.Seed), _cfg.SampleRate);

            List<float[]> pieces = new(segments.Count);
            for (int i = 0; i < segments.Count; i++)
            {
                // Distinct-but-deterministic per-segment seed: reusing the exact same seed for every segment would
                // make every segment's sampling trajectory start identically, which is more repetitive than the
                // reference's own per-segment-independent generation.
                ulong segSeed = unchecked(opts.Seed + (ulong)i);
                pieces.Add(SynthesizeSegment(backend, prefix, refMel, refMelLen, segments[i], opts, segSeed));
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
            prefix.Dispose();
            refMel.Dispose();
        }
    }

    /// <summary>Ramps the final <paramref name="fadeMs"/> of <paramref name="pcm"/> down to zero with a
    /// raised-cosine (zero slope at the end, so the fade introduces no new derivative discontinuity) — matching
    /// the reference's own <c>fade_out_tail</c> (<c>utils/common.py</c>). IndexTTS's output length is derived from
    /// the sampled stop token, not from acoustic content; decoding is stochastic, so the stop token occasionally
    /// lands early and the waveform's last sample is far from zero, producing an audible click (and, since BigVGAN's
    /// receptive field is incomplete right at that boundary, sometimes a burst of noise on top of it). On a normal
    /// generation this is free — the tail is already silence — so it's applied unconditionally rather than gated
    /// behind a detection threshold. Mutates and returns <paramref name="pcm"/> (the caller is this method's only
    /// owner at the call site, unlike the reference which returns a new tensor).</summary>
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

    /// <summary>Runs the AR decode + vocoder for one already-tokenized segment against a shared speaker
    /// <paramref name="prefix"/>/<paramref name="refMel"/> (both owned by the caller).</summary>
    private float[] SynthesizeSegment(IBackend backend, Tensor prefix, Tensor refMel, int refMelLen, int[] textIds, IndexTtsOptions opts, ulong seed)
    {
        uint rngState = DeterministicRng.Seed(unchecked((int)seed));
        Tensor latent = _t2s.Generate(backend, prefix, textIds, opts, ref rngState);
        Tensor wave;
        try
        {
            wave = _bigVgan.Forward(backend, latent, (int)latent.Shape[1], refMel, refMelLen);
        }
        finally
        {
            latent.Dispose();
        }

        try
        {
            int n = (int)wave.ElementCount;
            float[] pcm = new float[n];
            unsafe
            {
                float* wp = (float*)wave.DataPointer;
                for (int i = 0; i < n; i++) pcm[i] = wp[i];
            }
            return pcm;
        }
        finally
        {
            wave.Dispose();
        }
    }

    /// <summary>Splits <paramref name="text"/> into token-id segments, each tokenizing to at most
    /// <paramref name="maxTokensPerSegment"/> ids: sentence-bounded first (splitting on <c>. ! ? 。！？</c>),
    /// falling back to comma/semicolon splitting for an overlong single sentence, and finally a hard token-count
    /// cut for an overlong single clause with no further punctuation to split on.</summary>
    private List<int[]> SplitIntoTokenSegments(string text, int maxTokensPerSegment)
    {
        List<int[]> result = [];
        List<string> pending = [];
        int pendingTokens = 0;

        void FlushPending()
        {
            if (pending.Count == 0) return;
            int[] ids = EncodeNormalized(string.Concat(pending));
            if (ids.Length > 0) result.Add(ids);
            pending.Clear();
            pendingTokens = 0;
        }

        foreach (string sentence in SplitOnPattern(text, SentenceBoundaryPattern))
        {
            int[] sentIds = EncodeNormalized(sentence);
            if (sentIds.Length == 0) continue;
            if (sentIds.Length > maxTokensPerSegment)
            {
                FlushPending();
                foreach (string clause in SplitOnPattern(sentence, ClauseBoundaryPattern))
                {
                    int[] clauseIds = EncodeNormalized(clause);
                    if (clauseIds.Length == 0) continue;
                    if (clauseIds.Length > maxTokensPerSegment)
                    {
                        for (int offset = 0; offset < clauseIds.Length; offset += maxTokensPerSegment)
                            result.Add(clauseIds[offset..Math.Min(offset + maxTokensPerSegment, clauseIds.Length)]);
                    }
                    else if (pendingTokens + clauseIds.Length > maxTokensPerSegment)
                    {
                        FlushPending();
                        pending.Add(clause);
                        pendingTokens = clauseIds.Length;
                    }
                    else
                    {
                        pending.Add(clause);
                        pendingTokens += clauseIds.Length;
                    }
                }
                continue;
            }
            if (pendingTokens + sentIds.Length > maxTokensPerSegment) FlushPending();
            pending.Add(sentence);
            pendingTokens += sentIds.Length;
        }
        FlushPending();
        return result;
    }

    private int[] EncodeNormalized(string segment) => _tokenizer.Encode(IndexTtsTextNormalizer.InjectCjkBoundaries(segment));

    private static readonly Regex SentenceBoundaryPattern = new(@"[^.!?。！？]+[.!?。！？]*\s*", RegexOptions.Compiled);
    private static readonly Regex ClauseBoundaryPattern = new(@"[^,;，；]+[,;，；]*\s*", RegexOptions.Compiled);

    private static List<string> SplitOnPattern(string text, Regex pattern)
    {
        List<string> parts = [];
        foreach (Match m in pattern.Matches(text)) if (m.Value.Trim().Length > 0) parts.Add(m.Value);
        return parts.Count > 0 ? parts : [text];
    }

    /// <summary>Runs the shared mel front end and transposes its channels-first <c>[nMels, T]</c> output into the
    /// channels-last <c>[1, T, nMels]</c> layout <see cref="IndexTtsConformerEncoder"/>/<see cref="IndexTtsEcapaTdnn"/> expect.</summary>
    private (Tensor, int) ComputeMelChannelsLast(float[] audio)
    {
        float[,] mel = _melExtractor.Compute(audio);
        int nMels = mel.GetLength(0), t = mel.GetLength(1);
        Tensor outT = new(new TensorShape(1, t, nMels), DType.F32);
        unsafe
        {
            float* op = (float*)outT.DataPointer;
            for (int f = 0; f < t; f++)
                for (int m = 0; m < nMels; m++)
                    op[(long)f * nMels + m] = mel[m, f];
        }
        return (outT, t);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _speakerEncoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _t2s.EnumerateWeights()) yield return t;
        foreach (Tensor t in _bigVgan.EnumerateWeights()) yield return t;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(IndexTtsPipeline));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _tokenizer.Dispose();
        _speakerEncoder.Dispose();
        _t2s.Dispose();
        _bigVgan.Dispose();
        // The loaders own the original BF16 weight-tensor memory; _convertedWeights owns every genuinely-new F32
        // conversion ToF32 produced from them (see LoadAsync's remarks) -- dispose both last so nothing above
        // touches freed tensors mid-teardown.
        _gptLoader.Dispose();
        _bigVganLoader.Dispose();
        foreach (Tensor t in _convertedWeights) t.Dispose();
        GC.SuppressFinalize(this);
    }
}
