using System.Runtime.CompilerServices;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.VibeVoice;
using static HartsyInference.Audio.Models.VibeVoice.VibeVoiceOps;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.LLM.Transformer;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>VibeVoice-Realtime-0.5B — the single-speaker, low-latency streaming variant. A genuinely different
/// architecture from <see cref="VibeVoicePipeline"/> (1.5B/7B), not a smaller config of the same one:
///
/// <list type="bullet">
///   <item><b>Split LM</b>: two independently-trained Qwen2 stacks, confirmed from the real checkpoint's own
///         weight keys (not the architecture doc's prose) — a 4-layer <c>language_model</c> (own
///         <c>embed_tokens</c>, genuinely no final norm: no <c>norm.weight</c> key exists) that encodes
///         incoming text, and a 20-layer <c>tts_language_model</c> (own <c>embed_tokens</c> + own final norm)
///         that drives generation. The lower model's output for a text window is spliced into the upper
///         model's input sequence with a learned <c>tts_input_types</c> row added (index 1 = spliced text,
///         index 0 = the upper model's own native speech-embedding positions) — confirmed against real
///         <c>tts_input_types.weight [2, 896]</c>.</item>
///   <item><b>Windowed generation</b>: <see cref="TtsTextWindowSize"/> (5) text tokens feed
///         <see cref="TtsSpeechWindowSize"/> (6) speech-diffusion steps, repeating until the text is exhausted
///         and the binary EOS classifier fires (or a safety frame cap is hit).</item>
///   <item><b>No semantic VAE, no lm_head anywhere</b> — confirmed by the real checkpoint: only the acoustic
///         VAE (shared code with <see cref="VibeVoicePipeline"/>, same weight layout) and a top-level
///         <c>tts_eos_classifier</c> (<see cref="VibeVoiceEosClassifier"/>) instead of any token-based stop.</item>
/// </list>
///
/// <para><b>Voice conditioning is NOT wired yet — a hard blocker, not a TODO.</b> Checked both routes:
/// (1) <i>Zero-shot from a reference clip</i> is architecturally impossible against this release — the real
/// checkpoint's own 608 weight keys have zero <c>model.acoustic_tokenizer.encoder.*</c> entries (confirmed
/// 2026-08-10 loading the real checkpoint, which throws a `KeyNotFoundException` for exactly that reason);
/// this variant ships a decode-only acoustic VAE (<see cref="VibeVoiceAcousticTokenizerModel.LoadWeights"/>'s
/// <c>decodeOnly</c> flag), so there is no encoder to build a live-priming path from, adapted or otherwise.
/// (2) <i>Upstream's own fixed presets</i> — Microsoft's demo loads ~9 precomputed per-speaker <c>.pt</c>
/// snapshots (<c>demo/voices/streaming_model/*.pt</c>, confirmed via a real download + pickle disassembly:
/// each is <c>{"lm": BaseModelOutputWithPast(last_hidden_state, DynamicCache(4 layers)), "tts_lm":
/// BaseModelOutputWithPast(last_hidden_state, DynamicCache(20 layers))}</c> — the exact 4-lower/20-upper
/// split this class already implements, seeded once offline using a checkpoint that DID have an encoder,
/// not reproducible from the released weights) — buildable, but needs a bespoke pickle deserializer for
/// this nested-object-graph shape (`PytorchPickleLoader` only handles flat/near-flat state dicts) plus
/// seeding <see cref="IKvCache.KeyPrefix"/>/<see cref="IKvCache.ValuePrefix"/> from the parsed tensors — a
/// separate, scoped follow-up, not started. Until either lands, <see cref="Synthesize"/>/
/// <see cref="SynthesizeStream"/> throw <see cref="NotSupportedException"/> up front rather than silently
/// producing unconditioned/garbage audio. Everything below this point — the split-LM forward, the windowed
/// generation loop, the EOS classifier, the CFG-batched diffusion sampling — is real, loads against the
/// real checkpoint (verified), and is ready to run the moment a cache is available to seed
/// <c>ttsCache</c>/<c>textCache</c> instead of the (removed) live-encode priming step.</para></summary>
public sealed class VibeVoiceStreamingPipeline : IDisposable
{
    /// <summary>Output sample rate — shares the acoustic VAE with <see cref="VibeVoicePipeline"/>.</summary>
    public const int SampleRate = 24_000;

    /// <summary>Text tokens consumed per window before the next batch of speech-diffusion steps.</summary>
    private const int TtsTextWindowSize = 5;

    /// <summary>Speech-diffusion steps run per text window.</summary>
    private const int TtsSpeechWindowSize = 6;

    /// <summary>EOS classifier stop threshold (upstream: <c>sigmoid(logit) &gt; 0.5</c>).</summary>
    private const float EosThreshold = 0.5f;

    private readonly VibeVoiceConfig _cfg;
    private readonly Qwen2Config _textLmCfg;
    private readonly Qwen2Config _ttsLmCfg;
    private readonly VibeVoiceTokenizer _tokenizer;
    private readonly Qwen2Model _textLm;
    private readonly Qwen2Model _ttsLm;
    private readonly VibeVoiceAcousticTokenizerModel _acoustic;
    private readonly SpeechConnector _acousticConnector;
    private readonly VibeVoiceDiffusionHead _diffusionHead;
    private readonly VibeVoiceEosClassifier _eosClassifier;
    private readonly Tensor _ttsInputTypes;    // [2, hidden]: row 0 = speech position, row 1 = spliced-text position.

    private readonly float _speechScalingFactor;
    private readonly float _speechBiasFactor;

    private bool _preloaded;
    private int _disposed;

    public VibeVoiceConfig Config => _cfg;

    private VibeVoiceStreamingPipeline(VibeVoiceConfig cfg, Qwen2Config textLmCfg, Qwen2Config ttsLmCfg,
        VibeVoiceTokenizer tokenizer, Qwen2Model textLm, Qwen2Model ttsLm, VibeVoiceAcousticTokenizerModel acoustic,
        SpeechConnector acousticConnector, VibeVoiceDiffusionHead diffusionHead, VibeVoiceEosClassifier eosClassifier,
        Tensor ttsInputTypes, float scaling, float bias)
    {
        _cfg = cfg; _textLmCfg = textLmCfg; _ttsLmCfg = ttsLmCfg;
        _tokenizer = tokenizer; _textLm = textLm; _ttsLm = ttsLm;
        _acoustic = acoustic; _acousticConnector = acousticConnector;
        _diffusionHead = diffusionHead; _eosClassifier = eosClassifier;
        _ttsInputTypes = ttsInputTypes;
        _speechScalingFactor = scaling; _speechBiasFactor = bias;
    }

    /// <summary>Loads VibeVoice-Realtime-0.5B from the HartsyInference cache (downloads on first use).</summary>
    public static async Task<VibeVoiceStreamingPipeline> LoadAsync(CancellationToken ct = default)
    {
        const string repo = "microsoft/VibeVoice-Realtime-0.5B";
        string repoDir = AudioModelCache.GetRepoDirectory(repo, "tts");
        await Task.WhenAll(
            AudioModelCache.GetAsync(repo, "model.safetensors", category: "tts", ct: ct),
            AudioModelCache.GetAsync(repo, "config.json", category: "tts", ct: ct))
            .ConfigureAwait(false);

        VibeVoiceConfig cfg = VibeVoiceConfig.Streaming05B;
        int lowerLayers = cfg.Decoder.NumHiddenLayers - cfg.TtsBackboneNumHiddenLayers;   // 24 - 20 = 4

        Qwen2Config textLmCfg = SubConfig(cfg.Decoder, lowerLayers);
        Qwen2Config ttsLmCfg = SubConfig(cfg.Decoder, cfg.TtsBackboneNumHiddenLayers);

        SafeTensorsShardLoader loader = new();
        loader.LoadDirectory(repoDir);
        IReadOnlyDictionary<string, Tensor> weights = loader.GetAllTensors();

        VibeVoiceTokenizer tokenizer = new();

        // Two genuinely separate weight-bearing Qwen2 stacks — real checkpoint keys confirm this (each has
        // its own embed_tokens; language_model has no norm.weight at all, tts_language_model has its own).
        Qwen2Model textLm = new(textLmCfg);
        textLm.LoadWeightsNoFinalNorm(weights, "model.language_model");
        Qwen2Model ttsLm = new(ttsLmCfg);
        ttsLm.LoadWeightsNoLmHead(weights, "model.tts_language_model");

        // decodeOnly: true — confirmed against the real checkpoint's own key list (608 tensors, zero under
        // model.acoustic_tokenizer.encoder.*) that this release ships no encoder at all. It only ever decodes
        // latents (from a generation step or, upstream, a precomputed voice cache) — see the class doc.
        VibeVoiceAcousticTokenizerModel acoustic = new(cfg.AcousticTokenizer, "model.acoustic_tokenizer");
        acoustic.LoadWeights(weights, decodeOnly: true);

        SpeechConnector acousticConnector = new(cfg.AcousticVaeDim, cfg.Decoder.HiddenSize);
        acousticConnector.LoadWeights(weights, "model.acoustic_connector");

        VibeVoiceDiffusionHead diffusionHead = new(cfg.DiffusionHead, "model.prediction_head");
        diffusionHead.LoadWeights(weights);

        VibeVoiceEosClassifier eosClassifier = new(cfg.Decoder.HiddenSize);
        eosClassifier.LoadWeights(weights);

        Tensor ttsInputTypes = HartsyInference.Audio.Models.Whisper.WhisperOps.EnsureF32(weights["model.tts_input_types.weight"]);

        float scaling = ReadScalar(weights, "model.speech_scaling_factor");
        float bias = ReadScalar(weights, "model.speech_bias_factor");

        // Do NOT dispose the loader — see VibeVoicePipeline.LoadAsync's identical note (borrowed bf16 mmap views).
        return new VibeVoiceStreamingPipeline(cfg, textLmCfg, ttsLmCfg, tokenizer, textLm, ttsLm, acoustic,
            acousticConnector, diffusionHead, eosClassifier, ttsInputTypes, scaling, bias);
    }

    private static Qwen2Config SubConfig(VibeVoiceDecoderConfig d, int layers) => new()
    {
        HiddenSize = d.HiddenSize,
        NumHiddenLayers = layers,
        NumAttentionHeads = d.NumAttentionHeads,
        NumKeyValueHeads = d.NumKeyValueHeads,
        IntermediateSize = d.IntermediateSize,
        VocabSize = d.VocabSize,
        MaxPositionEmbeddings = d.MaxPositionEmbeddings,
        RopeTheta = d.RopeTheta,
        RmsNormEps = d.RmsNormEps,
    };

    /// <summary>Synthesizes 24 kHz audio for <paramref name="text"/> in the voice of <paramref name="voiceWavPath"/>
    /// (zero-shot from a reference clip — see the class doc's voice-conditioning note). Single speaker only,
    /// matching upstream's own <c>assert batch_size == 1</c> constraint.</summary>
    public float[] Synthesize(IBackend backend, string text, string voiceWavPath, int maxNewFrames = 600,
        float cfgScale = 1.5f, int? diffusionSteps = null, int seed = 0)
    {
        List<float[]> audioChunks = new();
        SynthesizeCore(backend, text, voiceWavPath, maxNewFrames, cfgScale, diffusionSteps, seed, audioChunks.Add);

        int totalSamples = 0;
        foreach (float[] c in audioChunks) totalSamples += c.Length;
        float[] audio = new float[totalSamples];
        int offset = 0;
        foreach (float[] c in audioChunks)
        {
            Array.Copy(c, 0, audio, offset, c.Length);
            offset += c.Length;
        }
        return audio;
    }

    /// <summary>Streaming counterpart to <see cref="Synthesize"/> — same generation loop, chunks handed to
    /// the caller as each is decoded. Mirrors <see cref="VibeVoicePipeline.SynthesizeStream"/>'s shape.</summary>
    public async IAsyncEnumerable<AudioChunk> SynthesizeStream(IBackend backend, string text, string voiceWavPath,
        int maxNewFrames, float cfgScale, int? diffusionSteps, int seed,
        [EnumeratorCancellation] CancellationToken cancel = default)
    {
        using AudioStreamer streamer = new();
        long sampleOffset = 0;
        Task producer = Task.Run(() =>
        {
            try
            {
                SynthesizeCore(backend, text, voiceWavPath, maxNewFrames, cfgScale, diffusionSteps, seed,
                    onChunk: chunk =>
                    {
                        streamer.Put(new AudioChunk(chunk, SampleRate, 1, sampleOffset), cancel).AsTask().GetAwaiter().GetResult();
                        sampleOffset += chunk.Length;
                    });
            }
            finally
            {
                streamer.Complete();
            }
        }, cancel);

        try
        {
            await foreach (AudioChunk chunk in streamer.ReadAllAsync(cancel).ConfigureAwait(false))
            {
                yield return chunk;
            }
        }
        finally
        {
            await producer.ConfigureAwait(false);
        }
    }

    /// <summary>The shared windowed generation loop behind both entry points.</summary>
    private unsafe void SynthesizeCore(IBackend backend, string text, string voiceWavPath, int maxNewFrames,
        float cfgScale, int? diffusionSteps, int seed, Action<float[]> onChunk)
    {
        ThrowIfDisposed();
        // See the class doc: neither voice-conditioning path (zero-shot encode — impossible against this
        // release's decode-only acoustic VAE; upstream's precomputed .pt presets — needs a deserializer that
        // doesn't exist yet) is wired. Throw here, not deeper in a confusing missing-key crash.
        throw new NotSupportedException(
            "VibeVoice-Realtime-0.5B voice conditioning is not wired yet: this release ships no acoustic-VAE "
            + "encoder (zero-shot cloning is architecturally impossible against it), and upstream's precomputed "
            + "per-speaker .pt voice caches need a dedicated deserializer that hasn't been built. The split-LM "
            + "generation loop below is real and loads real weights, but has nothing to seed voice conditioning "
            + "from yet.");
#pragma warning disable CS0162 // Unreachable code — kept intact and ready for when voice-cache loading lands.
        PreloadWeights(backend);
        int h = _cfg.Decoder.HiddenSize;
        int runSteps = diffusionSteps is > 0 ? diffusionSteps.Value : _cfg.DiffusionHead.DdpmNumInferenceSteps;
        // No discrete token sampling anywhere in this variant (no lm_head at all) — only the noise draw below.
        uint noiseRng = HartsyInference.Audio.Dsp.DeterministicRng.Seed(seed ^ 0x51ED270B);

        int[] ttsTextIds = _tokenizer.Encode(text.Trim() + "\n");

        // Generous cap: text windows + a frame per window's speech budget, rounded up for the voice-priming
        // prefill and the CFG negative stream's own smaller cache.
        int textWindows = (ttsTextIds.Length + TtsTextWindowSize - 1) / Math.Max(1, TtsTextWindowSize);
        int cacheCap = Math.Min(_cfg.Decoder.MaxPositionEmbeddings,
            ttsTextIds.Length + maxNewFrames + textWindows + 32);
        using IKvCache textCache = _textLm.CreateDecodeCache(Math.Max(64, textWindows * TtsTextWindowSize + 8));
        using IKvCache negTextCache = _textLm.CreateDecodeCache(8);
        using IKvCache ttsCache = _ttsLm.CreateDecodeCache(cacheCap);
        using IKvCache negTtsCache = _ttsLm.CreateDecodeCache(maxNewFrames + 8);

        using VibeVoiceTokenizerStreamingCache acousticCache = new();
        int[] sampleIndices = [0];

        // ── Voice priming — UNREACHABLE placeholder, not a working path (see the class doc: this would need
        // a working acoustic-VAE encoder, which this checkpoint doesn't ship). Left in this shape — encode a
        // reference-derived latent, project, tag as native speech positions (tts_input_types[0]), prefill the
        // positive TTS-backbone cache — because a real preset-cache loader will prime the exact same
        // `lastHidden`/`ttsCache` state from a parsed .pt snapshot instead; everything from here down (the
        // windowed loop, CFG, diffusion, EOS) stays unchanged once that swap happens.
        Tensor? lastHidden = null;
        {
            float[] voicePcm = LoadVoicePcm24k(voiceWavPath);
            int latentCount = Math.Max(1, (int)Math.Ceiling(voicePcm.Length / 3_200.0));
            using Tensor voiceLatent = EncodeVoicePcm(backend, voicePcm, latentCount);
            NormalizeLatentInPlace(voiceLatent, _speechScalingFactor, _speechBiasFactor);
            using Tensor voiceEmbed = _acousticConnector.Forward(backend, voiceLatent, batch: 1, seqLen: latentCount);
            using Tensor voiceEmbedTyped = AddTypeEmbed(voiceEmbed, latentCount, h, typeIndex: 0);
            Tensor primed = _ttsLm.ForwardEmbeds(backend, voiceEmbedTyped, batch: 1, t: latentCount, posStart: 0, ttsCache);
            lastHidden = SliceLastFrame(primed, h);
            primed.Dispose();
        }

        // ── CFG negative stream: a single <|image_pad|> sentinel through the text encoder, spliced into its
        // own tiny TTS-backbone cache — no voice/text conditioning at all, matching VibeVoicePipeline's own
        // "unlike the positive stream it holds NO text/voice prefill" negative-stream design.
        bool useCfg = MathF.Abs(cfgScale - 1f) > 1e-6f;
        Tensor? negLastHidden = null;
        if (useCfg)
        {
            int[] negTok = [VibeVoiceTokenizer.ImagePadTokenId];
            using Tensor negTextEmbeds = new(new TensorShape(1, 1, h), DType.F32);
            _textLm.EmbedLookup(negTextEmbeds, negTok, batch: 1, t: 1);
            using Tensor negTextHidden = _textLm.ForwardEmbeds(backend, negTextEmbeds, batch: 1, t: 1,
                posStart: 0, negTextCache, applyFinalNorm: false);
            using Tensor negSpliced = AddTypeEmbed(negTextHidden, 1, h, typeIndex: 1);
            Tensor negPrimed = _ttsLm.ForwardEmbeds(backend, negSpliced, batch: 1, t: 1, posStart: 0, negTtsCache);
            negLastHidden = SliceLastFrame(negPrimed, h);
            negPrimed.Dispose();
        }

        int framesEmitted = 0;
        int textWindowIndex = 0;
        bool stop = false;
        try
        {
            while (!stop)
            {
                int start = textWindowIndex * TtsTextWindowSize;
                int windowLen = Math.Max(0, Math.Min(TtsTextWindowSize, ttsTextIds.Length - start));
                textWindowIndex++;

                if (windowLen > 0)
                {
                    ReadOnlySpan<int> window = ttsTextIds.AsSpan(start, windowLen);
                    // Lower 4-layer text encoder: its own embed_tokens + layers, no final norm applied
                    // (the checkpoint has none — confirmed by the missing norm.weight key).
                    using Tensor windowEmbeds = new(new TensorShape(1, windowLen, h), DType.F32);
                    _textLm.EmbedLookup(windowEmbeds, window, batch: 1, t: windowLen);
                    Tensor lowerHidden = _textLm.ForwardEmbeds(backend, windowEmbeds, batch: 1, t: windowLen,
                        posStart: textCache.CurrentLength, textCache, applyFinalNorm: false);
                    using Tensor spliced = AddTypeEmbed(lowerHidden, windowLen, h, typeIndex: 1);
                    lowerHidden.Dispose();
                    Tensor newHidden = _ttsLm.ForwardEmbeds(backend, spliced, batch: 1, t: windowLen, posStart: ttsCache.CurrentLength, ttsCache);
                    lastHidden!.Dispose();
                    lastHidden = SliceLastFrame(newHidden, h);
                    newHidden.Dispose();
                }
                else if (textWindowIndex > textWindows + 2)
                {
                    // No more text and we've already given the tail a couple of empty windows to let the
                    // classifier catch up — stop even if it never fires (safety net, mirrors maxNewFrames).
                    stop = true;
                }

                for (int s = 0; s < TtsSpeechWindowSize && !stop; s++)
                {
                    using Tensor cond = ExpandTo3D(lastHidden!, h);
                    Tensor? negCond = useCfg ? ExpandTo3D(negLastHidden!, h) : null;

                    using Tensor noiseLatent = SampleNoise(_cfg.AcousticVaeDim, ref noiseRng);
                    Tensor denoised = DenoiseLatent(backend, _diffusionHead, runSteps, useCfg ? cfgScale : 1f,
                        noiseLatent, cond, negCond, _cfg.AcousticVaeDim, h);
                    negCond?.Dispose();

                    UnnormalizeLatentInPlace(denoised, _speechScalingFactor, _speechBiasFactor);
                    using Tensor frameAudio = _acoustic.Decode(backend, denoised, batch: 1, acousticCache, sampleIndices);
                    float[] chunk = TensorToPcm(frameAudio);
                    onChunk(chunk);
                    framesEmitted++;

                    Tensor latentNormalized = ReNormalizeLatent(denoised, _speechScalingFactor, _speechBiasFactor);
                    denoised.Dispose();
                    Tensor acousticEmbed = _acousticConnector.Forward(backend, latentNormalized, batch: 1, seqLen: 1);
                    latentNormalized.Dispose();
                    using Tensor stepEmbed = AddTypeEmbed(acousticEmbed, 1, h, typeIndex: 0);
                    acousticEmbed.Dispose();

                    Tensor newHidden = _ttsLm.ForwardEmbeds(backend, stepEmbed, batch: 1, t: 1, posStart: ttsCache.CurrentLength, ttsCache);
                    lastHidden!.Dispose();
                    lastHidden = SliceLastFrame(newHidden, h);

                    // Binary EOS: sigmoid(fc2(relu(fc1(last_hidden)))) > 0.5 stops generation — there is no
                    // token-based stop on this variant at all (no lm_head in the checkpoint).
                    float stopProb = _eosClassifier.StopProbability(backend, lastHidden);
                    newHidden.Dispose();

                    if (useCfg)
                    {
                        Tensor negNew = _ttsLm.ForwardEmbeds(backend, stepEmbed, batch: 1, t: 1, posStart: negTtsCache.CurrentLength, negTtsCache);
                        negLastHidden!.Dispose();
                        negLastHidden = SliceLastFrame(negNew, h);
                        negNew.Dispose();
                    }

                    if (stopProb > EosThreshold && windowLen == 0)
                    {
                        // Only honor EOS once the real text is exhausted — an early false-positive mid-utterance
                        // would truncate speech the model still has text left to say.
                        stop = true;
                    }
                    if (framesEmitted >= maxNewFrames || ttsCache.CurrentLength >= cacheCap - 2) { stop = true; break; }
                }
            }
        }
        finally
        {
            lastHidden?.Dispose();
            negLastHidden?.Dispose();
        }
#pragma warning restore CS0162
    }

    /// <summary>Adds the learned <c>tts_input_types[typeIndex]</c> row (<c>[hidden]</c>) to every one of the
    /// <paramref name="t"/> positions in <paramref name="embeds"/> — matches upstream's per-position
    /// type-embedding add; every position in one call here shares the same type, so this is a broadcast add,
    /// not a gather.</summary>
    private unsafe Tensor AddTypeEmbed(Tensor embeds, int t, int hidden, int typeIndex)
    {
        Tensor result = CopyOf(embeds);
        float* rp = (float*)result.DataPointer;
        float* typeRow = (float*)_ttsInputTypes.DataPointer + (long)typeIndex * hidden;
        for (int i = 0; i < t; i++)
        {
            float* row = rp + (long)i * hidden;
            for (int k = 0; k < hidden; k++) row[k] += typeRow[k];
        }
        return result;
    }

    private unsafe Tensor EncodeVoicePcm(IBackend backend, float[] pcm, int latentCount)
    {
        int tPcm = latentCount * 3_200;
        using Tensor pcmTensor = new(new TensorShape(1, 1, tPcm), DType.F32);
        float* dst = (float*)pcmTensor.DataPointer;
        int copy = Math.Min(pcm.Length, tPcm);
        fixed (float* src = pcm) Buffer.MemoryCopy(src, dst, copy * 4, copy * 4);
        for (int j = copy; j < tPcm; j++) dst[j] = 0f;
        return _acoustic.Encode(backend, pcmTensor, batch: 1, tPcm: tPcm);
    }

    private static float[] LoadVoicePcm24k(string wavPath)
    {
        HartsyInference.Audio.Io.WavFile.DecodedAudio wav = HartsyInference.Audio.Io.WavFile.Read(wavPath);
        float[] mono = wav.Channels[0];
        if (wav.SampleRate != SampleRate)
        {
            HartsyInference.Audio.Io.Resampler rs = HartsyInference.Audio.Io.Resampler.Create(wav.SampleRate, SampleRate);
            mono = rs.Resample(mono);
        }
        return mono;
    }

    public void PreloadWeights(IBackend backend)
    {
        if (_preloaded) return;
        _preloaded = true;
        backend.PreloadWeights(EnumerateWeights());
    }

    private IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _textLm.EnumerateWeights()) yield return t;
        foreach (Tensor t in _ttsLm.EnumerateWeights()) yield return t;
        foreach (Tensor t in _acoustic.EnumerateWeights()) yield return t;
        foreach (Tensor t in _acousticConnector.EnumerateWeights()) yield return t;
        foreach (Tensor t in _diffusionHead.EnumerateWeights()) yield return t;
        foreach (Tensor t in _eosClassifier.EnumerateWeights()) yield return t;
        yield return _ttsInputTypes;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(VibeVoiceStreamingPipeline));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _textLm.Dispose();
        _ttsLm.Dispose();
        _tokenizer.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }
}
