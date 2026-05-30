using SharpInference.Audio.Cache;
using SharpInference.Audio.Models.Kokoro;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.Audio.Pipelines;

/// <summary>End-to-end Kokoro-82M TTS pipeline. Wires the four submodules (PLBERT,
/// TextEncoder, ProsodyPredictor, iSTFTNetDecoder) and the voice-pack-driven style
/// path into a single <see cref="Synthesize"/> call.
///
/// <para>The pipeline is phoneme-in / audio-out. English text → IPA phoneme G2P is
/// <i>not</i> bundled — the upstream Kokoro release uses misaki (Python + spaCy +
/// espeak-ng), which we don't have in pure-C# yet. Callers should phonemize externally
/// (e.g. via the misaki CLI or eSpeak-NG); for a quick demo, supply a known-good IPA
/// string and select the matching voice pack.</para>
///
/// <para>Pipeline shape:</para>
/// <code>
///   IPA phonemes
///       │  KokoroPhonemeTokenizer
///       ▼
///   token_ids [1, T+2]                          ← BOS/EOS padded
///       │
///       ▼  voice_pack.GetStyle(T+2)
///   style [1, 256] ──┬──→ s_dec  [1, 128]
///                    └──→ s_pred [1, 128]
///       │
///       ├──► PLBERT(token_ids) → d_bert [1, T+2, 512]
///       │       │
///       │       ▼  + s_pred  → DurationEncoder + LSTM + duration_proj
///       │      durations [T+2] (int)
///       │       │
///       │       ▼  build alignment matrix from durations → length regulator
///       │      d_bert_expanded [1, 512, T_total]      T_total = sum(durations)
///       │       │
///       │       ▼  + s_pred  → predictor.shared + F0 chain + N chain
///       │      F0, N  [1, 1, 2*T_total]
///       │
///       └──► TextEncoder(token_ids) → text_features [1, T+2, 512]
///              │
///              ▼  length-regulate via SAME alignment as d_bert
///             asr [1, 512, T_total]
///
///   asr, F0, N, s_dec  →  KokoroIStftNetDecoder.Forward  →  audio (24 kHz float)
/// </code>
///
/// <para>The current decoder forward pass is the documented placeholder described in
/// <see cref="KokoroIStftNetDecoder"/> — it produces F0-modulated audio for prosody QA.
/// The full HiFi-GAN+iSTFT generator is the staged follow-up; the predictor + alignment
/// + style paths are however fully wired.</para></summary>
public sealed class KokoroPipeline : IDisposable
{
    private readonly KokoroConfig _cfg;
    private readonly KokoroPhonemeTokenizer _tokenizer;
    private readonly KokoroPlBert _plBert;
    private readonly KokoroTextEncoder _textEncoder;
    private readonly KokoroProsodyPredictor _predictor;
    private readonly KokoroIStftNetDecoder _decoder;
    private readonly SafeTensorsLoader? _loader;
    private readonly Dictionary<string, KokoroVoicePack> _voicePackCache = new(StringComparer.Ordinal);
    private readonly string _repoDir;
    private int _disposed;

    public KokoroConfig Config => _cfg;
    public string ModelName => "hexgrad/Kokoro-82M";

    private KokoroPipeline(KokoroConfig cfg, KokoroPhonemeTokenizer tok, KokoroPlBert plBert,
        KokoroTextEncoder textEnc, KokoroProsodyPredictor pred, KokoroIStftNetDecoder dec,
        SafeTensorsLoader? loader, string repoDir)
    {
        _cfg = cfg;
        _tokenizer = tok;
        _plBert = plBert;
        _textEncoder = textEnc;
        _predictor = pred;
        _decoder = dec;
        _loader = loader;
        _repoDir = repoDir;
    }

    /// <summary>Constructs a pipeline from already-loaded Kokoro/StyleTTS2 modules, without the Kokoro
    /// repo cache. The voice-pack <see cref="Synthesize"/> path is unavailable (no <c>voices/</c> dir);
    /// use <see cref="SynthesizeFromStyle"/> with an externally-computed style vector. This is the reuse
    /// entry point for StyleTTS 2, which shares these exact modules but loads them from its own weights.</summary>
    public KokoroPipeline(KokoroConfig cfg, KokoroPhonemeTokenizer tok, KokoroPlBert plBert,
        KokoroTextEncoder textEnc, KokoroProsodyPredictor pred, KokoroIStftNetDecoder dec)
        : this(cfg, tok, plBert, textEnc, pred, dec, loader: null, repoDir: "")
    {
    }

    /// <summary>Loads the Kokoro-82M pipeline. Downloads the model.safetensors + config.json
    /// into the SharpInference cache on first use. Voice packs are loaded lazily by
    /// <see cref="Synthesize"/>.</summary>
    public static async Task<KokoroPipeline> LoadAsync(CancellationToken ct = default)
    {
        Task<string> tfWeights = AudioModelCache.GetAsync("hexgrad/Kokoro-82M", "model.safetensors", ct: ct);
        Task<string> tfConfig = AudioModelCache.GetAsync("hexgrad/Kokoro-82M", "config.json", ct: ct);
        await Task.WhenAll(tfWeights, tfConfig).ConfigureAwait(false);

        KokoroConfig cfg = KokoroConfig.V1;
        KokoroPhonemeTokenizer tok = KokoroPhonemeTokenizer.LoadFromConfig(tfConfig.Result);

        SafeTensorsLoader loader = new();
        loader.Load(tfWeights.Result);
        IReadOnlyDictionary<string, Tensor> weights = loader.GetAllTensors();

        KokoroPlBert plBert = new(cfg);
        plBert.LoadWeights(weights);

        KokoroTextEncoder textEnc = new(cfg);
        textEnc.LoadWeights(weights);

        KokoroProsodyPredictor pred = new(cfg);
        pred.LoadWeights(weights);

        KokoroIStftNetDecoder dec = new(cfg);
        dec.LoadWeights(weights);

        string repoDir = AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M");
        return new KokoroPipeline(cfg, tok, plBert, textEnc, pred, dec, loader, repoDir);
    }

    /// <summary>Synthesizes audio from an IPA phoneme string. <paramref name="voiceName"/>
    /// must match a file under <c>voices/{voiceName}.bin</c> in the Kokoro repo cache
    /// (e.g. <c>"af_heart"</c>). <paramref name="speed"/> scales the predicted durations
    /// (1.0 = natural; 1.5 = faster, 0.7 = slower).</summary>
    public float[] Synthesize(IBackend backend, string phonemes, string voiceName = "af_heart", float speed = 1f)
    {
        ThrowIfDisposed();
        KokoroVoicePack pack = GetOrLoadVoicePack(voiceName);
        int[] tokenIds = _tokenizer.Encode(phonemes);
        if (tokenIds.Length < 3)
            throw new ArgumentException("phonemes must encode to at least one visible token.");

        // Voice-pack slice for the predictor + decoder, then run the shared core.
        using Tensor style = pack.GetStyle(tokenIds.Length);
        (Tensor sDec, Tensor sPred) = KokoroVoicePack.SplitStyle(style);
        try
        {
            return SynthesizeCore(backend, tokenIds, sDec, sPred, speed);
        }
        finally
        {
            sDec.Dispose();
            sPred.Dispose();
        }
    }

    /// <summary>Synthesizes from an externally-provided 256-d style vector — the reuse entry point for
    /// StyleTTS 2, whose decoder/predictor style halves come from a reference-audio <c>StyleEncoder</c>
    /// or the diffusion style sampler rather than a Kokoro voice pack. The vector is split
    /// <c>[:128] → decoder (acoustic)</c>, <c>[128:] → predictor (prosodic)</c>, matching the voice-pack
    /// convention.</summary>
    public float[] SynthesizeFromStyle(IBackend backend, string phonemes, Tensor refStyle256, float speed = 1f)
    {
        ThrowIfDisposed();
        int[] tokenIds = _tokenizer.Encode(phonemes);
        if (tokenIds.Length < 3)
            throw new ArgumentException("phonemes must encode to at least one visible token.");
        (Tensor sDec, Tensor sPred) = KokoroVoicePack.SplitStyle(refStyle256);
        try
        {
            return SynthesizeCore(backend, tokenIds, sDec, sPred, speed);
        }
        finally
        {
            sDec.Dispose();
            sPred.Dispose();
        }
    }

    /// <summary>The shared PLBERT → TextEncoder → duration → length-regulate → F0/N → decoder path.
    /// Borrows <paramref name="sDec"/> / <paramref name="sPred"/> (the caller owns + disposes them).</summary>
    private float[] SynthesizeCore(IBackend backend, int[] tokenIds, Tensor sDec, Tensor sPred, float speed)
    {
        // PLBERT → d_bert [1, T, 512].
        using Tensor dBert = _plBert.Forward(backend, tokenIds);
        // TextEncoder → text_features [1, T, 512].
        using Tensor textFeatures = _textEncoder.Forward(backend, tokenIds);

        // Predict durations.
        (Tensor durFeatures, int[] durations) = _predictor.PredictDurations(backend, dBert, sPred, speed);
        durFeatures.Dispose();

        int tTotal = 0;
        for (int i = 0; i < durations.Length; i++) tTotal += durations[i];
        if (tTotal == 0) tTotal = durations.Length;

        // Length-regulate d_bert + text_features → channels-first [1, 512, T_total].
        Tensor dBertExpanded = LengthRegulate(dBert, durations, tTotal);
        Tensor asr = LengthRegulate(textFeatures, durations, tTotal);
        try
        {
            (Tensor f0, Tensor n) = _predictor.F0Ntrain(backend, dBertExpanded, sPred);
            try
            {
                return _decoder.Forward(backend, asr, f0, n, sDec);
            }
            finally
            {
                f0.Dispose();
                n.Dispose();
            }
        }
        finally
        {
            dBertExpanded.Dispose();
            asr.Dispose();
        }
    }

    /// <summary>Repeats each phoneme of a <c>[1, T, C]</c> channels-last feature tensor by
    /// the predicted duration count, producing a <c>[1, C, T_total]</c> channels-first
    /// tensor ready for the decoder's conv stack. This is the Kokoro length-regulator —
    /// equivalent to building a one-hot alignment matrix and matmul-ing, but a simple
    /// repeat is cheaper and exact.</summary>
    private static unsafe Tensor LengthRegulate(Tensor featuresCL, int[] durations, int tTotal)
    {
        int t = (int)featuresCL.Shape[1];
        int c = (int)featuresCL.Shape[2];
        if (durations.Length != t)
            throw new ArgumentException($"durations length {durations.Length} does not match T {t}.");
        Tensor expanded = new(new TensorShape(1, c, tTotal), DType.F32);
        float* src = (float*)featuresCL.DataPointer;
        float* dst = (float*)expanded.DataPointer;

        int outFrame = 0;
        for (int i = 0; i < t; i++)
        {
            int d = durations[i];
            if (d <= 0) continue;
            // Source row for phoneme i lives at offset i*C in the channels-last tensor.
            // Destination is channels-first, so for each channel cc we write across
            // [outFrame .. outFrame+d) at row cc * tTotal.
            for (int cc = 0; cc < c; cc++)
            {
                float val = src[i * c + cc];
                long rowBase = (long)cc * tTotal + outFrame;
                for (int j = 0; j < d; j++) dst[rowBase + j] = val;
            }
            outFrame += d;
        }
        return expanded;
    }

    private KokoroVoicePack GetOrLoadVoicePack(string voiceName)
    {
        if (_voicePackCache.TryGetValue(voiceName, out KokoroVoicePack? cached)) return cached;
        string path = Path.Combine(_repoDir, "voices", $"{voiceName}.bin");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Voice pack '{voiceName}' not found at '{path}'. Available voices live under {_repoDir}/voices/.");
        KokoroVoicePack pack = KokoroVoicePack.LoadFromFile(path);
        _voicePackCache[voiceName] = pack;
        return pack;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(KokoroPipeline));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (KokoroVoicePack pack in _voicePackCache.Values) pack.Dispose();
        _voicePackCache.Clear();
        _loader?.Dispose();
    }
}
