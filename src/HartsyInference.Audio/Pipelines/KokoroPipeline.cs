using System.IO;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Diagnostics;
using HartsyInference.Audio.Models.Kokoro;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Metadata;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>End-to-end Kokoro-82M TTS pipeline. Wires the four submodules (PLBERT,
/// TextEncoder, ProsodyPredictor, iSTFTNetDecoder) and the voice-pack-driven style
/// path into a single <see cref="Synthesize"/> call.
///
/// <para>The pipeline is phoneme-in / audio-out: it takes the misaki-style IPA that
/// <see cref="HartsyInference.Audio.Frontends.EnglishG2P"/> produces, the symbol set Kokoro was trained on.</para>
///
/// <para>Pipeline shape:</para>
/// <code>
///   IPA phonemes
///       │  KokoroPhonemeTokenizer
///       ▼
///   token_ids [1, T+2]                          ← BOS/EOS padded
///       │
///       ▼  voice_pack.GetStyle(T)
///   style [1, 256] ──┬──→ s_dec  [1, 128]
///                    └──→ s_pred [1, 128]
///       │
///       ├──► PLBERT(token_ids) → d_bert [1, T+2, 512]
///       │       │
///       │       ▼  + s_pred  → DurationEncoder → d [1, T+2, 640] → LSTM + duration_proj
///       │      durations [T+2] (int)
///       │       │
///       │       ▼  build alignment matrix from durations → length regulator
///       │      en = d expanded [1, 640, T_total]      T_total = sum(durations)
///       │       │
///       │       ▼  predictor.shared + F0 chain + N chain (s_pred)
///       │      F0, N  [1, 1, 2*T_total]
///       │
///       └──► TextEncoder(token_ids) → text_features [1, T+2, 512]
///              │
///              ▼  length-regulate via SAME alignment as d
///             asr [1, 512, T_total]
///
///   asr, F0, N, s_dec  →  KokoroIStftNetDecoder.Forward  →  audio (24 kHz float)
/// </code></summary>
public sealed class KokoroPipeline : IDisposable
{
    // Single-file repack hosted by tools/repack: the official hexgrad checkpoint flattened +
    // inner-`module.`-stripped into one safetensors with the flat keys the submodules expect.
    // Voice packs still come from the original hexgrad repo (see _repoDir), so only the heavy
    // weights load from here.
    private const string RepackRepo = "Hartsy/kokoro-82m-safetensors";
    private const string RepackFile = "kokoro-82m.safetensors";
    // sha256 of RepackFile, pinned from out/kokoro-82m.manifest.json once the file is published.
    // Empty until then; when non-empty, LoadAsync verifies the download against it.
    private const string RepackSha256 = "";

    private readonly KokoroConfig _cfg;
    private readonly KokoroPhonemeTokenizer _tokenizer;
    private readonly KokoroPlBert _plBert;
    private readonly KokoroTextEncoder _textEncoder;
    private readonly KokoroProsodyPredictor _predictor;
    private readonly KokoroIStftNetDecoder _decoder;
    private readonly IDisposable? _loader;
    private readonly Dictionary<string, KokoroVoicePack> _voicePackCache = new(StringComparer.Ordinal);
    private readonly string _repoDir;
    private IBackend? _residentBackend;
    private Action<string>? _testStageObserver;
    private int _disposed;

    public KokoroConfig Config => _cfg;

    /// <summary>Called with each stage's name as a synthesis passes the boundary after it (<c>"start"</c> before the
    /// first), ahead of that boundary's cancellation check, so a test can cancel a call at a known stage. Covers the
    /// decoder's own boundaries too.</summary>
    internal Action<string>? TestStageObserver
    {
        get => _testStageObserver;
        set
        {
            _testStageObserver = value;
            _decoder.TestStageObserver = value;
        }
    }

    /// <summary>Called with each chunk's predicted durations and its F0 and energy curves (<c>[1, 1, 2*T_total]</c>,
    /// borrowed for the call) as the prosody predictor returns them, so a parity test can diff them against the
    /// reference model.</summary>
    internal Action<int[], Tensor, Tensor>? TestProsodyObserver { get; set; }

    /// <summary>The token count <see cref="Synthesize"/> sees for <paramref name="phonemes"/> (diagnostics: the text
    /// encoder and the voice-pack row depend on it).</summary>
    internal int CountTokens(string phonemes) => _tokenizer.Encode(phonemes).Length;

    public string ModelName => "hexgrad/Kokoro-82M";

    private KokoroPipeline(KokoroConfig cfg, KokoroPhonemeTokenizer tok, KokoroPlBert plBert, KokoroTextEncoder textEnc,
        KokoroProsodyPredictor pred, KokoroIStftNetDecoder dec, IDisposable? loader, string repoDir)
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

    /// <summary>Loads the Kokoro-82M pipeline. Downloads the repacked weights (<c>kokoro-82m.safetensors</c>) +
    /// config.json into the HartsyInference cache on first use. Voice packs are loaded lazily by <see cref="Synthesize"/>.
    /// <para>The weights are the repacked single-file safetensors produced by <c>tools/repack</c>:
    /// the offline step did the recursive flatten + inner-<c>module.</c> strip that the runtime used to
    /// do, so the file already carries the fully-qualified dotted keys (<c>bert.embeddings.…</c>,
    /// <c>bert_encoder.weight</c>, …) the submodules' <c>LoadWeights</c> expect.</para></summary>
    public static async Task<KokoroPipeline> LoadAsync(CancellationToken ct = default)
    {
        // Config + voice packs come from the canonical hexgrad repo. Weights: prefer the pre-flattened
        // single-file repack, but if it isn't published/reachable, download the canonical kokoro-v1_0.pth and
        // do the same flatten + inner-`module.`-strip in-engine (cached once), so Kokoro always installs & runs
        // like any other model — no dependency on a separately-hosted repack that may not exist.
        string configPath = await AudioModelCache.GetAsync("hexgrad/Kokoro-82M", "config.json", category: "tts", ct: ct).ConfigureAwait(false);
        string weightsPath = await EnsureRepackedWeightsAsync(ct).ConfigureAwait(false);

        KokoroConfig cfg = KokoroConfig.V1;
        KokoroPhonemeTokenizer tok = KokoroPhonemeTokenizer.LoadFromConfig(configPath);

        SafeTensorsLoader loader = new();
        loader.Load(weightsPath);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();

        KokoroPlBert plBert = new(cfg);
        plBert.LoadWeights(weights);

        KokoroTextEncoder textEnc = new(cfg);
        textEnc.LoadWeights(weights);

        KokoroProsodyPredictor pred = new(cfg);
        pred.LoadWeights(weights);

        KokoroIStftNetDecoder dec = new(cfg);
        dec.LoadWeights(weights);

        string repoDir = AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M", "tts");
        return new KokoroPipeline(cfg, tok, plBert, textEnc, pred, dec, loader, repoDir);
    }

    /// <summary>Resolves the flattened single-file Kokoro weights. Prefers the published repack; if it isn't
    /// reachable (e.g. not published), downloads the canonical hexgrad <c>kokoro-v1_0.pth</c> and converts it
    /// once into <c>kokoro-82m.safetensors</c> beside it — recursive flatten of the dict-of-state-dicts + strip
    /// the inner <c>module.</c> wrapper (nn.DataParallel), the exact transform <c>tools/repack</c> bakes offline,
    /// producing the flat dotted keys the submodule <c>LoadWeights</c> methods expect.</summary>
    private static async Task<string> EnsureRepackedWeightsAsync(CancellationToken ct)
    {
        try
        {
            string repack = await AudioModelCache.GetAsync(RepackRepo, RepackFile, category: "tts", ct: ct).ConfigureAwait(false);
            if (RepackSha256.Length != 0)
                AudioModelCache.VerifySha256(repack, RepackSha256);
            return repack;
        }
        catch (Exception ex)
        {
            string pth = await AudioModelCache.GetAsync("hexgrad/Kokoro-82M", "kokoro-v1_0.pth", category: "tts", ct: ct).ConfigureAwait(false);
            string outPath = Path.Combine(Path.GetDirectoryName(pth)!, RepackFile);
            if (!File.Exists(outPath))
            {
                HartsyInference.Core.Logging.Logs.Info(
                    $"[Kokoro] Repack '{RepackRepo}/{RepackFile}' unavailable ({ex.Message}); converting canonical kokoro-v1_0.pth → {RepackFile} (one-time).");
                // Strip the nn.DataParallel `module.` wrapper — the exact transform tools/repack bakes offline.
                // Stamped as it is written: this file is what a user's SwarmUI scans, and an unstamped one classifies
                // as null, which hides every Kokoro parameter in the UI.
                PickleCheckpointRepacker.Repack(pth, outPath, k => k.Replace(".module.", "."), recursiveFlatten: true,
                    metadata: RepackMetadata(pth));
            }
            return outPath;
        }
    }

    /// <summary>Identity for the locally converted repack, so a file the engine produced on a user's machine is as
    /// self-describing as one we publish. Null when the catalog does not know this family, which leaves the file
    /// unstamped rather than stamped with a guess.</summary>
    private static IReadOnlyDictionary<string, string>? RepackMetadata(string sourcePath)
    {
        ArtifactIdentity? identity = ModelIdentityCatalog.Find("kokoro");
        if (identity is null)
        {
            return null;
        }
        return ArtifactMetadata.ForRepack(identity, ArtifactProvenance.FromSourceFile(
            "HartsyInference.PickleCheckpointRepacker", ArtifactProvenance.MainComponent, sourcePath,
            sourceRepo: "hexgrad/Kokoro-82M") with { ModelId = "default" });
    }

    /// <summary>Synthesizes audio from an IPA phoneme string. <paramref name="voiceName"/>
    /// must match a file under <c>voices/{voiceName}.bin</c> in the Kokoro repo cache
    /// (e.g. <c>"af_heart"</c>), or name a blend of such packs (<see cref="KokoroVoiceMix"/>, e.g.
    /// <c>"af_bella,af_sky"</c>). <paramref name="speed"/> scales the predicted durations
    /// (1.0 = natural; 1.5 = faster, 0.7 = slower). A string longer than PLBERT can take is split by
    /// <see cref="KokoroPhonemeChunker"/> and the pieces' audio concatenated, as the reference pipeline does.
    /// <paramref name="cancel"/> is checked at every stage boundary,
    /// from before PLBERT to just before the iSTFT head; a cancelled call throws
    /// <see cref="OperationCanceledException"/> at the next one.</summary>
    public float[] Synthesize(IBackend backend, string phonemes, string voiceName = "af_heart", float speed = 1f,
        CancellationToken cancel = default)
    {
        ThrowIfDisposed();
        KokoroVoicePack pack = GetOrLoadVoicePack(voiceName);
        return SynthesizeChunked(phonemes, tokenIds =>
        {
            // The reference picks the style row by phoneme count (pack[len(ps) - 1]), not counting the pads.
            using Tensor style = pack.GetStyle(tokenIds.Length - 2);
            (Tensor sDec, Tensor sPred) = KokoroVoicePack.SplitStyle(style);
            try
            {
                return SynthesizeCore(backend, tokenIds, sDec, sPred, speed, cancel);
            }
            finally
            {
                sDec.Dispose();
                sPred.Dispose();
            }
        });
    }

    /// <summary>Synthesizes from an externally-provided 256-d style vector — the reuse entry point for
    /// StyleTTS 2, whose decoder/predictor style halves come from a reference-audio <c>StyleEncoder</c>
    /// or the diffusion style sampler rather than a Kokoro voice pack. The vector is split
    /// <c>[:128] → decoder (acoustic)</c>, <c>[128:] → predictor (prosodic)</c>, matching the voice-pack
    /// convention. Long input is chunked and <paramref name="cancel"/> checked as <see cref="Synthesize"/> does.</summary>
    public float[] SynthesizeFromStyle(IBackend backend, string phonemes, Tensor refStyle256, float speed = 1f,
        CancellationToken cancel = default)
    {
        ThrowIfDisposed();
        (Tensor sDec, Tensor sPred) = KokoroVoicePack.SplitStyle(refStyle256);
        try
        {
            return SynthesizeChunked(phonemes, tokenIds => SynthesizeCore(backend, tokenIds, sDec, sPred, speed, cancel));
        }
        finally
        {
            sDec.Dispose();
            sPred.Dispose();
        }
    }

    /// <summary>Tokenizes each <see cref="KokoroPhonemeChunker"/> chunk, synthesizes it, and concatenates the audio;
    /// chunks with no in-vocab symbol are skipped.</summary>
    private float[] SynthesizeChunked(string phonemes, Func<int[], float[]> synthesize)
    {
        ArgumentNullException.ThrowIfNull(phonemes);
        List<float[]> parts = new();
        long total = 0;
        foreach (string chunk in KokoroPhonemeChunker.Split(phonemes))
        {
            int[] tokenIds = _tokenizer.Encode(chunk);
            if (tokenIds.Length < 3) continue;
            float[] audio = synthesize(tokenIds);
            parts.Add(audio);
            total += audio.Length;
        }
        if (parts.Count == 0)
            throw new ArgumentException("phonemes must encode to at least one visible token.", nameof(phonemes));
        if (parts.Count == 1) return parts[0];
        float[] joined = new float[total];
        int offset = 0;
        foreach (float[] part in parts)
        {
            part.CopyTo(joined, offset);
            offset += part.Length;
        }
        return joined;
    }

    /// <summary>The shared PLBERT → TextEncoder → duration → length-regulate → F0/N → decoder path.
    /// Borrows <paramref name="sDec"/> / <paramref name="sPred"/> (the caller owns + disposes them).</summary>
    private float[] SynthesizeCore(IBackend backend, int[] tokenIds, Tensor sDec, Tensor sPred, float speed,
        CancellationToken cancel)
    {
        // Ahead of any device work, so a call cancelled while it waited for the device costs nothing.
        KokoroOps.StageBoundary(_testStageObserver, "start", cancel);
        EnsureWeightsResident(backend);
        StageTimer? timer = StageTimer.Start(backend, "Kokoro");
        // The two style halves feed every AdaIN in the graph; resident for the call, they cost one upload
        // instead of one per consumer. Released before the caller disposes them.
        Tensor[] styles = [sDec, sPred];
        backend.PreloadWeights(styles);
        try
        {
            // PLBERT → d_bert [1, T, 512].
            using Tensor dBert = _plBert.Forward(backend, tokenIds);
            timer?.Mark("plbert");
            KokoroOps.StageBoundary(_testStageObserver, "plbert", cancel);
            // TextEncoder → text_features [1, T, 512].
            using Tensor textFeatures = _textEncoder.Forward(backend, tokenIds);
            timer?.Mark("textenc");
            KokoroOps.StageBoundary(_testStageObserver, "textenc", cancel);

            // Predict durations. durFeatures is the DurationEncoder output d [1, T, 640] the F0/N branch reads.
            (Tensor durFeatures, int[] durations) = _predictor.PredictDurations(backend, dBert, sPred, speed);
            timer?.Mark("durations");
            Tensor? enExpanded = null;
            Tensor? asr = null;
            try
            {
                KokoroOps.StageBoundary(_testStageObserver, "durations", cancel);

                // Every predicted duration is clamped to ≥ 1 frame, so T_total ≥ T and the alignment is total.
                int tTotal = 0;
                for (int i = 0; i < durations.Length; i++) tTotal += durations[i];

                // Length-regulate d + text_features → channels-first [1, C, T_total].
                int[] frameToPhoneme = AlignmentIndices(durations, tTotal);
                enExpanded = LengthRegulate(backend, durFeatures, frameToPhoneme);
                asr = LengthRegulate(backend, textFeatures, frameToPhoneme);
                timer?.Mark("regulate");
                KokoroOps.StageBoundary(_testStageObserver, "regulate", cancel);
                (Tensor f0, Tensor n) = _predictor.F0Ntrain(backend, enExpanded, sPred);
                timer?.Mark("f0n");
                try
                {
                    TestProsodyObserver?.Invoke(durations, f0, n);
                    KokoroOps.StageBoundary(_testStageObserver, "f0n", cancel);
                    float[] audio = _decoder.Forward(backend, asr, f0, n, sDec, cancel);
                    timer?.Mark("decoder");
                    timer?.Report($"synth T={tokenIds.Length} T_total={tTotal}");
                    return audio;
                }
                finally
                {
                    f0.Dispose();
                    n.Dispose();
                }
            }
            finally
            {
                durFeatures.Dispose();
                enExpanded?.Dispose();
                asr?.Dispose();
            }
        }
        finally
        {
            backend.FreeWeights(styles);
        }
    }

    /// <summary>Uploads every submodule weight to the backend once. Without this the small tensors (AdaIN
    /// projections, biases, Snake alphas — each under the auto-promotion floor) re-upload on every op of every call.
    /// Unsynchronized like the rest of the pipeline (one synthesis at a time, the voice-pack cache's contract). A
    /// repeated preload is a no-op for already-resident weights, and the backend is recorded only after the preload
    /// succeeds, so one that throws part-way is retried by the next call. The device copies live and die with the
    /// backend.</summary>
    private void EnsureWeightsResident(IBackend backend)
    {
        if (ReferenceEquals(_residentBackend, backend)) return;
        backend.PreloadWeights(EnumerateWeights());
        _residentBackend = backend;
    }

    /// <summary>Every device-side weight of the four submodules.</summary>
    internal IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _plBert.EnumerateWeights()) yield return t;
        foreach (Tensor t in _textEncoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _predictor.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
    }

    /// <summary>The phoneme index of every output frame: phoneme <c>i</c> owns <c>durations[i]</c> consecutive frames.</summary>
    private static int[] AlignmentIndices(int[] durations, int tTotal)
    {
        int[] indices = new int[tTotal];
        int frame = 0;
        for (int i = 0; i < durations.Length; i++)
        {
            for (int j = 0; j < durations[i]; j++) indices[frame++] = i;
        }
        if (frame != tTotal)
            throw new HartsyInferenceException($"Kokoro alignment covers {frame} frames but T_total is {tTotal}.");
        return indices;
    }

    /// <summary>Repeats each phoneme of a <c>[1, T, C]</c> channels-last feature tensor by
    /// the predicted duration count, producing a <c>[1, C, T_total]</c> channels-first
    /// tensor ready for the decoder's conv stack. This is the Kokoro length-regulator —
    /// equivalent to building a one-hot alignment matrix and matmul-ing, but a row gather
    /// is cheaper and exact. Runs as two backend ops, so the features never leave the device.</summary>
    private static Tensor LengthRegulate(IBackend backend, Tensor featuresCL, int[] frameToPhoneme)
    {
        int c = (int)featuresCL.Shape[2];
        int tTotal = frameToPhoneme.Length;
        Tensor gathered = new(new TensorShape(1, tTotal, c), DType.F32);
        backend.GatherRows(gathered, featuresCL, frameToPhoneme);
        Tensor expanded = new(new TensorShape(1, c, tTotal), DType.F32);
        backend.Transpose2D(expanded, gathered, tTotal, c);
        gathered.Dispose();
        return expanded;
    }

    private KokoroVoicePack GetOrLoadVoicePack(string voiceName)
    {
        if (_voicePackCache.TryGetValue(voiceName, out KokoroVoicePack? cached)) return cached;
        KokoroVoiceMix mix = KokoroVoiceMix.Parse(voiceName);
        if (mix.IsBlend)
        {
            // A blend such as "af_bella,af_sky" (KPipeline.load_voice): the component packs, then their mean.
            KokoroVoicePack[] parts = mix.Parts.Select(p => GetOrLoadVoicePack(p.Voice)).ToArray();
            KokoroVoicePack blend = mix.Blend(voiceName, parts);
            _voicePackCache[voiceName] = blend;
            return blend;
        }
        voiceName = mix.PrimaryVoice;
        if (_voicePackCache.TryGetValue(voiceName, out cached)) return cached;
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
