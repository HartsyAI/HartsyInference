using System.Text.Json;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Onnx;
using HartsyInference.Phonemizer;
using HartsyInference.Phonemizer.Espeak;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Piper TTS pipeline over the shared <see cref="VitsSynthesizer"/>. A Piper voice is an ONNX model plus an
/// <c>.onnx.json</c> config (espeak voice, inference noise scales, phoneme id map). <see cref="LoadFromFiles"/> /
/// <see cref="LoadAsync"/> load the voice, infer the VITS architecture from the weights, and wire the espeak
/// phonemizer; <see cref="SynthesizeText"/> then goes text → espeak IPA → token ids → audio.</summary>
public sealed class PiperPipeline : IDisposable
{
    private readonly VitsConfig _cfg;
    private readonly VitsSynthesizer _synth;
    private readonly OnnxWeightLoader? _loader;
    private readonly IPhonemizer? _phonemizer;
    private readonly PhonemeIdMap? _idMap;
    private readonly string _espeakVoice;
    private readonly float _noiseScale, _lengthScale;
    private int _disposed;

    public int SampleRate => _cfg.SampleRate;

    public PiperPipeline(VitsConfig cfg)
    {
        _cfg = cfg;
        _synth = new VitsSynthesizer(cfg);
        _espeakVoice = "en-us";
        _noiseScale = cfg.NoiseScale;
        _lengthScale = cfg.LengthScale;
    }

    private PiperPipeline(VitsConfig cfg, VitsSynthesizer synth, OnnxWeightLoader loader, IPhonemizer phonemizer,
        PhonemeIdMap idMap, string espeakVoice, float noiseScale, float lengthScale)
    {
        _cfg = cfg;
        _synth = synth;
        _loader = loader;
        _phonemizer = phonemizer;
        _idMap = idMap;
        _espeakVoice = espeakVoice;
        _noiseScale = noiseScale;
        _lengthScale = lengthScale;
    }

    /// <summary>Builds a pipeline from a local Piper <c>.onnx</c> + <c>.onnx.json</c>. The phonemizer defaults to the
    /// bundled pure-C# espeak port (<see cref="EspeakPhonemizer.FromCache"/>); pass one to override.</summary>
    public static PiperPipeline LoadFromFiles(string onnxPath, string configJsonPath, IPhonemizer? phonemizer = null)
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(configJsonPath));
        JsonElement root = doc.RootElement;

        int sampleRate = root.GetProperty("audio").GetProperty("sample_rate").GetInt32();
        string espeakVoice = root.GetProperty("espeak").GetProperty("voice").GetString() ?? "en-us";
        JsonElement inf = root.GetProperty("inference");
        float noiseScale = inf.GetProperty("noise_scale").GetSingle();
        float lengthScale = inf.GetProperty("length_scale").GetSingle();
        float noiseW = inf.GetProperty("noise_w").GetSingle();
        int numSymbols = root.TryGetProperty("num_symbols", out JsonElement ns) ? ns.GetInt32() : 256;

        OnnxWeightLoader loader = new();
        loader.Load(onnxPath);
        Dictionary<string, Tensor> weights = loader.GetResolvedTensors();

        VitsConfig cfg = InferConfig(weights, sampleRate, numSymbols, noiseScale, lengthScale, noiseW);
        VitsSynthesizer synth = new(cfg);
        synth.LoadWeights(weights);

        using FileStream cfgStream = File.OpenRead(configJsonPath);
        PhonemeIdMap idMap = PhonemeIdMap.FromPiperConfig(cfgStream);
        IPhonemizer phon = phonemizer ?? EspeakPhonemizer.FromCache();

        return new PiperPipeline(cfg, synth, loader, phon, idMap, espeakVoice, noiseScale, lengthScale);
    }

    /// <summary>Downloads a Piper voice from HuggingFace (<c>rhasspy/piper-voices</c> by default) and loads it.
    /// <paramref name="voiceId"/> is e.g. <c>"en_US-lessac-medium"</c>; the in-repo path is derived from it.</summary>
    public static async Task<PiperPipeline> LoadAsync(string voiceId = "en_US-lessac-medium",
        string repo = "rhasspy/piper-voices", IPhonemizer? phonemizer = null, CancellationToken ct = default)
    {
        string inRepo = VoiceRepoPath(voiceId);
        Task<string> onnx = AudioModelCache.GetAsync(repo, $"{inRepo}.onnx", ct: ct);
        Task<string> json = AudioModelCache.GetAsync(repo, $"{inRepo}.onnx.json", ct: ct);
        await Task.WhenAll(onnx, json).ConfigureAwait(false);
        return LoadFromFiles(onnx.Result, json.Result, phonemizer);
    }

    /// <summary>Synthesizes audio for <paramref name="text"/> end-to-end (text → IPA → token ids → waveform).</summary>
    public float[] SynthesizeText(IBackend backend, string text, float? lengthScale = null, float? noiseScale = null, int seed = 0)
    {
        ThrowIfDisposed();
        if (_phonemizer is null || _idMap is null)
            throw new InvalidOperationException("Use LoadFromFiles/LoadAsync to enable text synthesis.");
        int[] tokens = _phonemizer.PhonemizeToIds(text, _espeakVoice, _idMap);
        return _synth.Infer(backend, tokens, lengthScale ?? _lengthScale, noiseScale ?? _noiseScale, seed);
    }

    /// <summary>Synthesizes from a fully-formed Piper token sequence (bos/pad/eos already interleaved).</summary>
    public float[] SynthesizeTokens(IBackend backend, ReadOnlySpan<int> tokens, float? lengthScale = null, float? noiseScale = null, int seed = 0)
    {
        ThrowIfDisposed();
        return _synth.Infer(backend, tokens, lengthScale ?? _lengthScale, noiseScale ?? _noiseScale, seed);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w) => _synth.LoadWeights(w);

    public IEnumerable<Tensor> EnumerateWeights() => _synth.EnumerateWeights();

    // Infer the Piper "medium"/"high" architecture from the decoder weights (channel count + resblock type), then
    // overlay the runtime values from the voice config.
    private static VitsConfig InferConfig(IReadOnlyDictionary<string, Tensor> w, int sampleRate, int numSymbols,
        float noiseScale, float lengthScale, float noiseW)
    {
        int initialChannel = w.TryGetValue("dec.conv_pre.weight", out Tensor? pre) ? (int)pre.Shape[0] : 256;
        bool type1 = w.ContainsKey("dec.resblocks.0.convs2.0.weight");
        VitsConfig baseCfg = type1 || initialChannel >= 512 ? VitsConfig.PiperHigh : VitsConfig.PiperMedium;
        return baseCfg with
        {
            SampleRate = sampleRate,
            NumVocab = numSymbols,
            NoiseScale = noiseScale,
            LengthScale = lengthScale,
            NoiseScaleW = noiseW,
            UpsampleInitialChannel = initialChannel,
        };
    }

    // "en_US-lessac-medium" -> "en/en_US/lessac/medium/en_US-lessac-medium". Pass a path directly to bypass.
    private static string VoiceRepoPath(string voiceId)
    {
        if (voiceId.Contains('/')) return voiceId;
        string[] dash = voiceId.Split('-');
        if (dash.Length < 3) return voiceId;
        string locale = dash[0];                       // en_US
        string lang = locale.Split('_')[0];            // en
        string name = dash[1];                         // lessac
        string quality = dash[2];                      // medium
        return $"{lang}/{locale}/{name}/{quality}/{voiceId}";
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(PiperPipeline));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _synth.Dispose();
        _loader?.Dispose();
        GC.SuppressFinalize(this);
    }
}
