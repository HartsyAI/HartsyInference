using System.Diagnostics;
using System.Text;
using HartsyInference.Audio.Models.PocketTts;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Kyutai Pocket-TTS pipeline (continuous-latent flow-LM @ 12.5 Hz → 24 kHz). Wraps the real-weight
/// bit-exact cores — <see cref="PocketTtsFlowLm"/> (LUT text conditioner + input_linear + bos + streaming
/// transformer + out_norm + flow_net + out_eos) and <see cref="PocketTtsMimiDecoder"/> — plus the SentencePiece
/// tokenizer and predefined voices. Text → SPM ids → conditioner embed; the AR loop is <b>primed by a voice
/// KV state</b> (a voice is required — with none the model end-of-sequences into near-silence); per frame it
/// samples a 32-d latent (<c>noise + flow_net</c>) and stops after <c>out_eos</c> fires; latents are
/// de-normalised (<c>·emb_std + emb_mean</c>) and Mimi-decoded to PCM.</summary>
public sealed unsafe class PocketTtsPipeline : IDisposable
{
    private readonly PocketTtsConfig _cfg;
    private readonly PocketTtsFlowLm _flm;
    private readonly PocketTtsMimiDecoder _decoder;
    private readonly PocketTtsTokenizer _tokenizer;
    private readonly Dictionary<string, PocketTtsVoice> _voices = new(StringComparer.OrdinalIgnoreCase);
    private readonly List<IDisposable> _retain = new();
    private int _disposed;

    public int SampleRate => _cfg.SampleRate;
    public PocketTtsConfig Config => _cfg;

    private PocketTtsPipeline(PocketTtsConfig cfg, PocketTtsFlowLm flm, PocketTtsMimiDecoder dec, PocketTtsTokenizer tok)
    {
        _cfg = cfg;
        _flm = flm;
        _decoder = dec;
        _tokenizer = tok;
    }

    /// <summary>Loads the FlowLM + Mimi decoder from the released <c>tts_*.safetensors</c> and the SentencePiece
    /// <c>tokenizer.model</c>.</summary>
    public static PocketTtsPipeline LoadFromCheckpoint(string weightsPath, string spmPath, PocketTtsConfig? config = null)
    {
        PocketTtsConfig cfg = config ?? PocketTtsConfig.Default;
        SafeTensorsLoader wl = new();
        wl.Load(weightsPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();

        PocketTtsFlowLm flm = new("flow_lm");
        flm.LoadWeights(w);
        PocketTtsMimiDecoder dec = new("mimi");
        dec.LoadWeights(w);
        PocketTtsTokenizer tok = new(spmPath);

        PocketTtsPipeline p = new(cfg, flm, dec, tok);
        p._retain.Add(wl);
        return p;
    }

    /// <summary>Registers a predefined voice from its KV-state safetensors
    /// (<c>languages/&lt;lang&gt;/embeddings/&lt;name&gt;.safetensors</c>).</summary>
    public void RegisterVoiceFromFile(string name, string voiceSafetensorsPath)
    {
        ThrowIfDisposed();
        SafeTensorsLoader vl = new();
        vl.Load(voiceSafetensorsPath);
        PocketTtsVoice voice = PocketTtsVoice.Load(vl.GetAllTensors());
        vl.Dispose();   // Load copies into fresh F32 tensors, so the loader isn't retained.
        if (_voices.Remove(name, out PocketTtsVoice? old)) old.Dispose();
        _voices[name] = voice;
    }

    public bool HasVoice(string name) => _voices.ContainsKey(name);

    /// <summary>Synthesizes 24 kHz mono PCM for <paramref name="text"/> in the registered voice
    /// <paramref name="voiceName"/> (required).</summary>
    public float[] Synthesize(IBackend backend, string text, string voiceName, int seed = 0)
    {
        ThrowIfDisposed();
        if (!_voices.TryGetValue(voiceName, out PocketTtsVoice? voice))
            throw new ArgumentException($"unknown Pocket-TTS voice '{voiceName}'; register it first.", nameof(voiceName));
        Stopwatch sw = Stopwatch.StartNew();

        string prepared = PrepareText(text);
        int[] ids = [.. _tokenizer.Encode(prepared)];
        if (ids.Length == 0) return [];
        // Upper bound on frames from a token-rate estimate (EOS stops earlier). Mirrors _estimate_max_gen_len.
        int maxFrames = Math.Min(_cfg.MaxFrames, (int)((ids.Length / 3.0 + 2.0) * _cfg.FrameRateHz) + 4);

        using Tensor textEmb = _flm.EmbedText(ids);
        using Tensor latents = _flm.GenerateVoiced(backend, textEmb, voice.PrefixK, voice.PrefixV, voice.PrefixLen,
            temp: _cfg.Temperature, noiseClamp: null, eosThreshold: _cfg.EosThreshold,
            framesAfterEos: 3, maxFrames: maxFrames, seed: seed);
        int n = (int)latents.Shape[2];

        Denormalize(latents, n);
        using Tensor audioT = _decoder.Forward(backend, latents, 1, n);
        int samples = (int)(audioT.ElementCount);
        float[] audio = new float[samples];
        fixed (float* dst = audio)
            Buffer.MemoryCopy((void*)audioT.DataPointer, dst, samples * 4L, samples * 4L);
        Logs.Info($"Pocket-TTS: {ids.Length} tokens → {n} frames → {samples} samples " +
            $"({samples / (double)SampleRate:F1}s) voice '{voiceName}' in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    // latents [1,32,N] channel-major: x[c,f] = x[c,f]*emb_std[c] + emb_mean[c].
    private void Denormalize(Tensor latents, int n)
    {
        float* lp = (float*)latents.DataPointer;
        float* std = (float*)_flm.EmbStd.DataPointer;
        float* mean = (float*)_flm.EmbMean.DataPointer;
        int ld = (int)latents.Shape[1];
        for (int c = 0; c < ld; c++)
            for (int f = 0; f < n; f++)
                lp[(long)c * n + f] = lp[(long)c * n + f] * std[c] + mean[c];
    }

    // Mirrors upstream prepare_text_prompt: normalize whitespace, capitalize, ensure trailing punctuation.
    private static string PrepareText(string text)
    {
        string t = (text ?? "").Replace('\n', ' ').Replace('\r', ' ').Trim();
        while (t.Contains("  ")) t = t.Replace("  ", " ");
        if (t.Length == 0) return t;
        StringBuilder sb = new(t);
        if (char.IsLower(sb[0])) sb[0] = char.ToUpperInvariant(sb[0]);
        char last = sb[sb.Length - 1];
        if (char.IsLetterOrDigit(last)) sb.Append('.');
        return sb.ToString();
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _flm.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (PocketTtsVoice v in _voices.Values) v.Dispose();
        _voices.Clear();
        foreach (IDisposable d in _retain) d.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(PocketTtsPipeline));
    }
}
