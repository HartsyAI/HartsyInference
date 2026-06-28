using System.Diagnostics;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.SparkTts;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Spark-TTS-0.5B text-to-speech pipeline. The Qwen2.5-0.5B LM emits the unified global +
/// semantic BiCodec token sequence; BiCodec decodes it to a 16 kHz waveform. <see cref="LoadFromDirectory"/>
/// / <see cref="LoadAsync"/> wire the LM + BiCodec + the Spark prompt tokenizer so
/// <see cref="SynthesizeControllable"/> can go straight from text + a coarse style (gender / pitch / speed)
/// to audio. The lower-level <see cref="Synthesize"/> takes pre-built prompt token ids (e.g. for zero-shot
/// voice cloning, where the caller supplies the reference clip's global tokens).</summary>
public sealed class SparkTtsPipeline : IDisposable
{
    private static readonly string[] Levels = ["very_low", "low", "moderate", "high", "very_high"];

    private readonly SparkTtsConfig _cfg;
    private readonly SparkTtsLm _lm;
    private readonly BiCodecDecoder _bicodec;
    private readonly SparkTtsTokenizer? _tokenizer;
    private readonly IDisposable[] _retain;
    private int _disposed;

    public int SampleRate => _cfg.SampleRate;

    public SparkTtsPipeline(SparkTtsConfig cfg, SparkTtsLm lm, BiCodecDecoder bicodec,
        SparkTtsTokenizer? tokenizer = null, IDisposable[]? retain = null)
    {
        _cfg = cfg;
        _lm = lm;
        _bicodec = bicodec;
        _tokenizer = tokenizer;
        _retain = retain ?? [];
    }

    /// <summary>Builds the pipeline from a local Spark-TTS-0.5B directory (the one holding <c>LLM/</c> and
    /// <c>BiCodec/</c>). Loads both checkpoints + the prompt tokenizer; the weight mmaps are held for the
    /// pipeline's lifetime.</summary>
    public static SparkTtsPipeline LoadFromDirectory(string rootDir, SparkTtsConfig? config = null)
    {
        SparkTtsConfig cfg = config ?? SparkTtsConfig.V0_5B;
        List<IDisposable> retain = new();

        SafeTensorsLoader llm = new();
        llm.Load(Path.Combine(rootDir, "LLM", "model.safetensors"));
        retain.Add(llm);
        SparkTtsLm lm = new(cfg);
        lm.LoadWeights(llm.GetAllTensors());

        SafeTensorsLoader bicodec = new();
        bicodec.Load(Path.Combine(rootDir, "BiCodec", "model.safetensors"));
        retain.Add(bicodec);
        BiCodecDecoder dec = new(cfg.BiCodec);
        dec.LoadWeights(bicodec.GetAllTensors());

        SparkTtsTokenizer tokenizer = SparkTtsTokenizer.FromDirectory(Path.Combine(rootDir, "LLM"));
        return new SparkTtsPipeline(cfg, lm, dec, tokenizer, retain.ToArray());
    }

    /// <summary>Downloads Spark-TTS-0.5B from HuggingFace (<c>SparkAudio/Spark-TTS-0.5B</c>) and loads it.</summary>
    public static async Task<SparkTtsPipeline> LoadAsync(string repo = "SparkAudio/Spark-TTS-0.5B",
        SparkTtsConfig? config = null, CancellationToken ct = default)
    {
        Task<string> llm = AudioModelCache.GetAsync(repo, "LLM/model.safetensors", ct: ct);
        Task<string> bicodec = AudioModelCache.GetAsync(repo, "BiCodec/model.safetensors", ct: ct);
        Task<string> vocab = AudioModelCache.GetAsync(repo, "LLM/vocab.json", ct: ct);
        Task<string> merges = AudioModelCache.GetAsync(repo, "LLM/merges.txt", ct: ct);
        Task<string> added = AudioModelCache.GetAsync(repo, "LLM/added_tokens.json", ct: ct);
        await Task.WhenAll(llm, bicodec, vocab, merges, added).ConfigureAwait(false);
        // All files land under the same cached repo dir; pass its root (parent of LLM/).
        string root = Path.GetDirectoryName(Path.GetDirectoryName(llm.Result)!)!;
        return LoadFromDirectory(root, config);
    }

    /// <summary>Builds the controllable-TTS prompt string (mirrors <c>cli/SparkTTS.py::process_prompt_control</c>):
    /// <c>&lt;|task_controllable_tts|&gt;&lt;|start_content|&gt;{text}&lt;|end_content|&gt;&lt;|start_style_label|&gt;
    /// &lt;|gender_X|&gt;&lt;|pitch_label_X|&gt;&lt;|speed_label_X|&gt;&lt;|end_style_label|&gt;</c>. Gender is
    /// <c>female</c>|<c>male</c>; pitch/speed are one of very_low|low|moderate|high|very_high.</summary>
    public static string BuildControllablePrompt(string text, string gender = "female",
        string pitch = "moderate", string speed = "moderate")
    {
        int g = gender.Equals("male", StringComparison.OrdinalIgnoreCase) ? 1 : 0;
        int p = Math.Max(0, Array.IndexOf(Levels, pitch.ToLowerInvariant()));
        int s = Math.Max(0, Array.IndexOf(Levels, speed.ToLowerInvariant()));
        return $"<|task_controllable_tts|><|start_content|>{text}<|end_content|>" +
               $"<|start_style_label|><|gender_{g}|><|pitch_label_{p}|><|speed_label_{s}|><|end_style_label|>";
    }

    /// <summary>Synthesizes a 16 kHz waveform from text + a coarse style, entirely in-engine: builds the
    /// controllable prompt, tokenizes it (Spark's Qwen2.5 BPE + added tokens), runs the LM, and decodes via
    /// BiCodec. Requires the tokenizer (use <see cref="LoadFromDirectory"/>/<see cref="LoadAsync"/>).
    /// <paramref name="greedy"/> makes generation deterministic.</summary>
    public float[] SynthesizeControllable(IBackend backend, string text, string gender = "female",
        string pitch = "moderate", string speed = "moderate", int maxTokens = 3000, int seed = 0, bool greedy = false)
    {
        ThrowIfDisposed();
        if (_tokenizer is null)
            throw new InvalidOperationException("Use LoadFromDirectory/LoadAsync to enable text synthesis.");
        int[] prompt = _tokenizer.Encode(BuildControllablePrompt(text, gender, pitch, speed));
        return Synthesize(backend, prompt, maxTokens, seed, greedy);
    }

    /// <summary>Synthesizes a 16 kHz waveform from pre-built prompt token IDs (the chat-templated prompt, or a
    /// zero-shot clone prompt carrying the reference clip's global tokens).</summary>
    public float[] Synthesize(IBackend backend, int[] promptTokenIds, int maxTokens = 3000, int seed = 0, bool greedy = false)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();

        (List<int> global, List<int> semantic) = _lm.GenerateAudioTokens(backend, promptTokenIds, maxTokens, seed, greedy);
        Logs.Info($"Spark-TTS: LM emitted {global.Count} global + {semantic.Count} semantic tokens in {sw.ElapsedMilliseconds}ms.");

        float[] audio = _bicodec.Decode(backend, global, semantic);
        sw.Stop();
        Logs.Info($"Spark-TTS synthesis complete: {audio.Length} samples ({audio.Length / (double)_cfg.SampleRate:F2}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm.Dispose();
        _bicodec.Dispose();
        foreach (IDisposable d in _retain) d.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(SparkTtsPipeline));
    }
}
