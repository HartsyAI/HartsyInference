using HartsyInference.Audio.Cache;
using HartsyInference.ModelHandler.TextEncoders.Bert;
using HartsyInference.Audio.Models.MeloTts;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.PyTorch;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Audio.Pipelines;

/// <summary>High-level MeloTTS (MyShell) facade: ties the English text front-end (CMUdict g2p), the
/// bert-base-uncased prosody BERT, and the <see cref="MeloTtsPipeline"/> (VITS2/Bert-VITS2) into a single
/// <see cref="SynthesizeText"/> call. Loads the MeloTTS checkpoint (<c>.pth</c>/<c>.safetensors</c>) and the BERT
/// weights + vocab, then goes text → (phoneme/tone/language ids + word2ph) → BERT features → waveform.
///
/// <para>For English the model fills its <c>ja_bert</c> (768-channel) slot with the bert-base-uncased features and
/// leaves the 1024-channel <c>bert</c> slot zero (matching <c>melo/text/english_bert.py</c>).</para></summary>
public sealed class MeloTts : IDisposable
{
    private readonly MeloTtsConfig _cfg;
    private readonly MeloTtsPipeline _pipeline;
    private readonly MeloTtsEnglishFrontend _frontend;
    private readonly MeloTtsBertProvider _bert;
    private readonly BertWordPieceTokenizer _tokenizer;
    private readonly IDisposable[] _retain;     // keep the weight mmaps alive for the model's lifetime
    private int _disposed;

    public int SampleRate => _pipeline.SampleRate;

    private MeloTts(MeloTtsConfig cfg, MeloTtsPipeline pipeline, MeloTtsEnglishFrontend frontend,
        MeloTtsBertProvider bert, BertWordPieceTokenizer tokenizer, IDisposable[] retain)
    {
        _cfg = cfg;
        _pipeline = pipeline;
        _frontend = frontend;
        _bert = bert;
        _tokenizer = tokenizer;
        _retain = retain;
    }

    /// <summary>Builds the model from local files: the MeloTTS checkpoint (<c>.pth</c> or converted
    /// <c>.safetensors</c>), the bert-base-uncased weights (<c>.bin</c>/<c>.safetensors</c>), and its
    /// <c>vocab.txt</c>.</summary>
    public static MeloTts LoadFromFiles(string meloCheckpointPath, string bertModelPath, string bertVocabPath,
        MeloTtsConfig? config = null)
    {
        MeloTtsConfig cfg = config ?? MeloTtsConfig.EnglishV3;
        List<IDisposable> retain = new();

        Dictionary<string, Tensor> meloRaw = LoadCheckpoint(meloCheckpointPath, retain);
        MeloTtsPipeline pipeline = new(cfg);
        pipeline.LoadWeights(NormalizeMeloKeys(meloRaw));

        Dictionary<string, Tensor> bertRaw = LoadCheckpoint(bertModelPath, retain);
        (Dictionary<string, Tensor> bertW, string bertPrefix) = NormalizeBertKeys(bertRaw);
        BertWordPieceTokenizer tokenizer = new(bertVocabPath, lowerCase: true);
        MeloTtsBertProvider bert = new(tokenizer, BertConfig.BaseUncased, bertW, bertPrefix);
        MeloTtsEnglishFrontend frontend = new(tokenizer);

        return new MeloTts(cfg, pipeline, frontend, bert, tokenizer, retain.ToArray());
    }

    /// <summary>Downloads the MeloTTS-English checkpoint and the bert-base-uncased prosody BERT from HuggingFace and
    /// loads them. Defaults to <c>myshell-ai/MeloTTS-English-v3</c> and <c>bert-base-uncased</c>.</summary>
    public static async Task<MeloTts> LoadAsync(string meloRepo = "myshell-ai/MeloTTS-English-v3",
        string meloFile = "checkpoint.pth", string bertRepo = "bert-base-uncased",
        string bertFile = "pytorch_model.bin", MeloTtsConfig? config = null, CancellationToken ct = default)
    {
        Task<string> melo = AudioModelCache.GetAsync(meloRepo, meloFile, ct: ct);
        Task<string> bertModel = AudioModelCache.GetAsync(bertRepo, bertFile, ct: ct);
        Task<string> bertVocab = AudioModelCache.GetAsync(bertRepo, "vocab.txt", ct: ct);
        await Task.WhenAll(melo, bertModel, bertVocab).ConfigureAwait(false);
        return LoadFromFiles(melo.Result, bertModel.Result, bertVocab.Result, config);
    }

    /// <summary>Synthesizes a waveform for <paramref name="text"/> end-to-end (text → g2p + BERT → VITS → audio).
    /// <paramref name="speakerId"/> indexes the speaker table; <paramref name="lengthScale"/>/<paramref
    /// name="noiseScale"/> override the config defaults; <paramref name="seed"/> seeds the stochastic stages.</summary>
    public float[] SynthesizeText(IBackend backend, string text, int speakerId = 0, float? lengthScale = null,
        float? noiseScale = null, int seed = 0)
    {
        ThrowIfDisposed();
        MeloTtsEnglishFrontend.Result r = _frontend.Process(text);
        int t = r.PhoneIds.Length;

        // The BERT provider must tokenize the same number-normalized text the g2p / word2ph were built from,
        // else the token count won't match word2ph (numbers expand to multiple word tokens).
        using Tensor jaBert = _bert.GetFeatures(backend, r.NormalizedText, r.Word2Ph); // [1, 768, T] — English BERT slot
        using Tensor bert = new(new TensorShape(1, _cfg.BertDim, t), DType.F32); // 1024-channel slot, allocated zeroed

        return _pipeline.Synthesize(backend, r.PhoneIds, r.ToneIds, r.LangIds, bert, jaBert, speakerId,
            lengthScale, noiseScale, seed);
    }

    // Loads a checkpoint (safetensors mmap or torch pickle) and keeps the loader alive (the tensors borrow its memory).
    private static Dictionary<string, Tensor> LoadCheckpoint(string path, List<IDisposable> retain)
    {
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            SafeTensorsLoader loader = new();
            loader.Load(path);
            retain.Add(loader);
            return loader.GetAllTensors();
        }
        PytorchPickleLoader pickle = new();
        pickle.Load(path, recursiveFlatten: true);
        retain.Add(pickle);
        return pickle.GetAllTensors();
    }

    // MeloTTS .pth nests the state dict under "model."; the posterior encoder (enc_q.*) is training-only.
    private static Dictionary<string, Tensor> NormalizeMeloKeys(Dictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> outW = new(raw.Count);
        foreach ((string k, Tensor v) in raw)
        {
            string key = k.StartsWith("model.", StringComparison.Ordinal) ? k["model.".Length..] : k;
            if (key.StartsWith("enc_q.", StringComparison.Ordinal)) continue;
            outW[key] = v;
        }
        return outW;
    }

    // Raw HF bert-base-uncased uses the old LayerNorm.gamma/.beta names and may carry a "bert." prefix (+ an MLM head
    // we ignore). Remap gamma→weight / beta→bias and detect the prefix BertModel should load under.
    private static (Dictionary<string, Tensor> Weights, string Prefix) NormalizeBertKeys(Dictionary<string, Tensor> raw)
    {
        Dictionary<string, Tensor> outW = new(raw.Count);
        foreach ((string k, Tensor v) in raw)
        {
            string key = k;
            if (key.EndsWith(".gamma", StringComparison.Ordinal)) key = key[..^".gamma".Length] + ".weight";
            else if (key.EndsWith(".beta", StringComparison.Ordinal)) key = key[..^".beta".Length] + ".bias";
            outW[key] = v;
        }
        string prefix = outW.Keys.Any(k => k.StartsWith("bert.embeddings.", StringComparison.Ordinal)) ? "bert" : "";
        return (outW, prefix);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _pipeline.Dispose();
        _tokenizer.Dispose();
        foreach (IDisposable d in _retain) d.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(MeloTts));
    }
}
