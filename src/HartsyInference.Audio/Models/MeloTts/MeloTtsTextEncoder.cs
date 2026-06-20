using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.MeloTts;

/// <summary>MeloTTS text encoder = the shared VITS encoder layers fed a summed embedding of phoneme + tone +
/// language ids plus two projected BERT feature streams: <c>x = (emb(p) + tone_emb(t) + lang_emb(l) +
/// bert_proj(bert) + ja_bert_proj(ja_bert)) · √hidden</c>. Reuses <see cref="VitsTextEncoder"/> for the
/// transformer layers + projection via its embedding-in entry point.</summary>
public sealed unsafe class MeloTtsTextEncoder
{
    private readonly MeloTtsConfig _cfg;
    private readonly int _h;
    private readonly VitsTextEncoder _core;
    private Tensor? _emb, _toneEmb, _langEmb, _bertW, _bertB, _jaBertW, _jaBertB;

    public MeloTtsTextEncoder(MeloTtsConfig cfg)
    {
        _cfg = cfg;
        _h = cfg.Core.HiddenChannels;
        _core = new VitsTextEncoder(cfg.Core);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "enc_p")
    {
        _emb = VitsWeights.Conv(w, $"{prefix}.emb");
        _toneEmb = VitsWeights.Conv(w, $"{prefix}.tone_emb");
        _langEmb = VitsWeights.Conv(w, $"{prefix}.language_emb");
        _bertW = VitsWeights.Conv(w, $"{prefix}.bert_proj"); _bertB = VitsWeights.Bias(w, $"{prefix}.bert_proj");
        _jaBertW = VitsWeights.Conv(w, $"{prefix}.ja_bert_proj"); _jaBertB = VitsWeights.Bias(w, $"{prefix}.ja_bert_proj");
        _core.LoadWeightsLayersOnly(w, prefix);
    }

    /// <summary>Encodes phonemes/tones/languages + BERT features → encoder hidden + prior (m_p, logs_p).
    /// <paramref name="bert"/> is <c>[BertDim, T]</c> and <paramref name="jaBert"/> is <c>[JaBertDim, T]</c>
    /// (channels-first); pass zero tensors when a language's BERT stream is unused.</summary>
    public (Tensor Hidden, Tensor MP, Tensor LogsP) Forward(IBackend backend, ReadOnlySpan<int> phonemes,
        ReadOnlySpan<int> tones, ReadOnlySpan<int> languages, Tensor bert, Tensor jaBert)
    {
        int t = phonemes.Length;
        float scale = MathF.Sqrt(_h);

        // bert_proj / ja_bert_proj: Conv1d(dim → hidden, k=1) over [dim, T] → [hidden, T].
        Tensor bertProj = new(new TensorShape(1, _h, t), DType.F32);
        backend.Conv1d(bertProj, bert, _bertW!, _bertB, 1, 0, 0, 1, 1);
        Tensor jaProj = new(new TensorShape(1, _h, t), DType.F32);
        backend.Conv1d(jaProj, jaBert, _jaBertW!, _jaBertB, 1, 0, 0, 1, 1);

        Tensor embed = new(new TensorShape(1, _h, t), DType.F32);
        float* xp = (float*)embed.DataPointer;
        float* ep = (float*)_emb!.DataPointer;
        float* tp = (float*)_toneEmb!.DataPointer;
        float* lp = (float*)_langEmb!.DataPointer;
        float* bp = (float*)bertProj.DataPointer;
        float* jp = (float*)jaProj.DataPointer;
        for (int j = 0; j < t; j++)
            for (int c = 0; c < _h; c++)
            {
                float sum = ep[(long)phonemes[j] * _h + c] + tp[(long)tones[j] * _h + c] + lp[(long)languages[j] * _h + c]
                    + bp[(long)c * t + j] + jp[(long)c * t + j];
                xp[(long)c * t + j] = sum * scale;
            }
        bertProj.Dispose(); jaProj.Dispose();
        return _core.ForwardFromEmbedding(backend, embed, t);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_emb, _toneEmb, _langEmb, _bertW, _bertB, _jaBertW, _jaBertB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (Tensor t in _core.EnumerateWeights()) yield return t;
    }
}
