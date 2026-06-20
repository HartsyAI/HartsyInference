using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>Kyutai STT model: a headless Helium backbone (<see cref="Qwen2Model"/>) plus the single shared
/// <c>embed_tokens</c> table that holds both text rows and the 32 audio-codebook ranges. Per frame the input
/// embedding is the sum of the previous text token's row and the 32 audio codes' rows (each offset into its
/// codebook's range); the tied head projects the final hidden over the text-vocab rows only.</summary>
public sealed unsafe class KyutaiSttModel : IDisposable
{
    private readonly KyutaiSttConfig _cfg;
    private readonly Qwen2Model _backbone;
    private Tensor? _embed;
    private int _disposed;

    public KyutaiSttConfig Config => _cfg;
    public int HiddenSize => _cfg.Helium.HiddenSize;

    public KyutaiSttModel(KyutaiSttConfig cfg)
    {
        _cfg = cfg;
        _backbone = new Qwen2Model(cfg.Helium);
    }

    /// <summary>Loads the shared embedding table (<c>model.embed_tokens.embed_tokens.weight</c>) and the
    /// headless Helium body (layers + final norm).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model",
        string embedKey = "model.embed_tokens.embed_tokens.weight")
    {
        _embed = WhisperOps.EnsureF32(w[embedKey]);
        _backbone.LoadWeightsHeadless(w, prefix);
    }

    /// <summary>Builds the per-frame input embedding <c>[1,1,hidden]</c> = text-row + Σ audio-code rows.</summary>
    public Tensor EmbedFrame(int prevTextToken, ReadOnlySpan<int> audioCodes)
    {
        if (_embed is null) throw new InvalidOperationException("KyutaiSttModel weights not loaded.");
        int h = HiddenSize;
        Tensor outT = new(new TensorShape(1, 1, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        float* ep = (float*)_embed.DataPointer;
        AddRow(op, ep, prevTextToken, h);
        for (int k = 0; k < _cfg.NumCodebooks && k < audioCodes.Length; k++)
            AddRow(op, ep, _cfg.AudioOffset(k) + audioCodes[k], h);
        return outT;
    }

    /// <summary>Runs one Helium step over a prebuilt frame embedding and returns the final hidden state.</summary>
    public Tensor Step(IBackend backend, Tensor frameEmbed, int posStart, StreamingKvCache cache)
        => _backbone.ForwardEmbeds(backend, frameEmbed, batch: 1, t: 1, posStart, cache);

    /// <summary>Projects a final hidden state to text logits over the first <c>TextVocab</c> rows of the
    /// shared (tied) embedding.</summary>
    public Tensor ProjectText(IBackend backend, Tensor hidden)
    {
        if (_embed is null) throw new InvalidOperationException("KyutaiSttModel weights not loaded.");
        return WhisperOps.ProjectLinear(backend, hidden, _embed, null, 1, 1,
            _cfg.Helium.HiddenSize, _cfg.TextVocab);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embed is not null) yield return _embed;
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
    }

    private static void AddRow(float* dst, float* table, int row, int h)
    {
        float* src = table + (long)row * h;
        for (int i = 0; i < h; i++) dst[i] += src[i];
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _embed = null;
        _backbone.Dispose();
        GC.SuppressFinalize(this);
    }
}
