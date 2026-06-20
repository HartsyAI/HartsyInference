using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>Kyutai TTS model: the temporal Helium backbone (<see cref="Qwen2Model"/>, headless) plus the text
/// embedding, the per-codebook audio embeddings (for feeding generated codes back into the temporal stream),
/// and the <see cref="MoshiDepthTransformer">depformer</see>. Per frame the temporal input is the sum of the
/// current text token's embedding and the (delayed) previous audio codes' embeddings; the resulting context
/// drives the depformer to emit the 32 codebooks.
///
/// <para>Structural scaffold (validation-pending). Speaker cross-attention conditioning is deferred (the
/// decoder-only backbone has no cross-attention sublayer yet) — runs unconditioned for now.</para></summary>
public sealed unsafe class KyutaiTtsModel : IDisposable
{
    private readonly KyutaiTtsConfig _cfg;
    private readonly Qwen2Model _temporal;
    private readonly MoshiDepthTransformer _depth;
    private Tensor? _textEmb;
    private readonly Tensor?[] _audioEmb;
    private int _disposed;

    public KyutaiTtsConfig Config => _cfg;
    public MoshiDepthTransformer Depth => _depth;
    public int HiddenSize => _cfg.Temporal.HiddenSize;

    public KyutaiTtsModel(KyutaiTtsConfig cfg)
    {
        _cfg = cfg;
        _temporal = new Qwen2Model(cfg.Temporal);
        _depth = new MoshiDepthTransformer(cfg.Depth);
        _audioEmb = new Tensor?[cfg.NumCodebooks];
    }

    /// <summary>Loads the temporal body (headless), the text + per-codebook audio embeddings, and the
    /// depformer. Key spellings follow the moshi layout and are reconciled on first checkpoint load.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string temporalPrefix = "transformer")
    {
        _temporal.LoadWeightsHeadless(w, temporalPrefix);
        _textEmb = WhisperOps.EnsureF32(w["text_emb.weight"]);
        for (int k = 0; k < _cfg.NumCodebooks; k++)
            _audioEmb[k] = WhisperOps.EnsureF32(w[$"emb.{k}.weight"]);
        _depth.LoadWeights(w);
    }

    /// <summary>Builds the temporal input embedding <c>[1,1,hidden]</c> = text-row + Σ (delayed) audio rows.</summary>
    public Tensor EmbedStep(int textToken, ReadOnlySpan<int> delayedCodes)
    {
        if (_textEmb is null) throw new InvalidOperationException("KyutaiTtsModel weights not loaded.");
        int h = HiddenSize;
        Tensor outT = new(new TensorShape(1, 1, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        AddRow(op, (float*)_textEmb.DataPointer, textToken, h);
        for (int k = 0; k < _cfg.NumCodebooks && k < delayedCodes.Length; k++)
            AddRow(op, (float*)_audioEmb[k]!.DataPointer, delayedCodes[k], h);
        return outT;
    }

    /// <summary>One temporal step → context hidden <c>[1,1,hidden]</c>.</summary>
    public Tensor StepTemporal(IBackend backend, Tensor embed, int posStart, StreamingKvCache cache)
        => _temporal.ForwardEmbeds(backend, embed, batch: 1, t: 1, posStart, cache);

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_textEmb is not null) yield return _textEmb;
        foreach (Tensor? t in _audioEmb) if (t is not null) yield return t;
        foreach (Tensor t in _temporal.EnumerateWeights()) yield return t;
        foreach (Tensor t in _depth.EnumerateWeights()) yield return t;
    }

    private static void AddRow(float* dst, float* table, int row, int h)
    {
        float* src = table + (long)row * h;
        for (int i = 0; i < h; i++) dst[i] += src[i];
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _textEmb = null;
        _temporal.Dispose();
        _depth.Dispose();
        GC.SuppressFinalize(this);
    }
}
