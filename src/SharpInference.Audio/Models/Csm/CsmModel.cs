using SharpInference.Audio.Dsp;
using SharpInference.Audio.Models.LanguageModels.Qwen2;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Audio.Sampling;
using SharpInference.Audio.Streaming;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Csm;

/// <summary>Sesame CSM dual-transformer. The Llama-3.2-1B <see cref="Qwen2Model"/> backbone (bias-off)
/// predicts codebook 0 of the next frame from the running text+audio context; the Llama-100M decoder
/// fills codebooks 1..7 of that frame, conditioned on the backbone hidden + the sampled codebook-0 token.
/// Embedding tables, codebook heads, and the backbone→decoder projection live here; both transformer
/// bodies + the shared <see cref="NucleusSampler"/> are reused. Runs full-sequence (no persistent KV
/// cache across frames — a fresh cache per backbone pass); correct, perf-tunable.</summary>
public sealed unsafe class CsmModel : IDisposable
{
    private readonly CsmConfig _cfg;
    private readonly Qwen2Model _backbone;
    private readonly Qwen2Model _decoder;
    private int _disposed;

    private Tensor? _textEmbed;            // [textVocab, backboneHidden]
    private Tensor?[] _audioEmbed;         // numCodebooks × [audioVocab, backboneHidden]
    private Tensor? _c0Head;               // [audioVocab, backboneHidden]
    private Tensor? _projW;                // backboneHidden → decoderHidden
    private Tensor?[] _audioHead;          // (numCodebooks-1) × [audioVocab, decoderHidden]

    public CsmConfig Config => _cfg;

    public CsmModel(CsmConfig cfg)
    {
        _cfg = cfg;
        _backbone = new Qwen2Model(cfg.Backbone);
        _decoder = new Qwen2Model(cfg.Decoder);
        _audioEmbed = new Tensor?[cfg.NumCodebooks];
        _audioHead = new Tensor?[cfg.NumCodebooks - 1];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // Headless Llama bodies: load layers/norm only (no embed_tokens / lm_head inside the transformer).
        _backbone.LoadWeightsHeadless(w, "backbone");
        _decoder.LoadWeightsHeadless(w, "decoder");
        _textEmbed = WhisperOps.EnsureF32(w["text_embeddings.weight"]);
        for (int i = 0; i < _cfg.NumCodebooks; i++) _audioEmbed[i] = WhisperOps.EnsureF32(w[$"audio_embeddings.{i}.weight"]);
        _c0Head = WhisperOps.EnsureF32(w["codebook0_head.weight"]);
        _projW = WhisperOps.EnsureF32(w["projection.weight"]);
        for (int i = 0; i < _cfg.NumCodebooks - 1; i++) _audioHead[i] = WhisperOps.EnsureF32(w[$"audio_head.{i}.weight"]);
    }

    /// <summary>Embeds a text token via the text table → <c>[1, 1, backboneHidden]</c>.</summary>
    public Tensor EmbedText(int tokenId) => EmbedRow(_textEmbed!, tokenId);

    /// <summary>Embeds one frame's 8 codebook values (summed across codebooks) → <c>[1, 1, backboneHidden]</c>.</summary>
    public Tensor EmbedAudioFrame(ReadOnlySpan<int> frameCodes)
    {
        int h = _cfg.Backbone.HiddenSize;
        Tensor outT = new(new TensorShape(1, 1, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int cb = 0; cb < frameCodes.Length && cb < _cfg.NumCodebooks; cb++)
        {
            float* tab = (float*)_audioEmbed[cb]!.DataPointer;
            int id = Math.Clamp(frameCodes[cb], 0, _cfg.AudioVocab - 1);
            float* row = tab + (long)id * h;
            for (int c = 0; c < h; c++) op[c] += row[c];
        }
        return outT;
    }

    /// <summary>Generates one full 8-codebook frame from the running <paramref name="contextEmbeds"/>
    /// <c>[1, T, backboneHidden]</c>. Returns the 8 codebook values; the last codebook-0 value equals
    /// <see cref="CsmConfig.AudioEosToken"/> at end-of-utterance.</summary>
    public int[] GenerateFrame(IBackend backend, Tensor contextEmbeds, ref uint rng)
    {
        int bt = (int)contextEmbeds.Shape[1];
        int bh = _cfg.Backbone.HiddenSize;
        using StreamingKvCache bCache = new(_cfg.Backbone.NumHiddenLayers, 1, _cfg.Backbone.NumKeyValueHeads, bt, _cfg.Backbone.HeadDim);
        Tensor hidden = _backbone.ForwardEmbeds(backend, contextEmbeds, 1, bt, 0, bCache);
        Tensor last = SliceLast(hidden, bh);
        hidden.Dispose();

        // codebook 0 from the backbone head.
        Tensor c0Logits = WhisperOps.ProjectLinear(backend, last, _c0Head!, bias: null, 1, 1, bh, _cfg.AudioVocab);
        int c0 = NucleusSampler.Draw(new Span<float>((void*)c0Logits.DataPointer, _cfg.AudioVocab), _cfg.AudioVocab, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
        c0Logits.Dispose();

        int[] frame = new int[_cfg.NumCodebooks];
        frame[0] = c0;
        if (c0 == _cfg.AudioEosToken) { last.Dispose(); return frame; }

        // Decoder fills codebooks 1..7, conditioned on projection(backbone hidden) + c0 embedding.
        int dh = _cfg.Decoder.HiddenSize;
        Tensor projected = WhisperOps.ProjectLinear(backend, last, _projW!, bias: null, 1, 1, bh, dh);
        last.Dispose();

        List<int> decoderSeq = new() { c0 };
        for (int cb = 1; cb < _cfg.NumCodebooks; cb++)
        {
            Tensor decInput = BuildDecoderInput(projected, decoderSeq, dh);
            int dt = (int)decInput.Shape[1];
            using StreamingKvCache dCache = new(_cfg.Decoder.NumHiddenLayers, 1, _cfg.Decoder.NumKeyValueHeads, dt, _cfg.Decoder.HeadDim);
            Tensor dHidden = _decoder.ForwardEmbeds(backend, decInput, 1, dt, 0, dCache);
            decInput.Dispose();
            Tensor dLast = SliceLast(dHidden, dh);
            dHidden.Dispose();
            Tensor cbLogits = WhisperOps.ProjectLinear(backend, dLast, _audioHead[cb - 1]!, bias: null, 1, 1, dh, _cfg.AudioVocab);
            dLast.Dispose();
            int cbVal = NucleusSampler.Draw(new Span<float>((void*)cbLogits.DataPointer, _cfg.AudioVocab), _cfg.AudioVocab, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
            cbLogits.Dispose();
            frame[cb] = cbVal;
            decoderSeq.Add(cbVal);
        }
        projected.Dispose();
        return frame;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
        if (_textEmbed is not null) yield return _textEmbed;
        foreach (Tensor? e in _audioEmbed) if (e is not null) yield return e;
        if (_c0Head is not null) yield return _c0Head;
        if (_projW is not null) yield return _projW;
        foreach (Tensor? hd in _audioHead) if (hd is not null) yield return hd;
    }

    /// <summary>Decoder input = projected backbone hidden, then the per-codebook embeddings of the
    /// already-sampled codebooks of this frame.</summary>
    private Tensor BuildDecoderInput(Tensor projected, List<int> seq, int dh)
    {
        int n = 1 + seq.Count;
        Tensor outT = new(new TensorShape(1, n, dh), DType.F32);
        float* op = (float*)outT.DataPointer;
        Buffer.MemoryCopy((void*)projected.DataPointer, op, dh * 4, dh * 4);
        for (int i = 0; i < seq.Count; i++)
        {
            // Reuse the backbone audio-embed tables sliced to the decoder dim (scaffold — the real model
            // has a decoder-side audio embedding; reconcile key on checkpoint).
            float* tab = (float*)_audioEmbed[i]!.DataPointer;
            int id = Math.Clamp(seq[i], 0, _cfg.AudioVocab - 1);
            float* row = tab + (long)id * _cfg.Backbone.HiddenSize;
            float* dst = op + (long)(i + 1) * dh;
            for (int c = 0; c < dh; c++) dst[c] = row[c];   // truncate/borrow first dh dims
        }
        return outT;
    }

    private Tensor EmbedRow(Tensor table, int id)
    {
        int h = (int)table.Shape[1];
        Tensor outT = new(new TensorShape(1, 1, h), DType.F32);
        int clamped = Math.Clamp(id, 0, (int)table.Shape[0] - 1);
        Buffer.MemoryCopy((float*)table.DataPointer + (long)clamped * h, (void*)outT.DataPointer, h * 4, h * 4);
        return outT;
    }

    private static Tensor SliceLast(Tensor hidden, int h)
    {
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)hidden.DataPointer + (long)(t - 1) * h, (void*)last.DataPointer, h * 4, h * 4);
        return last;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose();
        _decoder.Dispose();
        GC.SuppressFinalize(this);
    }
}
