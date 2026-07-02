using HartsyInference.Audio.Models.LanguageModels.Qwen3;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.QwenTts;

/// <summary>Qwen3-TTS MTP (multi-token-prediction) code-predictor: a small 5-layer Qwen3 depth transformer
/// (plain 1D RoPE) that, conditioned on the talker's per-frame hidden, autoregressively predicts the 15
/// acoustic codebooks (1..15) for that frame. It owns 15 separate <c>codec_embedding</c> tables (one per
/// depth step, indexed by the previously emitted codebook token) and 15 <c>lm_head</c> output projections,
/// plus a <c>small_to_mtp_projection</c> that maps the talker hidden into the MTP hidden width.
///
/// <para>Structure mirrors the Moshi depformer (<see cref="HartsyInference.Audio.Models.Kyutai.MoshiDepthTransformer"/>):
/// per depth position the input is either the projected talker hidden (step 0) or the previous codebook's
/// embedding (steps &gt; 0); the shared <see cref="Qwen3Model"/> body runs over the growing depth sequence via an
/// <see cref="IKvCache"/> from <see cref="Qwen3Model.CreateDecodeCache"/>; each step projects to its own per-codebook logits and samples.</para></summary>
public sealed unsafe class Qwen3MtpCodePredictor : IDisposable
{
    private readonly Qwen3TtsConfig _cfg;
    private readonly Qwen3Config _mtp;
    private readonly Qwen3Model _body;
    private readonly Tensor?[] _codecEmbedding;   // [MtpCodebooks] each [MtpVocab, mtpHidden]
    private readonly Tensor?[] _lmHeadW;          // [MtpCodebooks] each [MtpVocab, mtpHidden]
    private readonly Tensor?[] _lmHeadB;          // optional
    private Tensor? _smallToMtpW, _smallToMtpB;   // [mtpHidden, talkerHidden]
    private int _disposed;

    public Qwen3TtsConfig Config => _cfg;
    public int Codebooks => _cfg.MtpCodebooks;

    public Qwen3MtpCodePredictor(Qwen3TtsConfig cfg)
    {
        _cfg = cfg;
        _mtp = cfg.CodePredictor;
        _body = new Qwen3Model(_mtp);
        int n = cfg.MtpCodebooks;
        _codecEmbedding = new Tensor?[n];
        _lmHeadW = new Tensor?[n];
        _lmHeadB = new Tensor?[n];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "talker.code_predictor")
    {
        // Real layout: small_to_mtp_projection + the 15 lm_heads sit under `talker.code_predictor.*`; the body
        // and the 15 per-codebook codec_embedding tables sit under `talker.code_predictor.model.*`.
        _smallToMtpW = WhisperOps.EnsureF32(w[$"{prefix}.small_to_mtp_projection.weight"]);
        _smallToMtpB = TryGet(w, $"{prefix}.small_to_mtp_projection.bias");
        for (int k = 0; k < _cfg.MtpCodebooks; k++)
        {
            _codecEmbedding[k] = WhisperOps.EnsureF32(w[$"{prefix}.model.codec_embedding.{k}.weight"]);
            _lmHeadW[k] = WhisperOps.EnsureF32(w[$"{prefix}.lm_head.{k}.weight"]);
            _lmHeadB[k] = TryGet(w, $"{prefix}.lm_head.{k}.bias");
        }
        _body.LoadWeightsHeadless(w, $"{prefix}.model");
    }

    /// <summary>Adds the 15 acoustic codebook embeddings of a frame to a talker-hidden vector <paramref name="dst"/>
    /// (<c>[1,1,talkerHidden]</c>). Together with the talker's codebook-0 embed this forms the per-frame codec
    /// feedback the talker consumes for the next frame. The code_predictor codec_embedding rows are talker-hidden
    /// width.</summary>
    public unsafe void AddAcousticFeedback(Tensor dst, ReadOnlySpan<int> acoustic)
    {
        float* d = (float*)dst.DataPointer;
        int h = (int)dst.Shape[dst.Shape.Rank - 1];
        for (int g = 0; g < _cfg.MtpCodebooks && g < acoustic.Length; g++)
        {
            Tensor? emb = _codecEmbedding[g];
            if (emb is null) continue;
            int code = acoustic[g];
            if ((uint)code >= (uint)emb.Shape[0]) continue;
            float* e = (float*)emb.DataPointer + (long)code * h;
            for (int i = 0; i < h; i++) d[i] += e[i];
        }
    }

    /// <summary>Predicts the 15 acoustic codebook tokens (codebooks 1..15) for one frame, matching the reference
    /// code-predictor: a streaming depth transformer over positions [talker_hidden, code0_embed, code1_embed, ...],
    /// each input projected from talker width into MTP width by <c>small_to_mtp_projection</c>. Position 0 (the
    /// talker hidden) makes no prediction; position 1 (codebook-0 embedded through the talker's codec table)
    /// predicts codebook 1; each later position embeds the previous codebook through the code-predictor's own
    /// <c>codec_embedding</c> table. Greedy (argmax), as in the reference.</summary>
    public int[] PredictFrame(IBackend backend, Tensor talkerHidden, Tensor code0Embed)
    {
        int talkerH = _cfg.Talker.HiddenSize;
        int mtpHidden = _mtp.HiddenSize;
        int depth = _cfg.MtpCodebooks;
        int[] codes = new int[depth];

        using IKvCache cache = _body.CreateDecodeCache(depth + 2);

        // Position 0: projected talker hidden (builds KV, no prediction).
        Tensor cond = WhisperOps.ProjectLinear(backend, talkerHidden, _smallToMtpW!, _smallToMtpB, 1, 1, talkerH, mtpHidden);
        Tensor h0 = _body.ForwardEmbeds(backend, cond, 1, 0, cache);
        cond.Dispose(); h0.Dispose();

        // Position 1: codebook-0 embedded via the talker table, projected -> predict codebook 1.
        Tensor proj1 = WhisperOps.ProjectLinear(backend, code0Embed, _smallToMtpW!, _smallToMtpB, 1, 1, talkerH, mtpHidden);
        Tensor h1 = _body.ForwardEmbeds(backend, proj1, 1, 1, cache);
        proj1.Dispose();
        codes[0] = ArgmaxHead(backend, h1, 0, mtpHidden);
        h1.Dispose();

        // Positions 2..15: previous codebook embedded via code_predictor.codec_embedding[g-1], projected.
        for (int g = 1; g < depth; g++)
        {
            Tensor emb = CodecEmbedRow(g - 1, codes[g - 1], talkerH);
            Tensor proj = WhisperOps.ProjectLinear(backend, emb, _smallToMtpW!, _smallToMtpB, 1, 1, talkerH, mtpHidden);
            emb.Dispose();
            Tensor hidden = _body.ForwardEmbeds(backend, proj, 1, g + 1, cache);
            proj.Dispose();
            codes[g] = ArgmaxHead(backend, hidden, g, mtpHidden);
            hidden.Dispose();
        }
        return codes;
    }

    /// <summary>Projects the MTP hidden through <c>lm_head[g]</c> and returns the argmax (greedy) codebook token.</summary>
    private int ArgmaxHead(IBackend backend, Tensor hidden, int g, int mtpHidden)
    {
        Tensor logits = WhisperOps.ProjectLinear(backend, hidden, _lmHeadW[g]!, _lmHeadB[g], 1, 1, mtpHidden, _cfg.MtpVocabSize);
        float* p = (float*)logits.DataPointer;
        int best = 0; float bestV = float.NegativeInfinity;
        for (int i = 0; i < _cfg.MtpVocabSize; i++) { if (p[i] > bestV) { bestV = p[i]; best = i; } }
        logits.Dispose();
        return best;
    }

    /// <summary>Returns the code-predictor's <c>codec_embedding[g]</c> row for <paramref name="token"/> as
    /// <c>[1,1,talkerHidden]</c> (the table is talker-hidden width and is projected by the caller).</summary>
    private Tensor CodecEmbedRow(int g, int token, int width)
    {
        Tensor emb = new(new TensorShape(1, 1, width), DType.F32);
        Tensor table = _codecEmbedding[g]!;
        int vocab = (int)table.Shape[0];
        int clamped = (uint)token < (uint)vocab ? token : 0;
        float* src = (float*)table.DataPointer + (long)clamped * width;
        Buffer.MemoryCopy(src, (void*)emb.DataPointer, width * 4, width * 4);
        return emb;
    }

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_smallToMtpW is not null) yield return _smallToMtpW;
        if (_smallToMtpB is not null) yield return _smallToMtpB;
        for (int k = 0; k < _cfg.MtpCodebooks; k++)
        {
            if (_codecEmbedding[k] is not null) yield return _codecEmbedding[k]!;
            if (_lmHeadW[k] is not null) yield return _lmHeadW[k]!;
            if (_lmHeadB[k] is not null) yield return _lmHeadB[k]!;
        }
        foreach (Tensor t in _body.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _body.Dispose();
        GC.SuppressFinalize(this);
    }
}
