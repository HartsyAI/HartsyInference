using HartsyInference.Audio.Models.LanguageModels.Qwen3;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.QwenTts;

/// <summary>Qwen3-TTS MTP (multi-token-prediction) code-predictor: a small 5-layer Qwen3 depth transformer
/// (plain 1D RoPE) that, conditioned on the talker's per-frame hidden, autoregressively predicts the 15
/// acoustic codebooks (1..15) for that frame. It owns 15 separate <c>codec_embedding</c> tables (one per
/// depth step, indexed by the previously emitted codebook token) and 15 <c>lm_head</c> output projections,
/// plus a <c>small_to_mtp_projection</c> that maps the talker hidden into the MTP hidden width.
///
/// <para>Structure mirrors the Moshi depformer (<see cref="HartsyInference.Audio.Models.Kyutai.MoshiDepthTransformer"/>):
/// per depth position the input is either the projected talker hidden (step 0) or the previous codebook's
/// embedding (steps &gt; 0); the shared <see cref="Qwen3Model"/> body runs over the growing depth sequence via a
/// <see cref="StreamingKvCache"/>; each step projects to its own per-codebook logits and samples.</para></summary>
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

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "code_predictor")
    {
        _smallToMtpW = WhisperOps.EnsureF32(w[$"{prefix}.small_to_mtp_projection.weight"]);
        _smallToMtpB = TryGet(w, $"{prefix}.small_to_mtp_projection.bias");
        for (int k = 0; k < _cfg.MtpCodebooks; k++)
        {
            _codecEmbedding[k] = WhisperOps.EnsureF32(w[$"{prefix}.codec_embedding.{k}.weight"]);
            _lmHeadW[k] = WhisperOps.EnsureF32(w[$"{prefix}.lm_head.{k}.weight"]);
            _lmHeadB[k] = TryGet(w, $"{prefix}.lm_head.{k}.bias");
        }
        _body.LoadWeightsHeadless(w, $"{prefix}.model");
    }

    /// <summary>Predicts the 15 acoustic codebook tokens for one frame given the talker hidden
    /// <c>[1,1,talkerHidden]</c> and the already-sampled codebook-0 token (used as the depth-0 condition).
    /// Returns the 15 codebook tokens in order (codebooks 1..15).</summary>
    public int[] PredictFrame(IBackend backend, Tensor talkerHidden, int codebook0Token, float temperature, int topK, float topP, ref uint rng)
    {
        int mtpHidden = _mtp.HiddenSize;
        int depth = _cfg.MtpCodebooks;
        int cap = depth + 1;
        int[] codes = new int[depth];

        // Depth-0 conditioning: the talker hidden projected into the MTP width.
        Tensor cond = WhisperOps.ProjectLinear(backend, talkerHidden, _smallToMtpW!, _smallToMtpB, 1, 1, _cfg.Talker.HiddenSize, mtpHidden);

        using StreamingKvCache cache = new(_mtp.NumHiddenLayers, 1, _mtp.NumKeyValueHeads, cap, _mtp.HeadDim);
        int prevToken = codebook0Token;
        for (int k = 0; k < depth; k++)
        {
            // Depth 0 input is the projected talker hidden; later steps embed the previous codebook token.
            Tensor stepInput = k == 0 ? cond : LookupEmbedding(k, prevToken, mtpHidden);
            Tensor hidden = _body.ForwardEmbeds(backend, stepInput, 1, k, cache);
            stepInput.Dispose();

            Tensor logits = WhisperOps.ProjectLinear(backend, hidden, _lmHeadW[k]!, _lmHeadB[k], 1, 1, mtpHidden, _cfg.MtpVocabSize);
            hidden.Dispose();
            int tok = NucleusSampler.Draw(new Span<float>((void*)logits.DataPointer, _cfg.MtpVocabSize),
                _cfg.MtpVocabSize, temperature, topK, topP, ref rng);
            logits.Dispose();
            codes[k] = tok;
            prevToken = tok;
        }
        return codes;
    }

    private Tensor LookupEmbedding(int k, int token, int mtpHidden)
    {
        Tensor emb = new(new TensorShape(1, 1, mtpHidden), DType.F32);
        int vocab = _cfg.MtpVocabSize;
        int clamped = (uint)token < (uint)vocab ? token : 0;
        float* src = (float*)_codecEmbedding[k]!.DataPointer + (long)clamped * mtpHidden;
        Buffer.MemoryCopy(src, (void*)emb.DataPointer, mtpHidden * 4, mtpHidden * 4);
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
