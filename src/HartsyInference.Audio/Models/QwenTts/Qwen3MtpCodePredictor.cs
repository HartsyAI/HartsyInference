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
        // 0.6B checkpoints omit the projection entirely (talker hidden == MTP hidden == 1024 → identity);
        // only the 2048-wide 1.7B talker ships it.
        _smallToMtpW = w.TryGetValue($"{prefix}.small_to_mtp_projection.weight", out Tensor? smallToMtp)
            ? WhisperOps.EnsureF32(smallToMtp) : null;
        _smallToMtpB = TryGet(w, $"{prefix}.small_to_mtp_projection.bias");
        for (int k = 0; k < _cfg.MtpCodebooks; k++)
        {
            _codecEmbedding[k] = WhisperOps.EnsureF32(w[$"{prefix}.model.codec_embedding.{k}.weight"]);
            _lmHeadW[k] = WhisperOps.EnsureF32(w[$"{prefix}.lm_head.{k}.weight"]);
            _lmHeadB[k] = TryGet(w, $"{prefix}.lm_head.{k}.bias");
            RequireDim(_codecEmbedding[k], 0, _cfg.MtpVocabSize, $"codec_embedding.{k} rows");
            RequireDim(_codecEmbedding[k], 1, _cfg.Talker.HiddenSize, $"codec_embedding.{k} dim (talker width)");
            RequireDim(_lmHeadW[k], 0, _cfg.MtpVocabSize, $"lm_head.{k} out-dim");
            RequireDim(_lmHeadW[k], 1, _mtp.HiddenSize, $"lm_head.{k} in-dim");
        }
        if (_smallToMtpW is null)
        {
            if (_cfg.Talker.HiddenSize != _mtp.HiddenSize)
            {
                throw new InvalidOperationException(
                    $"Qwen3-TTS MTP config mismatch: no small_to_mtp_projection in the checkpoint but talker hidden {_cfg.Talker.HiddenSize} != MTP hidden {_mtp.HiddenSize}.");
            }
        }
        else
        {
            RequireDim(_smallToMtpW, 0, _mtp.HiddenSize, "small_to_mtp_projection out-dim");
            RequireDim(_smallToMtpW, 1, _cfg.Talker.HiddenSize, "small_to_mtp_projection in-dim");
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
    /// <c>codec_embedding</c> table. Sampled per the reference (subtalker_dosample=true, temp 0.9/top_k 50/top_p 1,
    /// no repetition penalty).</summary>
    public int[] PredictFrame(IBackend backend, Tensor talkerHidden, Tensor code0Embed, ref uint rng)
    {
        int talkerH = _cfg.Talker.HiddenSize;
        int mtpHidden = _mtp.HiddenSize;
        int depth = _cfg.MtpCodebooks;
        int[] codes = new int[depth];

        using IKvCache cache = _body.CreateDecodeCache(depth + 2);

        // Position 0: projected talker hidden (builds KV, no prediction).
        Tensor cond = ProjectToMtp(backend, talkerHidden, talkerH, mtpHidden);
        Tensor h0 = _body.ForwardEmbeds(backend, cond, 1, 0, cache);
        cond.Dispose(); h0.Dispose();

        // Position 1: codebook-0 embedded via the talker table, projected -> predict codebook 1.
        Tensor proj1 = ProjectToMtp(backend, code0Embed, talkerH, mtpHidden);
        Tensor h1 = _body.ForwardEmbeds(backend, proj1, 1, 1, cache);
        proj1.Dispose();
        codes[0] = SampleHead(backend, h1, 0, mtpHidden, ref rng);
        h1.Dispose();

        // Positions 2..15: previous codebook embedded via code_predictor.codec_embedding[g-1], projected.
        for (int g = 1; g < depth; g++)
        {
            Tensor emb = CodecEmbedRow(g - 1, codes[g - 1], talkerH);
            Tensor proj = ProjectToMtp(backend, emb, talkerH, mtpHidden);
            emb.Dispose();
            Tensor hidden = _body.ForwardEmbeds(backend, proj, 1, g + 1, cache);
            proj.Dispose();
            codes[g] = SampleHead(backend, hidden, g, mtpHidden, ref rng);
            hidden.Dispose();
        }
        return codes;
    }

    /// <summary>Talker-width → MTP-width projection; identity copy when the checkpoint has no projection
    /// (0.6B: both widths are 1024).</summary>
    private Tensor ProjectToMtp(IBackend backend, Tensor x, int talkerH, int mtpHidden)
        => _smallToMtpW is null
            ? x.To(x.Device)
            : WhisperOps.ProjectLinear(backend, x, _smallToMtpW, _smallToMtpB, 1, 1, talkerH, mtpHidden);

    /// <summary>Projects the MTP hidden through <c>lm_head[g]</c> and samples a codebook token with the
    /// sub-talker params (greedy falls out of <c>SubTalkerTopK=1</c>).</summary>
    private int SampleHead(IBackend backend, Tensor hidden, int g, int mtpHidden, ref uint rng)
    {
        Tensor logits = WhisperOps.ProjectLinear(backend, hidden, _lmHeadW[g]!, _lmHeadB[g], 1, 1, mtpHidden, _cfg.MtpVocabSize);
        int tok = NucleusSampler.Draw(new Span<float>((void*)logits.DataPointer, _cfg.MtpVocabSize),
            _cfg.MtpVocabSize, _cfg.SubTalkerTemperature, _cfg.SubTalkerTopK, _cfg.SubTalkerTopP, ref rng);
        logits.Dispose();
        return tok;
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

    private static void RequireDim(Tensor? t, int axis, int expected, string what)
    {
        if (t is not null && (int)t.Shape[axis] != expected)
        {
            throw new InvalidOperationException(
                $"Qwen3-TTS MTP config mismatch: {what} is {t.Shape[axis]} in the checkpoint but the config says {expected}.");
        }
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
