using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.Music;

/// <summary>YuE Stage-1 LM — a LLaMA-2-7B decoder that emits the interleaved codebook-0 stream <c>[vocal_0, accomp_0, vocal_1, accomp_1, …]</c> (track-decoupled next-token prediction) from lyric+genre prompt tokens.</summary>
/// <remarks>Reuses <see cref="Qwen2Model"/> (Llama, bias-off) + the shared
/// <see cref="NucleusSampler"/>; the only YuE-specific logic is the mandatory repetition penalty and
/// parsing emitted absolute IDs into the two per-track codebook-0 streams by ID range. Same shape as
/// Spark-TTS / CosyVoice's codec-token LMs.</remarks>
public sealed unsafe class YueStage1Lm : IDisposable
{
    private readonly YueConfig _cfg;
    private readonly Qwen2Model _lm;
    private int _disposed;

    public YueStage1Lm(YueConfig cfg)
    {
        _cfg = cfg;
        _lm = new Qwen2Model(cfg.Stage1);
    }

    /// <summary>Optional layer-split placement for the 7B body — pools VRAM across GPUs so Stage-1 can run at checkpoint precision instead of being quantized down to fit one card. Null = single-backend on the per-call backend. Logits/sampling follow <see cref="LlmPlacement.LastBackend"/> when set.</summary>
    public LlmPlacement? Placement { get; set; }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model")
        => _lm.LoadWeights(w, prefix);

    /// <summary>Generates the two codebook-0 track streams. Returns <c>(vocal, accompaniment)</c> codec indices (equal length).</summary>
    /// <remarks>All YuE generation params are exposed per-call (default to the config):
    /// <paramref name="temperature"/> (1.0), <paramref name="topK"/> (50), <paramref name="topP"/> (0.93),
    /// <paramref name="repetitionPenalty"/> (1.1). Classifier-free guidance (YuE default 1.5) is applied when
    /// <paramref name="uncondTokenIds"/> is supplied (a negative/unconditional prompt) and
    /// <paramref name="guidanceScale"/> ≠ 1: at each step <c>logits = uncond + g·(cond − uncond)</c> via a parallel
    /// KV cache primed on the negative prompt and fed the same generated tokens.</remarks>
    public (List<int> Vocal, List<int> Accomp) GenerateCb0(IBackend backend, ReadOnlySpan<int> promptTokenIds,
        int maxFrames = 3000, int seed = 0, float? temperature = null, int? topK = null, float? topP = null,
        float? repetitionPenalty = null, float? guidanceScale = null, ReadOnlySpan<int> uncondTokenIds = default)
    {
        (List<int> v, List<int> a, _) = GenerateSegment(backend, promptTokenIds, maxFrames, seed,
            temperature, topK, topP, repetitionPenalty, guidanceScale, uncondTokenIds);
        return (v, a);
    }

    /// <summary>Generates ONE segment's cb0 from an accumulated context (infer.py's per-segment <c>model.generate</c>).</summary>
    /// <remarks>Returns <c>(vocal, accomp)</c> codec indices PLUS <paramref name="RawTokens"/> —
    /// the exact sampled LM ids for this segment (interleaved codec tokens, no prompt, no trailing &lt;EOA&gt;) so
    /// the pipeline can build the next segment's context. Stops at &lt;EOA&gt; or after <paramref name="maxNewFrames"/>*2
    /// tokens. Per-token sampling / masking / vocal-accomp bucketing is byte-identical to the single-shot path.</remarks>
    public (List<int> Vocal, List<int> Accomp, List<int> RawTokens) GenerateSegment(IBackend backend,
        ReadOnlySpan<int> contextTokens, int maxNewFrames = 3000, int seed = 0,
        float? temperature = null, int? topK = null, float? topP = null, float? repetitionPenalty = null,
        float? guidanceScale = null, ReadOnlySpan<int> uncondTokenIds = default)
    {
        ThrowIfDisposed();
        float temp = temperature ?? _cfg.Temperature;
        int tk = topK ?? _cfg.TopK;
        float tp = topP ?? _cfg.TopP;
        float repPen = repetitionPenalty ?? _cfg.RepetitionPenalty;
        float g = guidanceScale ?? _cfg.GuidanceScale;
        bool useCfg = g != 1f && uncondTokenIds.Length > 0;

        int promptLen = contextTokens.Length;
        int maxTokens = maxNewFrames * 2;     // 2 interleaved tracks per frame
        int cacheCap = Math.Min(_cfg.Stage1.MaxPositionEmbeddings, promptLen + maxTokens + 8);
        // Efficient incremental decode cache (FixedKvCache): O(1) appends, no per-step prefix copy.
        using IKvCache cache = _lm.CreateDecodeCache(cacheCap);
        IKvCache? uncondCache = useCfg
            ? _lm.CreateDecodeCache(Math.Min(_cfg.Stage1.MaxPositionEmbeddings, uncondTokenIds.Length + maxTokens + 8))
            : null;

        uint rng = DeterministicRng.Seed(seed);
        List<int> vocal = new(maxNewFrames), accomp = new(maxNewFrames), raw = new(maxTokens);
        List<int> history = new(256);
        int vocab = _cfg.Stage1.VocabSize;
        bool wantVocal = true;

        // The layer-split path pools the body across the placement's backends; the head (logits) runs on the
        // last stage's backend, which owns lm_head + final norm (EnumerateStageWeights' contract).
        LlmPlacement? placement = Placement is { IsSingle: false } ? Placement : null;
        IBackend headBackend = placement?.LastBackend ?? backend;
        Tensor RunBody(ReadOnlySpan<int> tokens, int posStart, IKvCache kv) => placement is null
            ? _lm.Forward(backend, tokens, 1, posStart, kv) : _lm.ForwardStaged(placement, tokens, posStart, kv);

        int[] prompt = contextTokens.ToArray();
        Tensor hidden = RunBody(prompt, 0, cache);
        Tensor? uHidden = useCfg ? RunBody(uncondTokenIds.ToArray(), 0, uncondCache!) : null;

        try
        {
            for (int step = 0; step < maxTokens; step++)
            {
                Tensor last = SliceLast(hidden, _cfg.Stage1.HiddenSize);
                hidden.Dispose();
                Tensor logitsT = _lm.ProjectLogits(headBackend, last, 1, 1);
                last.Dispose();
                Span<float> logits = new((void*)logitsT.DataPointer, vocab);

                if (useCfg)
                {
                    Tensor uLast = SliceLast(uHidden!, _cfg.Stage1.HiddenSize);
                    uHidden!.Dispose();
                    Tensor uLogitsT = _lm.ProjectLogits(headBackend, uLast, 1, 1);
                    uLast.Dispose();
                    float* up = (float*)uLogitsT.DataPointer;
                    for (int v = 0; v < vocab; v++) logits[v] = up[v] + g * (logits[v] - up[v]);
                    uLogitsT.Dispose();
                }

                // Faithful Stage-1 masking (YuE infer.py BlockTokenRangeProcessor(0, 32002)): block ids
                // [0, AudioEosToken) so the model can only emit <EOA> or a codec token, never text.
                for (int v = 0; v < _cfg.AudioEosToken; v++) logits[v] = float.NegativeInfinity;
                ApplyRepetitionPenalty(logits, history, repPen);
                int next = NucleusSampler.Draw(logits, vocab, temp, tk, tp, ref rng);
                logitsT.Dispose();

                if (next == _cfg.AudioEosToken) break;
                raw.Add(next);     // accumulate the raw sampled ids (this segment's LM context tail; excludes <EOA>)
                history.Add(next);
                if (history.Count > 256) history.RemoveAt(0);

                // Both tracks share the cb0 range [VocalTokenBase, +CodebookSize); vocal/accomp is decided by
                // interleave parity (even step = vocal, odd = accomp), not by a per-track offset.
                if (next >= _cfg.VocalTokenBase && next < _cfg.VocalTokenBase + _cfg.CodebookSize)
                {
                    int code = next - _cfg.VocalTokenBase;
                    if (wantVocal) vocal.Add(code); else accomp.Add(code);
                }
                wantVocal = !wantVocal;

                int[] step1 = [next];
                hidden = RunBody(step1, cache.CurrentLength, cache);
                if (useCfg) uHidden = RunBody(step1, uncondCache!.CurrentLength, uncondCache);
                if (cache.CurrentLength >= cacheCap - 2) break;
            }
        }
        finally
        {
            hidden.Dispose();
            uHidden?.Dispose();
            uncondCache?.Dispose();
        }

        // Trim to a common frame count.
        int frames = Math.Min(vocal.Count, accomp.Count);
        return (vocal.GetRange(0, frames), accomp.GetRange(0, frames), raw);
    }

    /// <summary>HF-convention repetition penalty over the trailing history (>0 divide, &lt;0 multiply). Inline (small) — the windowed variant lives in CosyVoice's SpeechSampler.</summary>
    private static void ApplyRepetitionPenalty(Span<float> logits, List<int> history, float penalty)
    {
        if (penalty == 1f) return;
        foreach (int tok in history)
            if ((uint)tok < (uint)logits.Length)
                logits[tok] = logits[tok] > 0 ? logits[tok] / penalty : logits[tok] * penalty;
    }

    public IEnumerable<Tensor> EnumerateWeights() => _lm.EnumerateWeights();

    /// <summary>One placement stage's weight tensors (asymmetric preload for the layer-split path).</summary>
    public IEnumerable<Tensor> EnumerateStageWeights(int startLayer, int endLayer, bool isFirstStage, bool isLastStage,
        bool includeRedundantSplits = true)
        => _lm.EnumerateStageWeights(startLayer, endLayer, isFirstStage, isLastStage, includeRedundantSplits);

    private Tensor SliceLast(Tensor hidden, int h)
    {
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)hidden.DataPointer + (long)(t - 1) * h, (void*)last.DataPointer, h * 4, h * 4);
        return last;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(YueStage1Lm));
    }
}
