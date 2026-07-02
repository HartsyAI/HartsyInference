using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.Music;

/// <summary>YuE Stage-1 LM — a LLaMA-2-7B decoder that emits the interleaved codebook-0 stream
/// <c>[vocal_0, accomp_0, vocal_1, accomp_1, …]</c> (track-decoupled next-token prediction) from
/// lyric+genre prompt tokens. Reuses <see cref="Qwen2Model"/> (Llama, bias-off) + the shared
/// <see cref="NucleusSampler"/>; the only YuE-specific logic is the mandatory repetition penalty and
/// parsing emitted absolute IDs into the two per-track codebook-0 streams by ID range. Same shape as
/// Spark-TTS / CosyVoice's codec-token LMs.</summary>
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

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model")
        => _lm.LoadWeights(w, prefix);

    /// <summary>Generates the two codebook-0 track streams. Returns <c>(vocal, accompaniment)</c> codec
    /// indices (equal length). All YuE generation params are exposed per-call (default to the config):
    /// <paramref name="temperature"/> (1.0), <paramref name="topK"/> (50), <paramref name="topP"/> (0.93),
    /// <paramref name="repetitionPenalty"/> (1.1). Classifier-free guidance (YuE default 1.5) is applied when
    /// <paramref name="uncondTokenIds"/> is supplied (a negative/unconditional prompt) and
    /// <paramref name="guidanceScale"/> ≠ 1: at each step <c>logits = uncond + g·(cond − uncond)</c> via a parallel
    /// KV cache primed on the negative prompt and fed the same generated tokens.</summary>
    public (List<int> Vocal, List<int> Accomp) GenerateCb0(IBackend backend,
        ReadOnlySpan<int> promptTokenIds, int maxFrames = 3000, int seed = 0,
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

        int promptLen = promptTokenIds.Length;
        int maxTokens = maxFrames * 2;     // 2 interleaved tracks per frame
        int cacheCap = Math.Min(_cfg.Stage1.MaxPositionEmbeddings, promptLen + maxTokens + 8);
        // Efficient incremental decode cache (FixedKvCache): O(1) appends, no per-step prefix copy.
        using IKvCache cache = _lm.CreateDecodeCache(cacheCap);
        IKvCache? uncondCache = useCfg
            ? _lm.CreateDecodeCache(Math.Min(_cfg.Stage1.MaxPositionEmbeddings, uncondTokenIds.Length + maxTokens + 8))
            : null;

        uint rng = DeterministicRng.Seed(seed);
        List<int> vocal = new(maxFrames), accomp = new(maxFrames);
        List<int> history = new(256);
        int vocab = _cfg.Stage1.VocabSize;
        bool wantVocal = true;

        int[] prompt = promptTokenIds.ToArray();
        Tensor hidden = _lm.Forward(backend, prompt, 1, 0, cache);
        Tensor? uHidden = useCfg ? _lm.Forward(backend, uncondTokenIds.ToArray(), 1, 0, uncondCache!) : null;

        try
        {
            for (int step = 0; step < maxTokens; step++)
            {
                Tensor last = SliceLast(hidden, _cfg.Stage1.HiddenSize);
                hidden.Dispose();
                Tensor logitsT = _lm.ProjectLogits(backend, last, 1, 1);
                last.Dispose();
                Span<float> logits = new((void*)logitsT.DataPointer, vocab);

                if (useCfg)
                {
                    Tensor uLast = SliceLast(uHidden!, _cfg.Stage1.HiddenSize);
                    uHidden!.Dispose();
                    Tensor uLogitsT = _lm.ProjectLogits(backend, uLast, 1, 1);
                    uLast.Dispose();
                    float* up = (float*)uLogitsT.DataPointer;
                    for (int v = 0; v < vocab; v++) logits[v] = up[v] + g * (logits[v] - up[v]);
                    uLogitsT.Dispose();
                }

                ApplyRepetitionPenalty(logits, history, repPen);
                int next = NucleusSampler.Draw(logits, vocab, temp, tk, tp, ref rng);
                logitsT.Dispose();

                if (next == _cfg.AudioEosToken) break;
                history.Add(next);
                if (history.Count > 256) history.RemoveAt(0);

                if (wantVocal && next >= _cfg.VocalTokenBase && next < _cfg.VocalTokenBase + _cfg.CodebookSize)
                    vocal.Add(next - _cfg.VocalTokenBase);
                else if (!wantVocal && next >= _cfg.AccompTokenBase && next < _cfg.AccompTokenBase + _cfg.CodebookSize)
                    accomp.Add(next - _cfg.AccompTokenBase);
                wantVocal = !wantVocal;

                int[] step1 = [next];
                hidden = _lm.Forward(backend, step1, 1, cache.CurrentLength, cache);
                if (useCfg) uHidden = _lm.Forward(backend, step1, 1, uncondCache!.CurrentLength, uncondCache);
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
        return (vocal.GetRange(0, frames), accomp.GetRange(0, frames));
    }

    /// <summary>HF-convention repetition penalty over the trailing history (>0 divide, &lt;0 multiply).
    /// Inline (small) — the windowed variant lives in CosyVoice's SpeechSampler.</summary>
    private static void ApplyRepetitionPenalty(Span<float> logits, List<int> history, float penalty)
    {
        if (penalty == 1f) return;
        foreach (int tok in history)
            if ((uint)tok < (uint)logits.Length)
                logits[tok] = logits[tok] > 0 ? logits[tok] / penalty : logits[tok] * penalty;
    }

    public IEnumerable<Tensor> EnumerateWeights() => _lm.EnumerateWeights();

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
