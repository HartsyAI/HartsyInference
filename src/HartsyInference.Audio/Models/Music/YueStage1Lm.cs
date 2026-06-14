using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

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
    /// indices (equal length).</summary>
    public (List<int> Vocal, List<int> Accomp) GenerateCb0(IBackend backend,
        ReadOnlySpan<int> promptTokenIds, int maxFrames = 3000, int seed = 0)
    {
        ThrowIfDisposed();
        int promptLen = promptTokenIds.Length;
        int maxTokens = maxFrames * 2;     // 2 interleaved tracks per frame
        int cacheCap = Math.Min(_cfg.Stage1.MaxPositionEmbeddings, promptLen + maxTokens + 8);
        using StreamingKvCache cache = new(_cfg.Stage1.NumHiddenLayers, 1, _cfg.Stage1.NumKeyValueHeads, cacheCap, _cfg.Stage1.HeadDim);

        uint rng = DeterministicRng.Seed(seed);
        List<int> vocal = new(maxFrames), accomp = new(maxFrames);
        List<int> history = new(256);
        int vocab = _cfg.Stage1.VocabSize;
        bool wantVocal = true;

        int[] prompt = promptTokenIds.ToArray();
        Tensor hidden = _lm.Forward(backend, prompt, 1, 0, cache);

        for (int step = 0; step < maxTokens; step++)
        {
            int t = (int)hidden.Shape[1];
            Tensor last = SliceLast(hidden, _cfg.Stage1.HiddenSize);
            hidden.Dispose();
            Tensor logitsT = _lm.ProjectLogits(backend, last, 1, 1);
            last.Dispose();

            Span<float> logits = new((void*)logitsT.DataPointer, vocab);
            ApplyRepetitionPenalty(logits, history, _cfg.RepetitionPenalty);
            int next = NucleusSampler.Draw(logits, vocab, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
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
            if (cache.CurrentLength >= cacheCap - 2) break;
        }
        hidden.Dispose();

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
