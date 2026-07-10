using System.Diagnostics;
using HartsyInference.Audio.Models.Bark;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Bark (Suno) text-to-audio pipeline — the three-stage GPT cascade (semantic → coarse → fine)
/// + EnCodec 24 kHz decode. Token-IDs-in: the caller BERT-tokenizes the text and applies Bark's
/// <c>TEXT_ENCODING_OFFSET</c> + the semantic-infer prefix (an optional speaker-prompt history can be
/// prepended at each stage). Reuses the shared <see cref="Models.LanguageModels.Gpt.GptBackbone"/> (all
/// three stages), the shared <see cref="Sampling.NucleusSampler"/>, and the built <see cref="EnCodec"/>
/// 24 kHz decoder.</summary>
public sealed unsafe class BarkPipeline : IDisposable
{
    private readonly BarkConfig _cfg;
    private readonly BarkCausalStage _semantic;
    private readonly BarkCausalStage _coarse;
    private readonly BarkFineModel _fine;
    private readonly EnCodec _encodec;
    private int _disposed;

    public BarkPipeline(BarkConfig cfg, BarkCausalStage semantic, BarkCausalStage coarse,
        BarkFineModel fine, EnCodec encodec)
    {
        _cfg = cfg;
        _semantic = semantic;
        _coarse = coarse;
        _fine = fine;
        _encodec = encodec;
    }

    /// <summary>Synthesizes 24 kHz audio. <paramref name="textTokenIds"/> are BERT WordPiece ids already
    /// shifted by <see cref="BarkConfig.TextEncodingOffset"/> (caller's tokenizer); the semantic-infer
    /// prefix is appended here.</summary>
    public float[] Synthesize(IBackend backend, int[] textTokenIds, int seed = 0,
        int maxSemantic = 768)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Semantic stage: merged 256-text/256-history context + infer, min_eos_p early stop ──
        List<int> semantic = _semantic.GenerateSemantic(backend, textTokenIds, _cfg, seed, maxSemantic);
        if (semantic.Count == 0)
        {
            throw new InvalidOperationException("Bark semantic stage produced no tokens (immediate EOS).");
        }
        Logs.Info($"Bark: {semantic.Count} semantic tokens.");

        // ── 2. Coarse stage: ratio-derived step count, 60-step sliding windows, per-book constrained sampling ──
        int[,] coarse = _coarse.GenerateCoarse(backend, semantic, _cfg, seed + 1);
        int t = coarse.GetLength(1);
        Logs.Info($"Bark: {t} coarse frames.");

        // ── 3. Fine stage: fill codebooks 2..7 → [8, T] ──
        int[,] codes = _fine.Refine(backend, coarse, seed + 2);

        // ── 4. EnCodec decode ──
        // Codes are [nQ, batch, T_frames] (EnCodec/RVQ.Decode reads nQ = Shape[0]); Bark is single-batch so
        // the buffer is codebook-major cp[cb*t + j]. Declaring [1, NumCodebooks, t] here would make RVQ read
        // nQ=1 and decode ONLY codebook 0 — dropping all 7 residual/fine codebooks (heavy broadband HF noise).
        Tensor codesT = new(new TensorShape(_cfg.NumCodebooks, 1, t), DType.I32);
        int* cp = (int*)codesT.DataPointer;
        for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
            for (int j = 0; j < t; j++) cp[(long)cb * t + j] = codes[cb, j];
        Tensor audioT = _encodec.Decode(backend, codesT, batch: 1, tFrames: t);
        codesT.Dispose();

        int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
        float[] audio = new float[n];
        float* ap = (float*)audioT.DataPointer;
        for (int i = 0; i < n; i++) audio[i] = ap[i];
        audioT.Dispose();

        sw.Stop();
        Logs.Info($"Bark synthesis complete: {audio.Length} samples ({audio.Length / (double)_cfg.SampleRate:F2}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _semantic.Dispose();
        _coarse.Dispose();
        _fine.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(BarkPipeline));
    }
}
