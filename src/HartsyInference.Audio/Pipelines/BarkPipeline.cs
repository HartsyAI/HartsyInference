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
        int maxSemantic = 768, int maxCoarse = 1536)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Semantic stage ──
        List<int> semPrompt = new(textTokenIds) { _cfg.SemanticInferToken };
        List<int> semantic = _semantic.Generate(backend, semPrompt, maxSemantic,
            _cfg.SemanticTemperature, _cfg.TopK, _cfg.TopP, _cfg.SemanticPadToken, seed);
        Logs.Info($"Bark: {semantic.Count} semantic tokens.");

        // ── 2. Coarse stage: semantic tokens (offset) + infer prefix → interleaved 2-codebook stream ──
        List<int> coarsePrompt = new(semantic.Count + 1);
        foreach (int s in semantic) coarsePrompt.Add(s);
        coarsePrompt.Add(_cfg.CoarseInferToken);
        List<int> coarseFlat = _coarse.Generate(backend, coarsePrompt, maxCoarse,
            _cfg.CoarseTemperature, _cfg.TopK, _cfg.TopP, eosToken: -1, seed + 1);

        // De-interleave [book0, book1, book0, book1, ...] → [2, T]; map back to [0, CodebookSize).
        int t = coarseFlat.Count / _cfg.NumCoarseCodebooks;
        int[,] coarse = new int[_cfg.NumCoarseCodebooks, t];
        for (int j = 0; j < t; j++)
            for (int cb = 0; cb < _cfg.NumCoarseCodebooks; cb++)
            {
                int v = coarseFlat[j * _cfg.NumCoarseCodebooks + cb] - cb * _cfg.CodebookSize;
                coarse[cb, j] = Math.Clamp(v, 0, _cfg.CodebookSize - 1);
            }
        Logs.Info($"Bark: {t} coarse frames.");

        // ── 3. Fine stage: fill codebooks 2..7 → [8, T] ──
        int[,] codes = _fine.Refine(backend, coarse);

        // ── 4. EnCodec decode ──
        Tensor codesT = new(new TensorShape(1, _cfg.NumCodebooks, t), DType.F32);
        float* cp = (float*)codesT.DataPointer;
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
