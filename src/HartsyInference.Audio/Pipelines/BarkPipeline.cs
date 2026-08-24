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
    public float[] Synthesize(IBackend backend, int[] textTokenIds, int seed = 0, int maxSemantic = 768)
        => Synthesize(backend, textTokenIds, seed, maxSemantic, null, null);

    /// <summary>As above, overriding Bark's two sampling temperatures. Upstream <c>generate_audio</c>
    /// defaults both to 0.7 (<c>text_temp</c> = semantic stage, <c>waveform_temp</c> = coarse stage);
    /// null keeps the config value. The stages take the config per call, so a modified copy is enough —
    /// the weights live in the stage objects, not the config.</summary>
    public float[] Synthesize(IBackend backend, int[] textTokenIds, int seed, int maxSemantic,
        double? textTemp, double? waveformTemp)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();

        // ── 1. Semantic stage: merged 256-text/256-history context + infer, min_eos_p early stop ──
        BarkConfig runCfg = _cfg with
        {
            SemanticTemperature = textTemp is > 0 ? (float)textTemp.Value : _cfg.SemanticTemperature,
            CoarseTemperature = waveformTemp is > 0 ? (float)waveformTemp.Value : _cfg.CoarseTemperature,
        };
        List<int> semantic = _semantic.GenerateSemantic(backend, textTokenIds, runCfg, seed, maxSemantic);
        if (semantic.Count == 0)
        {
            throw new InvalidOperationException("Bark semantic stage produced no tokens (immediate EOS).");
        }
        Logs.Info($"Bark: {semantic.Count} semantic tokens.");

        // ── 2. Coarse stage: ratio-derived step count, 60-step sliding windows, per-book constrained sampling ──
        int[,] coarse = _coarse.GenerateCoarse(backend, semantic, runCfg, seed + 1);
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

    /// <summary>Streaming counterpart to <see cref="Synthesize"/>. Bark's fine stage is NOT autoregressive — it's
    /// an iterative full-context refinement over the whole coarse-stage output (<see cref="BarkFineModel.Refine"/>
    /// needs every coarse frame before it can start), so unlike Kyutai/CSM/Orpheus there is no way to start
    /// emitting audio while generation is still running: all three stages (semantic → coarse → fine) must finish
    /// first, exactly as in <see cref="Synthesize"/>. The streaming value here is narrower and specific to the
    /// FINAL step — chunking the EnCodec decode itself so a long utterance's audio starts arriving before the
    /// entire decode finishes, instead of waiting for one monolithic <c>_encodec.Decode</c> call.
    ///
    /// <para>EnCodec's decoder has a genuinely stateless LSTM bottleneck in every known reference implementation
    /// (Meta's two repos and HF transformers all zero-init the hidden state per call — confirmed during research,
    /// not assumed) — no fork solves this, so this uses the same chunked-independent-decode approach real-world
    /// long-form Bark generation uses (regenerate/redecode per chunk, accept a boundary transient), just applied
    /// to the codec's OWN frame axis rather than re-running all three stages per text chunk. Chunk size is
    /// deliberately coarse (~1s of audio) rather than Kyutai's ~480ms — bigger chunks mean fewer LSTM-restart
    /// boundaries for a given utterance length, which matters more here since each boundary is a real (if usually
    /// small) discontinuity, not an exact reconstruction seam.</para></summary>
    public IEnumerable<float[]> SynthesizeStreamChunks(IBackend backend, int[] textTokenIds, int seed = 0,
        int maxSemantic = 768, double? textTemp = null, double? waveformTemp = null)
    {
        ThrowIfDisposed();
        BarkConfig runCfg = _cfg with
        {
            SemanticTemperature = textTemp is > 0 ? (float)textTemp.Value : _cfg.SemanticTemperature,
            CoarseTemperature = waveformTemp is > 0 ? (float)waveformTemp.Value : _cfg.CoarseTemperature,
        };
        List<int> semantic = _semantic.GenerateSemantic(backend, textTokenIds, runCfg, seed, maxSemantic);
        if (semantic.Count == 0)
        {
            throw new InvalidOperationException("Bark semantic stage produced no tokens (immediate EOS).");
        }
        int[,] coarse = _coarse.GenerateCoarse(backend, semantic, runCfg, seed + 1);
        int[,] codes = _fine.Refine(backend, coarse, seed + 2);
        int totalFrames = codes.GetLength(1);

        const int framesPerChunk = 75; // ~1s @ EnCodec 24kHz's ~75 Hz frame rate.
        for (int start = 0; start < totalFrames; start += framesPerChunk)
        {
            int n = Math.Min(framesPerChunk, totalFrames - start);
            yield return DecodeChunk(backend, codes, start, n);
        }
    }

    /// <summary>Decodes one frame-range of the fine-stage code grid to PCM. A separate (non-iterator) method,
    /// not inlined into <see cref="SynthesizeStreamChunks"/> — C# disallows unsafe/pointer code directly inside
    /// an iterator method's body even though this class is declared <c>unsafe</c> overall.</summary>
    private Tensor DecodeChunkTensor(IBackend backend, int[,] codes, int start, int n)
    {
        Tensor codesT = new(new TensorShape(_cfg.NumCodebooks, 1, n), DType.I32);
        int* cp = (int*)codesT.DataPointer;
        for (int cb = 0; cb < _cfg.NumCodebooks; cb++)
            for (int j = 0; j < n; j++) cp[(long)cb * n + j] = codes[cb, start + j];
        Tensor audioT = _encodec.Decode(backend, codesT, batch: 1, tFrames: n);
        codesT.Dispose();
        return audioT;
    }

    private float[] DecodeChunk(IBackend backend, int[,] codes, int start, int n)
    {
        Tensor audioT = DecodeChunkTensor(backend, codes, start, n);
        int samples = (int)audioT.Shape[audioT.Shape.Rank - 1];
        float[] chunk = new float[samples];
        float* ap = (float*)audioT.DataPointer;
        for (int i = 0; i < samples; i++) chunk[i] = ap[i];
        audioT.Dispose();
        return chunk;
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
