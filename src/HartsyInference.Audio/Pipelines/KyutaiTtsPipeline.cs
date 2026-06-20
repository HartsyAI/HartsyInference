using System.Diagnostics;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Kyutai;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using MimiModel = HartsyInference.Audio.Models.Codecs.Mimi.Mimi;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Kyutai TTS pipeline (Moshi RQ-Transformer): per 12.5 Hz frame the temporal Helium backbone steps
/// over the summed text + previous-audio embeddings, and the depformer autoregressively emits the 32 Mimi
/// codebooks, which the built Mimi decodes to 24 kHz audio. Takes a per-frame text token stream (caller maps
/// words → frames via the PAD/EPAD/WORD schedule; the Audio package carries no text-vocab dependency).
///
/// <para>Structural scaffold (validation-pending): the exact delayed-coordinate handling (<see cref="MoshiDelay"/>),
/// the text state machine, and speaker cross-attention conditioning are checkpoint-gated reconcile items.</para></summary>
public sealed unsafe class KyutaiTtsPipeline : IDisposable
{
    private readonly KyutaiTtsConfig _cfg;
    private readonly KyutaiTtsModel _model;
    private readonly MimiModel _codec;
    private int _disposed;

    public KyutaiTtsPipeline(KyutaiTtsConfig cfg)
    {
        _cfg = cfg;
        _model = new KyutaiTtsModel(cfg);
        _codec = new MimiModel(cfg.Codec);
    }

    public int SampleRate => _cfg.Codec.SampleRate;

    /// <summary>Loads the temporal backbone + embeddings + depformer and the Mimi codec.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> backbone, IReadOnlyDictionary<string, Tensor> mimi,
        string temporalPrefix = "transformer")
    {
        _model.LoadWeights(backbone, temporalPrefix);
        _codec.LoadWeights(mimi);
    }

    /// <summary>Synthesizes 24 kHz mono PCM. <paramref name="perFrameText"/> is the text token fed at each
    /// 12.5 Hz frame (use the model's TextPad on hold frames). Generates one frame per entry.</summary>
    public float[] Synthesize(IBackend backend, ReadOnlySpan<int> perFrameText, int seed = 0,
        Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        int nFrames = perFrameText.Length;
        int nQ = _cfg.NumCodebooks;

        int cacheCap = Math.Min(_cfg.Temporal.MaxPositionEmbeddings, nFrames + 4);
        using StreamingKvCache cache = new(_cfg.Temporal.NumHiddenLayers, batch: 1,
            _cfg.Temporal.NumKeyValueHeads, cacheCap, _cfg.Temporal.HeadDim);

        uint rng = DeterministicRng.Seed(seed);
        int[,] grid = new int[nQ, nFrames];
        Span<int> prevCodes = stackalloc int[nQ];
        for (int k = 0; k < nQ; k++) prevCodes[k] = _cfg.AudioInit;

        for (int t = 0; t < nFrames && t < cacheCap - 1; t++)
        {
            Tensor embed = _model.EmbedStep(perFrameText[t], prevCodes);
            Tensor context = _model.StepTemporal(backend, embed, posStart: t, cache);
            embed.Dispose();

            int[] codes = _model.Depth.GenerateFrame(backend, context, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
            context.Dispose();
            for (int k = 0; k < nQ; k++) { grid[k, t] = codes[k]; prevCodes[k] = codes[k]; }
            if (progress != null && (t & 31) == 0) progress(new(t, nFrames, sw.Elapsed.TotalMilliseconds));
        }

        Tensor codesT = new(new TensorShape(nQ, 1, nFrames), DType.I32);
        int* cp = (int*)codesT.DataPointer;
        for (int k = 0; k < nQ; k++)
            for (int t = 0; t < nFrames; t++)
                cp[(long)k * nFrames + t] = grid[k, t];

        Tensor audioT = _codec.Decode(backend, codesT, batch: 1, tFrames: nFrames);
        codesT.Dispose();

        int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
        float[] audio = new float[n];
        Buffer.MemoryCopy((void*)audioT.DataPointer,
            System.Runtime.CompilerServices.Unsafe.AsPointer(ref audio[0]), n * 4, n * 4);
        audioT.Dispose();
        sw.Stop();
        Logs.Info($"Kyutai TTS: {nFrames} frames → {audio.Length} samples ({audio.Length / (double)SampleRate:F1}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _model.EnumerateWeights()) yield return t;
        foreach (Tensor t in _codec.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _model.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(KyutaiTtsPipeline));
    }
}
