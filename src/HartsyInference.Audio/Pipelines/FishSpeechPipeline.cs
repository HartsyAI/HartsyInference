using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.FishSpeech;
using HartsyInference.Core.Backends;
using HartsyInference.LLM.Transformer;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Fish-Speech 1.5 pipeline: the DualAR model AR-generates frames (a semantic token + 8 audio
/// codebooks each) over the text prompt, then the firefly decoder turns the codebook grid into 44.1 kHz audio.
/// Takes pre-tokenized text ids (caller tokenizes); stops at <paramref name="endToken"/> or the frame cap.</summary>
public sealed unsafe class FishSpeechPipeline : IDisposable
{
    private readonly FishSpeechConfig _cfg;
    private readonly FishSpeechDualAr _model;
    private readonly FireflyDecoder _codec;
    private int _disposed;

    public FishSpeechPipeline(FishSpeechConfig cfg)
    {
        _cfg = cfg;
        _model = new FishSpeechDualAr(cfg);
        _codec = new FireflyDecoder(cfg.NumCodebooks, cfg.CodebookSize);
    }

    public int SampleRate => _codec.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> model, IReadOnlyDictionary<string, Tensor> codec)
    {
        _model.LoadWeights(model);
        _codec.LoadWeights(codec);
    }

    /// <summary>Synthesizes audio from pre-tokenized text ids.</summary>
    public float[] Synthesize(IBackend backend, ReadOnlySpan<int> textTokens, int endToken, int maxFrames = 0,
        int seed = 0)
    {
        ThrowIfDisposed();
        int n = _cfg.NumCodebooks;
        int max = maxFrames > 0 ? maxFrames : _cfg.MaxNewTokens;
        int cap = textTokens.Length + max + 2;
        using IKvCache slow = _model.CreateSlowCache(cap);
        uint rng = DeterministicRng.Seed(seed);

        // Prefill text prompt (row-0 = text token, codebooks = 0).
        Span<int> zero = stackalloc int[n];
        int pos = 0;
        for (int i = 0; i < textTokens.Length; i++)
        {
            Tensor e = _model.EmbedFrame(textTokens[i], zero);
            _model.GenerateFrame(backend, e, pos++, slow, ref rng).Codes.AsSpan().Clear();
            e.Dispose();
        }

        List<int[]> frames = new(max);
        int prevSemantic = textTokens.Length > 0 ? textTokens[^1] : 0;
        int[] prevCodes = new int[n];
        for (int f = 0; f < max && pos < cap - 1; f++)
        {
            Tensor e = _model.EmbedFrame(prevSemantic, prevCodes);
            (int sem, int[] codes) = _model.GenerateFrame(backend, e, pos++, slow, ref rng);
            e.Dispose();
            if (sem == endToken) break;
            frames.Add(codes);
            prevSemantic = sem; prevCodes = codes;
        }

        if (frames.Count == 0) { Logs.Warning("FishSpeech: no audio frames generated."); return []; }
        int t = frames.Count;
        int[,] grid = new int[n, t];
        for (int j = 0; j < t; j++)
            for (int i = 0; i < n; i++) grid[i, j] = frames[j][i];
        return _codec.Decode(backend, grid, t);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _model.EnumerateWeights()) yield return t;
        foreach (Tensor t in _codec.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _model.Dispose(); _codec.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(FishSpeechPipeline));
    }
}
