using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Csm;
using HartsyInference.Audio.Models.HeartMula;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>HeartMuLa music-generation pipeline. The LM is the **already-built+verified** CSM dual-transformer
/// (<see cref="CsmModel"/>): a Llama-3B global backbone + Llama-300M depth decoder emit 8 RVQ codebooks per
/// 12.5 Hz frame, conditioned on lyrics tokens (+ a MuQ-MuLan style embedding, staged). Returns the codebook
/// grid; HeartCodec waveform decode (48 kHz flow-matching codec) is staged.</summary>
public sealed unsafe class HeartMulaPipeline : IDisposable
{
    private readonly HeartMulaConfig _cfg;
    private readonly CsmModel _lm;
    private int _disposed;

    public HeartMulaPipeline(HeartMulaConfig cfg)
    {
        _cfg = cfg;
        _lm = new CsmModel(cfg.Lm);
    }

    public int SampleRate => _cfg.Lm.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w) => _lm.LoadWeights(w);

    /// <summary>Generates the 8-codebook grid <c>[NumCodebooks, T]</c> from lyrics token ids (the audio EOS
    /// frame stops generation). HeartCodec waveform decode is the staged follow-up.</summary>
    public int[,] GenerateCodes(IBackend backend, ReadOnlySpan<int> lyricsTokens, int maxFrames, int seed = 0)
    {
        ThrowIfDisposed();
        int nb = _cfg.Lm.NumCodebooks;
        uint rng = DeterministicRng.Seed(seed);
        List<int[]> frames = new(maxFrames);

        // Context = lyrics text embeddings; then AR audio frames re-fed (CSM's re-embedded-context path).
        for (int f = 0; f < maxFrames; f++)
        {
            Tensor ctx = BuildContext(lyricsTokens, frames);
            int[] codes = _lm.GenerateFrame(backend, ctx, ref rng);
            ctx.Dispose();
            if (codes[0] == _cfg.Lm.AudioEosToken) break;
            frames.Add(codes);
        }

        int t = frames.Count;
        int[,] grid = new int[nb, t];
        for (int j = 0; j < t; j++)
            for (int i = 0; i < nb; i++) grid[i, j] = frames[j][i];
        return grid;
    }

    private Tensor BuildContext(ReadOnlySpan<int> lyrics, List<int[]> frames)
    {
        // Concatenate lyrics text embeddings + prior audio-frame embeddings (re-embedded each step).
        int h = _cfg.Lm.Backbone.HiddenSize;
        int total = lyrics.Length + frames.Count;
        Tensor ctx = new(new TensorShape(1, Math.Max(1, total), h), DType.F32);
        float* cp = (float*)ctx.DataPointer;
        int row = 0;
        for (int i = 0; i < lyrics.Length; i++)
        {
            Tensor e = _lm.EmbedText(lyrics[i]);
            Buffer.MemoryCopy((void*)e.DataPointer, cp + (long)row++ * h, h * 4, h * 4); e.Dispose();
        }
        foreach (int[] fr in frames)
        {
            Tensor e = _lm.EmbedAudioFrame(fr);
            Buffer.MemoryCopy((void*)e.DataPointer, cp + (long)row++ * h, h * 4, h * 4); e.Dispose();
        }
        return ctx;
    }

    public IEnumerable<Tensor> EnumerateWeights() => _lm.EnumerateWeights();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(HeartMulaPipeline));
    }
}
